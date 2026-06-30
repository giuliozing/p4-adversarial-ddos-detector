import tensorflow as tf
import hickle as hkl
import numpy as np
import larq as lq
import os
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import KerasClassifier
from keras import backend as K

def make_consistent(adv_example):
    adv_example[0] = np.round(adv_example[0]).astype(np.int32)
    adv_example[1] = np.round(adv_example[1]).astype(np.int32)
    adv_example[2] = np.clip(np.round(adv_example[2]), 0, 15)
    adv_example[3] = np.clip(np.round(adv_example[3]), 0, 255)
    return adv_example

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Layer personalizzati
custom_objects = {'DoReFa': lq.quantizers.DoReFa, 'f1_metric': f1_metric}

# Caricamento modello

original_modelAdv = tf.keras.models.load_model("modelliAddestrati/AdvDetectorFullPrec4.h5", custom_objects=custom_objects)

# Costruzione modello wrapper che accetta (N, 4) come input
inpAdv = tf.keras.Input(shape=(4,), dtype=tf.float32, name="wrapper_input")
split_inputsAdv = [tf.expand_dims(inpAdv[:, i], axis=-1) for i in range(4)]
outAdv = original_modelAdv(split_inputsAdv)
wrapped_modelAdv = tf.keras.Model(inputs=inpAdv, outputs=outAdv, name="wrapped_modelAdv")

# KerasClassifier sul modello wrapper
classifierAdv = KerasClassifier(
    model=wrapped_modelAdv,
    use_logits=True,
    clip_values=([0,0,0,0], [4294967295,4294967295,15,255])
)


# Caricamento modello

original_modelFP = tf.keras.models.load_model("modelliAddestrati/DDoSDetectorFullPrec4.h5", custom_objects=custom_objects)

# Costruzione modello wrapper che accetta (N, 4) come input
inpFP = tf.keras.Input(shape=(4,), dtype=tf.float32, name="wrapper_input")
split_inputsFP = [tf.expand_dims(inpFP[:, i], axis=-1) for i in range(4)]
outFP = original_modelFP(split_inputsFP)
wrapped_modelFP = tf.keras.Model(inputs=inpFP, outputs=outFP, name="wrapped_modelFP")

# KerasClassifier sul modello wrapper
classifierFP = KerasClassifier(
    model=wrapped_modelFP,
    use_logits=True,
    clip_values=([0,0,0,0], [4294967295,4294967295,15,255])
)

# Caricamento modello

original_modelQ = tf.keras.models.load_model("modelliAddestrati/DDoSDetectorQuant4.h5", custom_objects=custom_objects)

# Costruzione modello wrapper che accetta (N, 4) come input
inpQ = tf.keras.Input(shape=(4,), dtype=tf.float32, name="wrapper_input")
split_inputsQ = [tf.expand_dims(inpQ[:, i], axis=-1) for i in range(4)]
outQ = original_modelQ(split_inputsQ)
wrapped_modelQ = tf.keras.Model(inputs=inpQ, outputs=outQ, name="wrapped_modelQ")

# KerasClassifier sul modello wrapper
classifierQ = KerasClassifier(
    model=wrapped_modelQ,
    use_logits=True,
    clip_values=([0,0,0,0], [4294967295,4294967295,15,255])
)

# Caricamento dei dati
datiOriginali = hkl.load("Edge-IIoT_Chi_Square.hkl")["xattack"]
etichette = hkl.load("Edge-IIoT_Chi_Square.hkl")["yattack"]
# Genera una maschera per le etichette che valgono 1
mask_labels_1 = (etichette == 1)


datiAdvFP = hkl.load("EADclipped_advs_DDoSDetectorFullPrec4.h5Bin.hkl")
datiAdvQ = hkl.load("EADclipped_advs_DDoSDetectorQuant4.h5Bin.hkl")
for i in range(len(datiAdvFP)):
    datiAdvFP[i] = datiAdvFP[i][mask_labels_1]
    datiAdvQ[i] = datiAdvQ[i][mask_labels_1]

print("Dati Originali")

# Calcolo le predizioni del modello DDoS detector
predictionsFP_Orig = classifierFP.predict(datiOriginali)
predictionsQ_Orig = classifierQ.predict(datiOriginali)

# Calcolo le predizioni del modello Adv detector
predictionsFP_Adv_Orig = classifierAdv.predict(datiOriginali)
predictionsQ_Adv_Orig = classifierAdv.predict(datiOriginali)

# Calcolo le etichette predette
predicted_labelsFP_Orig = np.argmax(predictionsFP_Orig, axis=1)
predicted_labelsQ_Orig = np.argmax(predictionsQ_Orig, axis=1)

# Calcolo le etichette predette del modello Adv
predicted_labelsFP_Adv_Orig = np.argmax(predictionsFP_Adv_Orig, axis=1)
predicted_labelsQ_Adv_Orig = np.argmax(predictionsQ_Adv_Orig, axis=1)

# Unisco le predizioni del DDoS detector con quelle dell'Adv detector
for j in range(predicted_labelsFP_Adv_Orig.shape[0]):
    predicted_labelsFP_Orig[j] = predicted_labelsFP_Orig[j] or predicted_labelsFP_Adv_Orig[j]
    predicted_labelsQ_Orig[j] = predicted_labelsQ_Orig[j] or predicted_labelsQ_Adv_Orig[j]
    
# Calcolo le accuracy del DDoS detector + Adv detector nel riconoscimento dei DDoS
accuracyFP_ddos_Orig = np.mean(predicted_labelsFP_Orig == etichette)
accuracyQ_ddos_Orig = np.mean(predicted_labelsQ_Orig == etichette)

print("FP DDoS Orig {:.2f}%, Q DDoS Orig {:.2f}%".format(accuracyFP_ddos_Orig * 100, accuracyQ_ddos_Orig * 100))


for i in range(1, 11):
    k = i/10
    print("AMPIEZZA DELLA PERTURBAZIONE ", k)
    # recupero i dati clipped
    datiAdvFP_clipped = datiAdvFP[i-1].copy()
    datiAdvQ_clipped = datiAdvQ[i-1].copy()
    '''
    #mischio con i dati originali
    datiAdvFP_clipped = np.concatenate([datiAdvFP_clipped, datiOriginali], axis=0)
    labelsFP = np.concatenate([np.ones(len(datiAdvFP_clipped) - len(datiOriginali)), etichette], axis=0)
    indicesFP = np.random.permutation(len(datiAdvFP_clipped))
    datiAdvFP_clipped = datiAdvFP_clipped[indicesFP]
    labelsFP = labelsFP[indicesFP]
    
    datiAdvQ_clipped = np.concatenate([datiAdvQ_clipped, datiOriginali], axis=0)
    labelsQ = np.concatenate([np.ones(len(datiAdvQ_clipped) - len(datiOriginali)), etichette], axis=0)
    indicesQ = np.random.permutation(len(datiAdvQ_clipped))
    datiAdvQ_clipped = datiAdvQ_clipped[indicesQ]
    labelsQ = labelsQ[indicesQ]
    '''
    # istanzio le copie conservative
    datiAdvFP_clipped_cons = datiAdvFP_clipped.copy()
    datiAdvQ_clipped_cons = datiAdvQ_clipped.copy()
    for j in range(len(datiAdvFP_clipped_cons)):
        datiAdvFP_clipped_cons[j] = make_consistent(datiAdvFP_clipped_cons[j])
        datiAdvQ_clipped_cons[j] = make_consistent(datiAdvQ_clipped_cons[j])

    # Calcolo le predizioni del modello DDoS detector
    predictionsFP = classifierFP.predict(datiAdvFP_clipped)
    predictionsQ = classifierQ.predict(datiAdvQ_clipped)
    predictionsFP_cons = classifierFP.predict(datiAdvFP_clipped_cons)
    predictionsQ_cons = classifierQ.predict(datiAdvQ_clipped_cons)

    # Calcolo le predizioni del modello Adv detector
    predictionsFP_Adv = classifierAdv.predict(datiAdvFP_clipped)
    predictionsQ_Adv = classifierAdv.predict(datiAdvQ_clipped)
    predictionsFP_cons_Adv = classifierAdv.predict(datiAdvFP_clipped_cons)
    predictionsQ_cons_Adv = classifierAdv.predict(datiAdvQ_clipped_cons)

    # Calcolo le etichette predette del modello DDoS
    predicted_labelsFP = np.argmax(predictionsFP, axis=1)
    predicted_labelsQ = np.argmax(predictionsQ, axis=1)
    predicted_labelsFP_cons = np.argmax(predictionsFP_cons, axis=1)
    predicted_labelsQ_cons = np.argmax(predictionsQ_cons, axis=1)
    
    # Calcolo le etichette predette del modello Adv
    predicted_labelsFP_Adv = np.argmax(predictionsFP_Adv, axis=1)
    predicted_labelsQ_Adv = np.argmax(predictionsQ_Adv, axis=1)
    predicted_labelsFP_cons_Adv = np.argmax(predictionsFP_cons_Adv, axis=1)
    predicted_labelsQ_cons_Adv = np.argmax(predictionsQ_cons_Adv, axis=1)


    # Calcolo le accuracy preliminari
    accuracyFP_prelim = np.mean( (predicted_labelsFP == 1))
    accuracyQ_prelim = np.mean( (predicted_labelsQ == 1))
    accuracyFP_cons_prelim = np.mean( (predicted_labelsFP_cons == 1))
    accuracyQ_cons_prelim = np.mean( (predicted_labelsQ_cons == 1))
    '''
    accuracyFP_Adv_prelim = np.mean(predicted_labelsFP_Adv == labelsFP)
    accuracyQ_Adv_prelim = np.mean(predicted_labelsQ_Adv == labelsQ)
    accuracyFP_cons_Adv_prelim = np.mean(predicted_labelsFP_cons_Adv == labelsFP)
    accuracyQ_cons_Adv_prelim = np.mean(predicted_labelsQ_cons_Adv == labelsQ)
    '''
    print("FP Cons Prelim {:.2f}%, FP Prelim {:.2f}%, Q Cons Prelim {:.2f}%, Q Prelim {:.2f}%".format(accuracyFP_cons_prelim * 100, accuracyFP_prelim * 100, accuracyQ_cons_prelim * 100, accuracyQ_prelim * 100))
    #print("FP Adv Cons Prelim {:.2f}%, FP Adv Prelim {:.2f}%, Q Adv Cons Prelim {:.2f}%, Q Adv Prelim {:.2f}%".format(accuracyFP_cons_Adv_prelim * 100, accuracyFP_Adv_prelim * 100, accuracyQ_cons_Adv_prelim * 100, accuracyQ_Adv_prelim * 100))

    # Unisco le predizioni del DDoS detector con quelle dell'Adv detector
    for j in range(predicted_labelsFP_Adv.shape[0]):
        predicted_labelsFP[j] = predicted_labelsFP[j] or predicted_labelsFP_Adv[j]
        predicted_labelsQ[j] = predicted_labelsQ[j] or predicted_labelsQ_Adv[j]
        predicted_labelsFP_cons[j] = predicted_labelsFP_cons[j] or predicted_labelsFP_cons_Adv[j]
        predicted_labelsQ_cons[j] = predicted_labelsQ_cons[j] or predicted_labelsQ_cons_Adv[j]

    # Calcolo le accuracy del DDoS detector + Adv detector nel riconoscimento dei DDoS o degli Adversarial
    accuracyFP = np.mean( (predicted_labelsFP == 1))
    accuracyQ = np.mean( (predicted_labelsQ == 1))
    accuracyFP_cons = np.mean( (predicted_labelsFP_cons == 1))
    accuracyQ_cons = np.mean( (predicted_labelsQ_cons == 1))

    # Stampo i risultati
    #print("FP Consistent SS Adv {:.2f}%, FP SS Adv {:.2f}%, Q Consistent SS Adv {:.2f}%, Q SS Adv {:.2f}%".format(accuracy_FP_cons_ss_adv * 100, accuracy_FP_ss_adv * 100, accuracy_Q_cons_ss_adv * 100, accuracy_Q_ss_adv * 100))
    print("FP Adv Cons {:.2f}%, FP Adv {:.2f}%, Q Adv Cons {:.2f}%, Q  Adv {:.2f}%".format(accuracyFP_cons * 100, accuracyFP * 100, accuracyQ_cons * 100, accuracyQ * 100))
    #print("FP SS DDoS Cons {:.2f}%, FP SS DDoS {:.2f}%, Q SS DDoS Cons {:.2f}%, Q SS DDoS {:.2f}%".format(accuracyFP_cons_ss_ddos * 100, accuracyFP_ss_ddos * 100, accuracyQ_cons_ss_ddos * 100, accuracyQ_ss_ddos * 100))
    #print("FP DDoS Cons {:.2f}%, FP DDoS {:.2f}%, Q DDoS Cons {:.2f}%, Q DDoS {:.2f}%".format(accuracyFP_cons_ddos * 100, accuracyFP_ddos * 100, accuracyQ_cons_ddos * 100, accuracyQ_ddos * 100))