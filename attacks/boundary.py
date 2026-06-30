import tensorflow as tf
import hickle as hkl
import numpy as np
import larq as lq
import json
import pandas as pd
import os
from foolbox import TensorFlowModel, accuracy, samples, Model
from foolbox.attacks import BoundaryAttack
import eagerpy as ep


from keras.utils import to_categorical
from keras import backend as K


# Il tuo modello originale che accetta due input: model([input1, input2])
# Adesso definiamo una funzione wrapper:
def forward_fn(x):
    # x è un tensore EagerPy; otteniamo il tensore TensorFlow sottostante

    
    # Separa i due input lungo l'asse 1:
    input1 = x[:, 0]  # Primo input di shape (batch_size, n)
    input2 = x[:, 1]  # Secondo input di shape (batch_size, n)
    input3 = x[:, 2]
    input4 = x[:, 3]
    input5 = x[:, 4]
    input6 = x[:, 5]
    input7 = x[:, 6]
    input8 = x[:, 7]
    
    
    # Passa gli input separati al modello originale
    logits = model([input1, input2, input3, input4, input5, input6, input7, input8])
    
    # Assicuriamoci di restituire un tensore EagerPy
    return ep.astensor(logits)


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Definisco il layer personalizzato DoReFa
custom_objects = {'DoReFa': lq.quantizers.DoReFa, 'f1_metric': f1_metric}

model_name = input("Inserisci il nome del modello da aprire, senza il numero di features e l'estensione: ")

num_features = int(input("Inserisci il numero di features del modello: "))
model_name = model_name + str(num_features) + '.h5'

model_path = os.path.join('modelliAddestrati', model_name)

# Carico il modello
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

fmodel: Model = TensorFlowModel(forward_fn, bounds=(-1, 1))

data_path = os.path.join('data', 'datiAttacchiTCP.hkl')

# Carico i dati
data = hkl.load(data_path)

print(f"Dati caricati da: {data_path}")

X_attack = data['xattack'].astype(np.float32).squeeze(axis=1)
print(X_attack.shape)
# Converte i dati di input in tensori di EagerPy
X_attack_1 = ep.astensor(tf.convert_to_tensor(X_attack[:, 0]))
X_attack_2 = ep.astensor(tf.convert_to_tensor(X_attack[:,1]))
X_attack_3 = ep.astensor(tf.convert_to_tensor(X_attack[:, 2]))
X_attack_4 = ep.astensor(tf.convert_to_tensor(X_attack[:,3]))
X_attack_5 = ep.astensor(tf.convert_to_tensor(X_attack[:, 4]))
X_attack_6 = ep.astensor(tf.convert_to_tensor(X_attack[:,5]))
X_attack_7 = ep.astensor(tf.convert_to_tensor(X_attack[:, 6]))
X_attack_8 = ep.astensor(tf.convert_to_tensor(X_attack[:,7]))
print("Dati di attacco caricati")
print(X_attack_1.shape)
# Combiniamo i due input lungo un nuovo asse, ottenendo un array di shape (batch_size, 2, n)
X_attack_combined = tf.stack([X_attack_1.raw, X_attack_2.raw, X_attack_3.raw, X_attack_4.raw, X_attack_5.raw, X_attack_6.raw, X_attack_7.raw, X_attack_8.raw], axis=1)

print("Dati di attacco combinati")
print(X_attack_combined.shape)
X_attack_tf = tf.convert_to_tensor(X_attack_combined)
X_attack_ep = ep.astensor(X_attack_tf)
print("Dati di attacco convertiti in EagerPy")
#Risolvere annoso problema delle liste



y_attack = data['yattack']
threshold = 1
y_attack = np.where(y_attack >= threshold, 1, 0)

y_attack = ep.astensor(tf.convert_to_tensor(y_attack))  # Converti anche y_attack
print("Label di attacco caricate")
print("Testo l'accuracy del modello originale sui dati di attacco")
# Ora dovrebbe funzionare
clean_acc = accuracy(fmodel, X_attack_ep, y_attack)

print(f"clean accuracy:  {clean_acc * 100:.1f} %")
eps = np.linspace(0.0, 1, num=11)
try:
    print("--------Provo Boundary------------")
    attack = BoundaryAttack()
    raw_advs, clipped_advs, success = attack(fmodel, X_attack_ep, y_attack, epsilons=eps)
    # Salva raw_advs e clipped_advs in file JSON
    
    raw_advs_list = [None] * len(raw_advs)
    clipped_advs_list = [None] * len(clipped_advs)
    for i in range(len(raw_advs)):
        print("Conversione numero ", i)
        raw_advs_list[i] = (raw_advs[i].numpy()).tolist()
        clipped_advs_list[i] = (clipped_advs[i].numpy()).tolist()
    
    
    
    with open('raw_advsNQ8.json', 'w') as raw_file:
        json.dump(raw_advs_list, raw_file)
    
    with open('clipped_advsNQ8.json', 'w') as clipped_file:
        json.dump(clipped_advs_list, clipped_file)
    
    print("raw_advs e clipped_advs salvati in file JSON")
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(eps, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
    
except Exception as e:
    print(e)  









