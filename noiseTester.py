import tensorflow as tf
import hickle as hkl
import numpy as np
import larq as lq
import json
import os
from deepdefend.attacks import fgsm


from keras.utils import to_categorical
from keras import backend as K

EPS_MAX = 2
EPS_PRECISION = 10
NUM_SIMULATIONS = 30

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

data_path = os.path.join('data', 'datiAttacchiTCP.hkl')

# Carico i dati
data = hkl.load(data_path)

print(f"Dati caricati da: {data_path}")

X_attack = data['xattack'].astype(np.float32).squeeze(axis=1)

X_attack = tf.convert_to_tensor(X_attack)

X_attack_list =[]
for i in range(num_features):
    X_attack_list.append(X_attack[:,i])

y_attack = data['yattack']
threshold = 1
y_attack = np.where(y_attack >= threshold, 1, 0)
y_attack = tf.convert_to_tensor(to_categorical(y_attack))
for trial in range(NUM_SIMULATIONS):
    resultVector = {"Noise": []}

    try:
        print("--------Aggiungo rumore bianco gaussiano------------")
        random_variable = np.random.normal(loc=0.0, scale=1.0)
        for eps in range(EPS_MAX * EPS_PRECISION):
            noise_data = []
            for i in range(num_features):
                noise_data.append(X_attack_list[i] + random_variable*eps/EPS_PRECISION)
            result = model.evaluate(noise_data ,y_attack,batch_size=128, verbose=2)
            print("loss: ", result[0], " accuracy: ", result[1])
            resultVector["Noise"].append(result)
    except Exception as e:
        print(e)  

    json_name = os.path.splitext(model_name)[0] + 'Noise'+str(trial)+'.json'
    file_path = os.path.join('performance', json_name)
    # Salva il dizionario resultVector su un file JSON
    with open(file_path, 'w') as json_file:
        json.dump(resultVector, json_file, indent=4)

    print(f"Dizionario salvato su {file_path}")






