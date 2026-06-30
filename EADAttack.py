import tensorflow as tf
import hickle as hkl
import numpy as np
import larq as lq
import json
import os
from foolbox import TensorFlowModel, accuracy, Model
from foolbox.criteria import Misclassification
from foolbox.attacks import EADAttack
import eagerpy as ep
from keras.utils import to_categorical
from keras import backend as K


def forward_fn(x):
    input1 = x[:, 0]
    input2 = x[:, 1]


    
    logits = model([input1, input2])
    return ep.astensor(logits)


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


custom_objects = {'DoReFa': lq.quantizers.DoReFa, 'f1_metric': f1_metric}

model_name = input("Inserisci il nome del modello da aprire, senza il numero di features e l'estensione: ")
num_features = int(input("Inserisci il numero di features del modello: "))
model_name = model_name + str(num_features) + '.h5'
model_path = os.path.join('modelliAddestrati', model_name)

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
fmodel: Model = TensorFlowModel(forward_fn, bounds=(-1, 1))

data_path = os.path.join('data', 'datiAttacchiTCP.hkl')
data = hkl.load(data_path)
print(f"Dati caricati da: {data_path}")

X_attack = data['xattack'].astype(np.float32).squeeze(axis=1)
X_attack_1 = ep.astensor(tf.convert_to_tensor(X_attack[:, 0]))
X_attack_2 = ep.astensor(tf.convert_to_tensor(X_attack[:, 1]))




print("Dati di attacco caricati")

X_attack_combined = tf.stack([X_attack_1.raw, X_attack_2.raw], axis=1)
X_attack_tf = tf.convert_to_tensor(X_attack_combined)
X_attack_ep = ep.astensor(X_attack_tf)
print("Dati di attacco convertiti in EagerPy")


y_attack = data['yattack']
threshold = 1
y_attack = np.where(y_attack >= threshold, 1, 0)
y_attack = ep.astensor(tf.convert_to_tensor(y_attack, dtype=tf.int64))  # Conversione esplicita

print("Label di attacco caricate")
print("Testo l'accuracy del modello originale sui dati di attacco")
clean_acc = accuracy(fmodel, X_attack_ep, y_attack)
print(f"clean accuracy:  {clean_acc * 100:.1f} %")
eps = np.linspace(0.0, 1, num=11)
print("--------Provo EAD------------")
attack = EADAttack()
raw_advs, clipped_advs, success = attack(fmodel, X_attack_ep, y_attack, epsilons=eps)
# calculate and report the robust accuracy (the accuracy of the model when
# it is attacked)
robust_accuracy = 1 - success.float32().mean(axis=-1)
print("robust accuracy for perturbations with")
for eps, acc in zip(eps, robust_accuracy):
    print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")