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
import csv

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


custom_objects = {'DoReFa': lq.quantizers.DoReFa, 'f1_metric': f1_metric}

model_name = "modelloTCPBinarioCompletoNonNormalizzato"
num_features = 2
bitwidth = 8 #len e ttl

model_name = model_name + str(num_features) + '.h5'
model_path = os.path.join('modelliAddestrati', model_name)
print(f"Loading model from {model_path}")
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("Model loaded successfully")
input = [[], []]
print(f"Generating LUT for {num_features} features with bitwidth {bitwidth}")
for i in range(pow(2, num_features*bitwidth)):
    for j in range(num_features):
        if(j==0):
            input[j].append((i // pow(2, bitwidth * j)) % pow(2, bitwidth)*pow(2, 16-bitwidth))
        else:
            input[j].append((i // pow(2, bitwidth * j)) % pow(2, bitwidth))

input_array = [np.array(input[0]).reshape(-1, 1), np.array(input[1]).reshape(-1, 1)]
print("Generating output for LUT")
output = model(input_array).numpy()
print(output)
output = output.reshape(-1, output.shape[-1])  # Reshape output to match input dimensions
output_max = np.argmax(output, axis=1)
lut = np.column_stack((input[0], input[1], output[:, 0], output[:, 1], output_max))
lut_file = 'LUTbit4feat2binaryNonNormalizzato.csv'
np.savetxt(lut_file, lut, delimiter=',', header='Input_0,Input_1, Output_Max', comments='', fmt='%f')

print(f"LUT saved to {lut_file}")

    