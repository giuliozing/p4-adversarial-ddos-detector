import tensorflow as tf
import hickle as hkl
import numpy as np
import larq as lq
import os
import sys
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import KerasClassifier
from keras import backend as K

# Importa il modulo con apply_randomization
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modelliScript'))
import new_nn_traffic_binary_randomized

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Layer personalizzati
custom_objects = {
    'DoReFa': lq.quantizers.DoReFa, 
    'f1_metric': f1_metric,
    'apply_randomization': new_nn_traffic_binary_randomized.apply_randomization
}

# Caricamento modello
model_name = input("Inserisci il nome del modello da aprire senza l'estensione: ") + '.h5'
model_path = os.path.join('modelliAddestrati', model_name)
original_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Costruzione modello wrapper che accetta (N, 4) come input
inp = tf.keras.Input(shape=(4,), dtype=tf.float32, name="wrapper_input")
split_inputs = [tf.expand_dims(inp[:, i], axis=-1) for i in range(4)]
out = original_model(split_inputs)
wrapped_model = tf.keras.Model(inputs=inp, outputs=out, name="wrapped_model")

# Patch del metodo call per accettare argomenti extra di ART
original_call = wrapped_model.call
def patched_call(inputs, training=None, mask=None, batch_size=None, verbose=None):
    # Ignora batch_size e verbose, passa solo training e mask
    return original_call(inputs, training=training, mask=mask)
wrapped_model.call = patched_call

# Inizializza il modello con una chiamata fittizia per definire output_shape
dummy_input = tf.zeros((1, 4), dtype=tf.float32)
_ = wrapped_model(dummy_input)

# KerasClassifier sul modello wrapper
classifier = KerasClassifier(
    model=wrapped_model,
    use_logits=True,
    clip_values=([0,0,0,0], [4294967295,4294967295,15,255])
)

# Caricamento dati
data = hkl.load('dataNotNormalized_pcap.hkl')
X_attack = data['xattack'].astype(np.float32)  # (N, 4)
y_attack = np.where(data['yattack'] >= 1, 1, 0).astype(np.int32)  # (N,)

print("Label di attacco caricate")

# Test con diversi valori di epsilon
epsilon_values = np.arange(0.1, 1.1, 0.1)
results = []

print("---------Boundary Attack ----------")
for eps in epsilon_values:
    print(f"\n### Testing epsilon = {eps:.1f} ###")
    
    attack = BoundaryAttack(estimator=classifier, targeted=False, epsilon=eps, verbose=True, 
                           max_iter=50, num_trial=5, sample_size=5, init_size=20)
    
    # Per debug: usare un sottoinsieme piccolo, es. [:100]
    x_test_adv = attack.generate(x=X_attack[:2000])
    
    # Predizioni e accuracy
    predictions = classifier.predict(x_test_adv)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == y_attack[:2000])
    
    results.append((eps, accuracy))
    print(f"Epsilon {eps:.1f}: Accuracy = {accuracy * 100:.2f}%")

# Stampa riepilogo finale
print("\n" + "="*50)
print("RIEPILOGO RISULTATI")
print("="*50)
for eps, acc in results:
    print(f"Epsilon {eps:.1f}: Accuracy = {acc * 100:.2f}%")
print("="*50)

