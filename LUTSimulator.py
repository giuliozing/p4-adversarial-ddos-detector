import numpy as np
import pandas as pd
import hickle as hkl

print("Caricamento dati da Edge-IIoT_Chi_Square4.hkl...")
data = hkl.load('Edge-IIoT_Chi_Square4.hkl')

X_train = data['xtrain']
X_test = data['xtest']
X_attack = data['xattack']
y_train = data['ytrain']
y_test = data['ytest']
y_attack = data['yattack']


X = np.vstack([X_train, X_test, X_attack])
y = np.concatenate([y_train, y_test, y_attack])

print(f"Totale pacchetti: {len(X)}")
print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")

print("\nCaricamento LUT...")
lut1 = pd.read_csv('LUT/LUT_Adversarial_bit8feat2binary_1.csv')
lut2 = pd.read_csv('LUT/LUT_Adversarial_bit8feat2binary_2.csv')
lut3 = pd.read_csv('LUT/LUT_Adversarial_bit8feat2binary_3.csv')

print(f"LUT1 shape: {lut1.shape}")
print(f"LUT2 shape: {lut2.shape}")
print(f"LUT3 shape: {lut3.shape}")

# Adattamento ai nomi colonne generati da LUTOptimizedGenerator.py
# LUT1: X_1=Ack, X_2=Seq -> Output=X1
lut1_dict = {(row['X_1'], row['X_2']): row['Output'] for _, row in lut1.iterrows()}
# LUT2: X_1=Len, X_2=Flags -> Output=X2
lut2_dict = {(row['X_1'], row['X_2']): row['Output'] for _, row in lut2.iterrows()}
# LUT3: X_1=X1, X_2=X2 -> Output=Prediction
lut3_dict = {(row['X_1'], row['X_2']): row['Output'] for _, row in lut3.iterrows()}

print("\nLUT caricate con successo!")

correct_predictions = 0
total_predictions = 0
correct_class_0 = 0
total_class_0 = 0
correct_class_1 = 0
total_class_1 = 0


for idx in range(len(X)):

    ack = int(X[idx, 0])
    seq = int(X[idx, 1])
    length = int(X[idx, 2])
    flags = int(X[idx, 3])
    label = int(y[idx])
    
    # MODIFICA: Revert to Original Pairing (Ack+Seq, Len+Flags) but with LSB
    # LUT1: Ack LSB, Seq LSB
    ack_lsb = ack & 0xFF
    seq_lsb = seq & 0xFF
    lut1_key = (ack_lsb, seq_lsb)
    
    if lut1_key in lut1_dict:
        X_1 = lut1_dict[lut1_key]
    else:
        continue
    
    # LUT2: Length, Flags
    length_8bit = length & 0xFF
    flags_8bit = flags & 0xFF
    lut2_key = (length_8bit, flags_8bit)
    
    if lut2_key in lut2_dict:
        X_2 = lut2_dict[lut2_key]
    else:
        continue
    
    lut3_key = (X_1, X_2)
    if lut3_key in lut3_dict:
        prediction = lut3_dict[lut3_key]
    else:
        continue

    total_predictions += 1
    if prediction == label:
        correct_predictions += 1

    if label == 0:
        total_class_0 += 1
        if prediction == label:
            correct_class_0 += 1
    else:  # label == 1
        total_class_1 += 1
        if prediction == label:
            correct_class_1 += 1
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    accuracy_class_0 = (correct_class_0 / total_class_0 * 100) if total_class_0 > 0 else 0
    accuracy_class_1 = (correct_class_1 / total_class_1 * 100) if total_class_1 > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RISULTATI FINALI")
    print(f"{'='*50}")
    print(f"Totale pacchetti processati: {total_predictions}")
    print(f"Predizioni corrette: {correct_predictions}")
    print(f"Predizioni errate: {total_predictions - correct_predictions}")
    print(f"Accuracy complessiva: {accuracy:.2f}%")
    print(f"\nAccuracy per classe:")
    print(f"  Classe 0 (Normal): {accuracy_class_0:.2f}% ({correct_class_0}/{total_class_0})")
    print(f"  Classe 1 (Attack): {accuracy_class_1:.2f}% ({correct_class_1}/{total_class_1})")
    print(f"{'='*50}")
else:
    print("\nNessun pacchetto processato con successo!")
