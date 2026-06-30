import numpy as np
import pandas as pd

print("Test LUT su CSV diretto")
print("="*60)

csv_file_path = "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

print("\n1. Loading CSV...")
# Colonne: 20=ack, 33=seq, 30=len, 28=flags, 61=label
data = pd.read_csv(csv_file_path, usecols=[20, 33, 30, 28, 61])
print(f"   Rows loaded: {len(data)}")

# Clean data
print("\n2. Cleaning data...")
rows_to_delete = []
for index, row in data.iterrows():    
    try:
        for col in data.columns:
            value = row[col]
            if isinstance(value, (float, int)):
                data.at[index, col] = int(value)
            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                data.at[index, col] = int(float(value))
            else:
                rows_to_delete.append(index)
                break
    except ValueError:
        rows_to_delete.append(index)

data.drop(index=rows_to_delete, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"   Valid rows after cleaning: {len(data)}")

# Rename columns
data.columns = ['ack', 'seq', 'length', 'flags', 'label']

# Convert to numpy array
X = data[['ack', 'seq', 'length', 'flags']].values
y = data['label'].values



# Load LUTs
print("\n4. Loading LUTs...")
lut1 = pd.read_csv('LUT/REVIEWED_LUT_bit8feat2binary_1.csv')
lut2 = pd.read_csv('LUT/REVIEWED_LUT_bit8feat2binary_2.csv')
lut3 = pd.read_csv('LUT/REVIEWED_LUT_bit8feat2binary_3.csv')



# Converti in dizionari
lut1_dict = {(row['Ack'], row['Seq']): row['X_1'] for _, row in lut1.iterrows()}
lut2_dict = {(row['Len'], row['Flags']): row['X_2'] for _, row in lut2.iterrows()}
lut3_dict = {(row['X_1'], row['X_2']): row['Output'] for _, row in lut3.iterrows()}



# Initialize counters
correct_predictions = 0
total_predictions = 0
correct_class_0 = 0
total_class_0 = 0
correct_class_1 = 0
total_class_1 = 0

# Process each packet
for idx in range(len(X)):
    # Extract features
    ack = int(X[idx, 0])
    seq = int(X[idx, 1])
    length = int(X[idx, 2])
    flags = int(X[idx, 3])
    label = int(y[idx])
    
    # Extract 8 most significant bits of Ack and Seq
    ack_msb = (ack >> 24) & 0xFF
    seq_msb = (seq >> 24) & 0xFF
    
    # Estrai i 4 bit più significativi non nulli della length
    if length == 0:
        length_4bit = 0
    else:
        bit_length = length.bit_length()
        if bit_length <= 4:
            length_4bit = length
        else:
            shift = bit_length - 4
            length_4bit = (length >> shift) & 0x0F
    
    # Lookup in LUT1
    lut1_key = (ack_msb, seq_msb)
    if lut1_key in lut1_dict:
        X_1 = lut1_dict[lut1_key]
    else:
        continue
    
    # 8-bit mask for flags (length già ridotta a 4 bit)
    flags_8bit = flags & 0xFF
    
    # Lookup in LUT2
    lut2_key = (length_4bit, flags_8bit)
    if lut2_key in lut2_dict:
        X_2 = lut2_dict[lut2_key]
    else:
        continue
    
    # Lookup in LUT3
    lut3_key = (X_1, X_2)
    if lut3_key in lut3_dict:
        prediction = lut3_dict[lut3_key]
    else:
        continue
    
    # Compare with label
    total_predictions += 1
    if prediction == label:
        correct_predictions += 1
    
    # Track per class
    if label == 0:
        total_class_0 += 1
        if prediction == label:
            correct_class_0 += 1
    else:  # label == 1
        total_class_1 += 1
        if prediction == label:
            correct_class_1 += 1

# Calculate and display results
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    accuracy_class_0 = (correct_class_0 / total_class_0 * 100) if total_class_0 > 0 else 0
    accuracy_class_1 = (correct_class_1 / total_class_1 * 100) if total_class_1 > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS - DIRECT CSV")
    print(f"{'='*50}")
    print(f"Total packets processed: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_predictions - correct_predictions}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"\nAccuracy per class:")
    print(f"  Class 0 (Normal): {accuracy_class_0:.2f}% ({correct_class_0}/{total_class_0})")
    print(f"  Class 1 (Attack): {accuracy_class_1:.2f}% ({correct_class_1}/{total_class_1})")
    print(f"{'='*50}")
else:
    print("\nNo packets processed successfully!")