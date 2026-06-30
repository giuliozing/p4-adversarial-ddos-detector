import numpy as np
import pandas as pd
from scapy.all import rdpcap, TCP

print("Loading data from PCAP files...")

# Load PCAP files
print("\n1. Reading Edge-IIoT_normal.pcap...")
packets_normal = rdpcap('data/Edge-IIoT_original.pcap')
print(f"   Normal packets loaded: {len(packets_normal)}")

print("\n2. Reading Edge-IIoT_attack.pcap...")
packets_attack = rdpcap('data/Edge-IIoT_adversarial.pcap')
print(f"   Attack packets loaded: {len(packets_attack)}")

# Extract features from packets
print("\n3. Extracting features from packets...")

def extract_features(packet):
    """Extract Ack, Seq, Len, Flags from a TCP packet"""
    if TCP in packet:
        tcp_layer = packet[TCP]
        ack = tcp_layer.ack if tcp_layer.ack else 0
        seq = tcp_layer.seq if tcp_layer.seq else 0
        
        # Length is encoded in the data offset (4 bits)
        length = tcp_layer.dataofs if hasattr(tcp_layer, 'dataofs') else 0
        
        flags = int(tcp_layer.flags)
        return [ack, seq, length, flags]
    return None

# Process normal packets (label=0)
X_normal = []
for pkt in packets_normal:
    features = extract_features(pkt)
    if features:
        X_normal.append(features)

y_normal = [0] * len(X_normal)
print(f"   Features extracted from normal packets: {len(X_normal)}")

# Process attack packets (label=1)
X_attack = []
for pkt in packets_attack:
    features = extract_features(pkt)
    if features:
        X_attack.append(features)

y_attack = [1] * len(X_attack)
print(f"   Features extracted from attack packets: {len(X_attack)}")

# Combine all data
X = np.array(X_normal + X_attack)
y = np.array(y_normal + y_attack)

print(f"\n4. Combined data:")
print(f"   Total packets: {len(X)}")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Distribution: Normal={len(y_normal)}, Attack={len(y_attack)}")

print("\n5. Loading LUTs...")
lut1 = pd.read_csv('LUT/LUT_Adversarial_bit8feat2binary_1.csv')
lut2 = pd.read_csv('LUT/LUT_Adversarial_bit8feat2binary_2.csv')
lut3 = pd.read_csv('LUT/LUT_Adversarial_bit8feat2binary_3.csv')

print(f"LUT1 shape: {lut1.shape}")
print(f"LUT2 shape: {lut2.shape}")
print(f"LUT3 shape: {lut3.shape}")


lut1_dict = {(row['Ack'], row['Seq']): row['X_1'] for _, row in lut1.iterrows()}
lut2_dict = {(row['Len'], row['Flags']): row['X_2'] for _, row in lut2.iterrows()}
lut3_dict = {(row['X_1'], row['X_2']): row['Output'] for _, row in lut3.iterrows()}

print("LUTs successfully loaded!")

correct_predictions = 0
total_predictions = 0
correct_class_0 = 0
total_class_0 = 0
correct_class_1 = 0
total_class_1 = 0

print("\n6. Processing packets...")
for idx in range(len(X)):
    if idx % 10000 == 0:
        print(f"Processed {idx}/{len(X)} packets...")

    ack = int(X[idx, 0])
    seq = int(X[idx, 1])
    length = int(X[idx, 2])
    flags = int(X[idx, 3])
    label = int(y[idx])
    
    # MODIFIED: Use LSB (Least Significant Byte) 
    ack_lsb = ack & 0xFF
    seq_lsb = seq & 0xFF

    lut1_key = (ack_lsb, seq_lsb)
    if lut1_key in lut1_dict:
        X_1 = lut1_dict[lut1_key]
    else:
        continue
    
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
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total processed packets: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_predictions - correct_predictions}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"\nAccuracy per class:")
    print(f"  Class 0 (Normal): {accuracy_class_0:.2f}% ({correct_class_0}/{total_class_0})")
    print(f"  Class 1 (Attack): {accuracy_class_1:.2f}% ({correct_class_1}/{total_class_1})")
    print(f"{'='*50}")
else:
    print("\nNo packets processed successfully!")
