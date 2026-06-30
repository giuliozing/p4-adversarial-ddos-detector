import numpy as np
import hickle as hkl
from scapy.all import rdpcap, TCP

print("Loading data from PCAP files...")

# Load PCAP files
print("\n1. Reading Edge-IIoT_normal.pcap...")
packets_normal = rdpcap('packetManagement/Edge-IIoT_normal.pcap')
print(f"   Normal packets loaded: {len(packets_normal)}")

print("\n2. Reading Edge-IIoT_attack.pcap...")
packets_attack = rdpcap('packetManagement/Edge-IIoT_attack.pcap')
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
X_attack_all = []
for pkt in packets_attack:
    features = extract_features(pkt)
    if features:
        X_attack_all.append(features)

y_attack_all = [1] * len(X_attack_all)
print(f"   Features extracted from attack packets: {len(X_attack_all)}")

# Convert to numpy arrays
X_normal = np.array(X_normal)
y_normal = np.array(y_normal)
X_attack_all = np.array(X_attack_all)
y_attack_all = np.array(y_attack_all)

print(f"\n4. Creating dataset splits (70% train, 15% test, 15% attack)...")

# Calculate split sizes
n_normal = len(X_normal)
n_attack_all = len(X_attack_all)

# For normal data: split into train (70%) and test (15%)
# The remaining 15% will come from attack data
n_train_normal = int(0.7 * n_normal)
n_test_normal = int(0.15 * n_normal)

# For attack data: use 15% of total normal data size
n_attack = int(0.15 * n_normal)

print(f"   Total normal packets: {n_normal}")
print(f"   Total attack packets: {n_attack_all}")
print(f"   Train normal packets: {n_train_normal}")
print(f"   Test normal packets: {n_test_normal}")
print(f"   Attack packets for dataset: {n_attack}")

# Split normal data into train and test
X_train_normal = X_normal[:n_train_normal]
y_train_normal = y_normal[:n_train_normal]

X_test_normal = X_normal[n_train_normal:n_train_normal + n_test_normal]
y_test_normal = y_normal[n_train_normal:n_train_normal + n_test_normal]

# Select attack packets
X_attack = X_attack_all[:n_attack]
y_attack = y_attack_all[:n_attack]

# Create final train set (only normal packets)
X_train = X_train_normal
y_train = y_train_normal

# Create final test set (only normal packets)
X_test = X_test_normal
y_test = y_test_normal

print(f"\n5. Final dataset shapes:")
print(f"   X_train: {X_train.shape} (all normal)")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape} (all normal)")
print(f"   y_test: {y_test.shape}")
print(f"   X_attack: {X_attack.shape} (all attack)")
print(f"   y_attack: {y_attack.shape}")

print(f"\n6. Dataset distribution:")
print(f"   Train: {len(y_train)} packets ({100*len(y_train)/n_normal:.1f}%)")
print(f"   Test: {len(y_test)} packets ({100*len(y_test)/n_normal:.1f}%)")
print(f"   Attack: {len(y_attack)} packets ({100*len(y_attack)/n_normal:.1f}%)")
print(f"   Total: {len(y_train) + len(y_test) + len(y_attack)} packets")

# Save to HKL file
print("\n7. Saving dataset to HKL file...")
data = {
    'xtrain': X_train, 
    'ytrain': y_train, 
    'xtest': X_test,
    'ytest': y_test, 
    'xattack': X_attack, 
    'yattack': y_attack
}

output_file = 'dataNotNormalized_pcap.hkl'
hkl.dump(data, output_file)
print(f"   Dataset saved to: {output_file}")

print("\nDone!")
