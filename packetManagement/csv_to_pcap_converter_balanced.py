import pandas as pd
import numpy as np
from scapy.all import wrpcap, IP, TCP, Ether
import random

print("CSV to PCAP Converter - BALANCED")
print("="*60)

csv_file_path = "../Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

print("\n1. Caricamento balanced dataset...")
# Carica 1000 normal (dall'inizio) + 1000 attack (dalla fine)
normal_data = pd.read_csv(csv_file_path, usecols=[20, 33, 30, 28, 61], nrows=1000)
attack_data = pd.read_csv(csv_file_path, usecols=[20, 33, 30, 28, 61], skiprows=range(1, 2218201), nrows=1000)

data = pd.concat([normal_data, attack_data], ignore_index=True)
print(f"   Righe caricate: {len(data)}")

# Pulisci
rows_to_delete = []
for index, row in data.iterrows():
    try:
        for col in data.columns:
            value = row[col]
            if not isinstance(value, (float, int)):
                if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                    data.at[index, col] = int(float(value))
                else:
                    rows_to_delete.append(index)
                    break
    except ValueError:
        rows_to_delete.append(index)

data.drop(index=rows_to_delete, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"   Righe valide: {len(data)}")

data.columns = ['ack', 'seq', 'length', 'flags', 'label']
print(f"   Normal: {len(data[data['label']==0])}, Attack: {len(data[data['label']==1])}")

print("\n2. Generazione pacchetti...")
packets_normal = []
packets_attack = []

for index, row in data.iterrows():
    ack_num = int(row['ack'])
    seq_num = int(row['seq'])
    tcp_len = int(row['length'])
    tcp_flags = int(row['flags'])
    label = int(row['label'])
    
    random.seed(index)
    src_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
    dst_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
    src_port = random.randint(1024, 65535)
    dst_port = random.randint(1, 1023) if label == 0 else random.randint(1024, 65535)
    
    flags_str = ""
    if tcp_flags & 0x01:
        flags_str += "F"
    if tcp_flags & 0x02:
        flags_str += "S"
    if tcp_flags & 0x04:
        flags_str += "R"
    if tcp_flags & 0x08:
        flags_str += "P"
    if tcp_flags & 0x10:
        flags_str += "A"
    if tcp_flags & 0x20:
        flags_str += "U"
    
    if tcp_len == 0:
        dataoffset_value = 0
    else:
        bit_length = tcp_len.bit_length()
        if bit_length <= 4:
            dataoffset_value = tcp_len
        else:
            shift = bit_length - 4
            dataoffset_value = (tcp_len >> shift) & 0x0F
    
    try:
        pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
            sport=src_port,
            dport=dst_port,
            seq=seq_num,
            ack=ack_num,
            flags=flags_str if flags_str else 0,
            dataofs=dataoffset_value
        )
        
        if label == 0:
            packets_normal.append(pkt)
        else:
            packets_attack.append(pkt)
    except Exception as e:
        print(f"   Errore pacchetto {index}: {e}")

print(f"\n3. Salvatagg io...")
print(f"   Normal: {len(packets_normal)}")
print(f"   Attack: {len(packets_attack)}")

wrpcap("Edge-IIoT_normal.pcap", packets_normal)
wrpcap("Edge-IIoT_attack.pcap", packets_attack)

print("\n✓ COMPLETATO")
