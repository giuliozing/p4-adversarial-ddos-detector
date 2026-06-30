import pandas as pd
import numpy as np
from scapy.all import wrpcap, IP, TCP, Ether
from scapy.layers.inet import UDP
import random

print("CSV to PCAP Converter")
print("="*60)

# Path to the input CSV file
csv_file_path = "../Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

# Leggi il CSV con le colonne necessarie
# Colonne: 20=ack, 33=seq, 30=len, 28=flags, 61=label
print("\n1. Caricamento CSV...")
data = pd.read_csv(csv_file_path, usecols=[20, 33, 30, 28, 61])
print(f"   Righe caricate: {len(data)}")

# Limita a 1000 pacchetti
print(f"   Pacchetti da convertire: {len(data)}")

# Pulisci i dati
print("\n2. Pulizia dati...")
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

# Rimuovi righe invalide
data.drop(index=rows_to_delete, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"   Righe valide dopo pulizia: {len(data)}")

# Rinomina le colonne per chiarezza
column_names = data.columns.tolist()
data.columns = ['ack', 'seq', 'length', 'flags', 'label']

print("\n3. Generazione pacchetti TCP/IP...")
packets_normal = []  # label = 0
packets_attack = []  # label = 1

for index, row in data.iterrows():
    
    # Estrai i valori
    ack_num = int(row['ack'])
    seq_num = int(row['seq'])
    tcp_len = int(row['length'])
    tcp_flags = int(row['flags'])
    label = int(row['label'])
    
    # Genera indirizzi IP casuali ma deterministici basati sull'indice
    random.seed(index)
    src_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
    dst_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
    src_port = random.randint(1024, 65535)
    dst_port = random.randint(1, 1023) if label == 0 else random.randint(1024, 65535)
    
    # Converti flags in formato Scapy
    # TCP flags: FIN=0x01, SYN=0x02, RST=0x04, PSH=0x08, ACK=0x10, URG=0x20
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
    
    # Se non ci sono flag, usa una stringa vuota
    if not flags_str:
        flags_str = ""
    
    # Usa il dataoffset per codificare la length
    # Il dataoffset è limitato a 4 bit (0-15)
    # Estrai i 4 bit più significativi non nulli della length
    if tcp_len == 0:
        dataoffset_value = 0
    else:
        # Trova la posizione del bit più significativo
        bit_length = tcp_len.bit_length()
        # Estrai i 4 bit più significativi shiftando
        if bit_length <= 4:
            dataoffset_value = tcp_len  # Se ha 4 bit o meno, usa il valore completo
        else:
            shift = bit_length - 4  # Shift necessario per ottenere i 4 MSB
            dataoffset_value = (tcp_len >> shift) & 0x0F
    
    # Costruisci il pacchetto TCP/IP
    try:
        pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
            sport=src_port,
            dport=dst_port,
            seq=seq_num,
            ack=ack_num,
            flags=flags_str if flags_str else 0,
            dataofs=dataoffset_value  # Imposta il data offset
        )
        
        # Aggiungi il pacchetto alla lista corretta in base alla label
        if label == 0:
            packets_normal.append(pkt)
        else:  # label == 1
            packets_attack.append(pkt)
    except Exception as e:
        print(f"   Errore nella creazione del pacchetto {index}: {e}")
        continue

print(f"\n4. Totale pacchetti generati:")
print(f"   Pacchetti normali (label=0): {len(packets_normal)}")
print(f"   Pacchetti di attacco (label=1): {len(packets_attack)}")
print(f"   Totale: {len(packets_normal) + len(packets_attack)}")

# Salva in due file PCAP separati
output_file_normal = "Edge-IIoT_normal.pcap"
output_file_attack = "Edge-IIoT_attack.pcap"

print(f"\n5. Salvataggio in file PCAP...")
if len(packets_normal) > 0:
    wrpcap(output_file_normal, packets_normal)
    print(f"   ✓ {output_file_normal} salvato ({len(packets_normal)} pacchetti)")
else:
    print(f"   ✗ Nessun pacchetto normale da salvare")

if len(packets_attack) > 0:
    wrpcap(output_file_attack, packets_attack)
    print(f"   ✓ {output_file_attack} salvato ({len(packets_attack)} pacchetti)")
else:
    print(f"   ✗ Nessun pacchetto di attacco da salvare")

print("\n" + "="*60)
print("CONVERSIONE COMPLETATA!")
print("="*60)
print(f"File PCAP generati:")
print(f"  - Traffico normale: {output_file_normal} ({len(packets_normal)} pacchetti)")
print(f"  - Traffico di attacco: {output_file_attack} ({len(packets_attack)} pacchetti)")
print("\nNOTA: Gli indirizzi IP e le porte sono stati generati casualmente")
print("      poiché non sono presenti nel CSV originale.")
print("      Il numero di sequenza, ACK, flags e lunghezza sono reali.")
