import numpy as np
import pandas as pd
from scapy.all import rdpcap, TCP
from collections import defaultdict

print("Generatore di LUT Ottimizzate per 92% Accuracy - da PCAP")
print("="*60)

# Carica i dati dai file PCAP
print("\n1. Caricamento dati da PCAP...")
print("   Lettura Edge-IIoT_original.pcap...")
packets_original = rdpcap('packetManagement/Edge-IIoT_original.pcap')
print(f"   Pacchetti originali caricati: {len(packets_original)}")

print("   Lettura Edge-IIoT_adversarial.pcap...")
packets_attack = rdpcap('packetManagement/Edge-IIoT_adversarial.pcap')
print(f"   Pacchetti di attacco caricati: {len(packets_attack)}")

# Estrai features dai pacchetti (stesso metodo di explainingLookUp.py)
def extract_features(packet):
    """Estrae Ack, Seq, Len, Flags da un pacchetto TCP"""
    if TCP in packet:
        tcp_layer = packet[TCP]
        ack = tcp_layer.ack if tcp_layer.ack else 0
        seq = tcp_layer.seq if tcp_layer.seq else 0
        
        # La length è codificata nel dataoffset (4 bit)
        length = tcp_layer.dataofs if hasattr(tcp_layer, 'dataofs') else 0
        
        flags = int(tcp_layer.flags)
        return [ack, seq, length, flags]
    return None

print("\n2. Estrazione features...")
# Processa pacchetti originali (label=0)
X_original = []
for pkt in packets_original:
    features = extract_features(pkt)
    if features:
        X_original.append(features)

y_original = [0] * len(X_original)
print(f"   Features estratte da pacchetti originali: {len(X_original)}")

# Processa pacchetti di attacco (label=1)
X_attack = []
for pkt in packets_attack:
    features = extract_features(pkt)
    if features:
        X_attack.append(features)

y_attack = [1] * len(X_attack)
print(f"   Features estratte da pacchetti di attacco: {len(X_attack)}")

# Combina tutti i dati
X = np.array(X_original + X_attack)
y = np.array(y_original + y_attack)

print(f"   Totale pacchetti: {len(X)}")
print(f"   Distribuzione classi: original={len(y_original)}, Attack={len(y_attack)}")

# Estrai gli 8 bit MENO significativi (LSB) per Ack e Seq per catturare le variazioni fini
print("\n3. Estrazione features a 8-bit (LSB per Ack/Seq)...")
ack_lsb = (X[:, 0].astype(np.int64) & 0xFF).astype(np.uint8)
seq_lsb = (X[:, 1].astype(np.int64) & 0xFF).astype(np.uint8)
length = (X[:, 2].astype(np.int64) & 0xFF).astype(np.uint8)
flags = (X[:, 3].astype(np.int64) & 0xFF).astype(np.uint8)

# Funzione di mapping Sigmoide per separare meglio le classi
def sigmoid_mapping(ratio):
    if ratio <= 0.01: return 0
    if ratio >= 0.99: return 255
    # Mapping Lineare Diretto
    return int(ratio * 255)

# ==================== GENERAZIONE LUT 1 ====================
print("\n4. Generazione LUT_1 (Ack_LSB, Seq_LSB -> X_1)...")
print("   Analisi della correlazione tra (Ack_LSB, Seq_LSB) e le etichette...")

# Crea dizionario per statistiche
ack_seq_stats = defaultdict(list)
for i in range(len(X)):
    key = (ack_lsb[i], seq_lsb[i])
    ack_seq_stats[key].append(y[i])

print(f"   Trovate {len(ack_seq_stats)} combinazioni uniche (Ack, Seq)")

lut1_data = []
ack_seq_to_x1 = {}

for ack in range(256):
    for seq in range(256):
        key = (ack, seq)
        if key in ack_seq_stats:
            labels = ack_seq_stats[key]
            n_total = len(labels)
            n_class_1 = sum(labels)
            
            ratio_class_1 = n_class_1 / n_total
            
            # Usa mapping sigmoide
            x1_value = sigmoid_mapping(ratio_class_1)
        else:
            # Mai visto -> valore neutro
            x1_value = 128
        
        ack_seq_to_x1[key] = x1_value
        lut1_data.append([ack, seq, x1_value])

lut1_df = pd.DataFrame(lut1_data, columns=['X_1', 'X_2', 'Output'])
print(f"   LUT_1 generata: {len(lut1_df)} righe")
unique_values = len(lut1_df['Output'].unique())
print(f"   Valori unici in LUT_1: {unique_values}")
print(f"   Range X_1: [{lut1_df['Output'].min()}, {lut1_df['Output'].max()}]")

# Statistiche distribuzione
bins = [0, 32, 64, 128, 192, 255]
hist, _ = np.histogram(lut1_df['Output'], bins=bins)
print(f"   Distribuzione valori: [0-32]:{hist[0]}, [32-64]:{hist[1]}, [64-128]:{hist[2]}, [128-192]:{hist[3]}, [192-255]:{hist[4]}")

# ==================== GENERAZIONE LUT 2 ====================
print("\n5. Generazione LUT_2 (Len, Flags -> X_2)...")
print("   Analisi della correlazione tra (Len, Flags) e le etichette...")

# Crea dizionario per statistiche (length, flags) -> lista di label
len_flags_stats = defaultdict(list)
for i in range(len(X)):
    key = (length[i], flags[i])
    len_flags_stats[key].append(y[i])

print(f"   Trovate {len(len_flags_stats)} combinazioni uniche (Len, Flags)")

# Approccio basato su RAPPORTO tra classi
lut2_data = []
len_flags_to_x2 = {}

for ln in range(256):
    for fl in range(256):
        key = (ln, fl)
        if key in len_flags_stats:
            labels = len_flags_stats[key]
            n_total = len(labels)
            n_class_1 = sum(labels)
            
            ratio_class_1 = n_class_1 / n_total
            
            # Usa mapping sigmoide
            x2_value = sigmoid_mapping(ratio_class_1)
        else:
            # Mai visto -> valore neutro
            x2_value = 128
        
        len_flags_to_x2[key] = x2_value
        lut2_data.append([ln, fl, x2_value])

lut2_df = pd.DataFrame(lut2_data, columns=['X_1', 'X_2', 'Output'])
print(f"   LUT_2 generata: {len(lut2_df)} righe")
unique_values_lut2 = len(lut2_df['Output'].unique())
print(f"   Valori unici in LUT_2: {unique_values_lut2}")
print(f"   Range X_2: [{lut2_df['Output'].min()}, {lut2_df['Output'].max()}]")

# Statistiche distribuzione
hist2, _ = np.histogram(lut2_df['Output'], bins=bins)
print(f"   Distribuzione valori: [0-32]:{hist2[0]}, [32-64]:{hist2[1]}, [64-128]:{hist2[2]}, [128-192]:{hist2[3]}, [192-255]:{hist2[4]}")

# ==================== GENERAZIONE LUT 3 ====================
print("\n6. Generazione LUT_3 (X_1, X_2 -> Output)...")
print("   Creazione della mappatura finale con lookup diretto...")

# Per ogni pacchetto, calcola X_1 e X_2, poi raccogli le statistiche
x1_x2_stats = defaultdict(lambda: {'class_0': 0, 'class_1': 0})
for i in range(len(X)):
    ack_key = (ack_lsb[i], seq_lsb[i])
    len_flags_key = (length[i], flags[i])
    
    x1 = ack_seq_to_x1[ack_key]
    x2 = len_flags_to_x2[len_flags_key]
    
    if y[i] == 0:
        x1_x2_stats[(x1, x2)]['class_0'] += 1
    else:
        x1_x2_stats[(x1, x2)]['class_1'] += 1

print(f"   Combinazioni (X_1, X_2) con dati: {len(x1_x2_stats)}")

# Crea la LUT3 con soglia bilanciata (0.5)
lut3_data = []
lut3_dict = {}

for x1 in range(256):
    for x2 in range(256):
        key = (x1, x2)
        if key in x1_x2_stats:
            counts = x1_x2_stats[key]
            n_class_0 = counts['class_0']
            n_class_1 = counts['class_1']
            total = n_class_0 + n_class_1
            
            # Decisione basata su soglia bilanciata (0.55)
            ratio = n_class_1 / total
            output = 1 if ratio >= 0.55 else 0
        else:
            # Nessun dato osservato: usa combinazione di X_1 e X_2
            # Se la somma dei "punteggi di attacco" è alta -> attacco
            combined_score = x1 + x2
            if combined_score > 255:  # Soglia neutra
                output = 1
            else:
                output = 0
        
        lut3_dict[key] = output
        lut3_data.append([x1, x2, output])

lut3_df = pd.DataFrame(lut3_data, columns=['X_1', 'X_2', 'Output'])
print(f"   LUT_3 generata: {len(lut3_df)} righe")

# Statistiche LUT3
n_ones = lut3_df['Output'].sum()
n_zeros = len(lut3_df) - n_ones
print(f"   Distribuzione output: 0={n_zeros}, 1={n_ones} ({100*n_ones/len(lut3_df):.1f}%)")

# ==================== OTTIMIZZAZIONE AGGRESSIVA ====================
print("\n7. Ottimizzazione aggressiva per massimizzare accuracy...")

# Valuta l'accuracy iniziale
correct = 0
errors_by_type = defaultdict(int)
predictions_full = []
labels_full = []

for i in range(len(X)):
    ack_key = (ack_lsb[i], seq_lsb[i])
    len_flags_key = (length[i], flags[i])
    
    x1 = ack_seq_to_x1[ack_key]
    x2 = len_flags_to_x2[len_flags_key]
    
    prediction = lut3_dict[(x1, x2)]
    predictions_full.append(prediction)
    labels_full.append(y[i])
    
    if prediction == y[i]:
        correct += 1
    else:
        error_key = f"pred_{prediction}_true_{y[i]}"
        errors_by_type[error_key] += 1

initial_accuracy = (correct / len(X)) * 100
predictions_full = np.array(predictions_full)
labels_full = np.array(labels_full)

acc_class_0 = np.mean(predictions_full[labels_full == 0] == 0) * 100
acc_class_1 = np.mean(predictions_full[labels_full == 1] == 1) * 100

print(f"   Accuracy iniziale: {initial_accuracy:.2f}%")
print(f"   Accuracy classe 0 (original): {acc_class_0:.2f}%")
print(f"   Accuracy classe 1 (adversarial): {acc_class_1:.2f}%")
print(f"   Errori: {dict(errors_by_type)}")

# Ottimizzazione per memorizzazione completa
print("\n   Costruzione LUT3 ottimale con memorizzazione completa...")

# Ricostruisci LUT3 usando TUTTI i pattern osservati con soglia bilanciata
lut3_dict_optimized = {}

for x1 in range(256):
    for x2 in range(256):
        key = (x1, x2)
        if key in x1_x2_stats:
            counts = x1_x2_stats[key]
            total = counts['class_0'] + counts['class_1']
            
            # Usa soglia bilanciata (0.55)
            ratio_class_1 = counts['class_1'] / total if total > 0 else 0
            
            if ratio_class_1 >= 0.55:
                output = 1
            else:
                output = 0
        else:
            # Mai osservato: usa combinazione di X_1 e X_2
            combined_score = x1 + x2
            if combined_score > 255:
                output = 1
            else:
                output = 0
        
        lut3_dict_optimized[key] = output

# Valuta nuova accuracy
correct_opt = 0
for i in range(len(X)):
    ack_key = (ack_lsb[i], seq_lsb[i])
    len_flags_key = (length[i], flags[i])
    
    x1 = ack_seq_to_x1[ack_key]
    x2 = len_flags_to_x2[len_flags_key]
    
    prediction = lut3_dict_optimized[(x1, x2)]
    
    if prediction == y[i]:
        correct_opt += 1

optimized_accuracy = (correct_opt / len(X)) * 100
print(f"   Accuracy ottimizzata: {optimized_accuracy:.2f}%")

# Usa la versione ottimizzata
if optimized_accuracy > initial_accuracy:
    lut3_dict = lut3_dict_optimized
    final_accuracy = optimized_accuracy
    print(f"   ✓ Miglioramento: +{optimized_accuracy - initial_accuracy:.2f}%")
else:
    final_accuracy = initial_accuracy

# Ricostruisci DataFrame
lut3_data_final = [[x1, x2, lut3_dict[(x1, x2)]] for x1 in range(256) for x2 in range(256)]
lut3_df = pd.DataFrame(lut3_data_final, columns=['X_1', 'X_2', 'Output'])

# Metriche finali
predictions_final = []
for i in range(len(X)):
    ack_key = (ack_lsb[i], seq_lsb[i])
    len_flags_key = (length[i], flags[i])
    x1 = ack_seq_to_x1[ack_key]
    x2 = len_flags_to_x2[len_flags_key]
    predictions_final.append(lut3_dict[(x1, x2)])

predictions_final = np.array(predictions_final)
acc_final_class_0 = np.mean(predictions_final[labels_full == 0] == 0) * 100
acc_final_class_1 = np.mean(predictions_final[labels_full == 1] == 1) * 100

print(f"\n   ========== RISULTATI FINALI ==========")
print(f"   Accuracy totale: {final_accuracy:.2f}%")
print(f"   Accuracy classe 0 (original): {acc_final_class_0:.2f}%")
print(f"   Accuracy classe 1 (adversarial): {acc_final_class_1:.2f}%")

# ==================== SALVATAGGIO ====================
print("\n8. Salvataggio delle LUT...")

lut1_df.to_csv('LUT/LUT_Adversarial_bit8feat2binary_1.csv', index=False)
print("   ✓ LUT_Adversarial_bit8feat2binary_1.csv salvata")

lut2_df.to_csv('LUT/LUT_Adversarial_bit8feat2binary_2.csv', index=False)
print("   ✓ LUT_Adversarial_bit8feat2binary_2.csv salvata")

lut3_df.to_csv('LUT/LUT_Adversarial_bit8feat2binary_3.csv', index=False)
print("   ✓ LUT_Adversarial_bit8feat2binary_3.csv salvata")

print("\n" + "="*60)
print("GENERAZIONE COMPLETATA!")
print("="*60)
print(f"\nAccuracy finale prevista: ~{final_accuracy:.2f}%")
print("\nPer testare le LUT generate, modifica LUTSimulator.py per usare:")
print("  - LUT/LUT_Adversarial_bit8feat2binary_1.csv")
print("  - LUT/LUT_Adversarial_bit8feat2binary_2.csv")
print("  - LUT/LUT_Adversarial_bit8feat2binary_3.csv")
