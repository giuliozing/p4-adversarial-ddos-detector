from scapy.all import rdpcap, TCP, IP
import pandas as pd

print("Analisi campo TCP Length: confronto PCAP vs CSV")
print("="*70)

# Carica il file PCAP
pcap_file = "Edge-IIoTset dataset/Attack traffic/Backdoor_attack.pcap"
print(f"\n1. Lettura PCAP: {pcap_file}")

packets = rdpcap(pcap_file)
print(f"   Totale pacchetti caricati: {len(packets)}")

# Filtra solo pacchetti TCP
tcp_packets = [pkt for pkt in packets if TCP in pkt]
print(f"   Pacchetti TCP: {len(tcp_packets)}")

# Carica il CSV corrispondente
csv_file = "Edge-IIoTset dataset/Attack traffic/Backdoor_attack.csv"
print(f"\n2. Lettura CSV: {csv_file}")

# La colonna 31 (indice 30) contiene tcp.len
csv_data = pd.read_csv(csv_file, nrows=20)
print(f"   Righe CSV caricate: {len(csv_data)}")
print(f"   Colonne totali: {len(csv_data.columns)}")

# Identifica la colonna tcp.len
tcp_len_col = csv_data.columns[30]  # Colonna 31 (indice 30)
print(f"   Colonna 31 (tcp.len): '{tcp_len_col}'")

# Analizza i primi 20 pacchetti
print("\n3. Confronto PCAP vs CSV (primi 20 pacchetti TCP):")
print("-" * 150)
print(f"{'Idx':<5} | {'CSV_tcp.len':<12} | {'DataOfs':<8} | {'IP.len':<8} | {'TCP_hdr':<9} | {'Payload':<8} | {'Calc_Data':<10} | {'Match?':<20}")
print("-" * 150)

matches = {
    'dataofs': 0,
    'ip_len': 0,
    'tcp_header': 0,
    'payload': 0,
    'calc_data': 0
}

for i in range(min(20, len(tcp_packets), len(csv_data))):
    pkt = tcp_packets[i]
    tcp_layer = pkt[TCP]
    
    # Estrai dal PCAP
    data_offset = tcp_layer.dataofs if hasattr(tcp_layer, 'dataofs') else 0
    tcp_header_len = data_offset * 4
    ip_total_len = pkt[IP].len if IP in pkt and hasattr(pkt[IP], 'len') else 0
    payload_len = len(tcp_layer.payload) if tcp_layer.payload else 0
    calc_tcp_data_len = ip_total_len - 20 - tcp_header_len if ip_total_len > 0 else 0
    
    # Estrai dal CSV
    csv_tcp_len = int(csv_data.iloc[i][tcp_len_col]) if not pd.isna(csv_data.iloc[i][tcp_len_col]) else 0
    
    # Verifica match
    match_field = ""
    if csv_tcp_len == data_offset:
        match_field = "DataOfs ✓"
        matches['dataofs'] += 1
    elif csv_tcp_len == ip_total_len:
        match_field = "IP.len ✓"
        matches['ip_len'] += 1
    elif csv_tcp_len == tcp_header_len:
        match_field = "TCP_hdr ✓"
        matches['tcp_header'] += 1
    elif csv_tcp_len == payload_len:
        match_field = "Payload ✓"
        matches['payload'] += 1
    elif csv_tcp_len == calc_tcp_data_len:
        match_field = "Calc_Data ✓"
        matches['calc_data'] += 1
    else:
        match_field = "Nessun match ✗"
    
    print(f"{i:<5} | {csv_tcp_len:<12} | {data_offset:<8} | {ip_total_len:<8} | {tcp_header_len:<9} | {payload_len:<8} | {calc_tcp_data_len:<10} | {match_field:<20}")

# Risultati
print("\n4. Riepilogo match:")
print(f"   DataOfs (dataofs):                    {matches['dataofs']}/20")
print(f"   IP Total Length (IP.len):             {matches['ip_len']}/20")
print(f"   TCP Header Length (dataofs * 4):      {matches['tcp_header']}/20")
print(f"   Payload Length (len(TCP.payload)):    {matches['payload']}/20")
print(f"   Calculated TCP Data (IP.len-20-hdr):  {matches['calc_data']}/20")

# Determina il campo corretto
max_matches = max(matches.values())
best_match = [k for k, v in matches.items() if v == max_matches][0]

print(f"\n5. CONCLUSIONE:")
print(f"   Il campo 'tcp.len' del CSV corrisponde a: {best_match.upper()}")
print(f"   Match: {max_matches}/20 pacchetti ({max_matches/20*100:.0f}%)")

if best_match == 'dataofs':
    print("\n   → Usa tcp_layer.dataofs per estrarre tcp.len dal PCAP")
elif best_match == 'payload':
    print("\n   → Usa len(tcp_layer.payload) per estrarre tcp.len dal PCAP")
elif best_match == 'calc_data':
    print("\n   → Usa (IP.len - 20 - TCP_header) per estrarre tcp.len dal PCAP")

print("\n" + "="*70)
print("ANALISI COMPLETATA")
print("="*70)
