import csv
import dpkt

print("PCAP to CSV Converter")
print("="*60)

pcap_file_path = "Edge-IIoT_adversarial.pcap"
output_csv_path = "Edge-IIoT_adversarial.csv"

print(f"\n1. Caricamento {pcap_file_path}...")

print("\n2. Estrazione campi...")
rows = []
skipped = 0

with open(pcap_file_path, "rb") as f:
    pcap = dpkt.pcap.Reader(f)
    total = 0
    for ts, buf in pcap:
        total += 1
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                skipped += 1
                continue
            ip = eth.data
            transport = ip.data

            if isinstance(transport, dpkt.tcp.TCP):
                ack = transport.ack
                seq = transport.seq
                length = len(transport.data)
                flags_int = transport.flags
            elif isinstance(transport, dpkt.udp.UDP):
                ack = 0
                seq = 0
                length = len(transport.data)
                flags_int = 0
            else:
                ack = 0
                seq = 0
                length = len(bytes(transport)) if transport else 0
                flags_int = 0

            rows.append([len(rows), ack, seq, length, flags_int])
        except Exception:
            skipped += 1

print(f"   Pacchetti totali: {total}")
print(f"   Pacchetti convertiti: {len(rows)}")
print(f"   Pacchetti saltati (non-IP o errore): {skipped}")

print(f"\n3. Salvataggio in {output_csv_path}...")
with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "ack", "seq", "len", "flags"])
    writer.writerows(rows)

print(f"   ✓ {output_csv_path} salvato ({len(rows)} righe)")
print("\n" + "="*60)
print("CONVERSIONE COMPLETATA!")
print("="*60)
