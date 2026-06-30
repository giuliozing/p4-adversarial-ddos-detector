import hickle as hkl
import numpy as np
from scapy.all import Ether, IP, TCP, wrpcap

print("Loading original attack samples from HKL file...")
data = hkl.load('dataNotNormalized_pcap.hkl')
x_attack = data['xattack']
y_attack = data['yattack']

print(f"Loaded {len(x_attack)} attack samples")
print(f"Shape: {x_attack.shape}")
print(f"Data type: {x_attack.dtype}")

# Feature order: [ack, seq, length, flags]
print("\nFeature ranges:")
print(f"  ack: [{x_attack[:, 0].min()}, {x_attack[:, 0].max()}]")
print(f"  seq: [{x_attack[:, 1].min()}, {x_attack[:, 1].max()}]")
print(f"  length: [{x_attack[:, 2].min()}, {x_attack[:, 2].max()}]")
print(f"  flags: [{x_attack[:, 3].min()}, {x_attack[:, 3].max()}]")

print("\nConverting original attack samples to PCAP format...")

packets = []

for idx in range(len(x_attack)):
    if idx % 10000 == 0:
        print(f"Processing packet {idx}/{len(x_attack)}...")
    
    # Extract features
    ack = int(x_attack[idx, 0])
    seq = int(x_attack[idx, 1])
    length = int(x_attack[idx, 2])  # 4-bit data_offset (0-15)
    flags = int(x_attack[idx, 3])   # 6-bit flags (0-63)
    
    # Create packet with encoded features
    # The length is already 4-bit encoded, so we use it directly as dataofs
    # The flags are already 6-bit encoded
    
    pkt = Ether(dst="00:00:00:00:00:01", src="00:00:00:00:00:02") / \
          IP(dst="192.168.1.1", src="192.168.1.2") / \
          TCP(
              ack=ack,           # Full 32-bit ack value
              seq=seq,           # Full 32-bit seq value
              dataofs=length,    # 4-bit encoded length (stored in data offset field)
              flags=flags        # 6-bit flags
          )
    
    packets.append(pkt)

print(f"\nTotal packets created: {len(packets)}")

# Save to PCAP file
output_file = 'packetManagement/Edge-IIoT_original.pcap'
print(f"\nSaving to {output_file}...")
wrpcap(output_file, packets)

print(f"✓ Successfully saved {len(packets)} packets to {output_file}")

# Verify the first few packets
print("\n========== Verification ==========")
print("Reading back first 5 packets to verify encoding...")

from scapy.all import rdpcap

verify_packets = rdpcap(output_file, count=5)

for i, pkt in enumerate(verify_packets):
    if TCP in pkt:
        tcp = pkt[TCP]
        print(f"\nPacket {i}:")
        print(f"  Original: ack={x_attack[i, 0]}, seq={x_attack[i, 1]}, len={x_attack[i, 2]}, flags={x_attack[i, 3]}")
        print(f"  In PCAP:  ack={tcp.ack}, seq={tcp.seq}, dataofs={tcp.dataofs}, flags={int(tcp.flags)}")
        
        # Verify they match
        assert tcp.ack == x_attack[i, 0], "ACK mismatch!"
        assert tcp.seq == x_attack[i, 1], "SEQ mismatch!"
        assert tcp.dataofs == x_attack[i, 2], "Length mismatch!"
        assert int(tcp.flags) == x_attack[i, 3], "Flags mismatch!"
        print("  ✓ Verified!")

print("\n✓ All verifications passed!")
print(f"\nOriginal attack PCAP file created: {output_file}")
