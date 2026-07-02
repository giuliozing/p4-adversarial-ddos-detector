# P4 / Tofino data-plane pipeline (description)

> The actual P4 source for the Intel Tofino was developed and executed on the
> hardware testbed and is **not included** here. This document describes the pipeline
> at a conceptual level.

## Target

- **Tofino Native Architecture (TNA)**, Tofino1 programmable ASIC.
- Compiled with the target-aware compiler, validated in the Tofino SDK emulator, then
  deployed on the physical switch.

## Ingress pipeline

```
parser ─► LUT1 (match-action) ─► LUT2 (match-action) ─► LUT3 (match-action) ─► deparser ─► forward
```

1. **Parser** — extracts Ethernet, IP and TCP headers and derives the four TCP
   features: `flags`, `data_offset`, `seq`, `ack`.
2. **Feature encoding** — features are cast to 8-bit identifiers and combined into two
   16-bit keys:
   - key A = (`seq`, `ack`)
   - key B = (`data_offset`, `flags`)
   Exact matching is used for `flags` and `data_offset`; range matching for `seq` and
   `ack`.
3. **LUT1 / LUT2** — two match-action tables, one per key, each producing an
   intermediate 8-bit output (`o1`, `o2`) stored in metadata. These correspond to the
   two first-stage subnetworks.
4. **LUT3** — `o1`‖`o2` form a third key into the final table (the DNN output layer),
   producing the boolean decision `o3` (attack / non-attack).
5. **Forwarding** — `o3` selects the output port (legitimate vs. malicious), so
   classification and steering happen entirely in ingress. The egress pipeline is
   unmodified.

The **Attack Detector** and the **Adversarial Detector** share this structure and run
in parallel; their decisions are combined with a logical **OR** (a packet is dropped if
either flags it).

## LUT entry format

All entries are derived **offline** from the trained Keras models via LUT distillation
and exported as CSV. Each row maps a compound input key to the table's output value;
loading these rows into the corresponding P4 match-action table reproduces the DNN
inference exactly.

## Measured performance (reference)

- Functional equivalence: software 81.24% vs Tofino 81.15% overall accuracy (~485k packets).
- Latency: ~26.6 µs (plain port forwarding) vs ~27.9 µs (P4-DNN + forwarding) at 30 Gbps,
  i.e. **+1–2 µs** for full in-switch inference.
