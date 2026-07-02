# DDoS Adversarial Attacks Detection with DNNs in P4 Programmable Switches

Reproduction code for the paper:

> **G. Zingrillo, L. Ismail, E. Paolini, F. Paolucci, F. Cugini, L. De Marinis,**
> *"DDoS Adversarial Attacks Detection with Deep Neural Networks in P4 Programmable
> Switches"*, **EuCNC**.

This work builds an **in-switch, DNN-based NIDS** that is resilient to adversarial
attacks. Alongside the primary **Attack Detector**, we train and deploy a dedicated
**Adversarial Detector** (via *adversarial training*). Both detectors are
**LUT-distilled** onto an **Intel Tofino** P4 switch and run in parallel, their outputs
combined with a logical OR. The framework keeps robust accuracy between **91% and 99%**
across white-, gray- and black-box threat models, adding only **1–2 µs** of latency over
plain forwarding.

- **Dataset:** Edge-IIoTset (14 threat categories; binary benign/malicious task)
- **Features (4 TCP):** `tcp.flags`, `tcp.data_offset`, `tcp.seq`, `tcp.ack` — selected
  from 46 candidates by **chi-square** analysis under P4-extractability and
  cross-domain-robustness constraints
- **Model:** two DoReFa 8-bit quantized 2-input subnetworks → final stage → binary
  softmax, distilled into a **3-tier LUT hierarchy** (LUT1, LUT2 → LUT3)

## Key result

The undefended model collapses to ~40% (EAD) / ~52% (Boundary) under attack, while the
adversarially-trained detector stays **>98%** in gray-/black-box and **85–91%** under a
white-box EAD attack.

| EAD (white-/gray-box) | Boundary (black-box) |
|---|---|
| ![EAD](docs/figures/EAD_adv_training.png) | ![Boundary](docs/figures/Boundary_adv_training.png) |

Hardware validation on Tofino: software accuracy **81.24%** vs compiled-P4 **81.15%**
on ~485k packets (≤0.1 pp deviation).

## Repository structure

```
models/            # DNN definitions (binary detector + randomized/adversarial variant)
training/          # train the (quantized) detector
attacks/           # EAD (foolbox/ART), Boundary (foolbox/ART), Gaussian-noise baseline
data_pipeline/     # Edge-IIoTset feature extraction, csv <-> pcap conversion,
                   #   consistent adversarial-packet generation + pcap reconstruction
analysis/          # chi-square feature selection, TCP-length analysis
p4/                # P4/Tofino data-plane pipeline — description (see p4/README.md)
docs/figures/      # rendered result figures
utils.py           # scaling / normalization helpers
```

Each code folder is a Python package; run scripts as modules from the repository root.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # Python 3.10
```

## Data (not included)

The Edge-IIoTset captures, serialized `.hkl` datasets and trained `.h5` models are
**not committed** (see `.gitignore`). Download Edge-IIoTset from its
official source (Ferrag et al., 2022) and place the `DNN-EdgeIIoT-dataset.csv` under
`Edge-IIoTset dataset/Selected dataset for ML and DL/` (the path expected by
`analysis/feature_selection.py`), then regenerate artifacts with the workflow below.

## How to run

Run every script **as a module, from the repository root**:

```bash
python -m analysis.feature_selection           # chi-square feature ranking
python -m data_pipeline.edge_iiot_extractor    # build the feature dataset
python -m training.train_quantized             # train the Attack Detector
python -m attacks.ead_generate                 # craft EAD adversarial corpus (eps 0.25 / 0.5)
python -m training.train_quantized             # train the Adversarial Detector (adversarial training)
python -m attacks.ead                          # white-/gray-box evaluation
python -m attacks.boundary                     # black-box evaluation
```

The consistent-adversarial-packet tooling in `data_pipeline/` (`adversarial_to_pcap.py`,
`original_to_pcap.py`, `dataset_from_pcap.py`, `adversarial_lookup_from_pcap.py`)
rebuilds protocol-valid `.pcap` traces from adversarial feature vectors, used for the
Tofino hardware test. Data paths inside the scripts are relative to the repository root.

## P4 / Tofino data plane

The switch-side P4 program (parser → LUT1/LUT2/LUT3 match-action tables → forwarding)
is **not included** in this repository (it was developed/run on the Tofino testbed). See
[`p4/README.md`](p4/README.md) for the pipeline description and the LUT entry format.

## Acknowledgements

Funded by the European Commission Horizon Europe SNS JU **NATWORK** project (g.a. No.
101139285). Carried out at CNIT and Scuola Superiore Sant'Anna, Pisa.

This repository is part of a broader research internship — see
[`cnit-internship-inswitch-nids`](https://github.com/giuliozing/cnit-internship-inswitch-nids).

## License

Code released under the [MIT License](LICENSE). The Edge-IIoTset dataset remains under
its own license and must be obtained from the official source.
