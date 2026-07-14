# Structural Reasoning vs. Statistical Estimation for Matrix Chain Multiplication

Research code accompanying the paper **"Graph Neural Networks for Learned Matrix
Chain Optimization: When Structural Reasoning Outperforms Statistical Estimation"**
(Ch. Abhilash, Dakshatha, Shaun Angel, Abhinav, Maneeshwar — Dept. of CSE, VNR VJIET).

This project compares two ways of approximating the classical O(n³) dynamic
programming (DP) solution to the Matrix Chain Multiplication problem:

- **Structural reasoning** — `GraphMCMNet` and `PointerMCMNet` predict the DP
  split table directly. The cost is then computed *exactly* from the predicted
  splits, which guarantees the result is always a mathematically valid
  parenthesization (proven in Theorem 1 of the paper).
- **Statistical estimation** — `XGBoost` and `Random Forest` baselines regress
  the scalar cost directly from 145 hand-engineered features, with no
  structural constraint.

> **Note on inference speed:** neural inference here is **not constant-time**.
> GraphMCMNet scales as O(n²K) and PointerMCMNet as O(n²) per chain. On a
> single chain, exact Python DP (0.05–3.97 ms) is actually faster than either
> neural model (27–349 ms) due to fixed per-chain overhead. The practical
> advantage appears under **GPU batching**, where learned models achieve
> >100× aggregate throughput over sequential DP — see Section V-F of the
> paper for the full latency and ablation analysis.

## Key Results (500-chain held-out benchmark, n=5–50)

| Model | MAPE (%) | Validity (%) | Exact Split Match (%) |
|---|---|---|---|
| **GraphMCMNet** | **0.029** [0.01, 0.05] | **100.0** | **93.0** |
| PointerMCMNet | 0.148 [0.08, 0.23] | 100.0 | 88.8 |
| XGBoost | 36.187 | 22.0 | — |
| Random Forest | 43.284 | 18.2 | — |

Both structural models also generalize to real matrix chains extracted from
production architectures (ResNet, BERT, GPT-2, ViT, EfficientNet, MobileNet,
DenseNet, T5) — see `benchmarks/` and Section V-E of the paper.

Full results, statistical significance tests, and ablations are in Tables
II–VI of the paper (`docs/paper/`).

## Repository Structure
.
├── src/                  # Core implementation
│   ├── graph_mcm_net.py       # GraphMCMNet: sub-problem graph construction + gated message passing
│   ├── pointer_mcm_net.py     # PointerMCMNet: Transformer encoder + Bahdanau pointer decoder
│   ├── baselines/              # XGBoost / Random Forest dual-head ensembles + 145-feature extractor
│   ├── dp_solver.py            # Exact O(n^3) DP reference solver (ground truth)
│   ├── hu_shing.py             # O(n log n) Hu-Shing exact solver (Table VI cross-check)
│   └── train.py                # Curriculum training loop (Stages 1-4, Table I)
│
├── data/                 # Synthetic dataset generation
│   └── generate_chains.py      # Uniform / Spiky / Bottleneck / Monotone chain generator (seed=42)
│
├── models/               # Trained checkpoints (final Stage-4 / Stage-3 weights per model)
│
├── benchmarks/           # Real-architecture evaluation (Section V-E, Table V)
│   └── extract_real_chains.py  # Pulls dimension chains from ResNet/BERT/GPT-2/ViT/etc. configs
│
├── scripts/              # Entry points — see "Reproducing the Paper" below
│
├── docs/                 # Paper source, figures, supplementary material
│   └── paper/                  # conference_101719.tex and compiled figures
│
├── exploratory/          # (formerly scratch/) working notebooks, not required for reproduction
│
├── requirements.txt
├── LICENSE
└── README.md

> `prototype_frontend/` is a standalone visualization demo and is **not**
> required to reproduce any table or figure in the paper.

## Setup

```bash
git clone https://github.com/abhilash0042/ml-matrix-chain-optimizer.git
cd ml-matrix-chain-optimizer
pip install -r requirements.txt
```

Neural models were trained on an NVIDIA RTX 4050 (6GB VRAM); CPU-only
inference and the tree baselines run without a GPU.

## Reproducing the Paper

Generate the evaluation dataset (seed=42, matches the paper exactly):
```bash
python -m data.generate_chains --seed 42 --n_range 5 50 --n_samples 120000
```

Train a model (curriculum stages defined in `src/train.py`, Table I):
```bash
python -m src.train --model graph_mcm_net   # or pointer_mcm_net
```

Run the full evaluation suite — reproduces Tables II–IV and Figures 4–7:
```bash
python -m scripts.evaluate_all_models --seed 42
```

Run the real-architecture benchmark — reproduces Table V:
```bash
python -m benchmarks.extract_real_chains
python -m scripts.evaluate_all_models --input benchmarks/real_chains.json
```

Run the latency/ablation study — reproduces Table VI:
```bash
python -m scripts.benchmark_latency
```

All scripts are deterministic given `--seed 42` and will reproduce the
numbers reported in the paper to within floating-point tolerance.

## Citation

```bibtex
@inproceedings{abhilash2026graphmcmnet,
  title     = {Graph Neural Networks for Learned Matrix Chain Optimization:
               When Structural Reasoning Outperforms Statistical Estimation},
  author    = {Abhilash, Ch. and Dakshatha and Angel, Shaun and Abhinav and Maneeshwar},
  booktitle = {IEEE Conference Proceedings},
  year      = {2026},
  organization = {Dept. of CSE, VNR VJIET, Hyderabad, India}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ch. Abhilash — abhilash00042@gmail.com — Dept. of CSE, VNR VJIET, Hyderabad
