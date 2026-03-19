# Protein Binder Evaluation & RL-Guided Design Pipeline

End-to-end pipeline for generating, evaluating, and optimising protein binder candidates for therapeutic targets. Built to mirror Latent Labs' Latent-X workflow.

## Overview

| Phase | What it builds | Maps to |
|-------|---------------|---------|
| 0 | Environment & data setup | — |
| 1 | Target preparation & structural analysis | Molecular data analysis, biology knowledge |
| 2 | Binder generation (RFdiffusion + ProteinMPNN) | Compare against external technologies |
| 3 | In silico evaluation pipeline | Design evaluation strategies & benchmarks |
| 4 | Analysis, visualisation & write-up | Deep-dives, feeding back results |
| 5 | RL expansion — reward-weighted sampling | Optimise sampling techniques & hyperparameters |

## Quick Start

```bash
conda create -n binder-pipeline python=3.10
conda activate binder-pipeline
pip install -r requirements.txt
```

### External Tools (Phase 2)

```bash
# RFdiffusion
git clone https://github.com/RosettaCommons/RFdiffusion
cd RFdiffusion && pip install -e ".[dev]"

# ProteinMPNN
git clone https://github.com/dauparas/ProteinMPNN
```

## Repository Structure

```
protein-binder-pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── targets/           # Downloaded PDB files
│   └── generated/         # RFdiffusion / ProteinMPNN outputs
├── src/
│   ├── target_prep.py     # Phase 1: PDB loading, hotspot ID
│   ├── generate.py        # Phase 2: Wrappers around RFdiffusion/ProteinMPNN
│   ├── evaluate.py        # Phase 3: Chai-1 scoring, metric extraction
│   ├── analyse.py         # Phase 4: Plotting, clustering, interface analysis
│   └── rl_sampler.py      # Phase 5: Reward-weighted sampling
├── notebooks/
│   ├── 01_target_exploration.ipynb
│   ├── 02_evaluation_results.ipynb
│   └── 03_rl_experiments.ipynb
└── configs/
    └── eval_thresholds.yaml
```

## Benchmark Targets (from Latent-X paper, Table 1)

| Target | PDB ID | Notes |
|--------|--------|-------|
| IL-7Rα | 3DI2 | Key cytokine receptor, immunotherapy target |
| TrkA | 2IFG | Neurotrophin receptor kinase |
| InsulinR | 7PG0 | Insulin receptor |
| EGFR | 3NJP | Epidermal growth factor receptor |
| FGFR2 | 1DJS | Fibroblast growth factor receptor 2 |
| PD-L1 | — | Immune checkpoint ligand |

## Key Metrics

- **iPTM** (interface predicted TM-score): primary metric for binding confidence
- **pLDDT**: per-residue confidence, averaged over binder chain
- **Interface RMSD**: agreement between generated and re-predicted binding mode
- **pAE**: predicted aligned error, captures inter-chain confidence

## Phase 5 — RL Approach

Reward-weighted sampling using REINFORCE:
1. Sample batch of N sequences from ProteinMPNN at temperature T
2. Score each with in silico evaluation pipeline (iPTM as reward)
3. Upweight log-probability of high-scoring sequences, downweight low-scoring
4. Iteratively refine sampling distribution

Lighter alternative: Bayesian optimisation over ProteinMPNN temperature, RFdiffusion steps, and backbone noise level.
