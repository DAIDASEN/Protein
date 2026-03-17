# MCTD-ME: Monte Carlo Tree Diffusion with Multiple Experts for Protein Design

Implementation of the MCTD-ME framework from:

> "Monte Carlo Tree Diffusion with Multiple Experts for Protein Design"
> arXiv:2509.15796

## Overview

MCTD-ME integrates masked diffusion protein language models (DPLM-2) with Monte Carlo Tree Search (MCTS) for protein design. It uses multiple expert models and multiple critics (ESMFold-based metrics) to explore the protein sequence space efficiently.

### Key Features

- **Multiple Experts**: DPLM-2 (150M, 650M, 3B) + optional ProteinMPNN
- **PH-UCT-ME Selection**: Predictive Heuristic UCB with epistemic uncertainty (BALD) and diversity bonuses
- **pLDDT-Guided Masking**: Progressive masking of low-confidence residues
- **Three Tasks**: Folding, Inverse Folding, Motif Scaffolding
- **Cached Evaluation**: Avoids re-computing ESMFold on repeated sequences

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mctd_me

# Install package
pip install -e .
```

## Quick Start

### Inverse Folding

```bash
python scripts/run_inverse_folding.py \
    --pdb data/target.pdb \
    --chain A \
    --output ./outputs/inverse_folding/ \
    --num_iters 100
```

### Folding (Lead Optimization)

```bash
python scripts/run_folding.py \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL \
    --output ./outputs/folding/ \
    --num_iters 100
```

### Motif Scaffolding

```bash
python scripts/run_motif_scaffolding.py \
    --pdb data/motif.pdb \
    --chain A \
    --motif_residues 10-25 \
    --scaffold_length 100 \
    --output ./outputs/motif/
```

### Download Benchmark Data

```bash
python scripts/download_data.py --all --output ./data/
```

### Evaluation

```bash
python scripts/evaluate.py \
    --fasta outputs/inverse_folding/designs.fasta \
    --reference data/target.pdb \
    --chain A \
    --output ./eval_results/
```

## Package Structure

```
mctd_me/
├── mctd_me/
│   ├── __init__.py       # Package entry point
│   ├── config.py         # MCTDMEConfig dataclass
│   ├── tree.py           # MCTSNode with Q-values, bonuses
│   ├── selection.py      # PH-UCT-ME (Eq. 2-6)
│   ├── masking.py        # pLDDT-guided progressive masking (Eq. 7)
│   ├── experts.py        # DPLM-2 and ProteinMPNN wrappers
│   ├── critics.py        # Composite reward functions (Appendix A.3)
│   ├── diffusion.py      # Masked diffusion rollout engine
│   ├── mcts.py           # Main MCTD-ME algorithm (Algorithm 1)
│   ├── metrics.py        # TM-score, RMSD, pLDDT, AAR, scTM
│   └── utils.py          # Protein I/O, caching, utilities
├── scripts/
│   ├── run_inverse_folding.py
│   ├── run_folding.py
│   ├── run_motif_scaffolding.py
│   ├── evaluate.py
│   └── download_data.py
├── tests/
│   ├── test_tree.py
│   ├── test_selection.py
│   ├── test_masking.py
│   ├── test_metrics.py
│   ├── test_critics.py
│   ├── test_utils.py
│   ├── test_config.py
│   └── test_mcts.py
├── environment.yml
└── setup.py
```

## Algorithm Details

### PH-UCB-ME Selection Rule (Eq. 2-6)

```
PH-UCB-ME(s_t, a) = Q(s_t, a)
    + c_p * sqrt(log N(s_t) / (1 + N(s_t, a)))
      * π_cons,τ(a|s_t)
      * (w_ent * U_ent(s_t, a) + w_div * U_div(s_t, a))
```

Where:
- `π_cons,τ` = temperature-controlled geometric mean of expert distributions
- `U_ent` = BALD epistemic uncertainty (mixture entropy minus mean entropy)
- `U_div` = normalised Hamming distance from parent

### Hyperparameters (Table 10)

| Parameter | Value |
|-----------|-------|
| k_roll (rollouts/expert) | 3 |
| K (children kept) | 3 |
| max_depth | 5 |
| c_p (exploration) | 1.414 |
| T (MCTS iterations) | 100 |
| diffusion_steps | 150 |
| temperature τ | 1.0 |
| w_ent, w_div | 1.0 each |
| backup_rule | max |

### Reward Functions (Appendix A.3)

**Folding:** `R = 0.60·TM + 0.40·(1 - min(RMSD/10,1)) + 0.05·pLDDT`

**Inverse Folding:** `R = 0.60·AAR + 0.35·scTM + 0.05·B`

**Motif Scaffolding:** `R = 0.40·pLDDT_scaf + 0.30·g(RMSD_motif) + 0.30·scTM + 0.20·I[success]`

## Running Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{mctdme2025,
  title={Monte Carlo Tree Diffusion with Multiple Experts for Protein Design},
  author={...},
  journal={arXiv:2509.15796},
  year={2025}
}
```
