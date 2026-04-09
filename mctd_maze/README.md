# MCTD-Maze: Monte Carlo Tree Diffusion for System 2 Planning

Implementation of the MCTD framework from:

> "Monte Carlo Tree Diffusion for System 2 Planning"
> arXiv:2502.07202 (ICML 2025 Spotlight)
> Jaesik Yoon, Hyeonseo Cho, Doojin Baek, Yoshua Bengio, Sungjin Ahn

Official code: https://github.com/ahn-ml/mctd

## Overview

MCTD integrates diffusion-based trajectory planners with Monte Carlo Tree Search
for long-horizon goal-conditioned planning. It reconceptualises the denoising
process as a **tree-structured rollout**, enabling inference-time scaling similar
to AlphaGo-style search.

### Key Ideas

- **Denoising as Tree-Rollout**: trajectory split into S sub-plans; earlier
  sub-plans denoise faster (semi-autoregressive)
- **Guidance Levels as Meta-Actions**: GUIDE / NO_GUIDE control
  exploitation vs exploration within the tree
- **Jumpy Denoising as Fast Simulation**: DDIM with skip-C intervals for
  cheap leaf evaluation

### Benchmarks (OGBench)

| Category | Environments |
|----------|-------------|
| Pointmaze | Medium / Large / Giant |
| Antmaze  | Medium / Large / Giant |
| Robot Arm | Single / Double / Triple / Quadruple cube |
| Visual Pointmaze | Medium / Large |

## Installation

```bash
conda env create -f environment.yml
conda activate mctd_maze
pip install -e .
```

## Quick Start

```bash
# Train diffusion planner
python scripts/train.py --env pointmaze-medium-navigate-v0

# Evaluate with MCTD search
python scripts/evaluate.py \
    --env pointmaze-medium-navigate-v0 \
    --checkpoint /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium/best.pt
```

## Package Structure

```
mctd_maze/
├── mctd_maze/
│   ├── __init__.py        # Package entry point
│   ├── config.py          # MCTDMazeConfig dataclass (paper Table 10)
│   ├── tree.py            # MCTSNode: Q-values, visit counts, guidance
│   ├── diffusion.py       # Transformer trajectory diffusion model
│   ├── planner.py         # MCTD Algorithm 1: selection→expansion→sim→backup
│   ├── envs.py            # OGBench environment wrappers + low-level controllers
│   └── utils.py           # Trajectory I/O, dataset loading, caching
├── scripts/
│   ├── train.py           # Train diffusion planner (200k steps, fp16)
│   └── evaluate.py        # MCTD inference + success-rate evaluation
├── tests/
├── environment.yml
└── setup.py
```

## Algorithm Details

### MCTD Four Steps (Algorithm 1)

```
Selection    → traverse tree with UCB, dynamically adjust guidance schedule
Expansion    → extend partial trajectory with GUIDE or NO_GUIDE meta-action
Simulation   → complete remaining sub-plans via jumpy DDIM (skip-C)
Backprop     → update Q-values and guidance schedules up the tree
```

### Hyperparameters (Table 10)

| Parameter | Value |
|-----------|-------|
| Sub-plans S | 5 (out of N=500 total steps) |
| Max search iterations | 500 |
| Partial denoising steps | 20 |
| Jumpy interval C | 10 |
| DDIM eta | 0.0 |
| Beta schedule | linear |
| Learning rate | 5e-4 |
| Batch size | 1024 |
| Training steps | 200,005 |
| Network hidden | 128 |
| Transformer layers | 12 |
| Attention heads | 4 |
| FFN dimension | 512 |
| Training precision | fp16-mixed |
| Inference precision | fp32 |

## Running Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{yoon2025mctd,
  title     = {Monte Carlo Tree Diffusion for System 2 Planning},
  author    = {Yoon, Jaesik and Cho, Hyeonseo and Baek, Doojin and Bengio, Yoshua and Ahn, Sungjin},
  booktitle = {ICML},
  year      = {2025}
}
```
