"""
MCTDMazeConfig — all hyperparameters from paper Table 10.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Transformer diffusion model architecture."""
    hidden_dim: int = 128
    n_layers: int = 12
    n_heads: int = 4
    ffn_dim: int = 512
    beta_schedule: str = "linear"
    n_diffusion_steps: int = 200   # T for training noise schedule


@dataclass
class TrainConfig:
    """Training hyperparameters (Table 10)."""
    lr: float = 5e-4
    batch_size: int = 1024
    train_steps: int = 200_005
    precision: str = "16-mixed"   # fp16 training, fp32 inference
    grad_clip: float = 1.0
    warmup_steps: int = 1_000
    save_every: int = 10_000
    log_every: int = 500


@dataclass
class MCTDConfig:
    """MCTD search hyperparameters (Table 10)."""
    num_subplans: int = 5          # S: number of sub-plans per trajectory
    max_search: int = 500          # maximum MCTS iterations
    partial_denoise_steps: int = 20  # denoising steps per tree node
    jumpy_interval: int = 10       # C: DDIM skip interval for simulation
    ddim_eta: float = 0.0          # 0 = deterministic DDIM


@dataclass
class MCTDMazeConfig:
    """Top-level config, mirrors official Hydra configs."""
    env: str = "pointmaze-medium-navigate-v0"
    seed: int = 0
    device: str = "cuda:0"
    save_dir: str = "/root/rivermind-data/checkpoints/mctd_maze"
    data_dir: str = "/root/rivermind-data/data/ogbench"
    output_dir: str = "/root/rivermind-data/outputs/mctd_maze"
    use_wandb: bool = True
    wandb_project: str = "mctd_maze"
    log_level: str = "INFO"

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mctd: MCTDConfig = field(default_factory=MCTDConfig)
