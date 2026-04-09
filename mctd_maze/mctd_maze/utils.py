"""
Utilities: dataset loading, trajectory I/O, logging setup.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_dir: str,
    name: str = "checkpoint",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, f"{name}_{step:07d}.pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )
    # Keep a "best.pt" symlink updated by caller
    logging.getLogger(__name__).info(f"Saved checkpoint: {path}")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logging.getLogger(__name__).info(f"Results saved to {path}")


def dataset_to_trajectories(
    dataset: Dict[str, np.ndarray],
    horizon: int = 200,
    stride: int = 1,
) -> torch.Tensor:
    """
    Convert flat (N, dim) dataset into trajectory tensors (M, horizon, dim).

    Uses sliding window with given stride.
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    tokens = np.concatenate([obs, act], axis=-1)  # (N, obs+act)
    N, D = tokens.shape
    trajs = []
    for start in range(0, N - horizon + 1, stride):
        trajs.append(tokens[start:start + horizon])
    return torch.tensor(np.stack(trajs), dtype=torch.float32)
