"""
OGBench environment wrappers.

Wraps ogbench.make_env_and_datasets() with:
  - Consistent dataset loading directed to /root/rivermind-data/data/ogbench
  - Reward function factory for MCTD simulation
  - Low-level controller interface (heuristic or DQL-based)
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def make_env_and_dataset(
    env_name: str,
    data_dir: str = "/root/rivermind-data/data/ogbench",
    seed: int = 0,
) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Load OGBench environment and offline dataset.

    Returns
    -------
    env : gymnasium.Env
    dataset : dict  with keys observations, actions, rewards, terminals, ...
    """
    os.environ.setdefault("OGBENCH_DATASETS_PATH", data_dir)

    import ogbench
    env, dataset = ogbench.make_env_and_datasets(env_name, seed=seed)
    return env, dataset


def goal_reaching_reward(
    traj: torch.Tensor,
    goal: torch.Tensor,
    obs_dim: int,
    threshold: float = 0.5,
) -> float:
    """
    Simple goal-reaching reward: 1.0 if final obs within threshold, else 0.

    traj  : (L, obs_dim + act_dim)
    goal  : (obs_dim,)
    """
    final_obs = traj[-1, :obs_dim]
    dist = torch.norm(final_obs - goal.to(final_obs.device))
    return 1.0 if dist.item() < threshold else 0.0


class HeuristicController:
    """
    Simple waypoint-following low-level controller.
    Extracts actions from planned trajectory (action tokens).
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def get_action(
        self, obs: np.ndarray, planned_traj: torch.Tensor, step: int
    ) -> np.ndarray:
        """Return action for current timestep from planned trajectory."""
        if step >= len(planned_traj):
            return np.zeros(self.act_dim)
        token = planned_traj[step]
        action = token[self.obs_dim:self.obs_dim + self.act_dim]
        return action.cpu().numpy()
