"""
MCTD — Monte Carlo Tree Diffusion planner.

Implements Algorithm 1 from the paper:
  Selection → Expansion → Simulation → Backpropagation

The denoising process is reconceptualised as a tree-structured rollout:
  - Each node holds a partially denoised sub-plan x_{1:s}
  - Meta-actions {GUIDE, NO_GUIDE} control guidance per sub-plan
  - Jumpy DDIM provides cheap leaf simulation
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import torch

from mctd_maze.config import MCTDConfig
from mctd_maze.diffusion import TrajDiffusion
from mctd_maze.tree import GuidanceLevel, MCTSNode

logger = logging.getLogger(__name__)


class MCTD:
    """
    Monte Carlo Tree Diffusion planner.

    Parameters
    ----------
    model : TrajDiffusion
        Trained trajectory diffusion model.
    reward_fn : Callable[[torch.Tensor], float]
        Evaluates a complete trajectory; higher is better.
    cfg : MCTDConfig
        Search hyperparameters.
    goal : torch.Tensor | None
        Goal observation for guided denoising.
    device : str
    """

    def __init__(
        self,
        model: TrajDiffusion,
        reward_fn: Callable[[torch.Tensor], float],
        cfg: MCTDConfig,
        goal: Optional[torch.Tensor] = None,
        device: str = "cuda:0",
    ) -> None:
        self.model = model.to(device).eval()
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.goal = goal.to(device) if goal is not None else None
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run MCTD search and return the best complete trajectory.

        Parameters
        ----------
        obs : torch.Tensor  shape (obs_dim,)
            Current observation (used as trajectory start conditioning).

        Returns
        -------
        best_traj : torch.Tensor  shape (horizon, obs_dim + act_dim)
        """
        root = self._make_root(obs)
        best_traj: Optional[torch.Tensor] = None
        best_reward = float("-inf")

        for i in range(self.cfg.max_search):
            # --- Selection ---
            node = self._select(root)

            # --- Expansion ---
            if not node.is_leaf() or node.subplan_idx >= self.cfg.num_subplans:
                pass  # fully expanded or terminal
            else:
                children = self._expand(node)
                if children:
                    node = children[0]

            # --- Simulation ---
            traj, reward = self._simulate(node)

            if reward > best_reward:
                best_reward = reward
                best_traj = traj
                logger.debug(f"iter {i:4d} | new best reward={reward:.4f}")

            # --- Backpropagation ---
            node.backpropagate(reward)

        logger.info(f"MCTD finished | best reward={best_reward:.4f}")
        return best_traj

    # ------------------------------------------------------------------
    # MCTS steps
    # ------------------------------------------------------------------

    def _make_root(self, obs: torch.Tensor) -> MCTSNode:
        """Create root node from initial noise."""
        traj_len = self._traj_length()
        x_noise = torch.randn(1, traj_len, self.model.model.token_dim, device=self.device)
        return MCTSNode(subplan_idx=0, partial_traj=x_noise)

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB until a leaf or unexpanded node."""
        node = root
        while not node.is_leaf() and node.subplan_idx < self.cfg.num_subplans:
            node = node.best_child()
        return node

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Generate child nodes for each guidance meta-action.
        Each child applies partial denoising to the next sub-plan segment.
        """
        children = []
        for guidance in list(GuidanceLevel):
            child_traj = self._partial_denoise_subplan(
                node.partial_traj,
                node.subplan_idx,
                guidance,
            )
            child = MCTSNode(
                subplan_idx=node.subplan_idx + 1,
                partial_traj=child_traj,
                guidance=guidance,
                parent=node,
            )
            node.children.append(child)
            children.append(child)
        return children

    def _simulate(self, node: MCTSNode) -> Tuple[torch.Tensor, float]:
        """
        Complete remaining sub-plans via jumpy DDIM, then evaluate reward.
        """
        x = node.partial_traj.clone()
        remaining_start = self._subplan_noise_level(node.subplan_idx)

        # Jumpy denoising for fast simulation
        x = self.model.jumpy_denoise(
            x,
            t_start=remaining_start,
            jumpy_interval=self.cfg.jumpy_interval,
            goal=self.goal,
            eta=self.cfg.ddim_eta,
        )
        reward = self.reward_fn(x.squeeze(0))
        return x.squeeze(0), reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _traj_length(self) -> int:
        """Total trajectory token length (obs+act per timestep)."""
        # Derived from environment; override via subclass or config
        return 200  # default horizon

    def _subplan_noise_level(self, subplan_idx: int) -> int:
        """
        Causal noise schedule: sub-plan s gets noise level proportional
        to its position.  Earlier sub-plans have lower noise (more denoised).
        """
        T = self.model.n_steps
        S = self.cfg.num_subplans
        # Earlier sub-plans are more denoised (lower t)
        frac = (S - subplan_idx) / S
        return max(1, int(T * frac))

    def _partial_denoise_subplan(
        self,
        x: torch.Tensor,
        subplan_idx: int,
        guidance: GuidanceLevel,
    ) -> torch.Tensor:
        """Apply partial denoising for one sub-plan."""
        t_start = self._subplan_noise_level(subplan_idx)
        goal = self.goal if guidance == GuidanceLevel.GUIDE else None
        return self.model.partial_denoise(
            x,
            t_start=t_start,
            n_steps=self.cfg.partial_denoise_steps,
            goal=goal,
            eta=self.cfg.ddim_eta,
        )
