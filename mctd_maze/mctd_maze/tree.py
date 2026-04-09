"""
MCTSNode — tree node for MCTD planning.

Each node represents a partially-denoised sub-plan x_{1:s} at denoising
step t_s.  Guidance meta-actions {GUIDE, NO_GUIDE} control the
exploration-exploitation balance per sub-plan.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional

import torch


class GuidanceLevel(Enum):
    NO_GUIDE = 0   # exploratory: sample from prior
    GUIDE = 1      # exploitative: goal-directed denoising


class MCTSNode:
    """
    A node in the MCTD search tree.

    Attributes
    ----------
    subplan_idx : int
        Which sub-plan s this node corresponds to (0-indexed).
    partial_traj : torch.Tensor
        Partially denoised trajectory x_{1:s}, shape (s*H, obs_dim+act_dim).
    guidance : GuidanceLevel
        Meta-action applied to reach this node.
    parent : MCTSNode | None
    children : list[MCTSNode]
    visit_count : int       N(s_t, a) in UCB formula
    total_value : float     cumulative Q-value
    """

    def __init__(
        self,
        subplan_idx: int,
        partial_traj: torch.Tensor,
        guidance: GuidanceLevel = GuidanceLevel.GUIDE,
        parent: Optional[MCTSNode] = None,
    ) -> None:
        self.subplan_idx = subplan_idx
        self.partial_traj = partial_traj
        self.guidance = guidance
        self.parent = parent
        self.children: List[MCTSNode] = []

        self.visit_count: int = 0
        self.total_value: float = 0.0

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean backed-up reward estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 1.414) -> float:
        """
        UCB1 score used during tree selection.

        score = Q(s,a) + c * sqrt(log N(s) / (1 + N(s,a)))
        """
        if self.parent is None:
            return float("inf")
        parent_n = self.parent.visit_count
        if parent_n == 0:
            return float("inf")
        exploration = c_puct * math.sqrt(
            math.log(parent_n + 1) / (1 + self.visit_count)
        )
        return self.q_value + exploration

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c_puct: float = 1.414) -> MCTSNode:
        return max(self.children, key=lambda n: n.ucb_score(c_puct))

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.total_value += value

    def backpropagate(self, value: float) -> None:
        """Walk up to root, updating visit counts and values."""
        node: Optional[MCTSNode] = self
        while node is not None:
            node.update(value)
            node = node.parent

    def __repr__(self) -> str:
        return (
            f"MCTSNode(s={self.subplan_idx}, "
            f"guide={self.guidance.name}, "
            f"N={self.visit_count}, Q={self.q_value:.3f})"
        )
