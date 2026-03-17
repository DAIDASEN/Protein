"""
MCTS Tree node definition for MCTD-ME.

Each node represents a fully-denoised (or partially-masked) protein sequence.
Stores Q-values, visit counts, cached uncertainty bonuses, and parent/child
links needed for PH-UCT-ME selection and max/sum backup.

Reference: Sec. 3.2 and Algorithm 1 of arXiv:2509.15796.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


class MCTSNode:
    """
    A node in the MCTS tree.

    Parameters
    ----------
    sequence : str
        Fully-denoised amino-acid sequence (single-letter codes).
        '[MASK]' characters are allowed for partially-masked nodes kept in
        memory before denoising, but leaf nodes after expansion should have
        no mask tokens.
    parent : MCTSNode | None
        Parent node in the tree (None for the root).
    depth : int
        Depth of this node (root = 0).
    reward : float
        Composite reward score cached from critic evaluation.
        Set to 0.0 until the node is evaluated.
    u_ent : float
        Epistemic-uncertainty bonus U_ent (Eq. 5) computed at expansion time
        and reused during UCB selection.
    u_div : float
        Diversity bonus U_div (Eq. 6) computed at expansion time.
    """

    def __init__(
        self,
        sequence: str,
        parent: Optional["MCTSNode"] = None,
        depth: int = 0,
        reward: float = 0.0,
        u_ent: float = 0.0,
        u_div: float = 0.0,
    ) -> None:
        self.sequence: str = sequence
        self.parent: Optional[MCTSNode] = parent
        self.depth: int = depth

        # MCTS statistics
        self._q_sum: float = 0.0   # sum of backed-up values (used for max or sum)
        self._q_max: float = 0.0   # running maximum of backed-up values
        self._visit_count: int = 0  # N(s, a) – number of times this node was visited

        # Composite reward from the critics (set once evaluated)
        self.reward: float = reward

        # Uncertainty / diversity bonuses cached at expansion time
        self.u_ent: float = u_ent
        self.u_div: float = u_div

        # Children list (ordered by insertion; no fixed branching factor)
        self.children: List[MCTSNode] = []

        # Flag: has this node been expanded yet?
        self.is_expanded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def visit_count(self) -> int:
        """Number of times this node has been visited (backed through)."""
        return self._visit_count

    @property
    def q_value(self) -> float:
        """
        Q(s, a): expected return from this node.

        Returns the *maximum* backed-up reward seen through this node
        (consistent with the max-backup rule in Table 10).  If the node
        has never been visited, returns 0.
        """
        if self._visit_count == 0:
            return 0.0
        return self._q_max

    @property
    def q_mean(self) -> float:
        """Mean Q-value (useful for diagnostic logging)."""
        if self._visit_count == 0:
            return 0.0
        return self._q_sum / self._visit_count

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children yet."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """True if this node has no parent."""
        return self.parent is None

    # ------------------------------------------------------------------
    # Update / backup
    # ------------------------------------------------------------------

    def update(self, value: float, rule: str = "max") -> None:
        """
        Update this node's statistics with a new backed-up value.

        Parameters
        ----------
        value : float
            The reward (or backed-up Q) to incorporate.
        rule : str
            "max"  – maintain the running maximum (paper default).
            "sum"  – accumulate the sum (mean-Q mode).
        """
        self._visit_count += 1
        self._q_sum += value
        if value > self._q_max:
            self._q_max = value

    def backpropagate(self, value: float, rule: str = "max") -> None:
        """
        Walk up the tree from this node to the root, updating all ancestors.

        Parameters
        ----------
        value : float
            The reward propagated upward.
        rule : str
            Backup rule ("max" or "sum").
        """
        node: Optional[MCTSNode] = self
        while node is not None:
            node.update(value, rule=rule)
            node = node.parent

    # ------------------------------------------------------------------
    # Child management
    # ------------------------------------------------------------------

    def add_child(
        self,
        sequence: str,
        reward: float = 0.0,
        u_ent: float = 0.0,
        u_div: float = 0.0,
    ) -> "MCTSNode":
        """
        Create and register a child node.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence of the new child.
        reward : float
            Pre-computed composite reward of the child.
        u_ent : float
            Epistemic uncertainty bonus cached for this child.
        u_div : float
            Diversity bonus cached for this child.

        Returns
        -------
        MCTSNode
            The newly created child node.
        """
        child = MCTSNode(
            sequence=sequence,
            parent=self,
            depth=self.depth + 1,
            reward=reward,
            u_ent=u_ent,
            u_div=u_div,
        )
        self.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def best_child(self) -> Optional["MCTSNode"]:
        """Return the child with the highest Q-value (greedy pick)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.q_value)

    def all_sequences(self) -> List[str]:
        """
        Return all sequences reachable from this node (BFS), including self.
        """
        result: List[str] = []
        queue: List[MCTSNode] = [self]
        while queue:
            node = queue.pop(0)
            result.append(node.sequence)
            queue.extend(node.children)
        return result

    def path_to_root(self) -> List["MCTSNode"]:
        """Return list of nodes from root to self (inclusive)."""
        path: List[MCTSNode] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def subtree_size(self) -> int:
        """Total number of nodes in the subtree rooted here (including self)."""
        count = 1
        for child in self.children:
            count += child.subtree_size()
        return count

    def __repr__(self) -> str:
        seq_short = (
            self.sequence[:20] + "…" if len(self.sequence) > 20 else self.sequence
        )
        return (
            f"MCTSNode(seq={seq_short!r}, depth={self.depth}, "
            f"visits={self._visit_count}, Q={self.q_value:.4f}, "
            f"reward={self.reward:.4f})"
        )
