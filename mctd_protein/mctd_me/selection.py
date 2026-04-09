"""
PH-UCT-ME selection rule for MCTD-ME.

Implements Equations 2-6 from the paper:
  "Monte Carlo Tree Diffusion with Multiple Experts for Protein Design"
  arXiv:2509.15796

The PH-UCB-ME score for action a from state s_t is (Eq. 2):

    PH-UCB-ME(s_t, a) = Q(s_t, a)
        + c_p * sqrt(log N(s_t) / (1 + N(s_t, a)))
          * π_cons,τ(a|s_t)
          * (w_ent * U_ent(s_t, a) + w_div * U_div(s_t, a))

where:
  - π_cons,τ  is the temperature-controlled consensus prior  (Eq. 3)
  - U_ent     is the BALD-style epistemic uncertainty bonus  (Eq. 5)
  - U_div     is the normalized Hamming diversity bonus      (Eq. 6)

U_ent and U_div are computed once at expansion time and cached on each
child node (node.u_ent, node.u_div, node.pi_cons).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from mctd_me.tree import MCTSNode


# ---------------------------------------------------------------------------
# Consensus prior  (Eq. 3)
# ---------------------------------------------------------------------------

def compute_consensus_prior(
    logprobs_per_expert: List[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute the temperature-controlled consensus prior π_cons,τ (Eq. 3).

    π_cons,τ(a|s) = (1/Z) * exp( (1/τ) * (1/E) * Σ_e log p_ϕ^e(a|s, M) )

    This is a temperature-scaled geometric mean of the expert distributions,
    normalised to a proper probability distribution.

    Parameters
    ----------
    logprobs_per_expert : List[torch.Tensor]
        Each tensor has shape (vocab_size,) and contains log-probabilities
        p_ϕ^e(·|s, M) for a single masked position from expert e.
    temperature : float
        Temperature τ > 0.  τ → 0 collapses to argmax; τ → ∞ → uniform.

    Returns
    -------
    torch.Tensor
        Shape (vocab_size,), normalised probability distribution.
    """
    # Stack → (E, V)
    log_p = torch.stack(logprobs_per_expert, dim=0)  # (E, V)
    # Geometric mean in log space: (1/E) * Σ_e log p^e
    mean_log_p = log_p.mean(dim=0)  # (V,)
    # Temperature scaling: divide by τ
    scaled = mean_log_p / temperature
    # Normalise
    pi_cons = F.softmax(scaled, dim=-1)  # (V,)
    return pi_cons


def compute_consensus_prior_sequence(
    logprobs_per_expert: List[torch.Tensor],
    mask_indices: List[int],
    sequence_length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-position consensus log-probability for an entire sequence.

    Parameters
    ----------
    logprobs_per_expert : List[torch.Tensor]
        Each tensor has shape (seq_len, vocab_size).
    mask_indices : List[int]
        Indices of masked positions.
    sequence_length : int
        Total sequence length L.
    temperature : float

    Returns
    -------
    torch.Tensor
        Shape (seq_len, vocab_size), log-probabilities under π_cons,τ.
    """
    # Stack → (E, L, V)
    log_p = torch.stack(logprobs_per_expert, dim=0)  # (E, L, V)
    mean_log_p = log_p.mean(dim=0)                   # (L, V)
    scaled = mean_log_p / temperature
    log_pi_cons = F.log_softmax(scaled, dim=-1)       # (L, V)
    return log_pi_cons


# ---------------------------------------------------------------------------
# Epistemic uncertainty  (Eq. 5)
# ---------------------------------------------------------------------------

def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Shannon entropy of a probability distribution.

    Parameters
    ----------
    probs : torch.Tensor
        Shape (..., V).

    Returns
    -------
    torch.Tensor
        Scalar entropy value.
    """
    log_p = torch.log(probs + eps)
    return -(probs * log_p).sum(dim=-1)


def compute_u_ent_multi(
    probs_per_expert: List[torch.Tensor],
) -> float:
    """
    BALD-style epistemic uncertainty U_ent for multiple experts (Eq. 5).

    U_ent(s_t, a) = H( (1/E) Σ_e p^e(·|s_{t-1}, M) )
                    - (1/E) Σ_e H( p^e(·|s_{t-1}, M) )

    Averaged over all masked positions.

    Parameters
    ----------
    probs_per_expert : List[torch.Tensor]
        Each tensor has shape (|M|, vocab_size) – probabilities over masked
        positions from expert e.

    Returns
    -------
    float
        Scalar U_ent value ≥ 0.
    """
    # Stack → (E, |M|, V)
    p = torch.stack(probs_per_expert, dim=0).float()
    # Mixture distribution: mean over experts
    p_mix = p.mean(dim=0)  # (|M|, V)
    # Entropy of mixture  H(mixture)
    h_mix = _entropy(p_mix)  # (|M|,)
    # Mean expert entropy  (1/E) Σ_e H(p^e)
    h_experts = torch.stack([_entropy(pe) for pe in p], dim=0)  # (E, |M|)
    h_mean = h_experts.mean(dim=0)  # (|M|,)
    # BALD = H(mixture) - mean expert entropy, averaged over masked positions
    u_ent = (h_mix - h_mean).mean().item()
    return max(0.0, u_ent)  # numerically clip to ≥ 0


def compute_u_ent_single(
    probs: torch.Tensor,
) -> float:
    """
    Single-expert epistemic uncertainty U_ent^(1) (Eq. 5, E=1 special case).

    U_ent^(1)(s_t, a) = (1/|M|) Σ_{i∈M} H( p_ϕ(x_i | s_{t-1}, M) )

    Shannon entropy over masked sites, averaged.

    Parameters
    ----------
    probs : torch.Tensor
        Shape (|M|, vocab_size).

    Returns
    -------
    float
    """
    h = _entropy(probs.float())  # (|M|,)
    return h.mean().item()


# ---------------------------------------------------------------------------
# Diversity bonus  (Eq. 6)
# ---------------------------------------------------------------------------

def compute_u_div(
    child_sequence: str,
    parent_sequence: str,
) -> float:
    """
    Normalised Hamming diversity U_div (Eq. 6).

    U_div(s_t, a) = (1/L) Σ_i 1{ y_i^child ≠ y_i^parent }

    Parameters
    ----------
    child_sequence : str
        Fully-denoised child amino-acid sequence.
    parent_sequence : str
        Parent node's amino-acid sequence.

    Returns
    -------
    float
        Value in [0, 1].
    """
    L = len(child_sequence)
    if L == 0:
        return 0.0
    # Pad/truncate to common length (defensive)
    min_len = min(len(child_sequence), len(parent_sequence))
    diff = sum(
        c != p for c, p in zip(child_sequence[:min_len], parent_sequence[:min_len])
    )
    return diff / L


# ---------------------------------------------------------------------------
# PH-UCB-ME score  (Eq. 2)
# ---------------------------------------------------------------------------

def ph_ucb_me_score(
    node: MCTSNode,
    parent_visit_count: int,
    exploration_constant: float = 1.414,
    w_ent: float = 1.0,
    w_div: float = 1.0,
    pi_cons: float = 1.0,
) -> float:
    """
    PH-UCB-ME selection score for a single child node (Eq. 2).

    PH-UCB-ME(s_t, a) = Q(s_t, a)
        + c_p * sqrt( log N(s_t) / (1 + N(s_t, a)) )
          * π_cons,τ(a|s_t)
          * ( w_ent * U_ent(s_t, a) + w_div * U_div(s_t, a) )

    Parameters
    ----------
    node : MCTSNode
        The child node representing action a.
    parent_visit_count : int
        N(s_t) – visit count of the parent node.
    exploration_constant : float
        c_p (paper default √2 ≈ 1.414).
    w_ent : float
        Weight for U_ent.
    w_div : float
        Weight for U_div.
    pi_cons : float
        Scalar consensus prior probability π_cons,τ(a|s_t).  If unknown,
        pass 1.0 to fall back to unweighted UCB.

    Returns
    -------
    float
        PH-UCB-ME score.
    """
    q = node.q_value

    # UCB exploration term
    if parent_visit_count == 0:
        ucb_factor = float("inf")
    else:
        ucb_factor = exploration_constant * math.sqrt(
            math.log(parent_visit_count) / (1.0 + node.visit_count)
        )

    # Uncertainty bonus
    uncertainty_bonus = w_ent * node.u_ent + w_div * node.u_div

    score = q + ucb_factor * pi_cons * uncertainty_bonus
    return score


def select_child_ph_uct_me(
    parent: MCTSNode,
    exploration_constant: float = 1.414,
    w_ent: float = 1.0,
    w_div: float = 1.0,
    pi_cons_map: Optional[dict] = None,
) -> MCTSNode:
    """
    Select the best child of *parent* according to PH-UCT-ME (Eq. 2).

    Parameters
    ----------
    parent : MCTSNode
        Parent node whose children are candidates.
    exploration_constant : float
        c_p.
    w_ent : float
        Weight for U_ent bonus.
    w_div : float
        Weight for U_div bonus.
    pi_cons_map : dict | None
        Optional mapping {child_sequence: pi_cons_scalar}.  If None, all
        children are treated with pi_cons = 1.0 (uniform prior).

    Returns
    -------
    MCTSNode
        The selected child.
    """
    if not parent.children:
        raise ValueError("Parent node has no children to select from.")

    best_score = float("-inf")
    best_child = parent.children[0]

    for child in parent.children:
        pi_cons = 1.0
        if pi_cons_map is not None:
            pi_cons = pi_cons_map.get(child.sequence, 1.0)

        score = ph_ucb_me_score(
            node=child,
            parent_visit_count=parent.visit_count,
            exploration_constant=exploration_constant,
            w_ent=w_ent,
            w_div=w_div,
            pi_cons=pi_cons,
        )
        if score > best_score:
            best_score = score
            best_child = child

    return best_child
