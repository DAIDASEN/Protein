"""
pLDDT-guided progressive masking for MCTD-ME.

Implements Eq. 7 of arXiv:2509.15796:

    y_{t-1} ~ p_ϕ(y_{t-1} | Mask(y_t; M_t))

where M_t is the set of residue positions with pLDDT below the current
threshold, which decreases over MCTS tree depth (progressive masking).

The masking strategy:
  1. Run ESMFold on the current sequence to obtain per-residue pLDDT.
  2. Mask positions whose pLDDT < threshold.
  3. High-pLDDT residues are frozen / preserved during diffusion.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch


# Standard amino acid alphabet (same order used by DPLM-2)
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
MASK_TOKEN = "<mask>"
MASK_CHAR = "X"  # single-char stand-in when displaying masked sequences


# ---------------------------------------------------------------------------
# Threshold schedule
# ---------------------------------------------------------------------------

def compute_plddt_threshold(
    depth: int,
    max_depth: int,
    threshold_init: float = 0.7,
    threshold_min: float = 0.4,
    progressive: bool = True,
) -> float:
    """
    Compute the pLDDT masking threshold at a given tree depth.

    With progressive masking the threshold decreases linearly from
    *threshold_init* at depth 0 to *threshold_min* at *max_depth*.
    This progressively focuses diffusion on increasingly uncertain regions.

    Parameters
    ----------
    depth : int
        Current depth in the MCTS tree.
    max_depth : int
        Maximum tree depth (used to normalise the schedule).
    threshold_init : float
        Initial threshold at the root (default 0.70).
    threshold_min : float
        Minimum threshold at max depth (default 0.40).
    progressive : bool
        If False, always use *threshold_init*.

    Returns
    -------
    float
        pLDDT threshold for this depth.
    """
    if not progressive or max_depth == 0:
        return threshold_init
    fraction = min(depth / max_depth, 1.0)
    threshold = threshold_init - fraction * (threshold_init - threshold_min)
    return threshold


# ---------------------------------------------------------------------------
# Mask set computation
# ---------------------------------------------------------------------------

def get_mask_indices(
    plddt_scores: Sequence[float],
    threshold: float,
    min_mask_frac: float = 0.05,
    max_mask_frac: float = 0.80,
) -> List[int]:
    """
    Identify residue positions to mask based on pLDDT.

    Positions with pLDDT < *threshold* are included in the mask set M.
    At least *min_mask_frac* of positions will be masked (randomly chosen
    from low-confidence residues) so the diffusion step is never trivial.

    Parameters
    ----------
    plddt_scores : Sequence[float]
        Per-residue pLDDT values in [0, 1].
    threshold : float
        Positions below this value are masked.
    min_mask_frac : float
        Minimum fraction of sequence to mask (ensures exploration).
    max_mask_frac : float
        Maximum fraction of sequence to mask (preserves context).

    Returns
    -------
    List[int]
        Sorted list of 0-based indices to mask.
    """
    L = len(plddt_scores)
    # Positions below threshold
    mask_set = [i for i, v in enumerate(plddt_scores) if v < threshold]

    # Enforce minimum masking
    min_n = max(1, int(min_mask_frac * L))
    if len(mask_set) < min_n:
        # Add lowest pLDDT positions not already in mask_set
        remaining = [i for i in range(L) if i not in set(mask_set)]
        remaining_sorted = sorted(remaining, key=lambda i: plddt_scores[i])
        needed = min_n - len(mask_set)
        mask_set = mask_set + remaining_sorted[:needed]

    # Enforce maximum masking
    max_n = max(1, int(max_mask_frac * L))
    if len(mask_set) > max_n:
        # Keep the lowest-pLDDT ones
        mask_set = sorted(mask_set, key=lambda i: plddt_scores[i])[:max_n]

    return sorted(mask_set)


def apply_mask(
    sequence: str,
    mask_indices: List[int],
    mask_token: str = MASK_TOKEN,
) -> List[str]:
    """
    Replace residues at *mask_indices* with the mask token.

    Parameters
    ----------
    sequence : str
        Amino-acid sequence (single-letter codes).
    mask_indices : List[int]
        Positions to replace.
    mask_token : str
        Token to insert (default "<mask>").

    Returns
    -------
    List[str]
        Token list with mask tokens inserted (suitable for model tokenizer).
    """
    tokens = list(sequence)
    for idx in mask_indices:
        tokens[idx] = mask_token
    return tokens


def apply_mask_str(sequence: str, mask_indices: List[int]) -> str:
    """
    Return a masked sequence string using 'X' as the mask character.

    Useful for display and hashing.
    """
    tokens = list(sequence)
    for idx in mask_indices:
        tokens[idx] = MASK_CHAR
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Progressive masking main entry point
# ---------------------------------------------------------------------------

def get_mask_set_for_node(
    sequence: str,
    plddt_scores: Sequence[float],
    depth: int,
    max_depth: int,
    threshold_init: float = 0.7,
    threshold_min: float = 0.4,
    progressive: bool = True,
    min_mask_frac: float = 0.05,
    max_mask_frac: float = 0.80,
) -> Tuple[List[int], float]:
    """
    Full pipeline: given current sequence and pLDDT scores, return the mask
    set M_t for a node at the given tree depth.

    Parameters
    ----------
    sequence : str
        Current amino-acid sequence.
    plddt_scores : Sequence[float]
        Per-residue pLDDT values in [0, 1].
    depth : int
        Current MCTS tree depth.
    max_depth : int
        Maximum tree depth.
    threshold_init : float
        Initial pLDDT threshold.
    threshold_min : float
        Minimum pLDDT threshold.
    progressive : bool
        Whether to use progressive masking schedule.
    min_mask_frac : float
        Minimum fraction of sequence to mask.
    max_mask_frac : float
        Maximum fraction of sequence to mask.

    Returns
    -------
    mask_indices : List[int]
        Sorted positions to mask.
    threshold : float
        The pLDDT threshold used.
    """
    threshold = compute_plddt_threshold(
        depth=depth,
        max_depth=max_depth,
        threshold_init=threshold_init,
        threshold_min=threshold_min,
        progressive=progressive,
    )
    mask_indices = get_mask_indices(
        plddt_scores=plddt_scores,
        threshold=threshold,
        min_mask_frac=min_mask_frac,
        max_mask_frac=max_mask_frac,
    )
    return mask_indices, threshold


# ---------------------------------------------------------------------------
# De-novo (all-masked) initialisation
# ---------------------------------------------------------------------------

def make_all_masked_sequence(length: int, mask_token: str = MASK_TOKEN) -> List[str]:
    """
    Return an all-masked token list for de-novo design.

    Parameters
    ----------
    length : int
        Desired sequence length.

    Returns
    -------
    List[str]
        Token list of *length* mask tokens.
    """
    return [mask_token] * length


# ---------------------------------------------------------------------------
# Mask-application utilities for batches
# ---------------------------------------------------------------------------

def mask_sequence_tensor(
    token_ids: torch.Tensor,
    mask_indices: List[int],
    mask_token_id: int,
) -> torch.Tensor:
    """
    Apply mask to a 1-D token-ID tensor in-place (returns a new tensor).

    Parameters
    ----------
    token_ids : torch.Tensor
        Shape (L,), long tensor of token IDs.
    mask_indices : List[int]
        Positions to mask.
    mask_token_id : int
        Token ID of the [MASK] token.

    Returns
    -------
    torch.Tensor
        Masked token-ID tensor, shape (L,).
    """
    masked = token_ids.clone()
    for idx in mask_indices:
        masked[idx] = mask_token_id
    return masked
