"""
Evaluation metrics for MCTD-ME.

Implements all metrics described in Sec. 4 and Appendix A of arXiv:2509.15796:
  - TM-score    : topology similarity [0, 1]
  - scTM        : self-consistency TM-score (ESMFold vs target backbone)
  - RMSD        : global Cα RMSD in Å
  - MotifRMSD   : Cα RMSD on motif region after motif-aligned superposition
  - pLDDT       : per-residue confidence / 100 → [0, 1]
  - AAR         : amino acid recovery (fraction matching native sequence)

All coordinate-based metrics use Kabsch superposition before computing RMSD.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------

def tm_score(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    d0_override: Optional[float] = None,
) -> float:
    """
    Compute TM-score between two sets of Cα coordinates (no alignment).

    TM-score = (1/L_ref) Σ_i 1 / (1 + (d_i/d0)^2)
    d0 = 1.24 * (L_ref - 15)^(1/3) - 1.8  for L_ref > 21, else d0=0.5

    Parameters
    ----------
    coords_pred : np.ndarray, shape (L_pred, 3)
    coords_ref  : np.ndarray, shape (L_ref, 3)
    d0_override : float | None
        Override d0 normalisation distance (useful for short proteins).

    Returns
    -------
    float in [0, 1].
    """
    L_ref = len(coords_ref)
    L_pred = len(coords_pred)
    L_min = min(L_pred, L_ref)
    if L_min == 0:
        return 0.0

    if d0_override is not None:
        d0 = d0_override
    elif L_ref > 21:
        d0 = 1.24 * ((L_ref - 15) ** (1.0 / 3.0)) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.1)

    p = coords_pred[:L_min]
    r = coords_ref[:L_min]
    d2 = np.sum((p - r) ** 2, axis=-1)
    score = np.sum(1.0 / (1.0 + d2 / d0 ** 2)) / L_ref
    return float(min(score, 1.0))


# ---------------------------------------------------------------------------
# Kabsch superposition
# ---------------------------------------------------------------------------

def kabsch_superpose(
    mobile: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Superpose *mobile* onto *target* via the Kabsch algorithm.

    Parameters
    ----------
    mobile : np.ndarray, shape (L, 3)
    target : np.ndarray, shape (L, 3)

    Returns
    -------
    mobile_aligned : np.ndarray, shape (L, 3)
    rmsd           : float, Å
    """
    assert len(mobile) == len(target), "mobile and target must have equal length"
    L = len(mobile)
    if L == 0:
        return mobile.copy(), 0.0

    mob_c = mobile - mobile.mean(axis=0)
    tgt_c = target - target.mean(axis=0)

    H = mob_c.T @ tgt_c
    U_mat, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U_mat.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U_mat.T

    aligned = (mob_c @ R.T) + target.mean(axis=0)
    rmsd_val = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1))))
    return aligned, rmsd_val


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def rmsd(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    superpose: bool = True,
) -> float:
    """
    Compute Cα RMSD.

    Parameters
    ----------
    coords_pred : np.ndarray, shape (L, 3)
    coords_ref  : np.ndarray, shape (L, 3)
    superpose   : bool
        If True, perform Kabsch superposition first (recommended).

    Returns
    -------
    float in Å.
    """
    L = min(len(coords_pred), len(coords_ref))
    if L == 0:
        return float("inf")
    p = coords_pred[:L]
    r = coords_ref[:L]
    if superpose:
        _, rmsd_val = kabsch_superpose(p, r)
        return rmsd_val
    diff = p - r
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))


# ---------------------------------------------------------------------------
# Motif RMSD
# ---------------------------------------------------------------------------

def motif_rmsd(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    motif_indices_pred: List[int],
    motif_indices_ref: Optional[List[int]] = None,
) -> float:
    """
    Compute Cα RMSD on the motif region after motif-aligned superposition.

    The superposition is performed using **only** the motif residues.
    The RMSD is then evaluated on those same motif residues.

    Parameters
    ----------
    coords_pred : np.ndarray, shape (L_pred, 3)
        Full predicted Cα coordinates.
    coords_ref : np.ndarray, shape (L_ref, 3)
        Full reference Cα coordinates.
    motif_indices_pred : List[int]
        Motif residue indices in the predicted structure.
    motif_indices_ref : List[int] | None
        Motif residue indices in the reference structure; defaults to
        *motif_indices_pred* (same positions in both).

    Returns
    -------
    float, Å.
    """
    if motif_indices_ref is None:
        motif_indices_ref = motif_indices_pred

    n_mot = min(len(motif_indices_pred), len(motif_indices_ref))
    if n_mot == 0:
        return float("inf")

    mob_motif = coords_pred[motif_indices_pred[:n_mot]]  # (n_mot, 3)
    tgt_motif = coords_ref[motif_indices_ref[:n_mot]]    # (n_mot, 3)

    _, rmsd_val = kabsch_superpose(mob_motif, tgt_motif)
    return rmsd_val


# ---------------------------------------------------------------------------
# scTM (self-consistency TM-score)
# ---------------------------------------------------------------------------

def sctm_score(
    coords_pred: np.ndarray,
    coords_target: np.ndarray,
) -> float:
    """
    Self-consistency TM-score:  TM-score between ESMFold(designed) and the
    target backbone, after Kabsch superposition.

    Parameters
    ----------
    coords_pred : np.ndarray
        Cα coords of ESMFold(designed_sequence), shape (L_pred, 3).
    coords_target : np.ndarray
        Cα coords of the target backbone, shape (L_ref, 3).

    Returns
    -------
    float in [0, 1].
    """
    L = min(len(coords_pred), len(coords_target))
    if L == 0:
        return 0.0
    aligned, _ = kabsch_superpose(coords_pred[:L], coords_target[:L])
    return tm_score(aligned, coords_target[:L])


# ---------------------------------------------------------------------------
# pLDDT
# ---------------------------------------------------------------------------

def mean_plddt(plddt_list: Sequence[float], scale: float = 100.0) -> float:
    """
    Compute mean pLDDT normalised to [0, 1].

    Parameters
    ----------
    plddt_list : Sequence[float]
        Per-residue pLDDT values (raw from ESMFold, typically in [0, 100]).
    scale : float
        Divisor to normalise to [0, 1] (100.0 for ESMFold output).

    Returns
    -------
    float in [0, 1].
    """
    if not plddt_list:
        return 0.0
    return float(np.mean(plddt_list)) / scale


def normalise_plddt(
    plddt_list: Sequence[float], scale: float = 100.0
) -> List[float]:
    """Normalise a per-residue pLDDT list to [0, 1]."""
    return [v / scale for v in plddt_list]


# ---------------------------------------------------------------------------
# Amino acid recovery (AAR)
# ---------------------------------------------------------------------------

def amino_acid_recovery(
    designed_sequence: str,
    native_sequence: str,
    mask_only: Optional[List[int]] = None,
) -> float:
    """
    Compute the amino acid recovery rate.

    AAR = (number of positions where designed matches native) / L

    Parameters
    ----------
    designed_sequence : str
    native_sequence   : str
    mask_only : List[int] | None
        If provided, compute recovery only over these positions.

    Returns
    -------
    float in [0, 1].
    """
    if mask_only is not None:
        positions = mask_only
        L = len(positions)
    else:
        L = min(len(designed_sequence), len(native_sequence))
        positions = list(range(L))

    if L == 0:
        return 0.0

    matches = sum(
        designed_sequence[i] == native_sequence[i]
        for i in positions
        if i < len(designed_sequence) and i < len(native_sequence)
    )
    return matches / L


# ---------------------------------------------------------------------------
# Unified metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    designed_sequence: str,
    native_sequence: Optional[str],
    coords_pred: np.ndarray,
    coords_target: np.ndarray,
    plddt_raw: Sequence[float],
    motif_indices: Optional[List[int]] = None,
    motif_coords_ref: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the full suite of MCTD-ME evaluation metrics.

    Parameters
    ----------
    designed_sequence : str
    native_sequence   : str | None
    coords_pred       : np.ndarray, shape (L, 3) – Cα of ESMFold(designed)
    coords_target     : np.ndarray, shape (L, 3) – Cα of target structure
    plddt_raw         : Sequence[float] – per-residue pLDDT from ESMFold
    motif_indices     : List[int] | None – motif positions
    motif_coords_ref  : np.ndarray | None – motif reference coordinates

    Returns
    -------
    dict with keys: tm, sc_tm, rmsd_global, rmsd_motif, plddt_mean, aar
    """
    metrics: Dict[str, float] = {}

    L = min(len(coords_pred), len(coords_target))
    if L > 0:
        aligned_pred, global_rmsd = kabsch_superpose(
            coords_pred[:L], coords_target[:L]
        )
        metrics["tm"] = tm_score(aligned_pred, coords_target[:L])
        metrics["sc_tm"] = sctm_score(coords_pred, coords_target)
        metrics["rmsd_global"] = global_rmsd
    else:
        metrics["tm"] = 0.0
        metrics["sc_tm"] = 0.0
        metrics["rmsd_global"] = float("inf")

    metrics["plddt_mean"] = mean_plddt(plddt_raw)

    if native_sequence is not None:
        metrics["aar"] = amino_acid_recovery(designed_sequence, native_sequence)
    else:
        metrics["aar"] = float("nan")

    if motif_indices is not None and motif_coords_ref is not None:
        metrics["rmsd_motif"] = motif_rmsd(
            coords_pred,
            motif_coords_ref,
            motif_indices_pred=motif_indices,
        )
    else:
        metrics["rmsd_motif"] = float("nan")

    return metrics
