"""
Composite reward functions for MCTD-ME.

Implements the three reward formulas from Appendix A.3 of arXiv:2509.15796:

  R_fold   (folding task)          – Eq. 8
  R_inv    (inverse folding task)  – Eq. 9
  R_motif  (motif scaffolding)     – Eq. 10

All rewards are normalised to [0, 1].

The ESMFold structure predictor is used as the primary critic to obtain:
  - Per-residue pLDDT scores
  - Predicted 3D backbone coordinates (for TM-score and RMSD)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ESMFold critic
# ---------------------------------------------------------------------------

class ESMFoldCritic:
    """
    Wraps ESMFold (facebook/esmfold_v1 via HuggingFace) to predict structure
    and per-residue pLDDT for a given amino-acid sequence.

    Provides:
      - predict(sequence)  →  dict with 'plddt', 'positions' (Cα coords)
    """

    def __init__(
        self,
        model_name: str = "facebook/esmfold_v1",
        device: Union[str, torch.device] = "cuda",
        cache_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import EsmForProteinFolding
        except ImportError as exc:
            raise ImportError(
                "Install `transformers>=4.31` to use ESMFold."
            ) from exc

        logger.info(f"Loading ESMFold: {self.model_name}")
        self._model = EsmForProteinFolding.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        if self.chunk_size is not None:
            self._model.esm.encoder.set_chunk_size(self.chunk_size)
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("ESMFold loaded.")

    def predict(self, sequence: str) -> Dict:
        """
        Fold *sequence* and return structure information.

        Returns
        -------
        dict with keys:
            'plddt'     : List[float]  – per-residue pLDDT in [0, 100]
            'positions' : np.ndarray   – Cα coordinates, shape (L, 3)
            'ptm'       : float        – predicted TM-score confidence
        """
        self._load()
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
        except Exception:
            from transformers import EsmTokenizer
            tokenizer = EsmTokenizer.from_pretrained(
                "facebook/esm2_t36_3B_UR50D", cache_dir=self.cache_dir
            )

        inputs = tokenizer(
            [sequence],
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # pLDDT: (1, L, 1) or (1, L) depending on transformers version
        plddt = outputs.plddt[0]  # (L,) or (L, 1)
        if plddt.dim() > 1:
            plddt = plddt.squeeze(-1)
        plddt_list = plddt.cpu().float().tolist()

        # Cα positions from atom37 representation (index 1 = Cα)
        # positions shape: (1, L, 37, 3) or (1, frames, L, 3)
        positions = outputs.positions  # varies by version
        if hasattr(positions, "shape"):
            pos = positions.cpu().float().numpy()
            # Handle different output shapes
            if pos.ndim == 4 and pos.shape[-2] == 37:
                # atom37 format: (1, L, 37, 3) – take Cα (index 1)
                ca_coords = pos[0, :, 1, :]  # (L, 3)
            elif pos.ndim == 4:
                # (1, frames, L, 3) – take last frame
                ca_coords = pos[0, -1, :, :]  # (L, 3)
            else:
                ca_coords = pos.reshape(-1, 3)
        else:
            ca_coords = np.zeros((len(sequence), 3), dtype=np.float32)

        ptm = float(getattr(outputs, "ptm", torch.tensor(0.0)).item())

        return {
            "plddt": plddt_list,          # List[float], values in [0, 100]
            "positions": ca_coords,        # np.ndarray (L, 3)
            "ptm": ptm,
        }


# ---------------------------------------------------------------------------
# Structural comparison utilities
# ---------------------------------------------------------------------------

def compute_tm_score(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
) -> float:
    """
    Compute TM-score between two sets of Cα coordinates.

    Uses the standard TM-score normalisation by the reference length:
        TM = (1/L_ref) Σ_i 1 / (1 + (d_i / d0)^2)
    where d0 = 1.24 * (L_ref - 15)^(1/3) - 1.8  (for L_ref > 15).

    Parameters
    ----------
    coords_pred : np.ndarray
        Shape (L_pred, 3).
    coords_ref : np.ndarray
        Shape (L_ref, 3).

    Returns
    -------
    float
        TM-score in [0, 1].
    """
    L_ref = len(coords_ref)
    L_pred = len(coords_pred)
    L_min = min(L_pred, L_ref)

    if L_min == 0:
        return 0.0

    # d0: normalisation distance
    if L_ref > 21:
        d0 = 1.24 * ((L_ref - 15) ** (1.0 / 3.0)) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.1)

    # Truncate to shorter length (no alignment performed here)
    p = coords_pred[:L_min]
    r = coords_ref[:L_min]

    d2 = np.sum((p - r) ** 2, axis=-1)  # (L_min,)
    tm = np.sum(1.0 / (1.0 + d2 / (d0 ** 2))) / L_ref
    return float(min(tm, 1.0))


def compute_rmsd(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
) -> float:
    """
    Compute Cα RMSD between two coordinate sets (no superposition).

    For a meaningful RMSD, caller should pre-align structures.

    Parameters
    ----------
    coords_pred : np.ndarray, shape (L, 3)
    coords_ref  : np.ndarray, shape (L, 3)

    Returns
    -------
    float
        RMSD in Å.
    """
    L = min(len(coords_pred), len(coords_ref))
    if L == 0:
        return float("inf")
    diff = coords_pred[:L] - coords_ref[:L]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))


def kabsch_superpose(
    mobile: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Superpose *mobile* onto *target* using the Kabsch algorithm.

    Returns the rotated mobile coordinates and the RMSD after superposition.

    Parameters
    ----------
    mobile : np.ndarray, shape (L, 3)
    target : np.ndarray, shape (L, 3)

    Returns
    -------
    mobile_aligned : np.ndarray, shape (L, 3)
    rmsd           : float
    """
    L = len(mobile)
    assert len(target) == L, "mobile and target must have equal length"

    # Centre
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)

    # Covariance matrix
    H = mobile_c.T @ target_c  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Ensure right-handed coordinate system
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T  # rotation matrix

    # Rotate mobile
    mobile_aligned = (mobile_c @ R.T) + target.mean(axis=0)
    rmsd = compute_rmsd(mobile_aligned, target)
    return mobile_aligned, rmsd


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

class CompositeReward:
    """
    Computes the task-specific composite reward for a designed sequence.

    Parameters
    ----------
    task : str
        One of {"folding", "inverse_folding", "motif_scaffolding"}.
    critic : ESMFoldCritic
        ESMFold critic used to predict structure.
    reward_weights_folding : tuple
        (alpha, beta, gamma) for R_fold.
    reward_weights_inv : tuple
        (w_aar, w_sctm, w_b) for R_inv.
    reward_weights_motif : tuple
        (w_plddt, w_rmsd, w_sctm, w_bonus) for R_motif.
    motif_rmsd_cutoff : float
        Hard cutoff for motif RMSD (Å).
    motif_sctm_cutoff : float
        Hard cutoff for motif scTM.
    """

    def __init__(
        self,
        task: str,
        critic: ESMFoldCritic,
        reward_weights_folding: Tuple[float, float, float] = (0.60, 0.40, 0.00),
        reward_weights_inv: Tuple[float, float, float] = (0.60, 0.35, 0.05),
        reward_weights_motif: Tuple[float, float, float, float] = (0.40, 0.30, 0.30, 0.20),
        motif_rmsd_cutoff: float = 1.0,
        motif_sctm_cutoff: float = 0.8,
        folding_plddt_bonus: float = 0.05,
    ) -> None:
        self.task = task
        self.critic = critic
        self.reward_weights_folding = reward_weights_folding
        self.reward_weights_inv = reward_weights_inv
        self.reward_weights_motif = reward_weights_motif
        self.motif_rmsd_cutoff = motif_rmsd_cutoff
        self.motif_sctm_cutoff = motif_sctm_cutoff
        self.folding_plddt_bonus = folding_plddt_bonus

    # ------------------------------------------------------------------
    # Task-specific rewards
    # ------------------------------------------------------------------

    def _reward_folding(
        self,
        sequence: str,
        target_coords: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """
        R_fold = α·TM + β·(1 - min(RMSD/10, 1)) + γ·pLDDT   (Appendix A.3)

        Default: (α, β, γ) = (0.60, 0.40, 0.00).
        If pLDDT is returned, γ = 0.05 and (α, β) are renormalised.
        """
        result = self.critic.predict(sequence)
        plddt_raw = result["plddt"]
        plddt_mean = float(np.mean(plddt_raw)) / 100.0  # normalise to [0,1]
        ca_pred = result["positions"]

        if target_coords is not None:
            # Superpose then compute TM / RMSD
            L = min(len(ca_pred), len(target_coords))
            if L > 0:
                aligned, rmsd = kabsch_superpose(ca_pred[:L], target_coords[:L])
                tm = compute_tm_score(aligned, target_coords[:L])
            else:
                rmsd, tm = 10.0, 0.0
        else:
            rmsd, tm = 10.0, 0.0

        alpha, beta, gamma = self.reward_weights_folding
        # If pLDDT available, include it with bonus weight and renormalise
        if gamma == 0.0 and plddt_mean > 0:
            g = self.folding_plddt_bonus
            total = alpha + beta + g
            alpha, beta, gamma = alpha / total, beta / total, g / total

        rmsd_term = 1.0 - min(rmsd / 10.0, 1.0)
        reward = alpha * tm + beta * rmsd_term + gamma * plddt_mean

        info = {
            "tm": tm,
            "rmsd": rmsd,
            "plddt": plddt_mean,
            "plddt_per_residue": [v / 100.0 for v in plddt_raw],
        }
        return float(reward), info

    def _reward_inverse_folding(
        self,
        sequence: str,
        native_sequence: str,
        target_coords: np.ndarray,
        biophysical_bonus: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        R_inv = 0.60·AAR + 0.35·scTM + 0.05·B   (Appendix A.3)

        AAR  = amino acid recovery vs native
        scTM = TM-score of ESMFold(designed_sequence) vs target backbone
        B    = small biophysical bonus (default 0)
        """
        # Amino acid recovery
        min_len = min(len(sequence), len(native_sequence))
        matches = sum(s == n for s, n in zip(sequence[:min_len], native_sequence[:min_len]))
        aar = matches / max(len(native_sequence), 1)

        # scTM: fold designed sequence and compare to target backbone
        result = self.critic.predict(sequence)
        ca_pred = result["positions"]
        plddt_raw = result["plddt"]

        L = min(len(ca_pred), len(target_coords))
        if L > 0:
            aligned, _ = kabsch_superpose(ca_pred[:L], target_coords[:L])
            sc_tm = compute_tm_score(aligned, target_coords[:L])
        else:
            sc_tm = 0.0

        w_aar, w_sctm, w_b = self.reward_weights_inv
        reward = w_aar * aar + w_sctm * sc_tm + w_b * biophysical_bonus

        info = {
            "aar": aar,
            "sc_tm": sc_tm,
            "biophysical_bonus": biophysical_bonus,
            "plddt": float(np.mean(plddt_raw)) / 100.0,
            "plddt_per_residue": [v / 100.0 for v in plddt_raw],
        }
        return float(reward), info

    def _reward_motif_scaffolding(
        self,
        sequence: str,
        motif_indices: List[int],
        motif_coords_ref: np.ndarray,
        motif_seq_ref: str,
    ) -> Tuple[float, Dict]:
        """
        R_motif  (Appendix A.3):

            g(x) = max(0, 1 - x/2)   if x < 1
                 = max(0, 0.2 - x/10) otherwise

            R_motif = 0.40·pLDDT_scaf
                    + 0.30·g(RMSD_motif)
                    + 0.30·scTM
                    + 0.20·I[RMSD_motif < 1 AND scTM > 0.8]

            R = 0 if motif not exactly preserved (hard constraint).
        """
        result = self.critic.predict(sequence)
        ca_pred = result["positions"]
        plddt_raw = result["plddt"]

        # Scaffold pLDDT (non-motif positions)
        scaf_indices = [i for i in range(len(sequence)) if i not in set(motif_indices)]
        if scaf_indices:
            plddt_scaf = np.mean([plddt_raw[i] for i in scaf_indices]) / 100.0
        else:
            plddt_scaf = float(np.mean(plddt_raw)) / 100.0

        # Motif RMSD: superpose motif Cα onto reference motif Cα
        motif_pred = ca_pred[motif_indices]  # (|motif|, 3)
        L_mot = min(len(motif_pred), len(motif_coords_ref))
        if L_mot > 0:
            _, motif_rmsd = kabsch_superpose(
                motif_pred[:L_mot], motif_coords_ref[:L_mot]
            )
        else:
            motif_rmsd = float("inf")

        # scTM: full-structure TM-score against any reference backbone
        # (here we use motif region for simplicity when full backbone unavailable)
        if len(ca_pred) > 0 and len(motif_coords_ref) > 0:
            L = min(len(ca_pred), len(motif_coords_ref))
            aligned, _ = kabsch_superpose(ca_pred[:L], motif_coords_ref[:L])
            sc_tm = compute_tm_score(aligned, motif_coords_ref[:L])
        else:
            sc_tm = 0.0

        # Hard constraint: motif must be exactly preserved
        # (check that sequence at motif positions matches reference)
        motif_preserved = all(
            sequence[idx] == motif_seq_ref[j]
            for j, idx in enumerate(motif_indices[:len(motif_seq_ref)])
        )
        if not motif_preserved:
            info = {
                "plddt_scaf": plddt_scaf,
                "motif_rmsd": motif_rmsd,
                "sc_tm": sc_tm,
                "motif_preserved": False,
                "plddt_per_residue": [v / 100.0 for v in plddt_raw],
            }
            return 0.0, info

        # g(x) function
        def g(x: float) -> float:
            if x < 1.0:
                return max(0.0, 1.0 - x / 2.0)
            else:
                return max(0.0, 0.2 - x / 10.0)

        w_plddt, w_rmsd, w_sctm, w_bonus = self.reward_weights_motif
        bonus_indicator = float(
            motif_rmsd < self.motif_rmsd_cutoff and sc_tm > self.motif_sctm_cutoff
        )
        reward = (
            w_plddt * plddt_scaf
            + w_rmsd * g(motif_rmsd)
            + w_sctm * sc_tm
            + w_bonus * bonus_indicator
        )

        info = {
            "plddt_scaf": plddt_scaf,
            "motif_rmsd": motif_rmsd,
            "sc_tm": sc_tm,
            "motif_preserved": True,
            "bonus_indicator": bonus_indicator,
            "plddt_per_residue": [v / 100.0 for v in plddt_raw],
        }
        return float(reward), info

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        sequence: str,
        target_coords: Optional[np.ndarray] = None,
        native_sequence: Optional[str] = None,
        motif_indices: Optional[List[int]] = None,
        motif_coords_ref: Optional[np.ndarray] = None,
        motif_seq_ref: Optional[str] = None,
        biophysical_bonus: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Compute the composite reward for *sequence*.

        Parameters depend on the task:
          - folding          : target_coords (optional, Cα of target structure)
          - inverse_folding  : native_sequence, target_coords
          - motif_scaffolding: motif_indices, motif_coords_ref, motif_seq_ref

        Returns
        -------
        reward : float
            Composite reward in [0, 1] (approximately).
        info   : dict
            Detailed metric breakdown.
        """
        if self.task == "folding":
            return self._reward_folding(sequence, target_coords=target_coords)
        elif self.task == "inverse_folding":
            if native_sequence is None or target_coords is None:
                raise ValueError(
                    "inverse_folding requires `native_sequence` and `target_coords`."
                )
            return self._reward_inverse_folding(
                sequence,
                native_sequence=native_sequence,
                target_coords=target_coords,
                biophysical_bonus=biophysical_bonus,
            )
        elif self.task == "motif_scaffolding":
            if any(x is None for x in [motif_indices, motif_coords_ref, motif_seq_ref]):
                raise ValueError(
                    "motif_scaffolding requires motif_indices, "
                    "motif_coords_ref, and motif_seq_ref."
                )
            return self._reward_motif_scaffolding(
                sequence,
                motif_indices=motif_indices,
                motif_coords_ref=motif_coords_ref,
                motif_seq_ref=motif_seq_ref,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
