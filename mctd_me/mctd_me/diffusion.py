"""
Masked diffusion rollout engine for MCTD-ME.

Implements the masked discrete diffusion process used in DPLM-2 and the
multi-step reverse diffusion rollout described in Algorithm 1.

Key concepts (Sec. 3.1, Eq. 7):
  - Forward process: gradually mask amino-acid tokens with probability β_t.
  - Reverse process: expert model predicts unmasked tokens at each step.
  - MCTD-ME uses a single reverse step (deterministic unmasking / argmax)
    from the current masked node to generate a child candidate.

The engine supports:
  1. Single-step denoising (used in PH-UCB-ME expansion).
  2. Multi-step reverse diffusion for deeper exploration.
  3. Temperature-controlled sampling.

Reference: arXiv:2509.15796 Sec. 3 and Appendix A.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from mctd_me.experts import BaseExpert
from mctd_me.masking import (
    get_mask_set_for_node,
    apply_mask,
    AA_VOCAB,
    MASK_TOKEN,
)
from mctd_me.selection import (
    compute_consensus_prior,
    compute_u_ent_multi,
    compute_u_ent_single,
    compute_u_div,
)

logger = logging.getLogger(__name__)

AA20 = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Forward masking schedules
# ---------------------------------------------------------------------------

def linear_noise_schedule(
    num_steps: int,
    beta_start: float = 0.001,
    beta_end: float = 0.999,
) -> torch.Tensor:
    """
    Linear schedule for the masking probability β_t.

    At step t (1-indexed) the cumulative mask probability is
    ᾱ_t = prod_{s=1}^{t} (1 - β_s).

    Returns
    -------
    torch.Tensor
        Shape (num_steps,), masking probabilities β_t.
    """
    return torch.linspace(beta_start, beta_end, num_steps)


def cosine_noise_schedule(
    num_steps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).

    Returns
    -------
    torch.Tensor
        Shape (num_steps,), cumulative mask fractions ᾱ_t.
    """
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # Clip to (0, 1)
    alphas_cumprod = torch.clamp(alphas_cumprod, 1e-6, 1.0 - 1e-6)
    return alphas_cumprod[1:]  # (num_steps,)


# ---------------------------------------------------------------------------
# Single-step denoising rollout
# ---------------------------------------------------------------------------

class MaskedDiffusionRollout:
    """
    Performs a single masked-diffusion rollout for a given expert.

    Algorithm (from Eq. 7):
      y_{t-1} ~ p_ϕ(y_{t-1} | Mask(y_t; M_t))

    In MCTD-ME the "rollout" starts from the current node's sequence, masks
    the low-pLDDT positions, and lets the expert fill them in once (or via
    multi-step iterative refinement).

    Parameters
    ----------
    expert : BaseExpert
        The protein language model used for denoising.
    num_diffusion_steps : int
        Number of reverse diffusion steps (default 150, Table 10).
    temperature : float
        Sampling temperature τ.
    deterministic : bool
        If True, use argmax decoding (paper default).
    """

    def __init__(
        self,
        expert: BaseExpert,
        num_diffusion_steps: int = 150,
        temperature: float = 1.0,
        deterministic: bool = True,
    ) -> None:
        self.expert = expert
        self.num_diffusion_steps = num_diffusion_steps
        self.temperature = temperature
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    # Multi-step iterative refinement
    # ------------------------------------------------------------------

    def _iterative_refinement(
        self,
        sequence: str,
        mask_indices: List[int],
        structure: Optional[object] = None,
    ) -> str:
        """
        Iteratively unmask residues over *num_diffusion_steps* reverse steps.

        Strategy: at each step unmask a fraction of the remaining masked
        positions (those with highest model confidence), until all are filled.
        This mirrors the MDLM / DPLM-2 reverse diffusion process.

        Parameters
        ----------
        sequence : str
            Input sequence with mask positions to fill.
        mask_indices : List[int]
            Indices of positions to be predicted.
        structure : optional
            Passed to the expert for structure-conditioned experts.

        Returns
        -------
        str
            Fully denoised sequence.
        """
        current_seq = list(sequence)
        remaining_mask = set(mask_indices)
        n_mask = len(mask_indices)
        if n_mask == 0:
            return sequence

        step_size = max(1, n_mask // self.num_diffusion_steps)

        for step in range(self.num_diffusion_steps):
            if not remaining_mask:
                break
            remaining_list = sorted(remaining_mask)
            # Get log-probs for all remaining masked positions
            log_probs = self.expert.get_logprobs(
                "".join(current_seq), remaining_list, structure=structure
            )  # (|remaining|, 20)
            probs = torch.exp(log_probs)  # (|remaining|, 20)

            # Confidence = max probability over amino acids
            max_probs, argmax_ids = probs.max(dim=-1)  # (|remaining|,)

            # Sort by confidence (descending) and unmask the most confident
            sorted_order = torch.argsort(max_probs, descending=True)
            to_unmask_count = min(step_size, len(remaining_list))
            to_unmask_indices = sorted_order[:to_unmask_count].tolist()

            for local_i in to_unmask_indices:
                global_pos = remaining_list[local_i]
                aa_id = argmax_ids[local_i].item()
                current_seq[global_pos] = AA20[aa_id]
                remaining_mask.discard(global_pos)

        # Fill any remaining (edge case)
        if remaining_mask:
            remaining_list = sorted(remaining_mask)
            log_probs = self.expert.get_logprobs(
                "".join(current_seq), remaining_list, structure=structure
            )
            for i, pos in enumerate(remaining_list):
                aa_id = log_probs[i].argmax().item()
                current_seq[pos] = AA20[aa_id]

        return "".join(current_seq)

    # ------------------------------------------------------------------
    # Single-step denoising (used in MCTS expansion)
    # ------------------------------------------------------------------

    def rollout(
        self,
        sequence: str,
        mask_indices: List[int],
        structure: Optional[object] = None,
        multi_step: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        """
        Perform one masked-diffusion rollout.

        Parameters
        ----------
        sequence : str
            Current sequence (parent node).
        mask_indices : List[int]
            Positions to re-generate.
        structure : optional
            Backbone structure for structure-conditioned experts.
        multi_step : bool
            If True use iterative refinement; if False use single-step argmax.

        Returns
        -------
        new_sequence : str
            Denoised child sequence.
        log_probs : torch.Tensor
            Shape (|M|, 20) – expert log-probabilities at masked positions
            (from the final denoising step, used to compute U_ent/U_div).
        """
        if not mask_indices:
            # Nothing to denoise
            dummy_lp = torch.zeros(0, 20)
            return sequence, dummy_lp

        if multi_step and self.num_diffusion_steps > 1:
            new_seq = self._iterative_refinement(
                sequence, mask_indices, structure=structure
            )
        else:
            new_seq = self.expert.sample(
                sequence,
                mask_indices,
                temperature=self.temperature if not self.deterministic else 1e-8,
                structure=structure,
            )

        # Compute log-probs at the final (unmasked) positions for uncertainty
        log_probs = self.expert.get_logprobs(new_seq, mask_indices, structure=structure)
        return new_seq, log_probs


# ---------------------------------------------------------------------------
# Multi-expert rollout engine
# ---------------------------------------------------------------------------

class MultiExpertRolloutEngine:
    """
    Orchestrates rollouts across multiple expert models.

    For each expert, runs *num_rollouts* stochastic diffusion rollouts.
    Aggregates results to compute:
      - Candidate child sequences
      - Expert log-probabilities (for U_ent computation via BALD)
      - Diversity bonuses (U_div via Hamming distance)
      - Consensus prior π_cons,τ (Eq. 3)

    Parameters
    ----------
    experts : List[BaseExpert]
        Ordered list of expert models.
    num_rollouts : int
        k_roll – rollouts per expert per expansion.
    num_diffusion_steps : int
        Reverse diffusion steps per rollout.
    temperature : float
        Sampling temperature τ.
    deterministic : bool
        Use argmax decoding (paper default: True).
    w_ent : float
        Weight for U_ent bonus.
    w_div : float
        Weight for U_div bonus.
    """

    def __init__(
        self,
        experts: List[BaseExpert],
        num_rollouts: int = 3,
        num_diffusion_steps: int = 150,
        temperature: float = 1.0,
        deterministic: bool = True,
        w_ent: float = 1.0,
        w_div: float = 1.0,
    ) -> None:
        self.experts = experts
        self.num_rollouts = num_rollouts
        self.num_diffusion_steps = num_diffusion_steps
        self.temperature = temperature
        self.deterministic = deterministic
        self.w_ent = w_ent
        self.w_div = w_div

        self._rollout_engines = [
            MaskedDiffusionRollout(
                expert=e,
                num_diffusion_steps=num_diffusion_steps,
                temperature=temperature,
                deterministic=deterministic,
            )
            for e in experts
        ]

    def expand(
        self,
        sequence: str,
        mask_indices: List[int],
        structure: Optional[object] = None,
        parent_sequence: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate candidate child sequences by running multi-expert rollouts.

        Returns a list of dicts, one per unique candidate sequence:
          {
            'sequence'  : str,
            'u_ent'     : float,
            'u_div'     : float,
            'pi_cons'   : float,     # scalar consensus prior for this action
            'log_probs_per_expert': List[torch.Tensor],  # (|M|, 20) per expert
          }

        Parameters
        ----------
        sequence : str
            Parent node sequence.
        mask_indices : List[int]
            Positions to re-generate.
        structure : optional
            Backbone coordinates (for structure-conditioned experts).
        parent_sequence : str | None
            Used for diversity bonus; defaults to *sequence* if None.

        Returns
        -------
        List[Dict]
            Candidate entries sorted by u_ent + u_div (descending).
        """
        if parent_sequence is None:
            parent_sequence = sequence

        candidates: Dict[str, Dict] = {}  # sequence → info dict

        # Collect log-probs from all experts (for BALD computation)
        all_expert_logprobs: List[torch.Tensor] = []
        all_expert_probs: List[torch.Tensor] = []

        for expert_idx, (expert, engine) in enumerate(
            zip(self.experts, self._rollout_engines)
        ):
            expert_logprobs_list: List[torch.Tensor] = []

            for roll in range(self.num_rollouts):
                new_seq, log_probs = engine.rollout(
                    sequence=sequence,
                    mask_indices=mask_indices,
                    structure=structure,
                    multi_step=True,
                )
                expert_logprobs_list.append(log_probs)

                if new_seq not in candidates:
                    candidates[new_seq] = {
                        "sequence": new_seq,
                        "log_probs_per_expert": [],
                        "_probs_for_bald": [],
                    }

            # Average expert log-probs across rollouts for this expert
            if expert_logprobs_list:
                avg_lp = torch.stack(expert_logprobs_list, dim=0).mean(dim=0)  # (|M|, 20)
                all_expert_logprobs.append(avg_lp)
                all_expert_probs.append(torch.exp(avg_lp))

        # ------------------------------------------------------------------
        # Compute consensus prior π_cons,τ  (Eq. 3)
        # ------------------------------------------------------------------
        if len(all_expert_logprobs) > 0 and len(mask_indices) > 0:
            # (E, |M|, V)
            log_p_stack = torch.stack(all_expert_logprobs, dim=0)
            mean_log_p = log_p_stack.mean(dim=0)  # (|M|, V)
            scaled = mean_log_p / self.temperature
            pi_cons_full = torch.softmax(scaled, dim=-1)  # (|M|, V)
            # Per-candidate scalar π_cons = geometric mean over mask positions
            # of the probability assigned to the chosen amino acid
        else:
            pi_cons_full = None

        # ------------------------------------------------------------------
        # Compute BALD epistemic uncertainty U_ent  (Eq. 5)
        # ------------------------------------------------------------------
        if len(all_expert_probs) > 1:
            u_ent_global = compute_u_ent_multi(all_expert_probs)
        elif len(all_expert_probs) == 1:
            u_ent_global = compute_u_ent_single(all_expert_probs[0])
        else:
            u_ent_global = 0.0

        # ------------------------------------------------------------------
        # Assign per-candidate scores
        # ------------------------------------------------------------------
        result_list = []
        for seq, info in candidates.items():
            u_div = compute_u_div(seq, parent_sequence)

            # Per-candidate π_cons: mean probability assigned to chosen tokens
            pi_cons_scalar = 1.0
            if pi_cons_full is not None and len(mask_indices) > 0:
                pi_vals = []
                for i, pos in enumerate(mask_indices):
                    if pos < len(seq):
                        aa_char = seq[pos]
                        aa_idx = AA20.find(aa_char)
                        if 0 <= aa_idx < 20:
                            pi_vals.append(pi_cons_full[i, aa_idx].item())
                if pi_vals:
                    pi_cons_scalar = float(sum(pi_vals) / len(pi_vals))

            info.update(
                {
                    "u_ent": u_ent_global,
                    "u_div": u_div,
                    "pi_cons": pi_cons_scalar,
                }
            )
            result_list.append(info)

        # Sort by combined bonus score (descending) for top-K selection
        result_list.sort(
            key=lambda x: self.w_ent * x["u_ent"] + self.w_div * x["u_div"],
            reverse=True,
        )
        return result_list
