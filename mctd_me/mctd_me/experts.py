"""
Expert model wrappers for MCTD-ME.

Supported experts:
  - DPLM2Expert  : wraps airkingbd/dplm_150m, dplm_650m, dplm_3b from HuggingFace.
                   These are masked discrete diffusion protein language models.
  - ProteinMPNNExpert : wraps the ProteinMPNN inverse-folding model.

All experts expose a common interface:
  expert.get_logprobs(sequence, mask_indices)  →  (|M|, vocab_size) log-prob tensor
  expert.sample(sequence, mask_indices, ...)   →  denoised sequence string

Reference: arXiv:2509.15796 Sec. 3 and Appendix A.
"""

from __future__ import annotations

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Amino acid alphabet (matches DPLM-2 tokenizer order)
# ---------------------------------------------------------------------------

AA20 = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseExpert(ABC):
    """Abstract base for expert models used in MCTD-ME."""

    @abstractmethod
    def get_logprobs(
        self,
        sequence: str,
        mask_indices: List[int],
        structure: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Return log-probabilities for each masked position.

        Parameters
        ----------
        sequence : str
            Full amino-acid sequence with some positions to be predicted.
        mask_indices : List[int]
            Indices of positions that are masked (to be predicted).
        structure : optional
            For structure-conditioned experts (e.g., ProteinMPNN).

        Returns
        -------
        torch.Tensor
            Shape (|mask_indices|, vocab_size), log-probabilities
            p_ϕ(x_i | s, M) for each masked position i ∈ M.
        """

    @abstractmethod
    def sample(
        self,
        sequence: str,
        mask_indices: List[int],
        temperature: float = 1.0,
        structure: Optional[object] = None,
    ) -> str:
        """
        Sample a fully-denoised sequence from this expert.

        Parameters
        ----------
        sequence : str
            Current sequence (may contain gaps / placeholder characters
            at *mask_indices*).
        mask_indices : List[int]
            Positions to fill in.
        temperature : float
            Sampling temperature.
        structure : optional
            For structure-conditioned experts.

        Returns
        -------
        str
            Denoised amino-acid sequence (same length as input).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name for logging."""


# ---------------------------------------------------------------------------
# DPLM-2 expert
# ---------------------------------------------------------------------------

class DPLM2Expert(BaseExpert):
    """
    Wrapper around a DPLM-2 masked protein language model.

    The model is loaded from HuggingFace (airkingbd/dplm_150m etc.) via the
    ``transformers`` library using ``AutoModelForMaskedLM``.  DPLM-2 is a
    masked discrete diffusion model trained on UniRef50 protein sequences.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID, e.g. "airkingbd/dplm_150m".
    device : str | torch.device
        Target device.
    cache_dir : str | None
        HuggingFace cache directory.
    half_precision : bool
        Load in fp16 to save GPU memory.
    """

    def __init__(
        self,
        model_name: str = "airkingbd/dplm_150m",
        device: Union[str, torch.device] = "cuda",
        cache_dir: Optional[str] = None,
        half_precision: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.half_precision = half_precision

        self._model = None
        self._tokenizer = None
        self._mask_token_id: Optional[int] = None
        self._vocab_size: Optional[int] = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmTokenizer, EsmForMaskedLM
        except ImportError as e:
            raise ImportError("Install `transformers` to use DPLM2Expert.") from e

        logger.info(f"Loading DPLM-2 model: {self.model_name}")

        # DPLM-2 uses the ESM tokenizer schema
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback to ESM tokenizer which DPLM-2 is compatible with
            self._tokenizer = EsmTokenizer.from_pretrained(
                "facebook/esm2_t6_8M_UR50D",
                cache_dir=self.cache_dir,
            )

        try:
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            self._model = EsmForMaskedLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

        if self.half_precision:
            self._model = self._model.half()
        self._model = self._model.to(self.device)
        self._model.eval()

        self._mask_token_id = self._tokenizer.mask_token_id
        self._vocab_size = self._tokenizer.vocab_size
        logger.info(
            f"DPLM-2 ({self.model_name}) loaded. "
            f"mask_token_id={self._mask_token_id}, vocab_size={self._vocab_size}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, sequence: str, mask_indices: List[int]) -> dict:
        """
        Tokenise *sequence* and apply mask tokens at *mask_indices*.

        Returns a dict ready to pass to the model.
        """
        self._load()
        tokens = list(sequence)
        for idx in mask_indices:
            tokens[idx] = self._tokenizer.mask_token
        masked_seq_str = " ".join(tokens)  # ESM tokenizer expects space-separated
        encoded = self._tokenizer(
            masked_seq_str,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _forward(self, sequence: str, mask_indices: List[int]) -> torch.Tensor:
        """
        Run a forward pass and return logits tensor, shape (seq_len, vocab_size).
        """
        self._load()
        inputs = self._encode(sequence, mask_indices)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # outputs.logits: (1, seq_len_with_special, vocab_size)
        logits = outputs.logits[0]  # (seq_len_with_special, V)
        return logits

    def _aa_logprobs_at_positions(
        self,
        logits: torch.Tensor,
        mask_indices: List[int],
    ) -> torch.Tensor:
        """
        Extract log-probabilities at masked positions for the 20 AA tokens.

        The ESM/DPLM-2 tokenizer adds [CLS] at position 0 and [EOS] at the
        end, so actual residue i corresponds to logits position i+1.

        Returns
        -------
        torch.Tensor
            Shape (|M|, 20) – log-probs for the 20 standard amino acids.
        """
        self._load()
        # Map 20 AA characters to their token IDs
        aa_ids = [
            self._tokenizer.convert_tokens_to_ids(aa)
            for aa in list(AA20)
        ]
        # Offset by 1 for [CLS] token
        rows = []
        for idx in mask_indices:
            pos_logits = logits[idx + 1]  # (V,)
            aa_logits = pos_logits[aa_ids]  # (20,)
            rows.append(F.log_softmax(aa_logits, dim=-1))
        if not rows:
            return torch.zeros(0, 20, device=self.device)
        return torch.stack(rows, dim=0)  # (|M|, 20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_logprobs(
        self,
        sequence: str,
        mask_indices: List[int],
        structure: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Return log p_ϕ^e(x_i | s, M) for each masked position i ∈ M.

        Returns
        -------
        torch.Tensor
            Shape (|M|, 20) over the 20 standard amino acids.
        """
        if not mask_indices:
            return torch.zeros(0, 20, device=self.device)
        logits = self._forward(sequence, mask_indices)
        return self._aa_logprobs_at_positions(logits, mask_indices)

    def sample(
        self,
        sequence: str,
        mask_indices: List[int],
        temperature: float = 1.0,
        structure: Optional[object] = None,
    ) -> str:
        """
        Sample a denoised sequence by filling in all masked positions.

        Uses temperature-controlled categorical sampling (or argmax when
        temperature → 0).  Deterministic unmasking (argmax) is the paper
        default (Table 10).

        Returns
        -------
        str
            Fully-denoised amino-acid sequence.
        """
        if not mask_indices:
            return sequence
        logits = self._forward(sequence, mask_indices)
        log_probs = self._aa_logprobs_at_positions(logits, mask_indices)  # (|M|, 20)
        aa_list = list(AA20)

        result = list(sequence)
        for i, idx in enumerate(mask_indices):
            if temperature <= 1e-6:
                # Deterministic argmax (paper default)
                aa_id = log_probs[i].argmax().item()
            else:
                probs = torch.exp(log_probs[i] / temperature)
                probs = probs / probs.sum()
                aa_id = torch.multinomial(probs, 1).item()
            result[idx] = aa_list[aa_id]

        return "".join(result)

    @property
    def name(self) -> str:
        return f"DPLM2({self.model_name.split('/')[-1]})"


# ---------------------------------------------------------------------------
# ProteinMPNN expert
# ---------------------------------------------------------------------------

class ProteinMPNNExpert(BaseExpert):
    """
    Wrapper around ProteinMPNN for inverse-folding.

    ProteinMPNN takes backbone coordinates as input and outputs amino-acid
    probabilities.  It is used as an additional expert alongside DPLM-2 in
    the inverse-folding task.

    Parameters
    ----------
    model_path : str
        Path to the cloned ProteinMPNN repository.
        (https://github.com/dauparas/ProteinMPNN)
    model_name : str
        ProteinMPNN model weights name (e.g. "v_48_020").
    device : str
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "v_48_020",
        device: Union[str, torch.device] = "cuda",
        ca_only: bool = False,
        backbone_noise: float = 0.00,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.device = torch.device(device)
        self.ca_only = ca_only
        self.backbone_noise = backbone_noise
        self._model = None
        self._alphabet = None

    def _load(self) -> None:
        """Lazy-load ProteinMPNN model."""
        if self._model is not None:
            return
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)
        try:
            from protein_mpnn_utils import (  # type: ignore[import]
                ProteinMPNN,
                tied_featurize,
                parse_PDB,
                alphabet,
            )
            self._alphabet = alphabet
        except ImportError as e:
            raise ImportError(
                f"Could not import ProteinMPNN from {self.model_path}. "
                "Clone https://github.com/dauparas/ProteinMPNN and set "
                "proteinmpnn_path in MCTDMEConfig."
            ) from e

        import glob
        ckpt_pattern = os.path.join(
            self.model_path,
            "vanilla_model_weights" if not self.ca_only else "ca_model_weights",
            f"{self.model_name}.pt",
        )
        matches = glob.glob(ckpt_pattern)
        if not matches:
            raise FileNotFoundError(f"ProteinMPNN checkpoint not found: {ckpt_pattern}")
        checkpoint = torch.load(matches[0], map_location=self.device)

        hidden_dim = 128
        num_layers = 3
        self._model = ProteinMPNN(
            ca_only=self.ca_only,
            num_letters=21,
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            augment_eps=self.backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info(f"ProteinMPNN loaded from {matches[0]}")

    def get_logprobs(
        self,
        sequence: str,
        mask_indices: List[int],
        structure: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Run ProteinMPNN and return log-probabilities at masked positions.

        Parameters
        ----------
        structure : object
            Expected to be a dict compatible with ProteinMPNN's tied_featurize
            (contains 'X', 'S', 'mask', 'chain_M', etc.).

        Returns
        -------
        torch.Tensor
            Shape (|M|, 20).
        """
        if structure is None:
            raise ValueError("ProteinMPNN requires backbone coordinates via `structure`.")
        self._load()

        with torch.no_grad():
            # ProteinMPNN returns log_probs shape (1, L, 21) – 21st is unknown
            log_probs = self._model(
                **structure,
                temperature=1.0,
                use_input_decoding_order=True,
            )  # (1, L, 21)

        # Take first 20 (standard AAs) and renormalise
        lp = log_probs[0, :, :20]  # (L, 20)
        lp = F.log_softmax(lp, dim=-1)

        rows = [lp[idx] for idx in mask_indices]
        if not rows:
            return torch.zeros(0, 20, device=self.device)
        return torch.stack(rows, dim=0)  # (|M|, 20)

    def sample(
        self,
        sequence: str,
        mask_indices: List[int],
        temperature: float = 1.0,
        structure: Optional[object] = None,
    ) -> str:
        """Sample a sequence from ProteinMPNN at the masked positions."""
        log_probs = self.get_logprobs(sequence, mask_indices, structure=structure)
        aa_list = list(AA20)
        result = list(sequence)
        for i, idx in enumerate(mask_indices):
            if temperature <= 1e-6:
                aa_id = log_probs[i].argmax().item()
            else:
                probs = torch.exp(log_probs[i] / temperature)
                probs = probs / probs.sum()
                aa_id = torch.multinomial(probs, 1).item()
            result[idx] = aa_list[aa_id]
        return "".join(result)

    @property
    def name(self) -> str:
        return f"ProteinMPNN({self.model_name})"


# ---------------------------------------------------------------------------
# Expert factory
# ---------------------------------------------------------------------------

def build_experts(
    expert_names: List[str],
    use_proteinmpnn: bool = False,
    proteinmpnn_path: Optional[str] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    half_precision: bool = False,
) -> List[BaseExpert]:
    """
    Construct a list of expert models from configuration.

    Parameters
    ----------
    expert_names : List[str]
        HuggingFace model IDs for DPLM-2 experts.
    use_proteinmpnn : bool
        Whether to add a ProteinMPNN expert.
    proteinmpnn_path : str | None
        Path to ProteinMPNN repository (required if use_proteinmpnn=True).
    device : str
    cache_dir : str | None
    half_precision : bool

    Returns
    -------
    List[BaseExpert]
    """
    experts: List[BaseExpert] = []
    for name in expert_names:
        experts.append(
            DPLM2Expert(
                model_name=name,
                device=device,
                cache_dir=cache_dir,
                half_precision=half_precision,
            )
        )
    if use_proteinmpnn:
        if proteinmpnn_path is None:
            raise ValueError(
                "proteinmpnn_path must be specified when use_proteinmpnn=True"
            )
        experts.append(
            ProteinMPNNExpert(model_path=proteinmpnn_path, device=device)
        )
    return experts
