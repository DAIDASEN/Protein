"""
Utilities for MCTD-ME.

Covers:
  - Protein sequence validation and manipulation
  - Structure I/O (PDB parsing with Biopython / Biotite)
  - Sequence caching
  - Logging / random seed setup
  - FASTA I/O
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random
import re
import string
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard 20 amino acids
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def validate_sequence(sequence: str) -> bool:
    """
    Return True if *sequence* contains only standard amino-acid characters.

    Parameters
    ----------
    sequence : str

    Returns
    -------
    bool
    """
    return all(c in AA_SET for c in sequence.upper())


def clean_sequence(sequence: str) -> str:
    """
    Remove whitespace, newlines, and non-AA characters.

    Parameters
    ----------
    sequence : str

    Returns
    -------
    str
        Cleaned sequence with only standard AA characters.
    """
    seq = sequence.strip().upper()
    seq = re.sub(r"\s+", "", seq)
    # Replace non-standard characters with 'A' as a fallback
    seq = "".join(c if c in AA_SET else "A" for c in seq)
    return seq


def sequence_hash(sequence: str) -> str:
    """
    Return a short SHA-256 hash for a protein sequence.

    Used as a cache key.
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def hamming_distance(seq_a: str, seq_b: str) -> int:
    """
    Compute the Hamming distance between two equal-length sequences.
    """
    L = min(len(seq_a), len(seq_b))
    return sum(a != b for a, b in zip(seq_a[:L], seq_b[:L]))


def identity(seq_a: str, seq_b: str) -> float:
    """
    Sequence identity (fraction of matching positions).
    """
    L = min(len(seq_a), len(seq_b))
    if L == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq_a[:L], seq_b[:L]))
    return matches / L


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def read_fasta(path: str) -> Dict[str, str]:
    """
    Parse a FASTA file into a dict {header: sequence}.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    records: Dict[str, str] = {}
    current_header: Optional[str] = None
    current_seq: List[str] = []

    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    records[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    if current_header is not None:
        records[current_header] = "".join(current_seq)
    return records


def write_fasta(sequences: Dict[str, str], path: str, line_width: int = 60) -> None:
    """
    Write sequences to a FASTA file.

    Parameters
    ----------
    sequences : dict {header: sequence}
    path : str
    line_width : int
        Maximum characters per line in the sequence block.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in sequences.items():
            fh.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")


# ---------------------------------------------------------------------------
# Structure I/O
# ---------------------------------------------------------------------------

def parse_pdb_ca(pdb_path: str, chain_id: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """
    Extract Cα coordinates and sequence from a PDB file.

    Uses Biopython if available, otherwise falls back to a lightweight parser.

    Parameters
    ----------
    pdb_path : str
    chain_id : str | None
        Chain identifier (e.g. 'A').  If None, the first chain is used.

    Returns
    -------
    ca_coords : np.ndarray, shape (L, 3)
    sequence  : str
        Single-letter amino-acid sequence.
    """
    try:
        return _parse_pdb_biopython(pdb_path, chain_id)
    except ImportError:
        return _parse_pdb_minimal(pdb_path, chain_id)


def _parse_pdb_biopython(
    pdb_path: str, chain_id: Optional[str]
) -> Tuple[np.ndarray, str]:
    """Parse PDB using Biopython."""
    from Bio import PDB
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    if chain_id is None:
        chain = next(iter(model))
    else:
        chain = model[chain_id]

    ca_list: List[np.ndarray] = []
    seq_list: List[str] = []

    for residue in chain:
        if residue.id[0] != " ":  # skip heteroatoms
            continue
        try:
            ca = residue["CA"].get_vector().get_array()
            resname = residue.get_resname()
            aa = protein_letters_3to1.get(resname, "X")
            ca_list.append(ca)
            seq_list.append(aa)
        except KeyError:
            continue

    if not ca_list:
        return np.zeros((0, 3), dtype=np.float32), ""

    return np.array(ca_list, dtype=np.float32), "".join(seq_list)


def _parse_pdb_minimal(
    pdb_path: str, chain_id: Optional[str]
) -> Tuple[np.ndarray, str]:
    """
    Lightweight PDB Cα parser that does not require Biopython.
    """
    _3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    ca_list: List[np.ndarray] = []
    seq_list: List[str] = []
    seen_res: set = set()

    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            chain = line[21].strip()
            if chain_id is not None and chain != chain_id:
                continue
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            ins_code = line[26].strip()
            res_key = (chain, resnum, ins_code)
            if res_key in seen_res:
                continue
            seen_res.add(res_key)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            aa = _3to1.get(resname, "X")
            ca_list.append([x, y, z])
            seq_list.append(aa)

    if not ca_list:
        return np.zeros((0, 3), dtype=np.float32), ""
    return np.array(ca_list, dtype=np.float32), "".join(seq_list)


def write_pdb_ca(
    ca_coords: np.ndarray,
    sequence: str,
    path: str,
    chain_id: str = "A",
) -> None:
    """
    Write a simplified PDB file with only Cα atoms.

    Parameters
    ----------
    ca_coords : np.ndarray, shape (L, 3)
    sequence  : str
    path      : str
    chain_id  : str
    """
    _1to3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    L = min(len(ca_coords), len(sequence))
    with open(path, "w") as fh:
        for i in range(L):
            x, y, z = ca_coords[i]
            aa3 = _1to3.get(sequence[i].upper(), "ALA")
            line = (
                f"ATOM  {i+1:5d}  CA  {aa3} {chain_id}{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            fh.write(line)
        fh.write("END\n")


# ---------------------------------------------------------------------------
# Sequence cache
# ---------------------------------------------------------------------------

class SequenceCache:
    """
    In-memory and optional on-disk cache for evaluated sequences.

    Maps sequence → (reward, info_dict).
    Avoids re-evaluating the same sequence with the expensive ESMFold critic.
    """

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self._load()

    def __contains__(self, sequence: str) -> bool:
        return sequence in self._cache

    def __setitem__(self, sequence: str, value: Tuple[float, dict]) -> None:
        self._cache[sequence] = value

    def __getitem__(self, sequence: str) -> Tuple[float, dict]:
        return self._cache[sequence]

    def get(self, sequence: str, default=None):
        return self._cache.get(sequence, default)

    def __len__(self) -> int:
        return len(self._cache)

    def items(self):
        return self._cache.items()

    def top_k(self, k: int) -> List[Tuple[str, float, dict]]:
        """Return the top-*k* sequences by reward (descending)."""
        sorted_items = sorted(
            self._cache.items(), key=lambda kv: kv[1][0], reverse=True
        )
        return [(seq, r, info) for seq, (r, info) in sorted_items[:k]]

    def save(self) -> None:
        """Persist cache to disk."""
        if self.cache_path is None:
            return
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as fh:
            pickle.dump(self._cache, fh)

    def _load(self) -> None:
        try:
            with open(self.cache_path, "rb") as fh:
                self._cache = pickle.load(fh)
            logger.info(f"Loaded {len(self._cache)} cached sequences from {self.cache_path}")
        except Exception as exc:
            logger.warning(f"Could not load cache from {self.cache_path}: {exc}")
            self._cache = {}


# ---------------------------------------------------------------------------
# Random seed / logging helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level : str
        Logging level string (e.g. "DEBUG", "INFO").
    log_file : str | None
        If provided, also write logs to this file.
    """
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt,
                        handlers=handlers)


# ---------------------------------------------------------------------------
# Motif utilities
# ---------------------------------------------------------------------------

def parse_motif_spec(spec: str) -> Tuple[str, List[int]]:
    """
    Parse a motif specification string of the form "PDBID:chain:start-end".

    Example: "1BCF:A:10-25" → ("1BCF_A", [10, 11, ..., 25])

    Parameters
    ----------
    spec : str

    Returns
    -------
    name : str
    residue_indices : List[int]  (0-based)
    """
    parts = spec.split(":")
    if len(parts) == 3:
        pdb_id, chain, range_str = parts
        name = f"{pdb_id}_{chain}"
        start, end = map(int, range_str.split("-"))
        indices = list(range(start - 1, end))  # convert 1-based to 0-based
    elif len(parts) == 2:
        pdb_id, range_str = parts
        name = pdb_id
        start, end = map(int, range_str.split("-"))
        indices = list(range(start - 1, end))
    else:
        raise ValueError(f"Cannot parse motif spec: {spec!r}")
    return name, indices


# EvoDiff motif benchmark names (Sec. 4.3, arXiv:2509.15796)
EVODIFF_MOTIF_IDS = [
    "1BCF", "1PRW", "1QJG", "1YCR", "2KL8", "3IXT", "4JHW", "4ZYP",
    "5IUS", "5TPN", "5TRV_long", "5TRV_med", "5TRV_short",
    "5WN9", "5YUI", "6E6R_long", "6E6R_med", "6E6R_short",
    "6EXZ_long", "6EXZ_med",
]
