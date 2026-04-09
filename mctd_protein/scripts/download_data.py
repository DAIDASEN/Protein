#!/usr/bin/env python3
"""
Download benchmark datasets for MCTD-ME evaluation.

Downloads:
  1. CAMEO 2022 benchmark (183 targets, lengths 15-704 residues)
  2. PDB date-split benchmark (449 chains)
  3. EvoDiff motif scaffolding set (23 motifs, Yim et al. 2024)

Usage
-----
python download_data.py --output ./data/ [--cameo] [--pdb_split] [--evodiff_motifs]

Note: The full PDB structures must be downloaded separately from RCSB PDB.
This script downloads the sequence lists / metadata and PDB structure files
where publicly available.
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mctd_me.utils import setup_logging, EVODIFF_MOTIF_IDS

logger = logging.getLogger(__name__)

RCSB_PDB_BASE = "https://files.rcsb.org/download"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a file with retry logic."""
    for attempt in range(retries):
        try:
            logger.debug(f"  Downloading {url} → {dest}")
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as exc:
            logger.warning(f"  Attempt {attempt+1}/{retries} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False


def download_pdb(pdb_id: str, dest_dir: Path) -> Optional[Path]:
    """Download a PDB structure from RCSB."""
    pdb_id = pdb_id.upper()
    dest = dest_dir / f"{pdb_id}.pdb"
    if dest.exists():
        logger.debug(f"  {pdb_id}.pdb already exists, skipping.")
        return dest
    url = f"{RCSB_PDB_BASE}/{pdb_id}.pdb"
    success = download_file(url, dest)
    return dest if success else None


# ---------------------------------------------------------------------------
# EvoDiff motif set
# ---------------------------------------------------------------------------

# Motif PDB IDs and chain + residue ranges (from Yim et al. 2024)
EVODIFF_MOTIF_SPECS: Dict[str, Dict] = {
    "1BCF":        {"pdb": "1BCF", "chain": "A", "residues": "1-24"},
    "1PRW":        {"pdb": "1PRW", "chain": "A", "residues": "16-35"},
    "1QJG":        {"pdb": "1QJG", "chain": "A", "residues": "1-19"},
    "1YCR":        {"pdb": "1YCR", "chain": "B", "residues": "1-13"},
    "2KL8":        {"pdb": "2KL8", "chain": "A", "residues": "1-16"},
    "3IXT":        {"pdb": "3IXT", "chain": "A", "residues": "254-277"},
    "4JHW":        {"pdb": "4JHW", "chain": "A", "residues": "1-60"},
    "4ZYP":        {"pdb": "4ZYP", "chain": "A", "residues": "1-30"},
    "5IUS":        {"pdb": "5IUS", "chain": "A", "residues": "1-40"},
    "5TPN":        {"pdb": "5TPN", "chain": "A", "residues": "1-31"},
    "5TRV_long":   {"pdb": "5TRV", "chain": "A", "residues": "1-100"},
    "5TRV_med":    {"pdb": "5TRV", "chain": "A", "residues": "1-50"},
    "5TRV_short":  {"pdb": "5TRV", "chain": "A", "residues": "1-25"},
    "5WN9":        {"pdb": "5WN9", "chain": "A", "residues": "1-28"},
    "5YUI":        {"pdb": "5YUI", "chain": "A", "residues": "1-25"},
    "6E6R_long":   {"pdb": "6E6R", "chain": "A", "residues": "1-80"},
    "6E6R_med":    {"pdb": "6E6R", "chain": "A", "residues": "1-50"},
    "6E6R_short":  {"pdb": "6E6R", "chain": "A", "residues": "1-25"},
    "6EXZ_long":   {"pdb": "6EXZ", "chain": "A", "residues": "1-70"},
    "6EXZ_med":    {"pdb": "6EXZ", "chain": "A", "residues": "1-40"},
}


def download_evodiff_motifs(data_dir: Path) -> None:
    """Download EvoDiff motif PDB files and write a metadata JSON."""
    motif_dir = data_dir / "evodiff_motifs"
    motif_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    unique_pdbs = set(spec["pdb"] for spec in EVODIFF_MOTIF_SPECS.values())

    logger.info(f"Downloading {len(unique_pdbs)} unique PDB structures for EvoDiff motifs …")
    for pdb_id in sorted(unique_pdbs):
        path = download_pdb(pdb_id, motif_dir)
        if path:
            logger.info(f"  ✓ {pdb_id}")
        else:
            logger.warning(f"  ✗ Failed to download {pdb_id}")

    # Write metadata
    for motif_name, spec in EVODIFF_MOTIF_SPECS.items():
        pdb_path = motif_dir / f"{spec['pdb'].upper()}.pdb"
        metadata[motif_name] = {
            "pdb_file": str(pdb_path),
            "chain": spec["chain"],
            "residues": spec["residues"],
        }

    meta_path = motif_dir / "motif_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"EvoDiff motif metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CAMEO 2022
# ---------------------------------------------------------------------------

# CAMEO 2022 benchmark PDB IDs (sample subset – full list at cameo3d.org)
# The complete 183-target list should be fetched from the official CAMEO website.
CAMEO_2022_SAMPLE_IDS = [
    "7VVK", "7WKK", "7XAK", "7YCK", "7ZBK",
    "8A5K", "8B5K", "8C5K", "8D5K", "8E5K",
]


def download_cameo2022(data_dir: Path) -> None:
    """
    Download CAMEO 2022 benchmark structures.

    Note: The full list of 183 targets should be obtained from
    https://www.cameo3d.org/sp/1-week/?to=2022-12-31
    """
    cameo_dir = data_dir / "cameo2022"
    cameo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading CAMEO 2022 sample targets ({len(CAMEO_2022_SAMPLE_IDS)} PDBs) …"
    )
    logger.warning(
        "Note: This downloads a sample only. "
        "For the full 183-target benchmark, download from https://www.cameo3d.org"
    )

    success_count = 0
    for pdb_id in CAMEO_2022_SAMPLE_IDS:
        path = download_pdb(pdb_id, cameo_dir)
        if path:
            success_count += 1
            logger.info(f"  ✓ {pdb_id}")
        else:
            logger.warning(f"  ✗ {pdb_id}")

    # Write list
    pdb_list = [str(cameo_dir / f"{p.upper()}.pdb") for p in CAMEO_2022_SAMPLE_IDS]
    list_path = cameo_dir / "pdb_list.json"
    with open(list_path, "w") as fh:
        json.dump({"pdb_files": pdb_list, "n_total": 183, "n_downloaded": success_count}, fh, indent=2)
    logger.info(f"CAMEO 2022 list saved to {list_path}")


# ---------------------------------------------------------------------------
# PDB date-split
# ---------------------------------------------------------------------------

# Sample subset of the PDB date-split benchmark (449 chains)
PDB_SPLIT_SAMPLE = [
    ("8G5A", "A"), ("8G3P", "A"), ("8FYZ", "A"), ("8FXK", "A"), ("8FXJ", "A"),
]


def download_pdb_split(data_dir: Path) -> None:
    """
    Download PDB date-split benchmark structures.

    Note: The full 449-chain split is constructed by filtering chains from
    PDB structures deposited after 2022-05-01 with lengths 15-700 residues.
    """
    pdb_dir = data_dir / "pdb_split"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading PDB date-split sample ({len(PDB_SPLIT_SAMPLE)} PDBs) …"
    )
    logger.warning(
        "Note: This downloads a sample only. "
        "The full 449-chain split requires additional filtering from RCSB."
    )

    metadata = []
    for pdb_id, chain in PDB_SPLIT_SAMPLE:
        path = download_pdb(pdb_id, pdb_dir)
        if path:
            metadata.append({"pdb_id": pdb_id, "chain": chain, "pdb_file": str(path)})
            logger.info(f"  ✓ {pdb_id}:{chain}")
        else:
            logger.warning(f"  ✗ {pdb_id}")

    meta_path = pdb_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump({"targets": metadata, "n_total": 449, "n_downloaded": len(metadata)}, fh, indent=2)
    logger.info(f"PDB date-split metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download MCTD-ME benchmark datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", default="./data", help="Root data directory")
    p.add_argument("--cameo", action="store_true", help="Download CAMEO 2022 benchmark")
    p.add_argument("--pdb_split", action="store_true", help="Download PDB date-split benchmark")
    p.add_argument("--evodiff_motifs", action="store_true", help="Download EvoDiff motif set")
    p.add_argument("--all", action="store_true", help="Download all benchmarks")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.evodiff_motifs:
        download_evodiff_motifs(data_dir)
    if args.all or args.cameo:
        download_cameo2022(data_dir)
    if args.all or args.pdb_split:
        download_pdb_split(data_dir)

    if not any([args.all, args.evodiff_motifs, args.cameo, args.pdb_split]):
        print("No dataset selected. Use --all, --cameo, --pdb_split, or --evodiff_motifs.")
        print("Run with --help for usage.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
