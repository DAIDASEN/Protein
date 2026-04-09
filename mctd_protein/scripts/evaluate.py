#!/usr/bin/env python3
"""
Evaluation script for MCTD-ME designed sequences.

Computes the full suite of metrics (TM-score, scTM, RMSD, MotifRMSD, pLDDT, AAR)
for a FASTA file of designed sequences against a reference PDB structure.

Usage
-----
python evaluate.py \\
    --fasta designs.fasta \\
    --reference target.pdb \\
    --chain A \\
    [--native_seq MKTAY...]  \\
    [--motif_residues 10-25] \\
    --output ./eval_results/

The script uses ESMFold to predict the structure of each designed sequence.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mctd_me.critics import ESMFoldCritic
from mctd_me.metrics import compute_all_metrics
from mctd_me.utils import parse_pdb_ca, read_fasta, setup_logging, write_fasta

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate MCTD-ME designed sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fasta", required=True, help="FASTA file with designed sequences")
    p.add_argument("--reference", required=True, help="Reference structure PDB file")
    p.add_argument("--chain", default="A", help="Chain ID of reference structure")
    p.add_argument("--native_seq", default=None,
                   help="Native/target amino-acid sequence (for AAR)")
    p.add_argument("--motif_residues", default=None,
                   help="Motif residue range (1-based, e.g. '10-25') for MotifRMSD")
    p.add_argument("--output", default="./eval_results", help="Output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--esmfold_model", default="facebook/esmfold_v1")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--top_k", type=int, default=None,
                   help="Only evaluate the first K sequences")
    return p.parse_args()


def parse_residue_range(spec: str) -> List[int]:
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-", 1)
            indices.extend(range(int(s) - 1, int(e)))
        else:
            indices.append(int(part) - 1)
    return sorted(set(indices))


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=args.log_level, log_file=str(out_dir / "eval.log"))

    # ---- Load reference ----
    logger.info(f"Loading reference structure: {args.reference}")
    ref_coords, ref_seq = parse_pdb_ca(args.reference, chain_id=args.chain)
    logger.info(f"Reference length: {len(ref_seq)}")

    native_seq = args.native_seq or ref_seq

    motif_indices: Optional[List[int]] = None
    motif_coords_ref: Optional[np.ndarray] = None
    if args.motif_residues:
        motif_indices = parse_residue_range(args.motif_residues)
        motif_coords_ref = ref_coords[motif_indices]
        logger.info(f"Motif: {len(motif_indices)} residues at {motif_indices[:5]}…")

    # ---- Load designed sequences ----
    sequences = read_fasta(args.fasta)
    if args.top_k is not None:
        sequences = dict(list(sequences.items())[: args.top_k])
    logger.info(f"Evaluating {len(sequences)} sequences …")

    # ---- ESMFold critic ----
    esmfold = ESMFoldCritic(
        model_name=args.esmfold_model,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # ---- Evaluate each sequence ----
    rows: List[dict] = []
    for header, seq in sequences.items():
        logger.info(f"  Folding {header[:40]} ({len(seq)} aa) …")
        try:
            pred = esmfold.predict(seq)
            ca_pred = pred["positions"]
            plddt_raw = pred["plddt"]

            metrics = compute_all_metrics(
                designed_sequence=seq,
                native_sequence=native_seq,
                coords_pred=ca_pred,
                coords_target=ref_coords,
                plddt_raw=plddt_raw,
                motif_indices=motif_indices,
                motif_coords_ref=motif_coords_ref,
            )
            row = {"header": header, "sequence": seq, **metrics}
        except Exception as exc:
            logger.warning(f"  Error evaluating {header}: {exc}")
            row = {
                "header": header,
                "sequence": seq,
                "tm": float("nan"),
                "sc_tm": float("nan"),
                "rmsd_global": float("nan"),
                "rmsd_motif": float("nan"),
                "plddt_mean": float("nan"),
                "aar": float("nan"),
            }
        rows.append(row)

    # ---- Aggregate ----
    def _avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])]
        return mean(vals)

    aggregate = {
        "n_sequences": len(rows),
        "mean_tm": _avg("tm"),
        "mean_sc_tm": _avg("sc_tm"),
        "mean_rmsd_global": _avg("rmsd_global"),
        "mean_plddt": _avg("plddt_mean"),
        "mean_aar": _avg("aar"),
    }
    if motif_indices is not None:
        aggregate["mean_rmsd_motif"] = _avg("rmsd_motif")

    # ---- Save ----
    results_path = out_dir / "metrics.json"
    with open(results_path, "w") as fh:
        json.dump({"aggregate": aggregate, "per_sequence": rows}, fh, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- Print summary ----
    print("\n=== Evaluation Summary ===")
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
