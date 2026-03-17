#!/usr/bin/env python3
"""
Motif scaffolding with MCTD-ME.

Design a protein scaffold that presents a functional motif at specified
residue positions, maximising:

    R_motif = 0.40·pLDDT_scaf + 0.30·g(RMSD_motif) + 0.30·scTM
            + 0.20·I[RMSD_motif < 1 AND scTM > 0.8]

where g(x) = max(0, 1 - x/2) if x < 1 else max(0, 0.2 - x/10).
R = 0 if motif residues are not exactly preserved.

The benchmark uses the 23 EvoDiff motif set (Yim et al. 2024):
  1BCF, 1PRW, 1QJG, 1YCR, 2KL8, 3IXT, 4JHW, 4ZYP, 5IUS, 5TPN,
  5TRV_long/med/short, 5WN9, 5YUI, 6E6R_long/med/short, 6EXZ_long/med

Usage
-----
python run_motif_scaffolding.py \\
    --pdb /path/to/motif.pdb \\
    --chain A \\
    --motif_residues 10-25 \\
    --scaffold_length 100 \\
    --output ./results/motif/

See --help for full option list.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mctd_me.config import MCTDMEConfig
from mctd_me.mcts import MCTDME
from mctd_me.utils import parse_pdb_ca, setup_logging, write_fasta

logger = logging.getLogger(__name__)


def parse_residue_range(spec: str) -> List[int]:
    """
    Parse a residue range string like "10-25" or "10,11,15-20" (0-based output).
    """
    indices: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            indices.extend(range(start - 1, end))  # 1-based → 0-based
        else:
            indices.append(int(part) - 1)
    return sorted(set(indices))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCTD-ME motif scaffolding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdb", required=True,
                   help="PDB file containing the motif")
    p.add_argument("--chain", default="A", help="Chain ID of the motif")
    p.add_argument("--motif_residues", required=True,
                   help="Residue range (1-based) e.g. '10-25' or '10,11,20-30'")
    p.add_argument("--scaffold_length", type=int, required=True,
                   help="Total length of the designed scaffold sequence")
    p.add_argument("--motif_start_in_scaffold", type=int, default=None,
                   help="Position in scaffold where motif is placed (1-based). "
                        "If None, motif is placed at the start.")
    p.add_argument("--output", default="./outputs/motif_scaffolding")
    p.add_argument("--experts", nargs="+",
                   default=["airkingbd/dplm_150m",
                            "airkingbd/dplm_650m",
                            "airkingbd/dplm_3b"])
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--num_rollouts", type=int, default=3)
    p.add_argument("--top_k_children", type=int, default=3)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--diffusion_steps", type=int, default=150)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--return_top_k", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--critic_device", default=None,
                   help="Device for ESMFold critic, e.g. 'cuda:1'. Defaults to --device.")
    p.add_argument("--esmfold_chunk_size", type=int, default=None,
                   help="ESMFold chunk size (lower = less VRAM, e.g. 64 or 32)")
    p.add_argument("--half_precision", action="store_true",
                   help="Load DPLM-2 experts in fp16")
    p.add_argument("--cache_dir", default=None, help="HuggingFace cache / local model dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def build_initial_scaffold(
    motif_seq: str,
    scaffold_length: int,
    motif_indices_in_scaffold: List[int],
) -> str:
    """
    Build an initial scaffold sequence with the motif embedded.
    Non-motif positions are initialised to alanine.

    Parameters
    ----------
    motif_seq : str
        Motif amino-acid sequence.
    scaffold_length : int
    motif_indices_in_scaffold : List[int]

    Returns
    -------
    str
        Scaffold sequence with motif residues fixed and the rest as 'A'.
    """
    scaffold = ["A"] * scaffold_length
    for i, scaf_idx in enumerate(motif_indices_in_scaffold):
        if i < len(motif_seq) and scaf_idx < scaffold_length:
            scaffold[scaf_idx] = motif_seq[i]
    return "".join(scaffold)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=args.log_level, log_file=str(out_dir / "run.log"))

    # ---- Load motif backbone ----
    logger.info(f"Loading motif from {args.pdb} (chain {args.chain})")
    full_coords, full_seq = parse_pdb_ca(args.pdb, chain_id=args.chain)

    # Motif residue indices in the source PDB (0-based)
    motif_indices_in_pdb = parse_residue_range(args.motif_residues)
    motif_seq = "".join(full_seq[i] for i in motif_indices_in_pdb if i < len(full_seq))
    motif_coords_ref = full_coords[motif_indices_in_pdb]
    n_motif = len(motif_indices_in_pdb)

    logger.info(
        f"Motif: {n_motif} residues | "
        f"Scaffold length: {args.scaffold_length}"
    )

    # ---- Place motif in scaffold ----
    if args.motif_start_in_scaffold is not None:
        start = args.motif_start_in_scaffold - 1  # 1-based → 0-based
    else:
        start = 0
    motif_indices_in_scaffold = list(range(start, start + n_motif))
    if max(motif_indices_in_scaffold) >= args.scaffold_length:
        logger.error(
            "Motif indices exceed scaffold length. "
            "Adjust --motif_start_in_scaffold or --scaffold_length."
        )
        sys.exit(1)

    initial_scaffold = build_initial_scaffold(
        motif_seq=motif_seq,
        scaffold_length=args.scaffold_length,
        motif_indices_in_scaffold=motif_indices_in_scaffold,
    )

    # ---- Config ----
    config = MCTDMEConfig(
        task="motif_scaffolding",
        experts=args.experts,
        num_rollouts=args.num_rollouts,
        top_k_children=args.top_k_children,
        max_depth=args.max_depth,
        num_mcts_iterations=args.num_iters,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        device=args.device,
        critic_device=args.critic_device,
        esmfold_chunk_size=args.esmfold_chunk_size,
        seed=args.seed,
        output_dir=str(out_dir),
        cache_dir=args.cache_dir or "~/.cache/mctd_me",
    )

    engine = MCTDME(config=config)

    critic_kwargs = {
        "motif_indices": motif_indices_in_scaffold,
        "motif_coords_ref": motif_coords_ref,
        "motif_seq_ref": motif_seq,
    }

    logger.info("Starting MCTD-ME motif scaffolding …")
    results = engine.run_lead_optimization(
        lead_sequence=initial_scaffold,
        critic_kwargs=critic_kwargs,
        return_top_k=args.return_top_k,
        verbose=True,
    )

    # ---- Save ----
    fasta_dict: dict = {}
    summary = []
    for rank, (seq, reward, info) in enumerate(results, start=1):
        tag = f"scaffold_{rank:03d}_r{reward:.4f}"
        fasta_dict[tag] = seq
        summary.append({
            "rank": rank,
            "sequence": seq,
            "reward": reward,
            "plddt_scaf": info.get("plddt_scaf"),
            "motif_rmsd": info.get("motif_rmsd"),
            "sc_tm": info.get("sc_tm"),
            "motif_preserved": info.get("motif_preserved"),
        })

    write_fasta(fasta_dict, str(out_dir / "scaffolds.fasta"))
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Results saved to {out_dir}")

    if results:
        best_seq, best_r, best_info = results[0]
        print(f"\nBest scaffold  reward={best_r:.4f}")
        for k in ("plddt_scaf", "motif_rmsd", "sc_tm", "motif_preserved"):
            v = best_info.get(k)
            if v is not None:
                print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
