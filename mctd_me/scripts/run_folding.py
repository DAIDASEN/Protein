#!/usr/bin/env python3
"""
Folding task with MCTD-ME.

Given an optional target structure (PDB), design amino-acid sequences
that fold into stable, well-structured proteins by maximising:

    R_fold = α·TM + β·(1 - min(RMSD/10, 1)) + γ·pLDDT

Defaults: (α, β, γ) = (0.60, 0.40, 0.00) with γ=0.05 if pLDDT is used.

Usage
-----
python run_folding.py \\
    --sequence MKTAYIAKQR...  \\
    [--target_pdb /path/to/target.pdb] \\
    [--length 100]             \\  # for de-novo design without seed sequence
    --output ./results/folding/

See --help for full option list.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mctd_me.config import MCTDMEConfig
from mctd_me.mcts import MCTDME
from mctd_me.utils import parse_pdb_ca, setup_logging, write_fasta

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCTD-ME folding: design stable protein sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--sequence", help="Starting amino-acid sequence (lead optimisation)")
    group.add_argument("--length", type=int, help="Sequence length for de-novo design")

    p.add_argument("--target_pdb", default=None, help="Optional target backbone PDB")
    p.add_argument("--chain", default="A", help="PDB chain ID")
    p.add_argument("--output", default="./outputs/folding", help="Output directory")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=args.log_level, log_file=str(out_dir / "run.log"))

    # ---- Load optional target backbone ----
    target_coords = None
    if args.target_pdb:
        logger.info(f"Loading target backbone from {args.target_pdb}")
        target_coords, _ = parse_pdb_ca(args.target_pdb, chain_id=args.chain)

    # ---- Config ----
    config = MCTDMEConfig(
        task="folding",
        experts=args.experts,
        num_rollouts=args.num_rollouts,
        top_k_children=args.top_k_children,
        max_depth=args.max_depth,
        num_mcts_iterations=args.num_iters,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
        output_dir=str(out_dir),
    )

    engine = MCTDME(config=config)
    critic_kwargs = {"target_coords": target_coords} if target_coords is not None else {}

    if args.sequence:
        logger.info("Lead optimisation mode")
        results = engine.run_lead_optimization(
            lead_sequence=args.sequence,
            critic_kwargs=critic_kwargs,
            return_top_k=args.return_top_k,
        )
    elif args.length:
        logger.info(f"De-novo design mode, length={args.length}")
        results = engine.run_denovo(
            sequence_length=args.length,
            critic_kwargs=critic_kwargs,
            return_top_k=args.return_top_k,
        )
    else:
        logger.error("Either --sequence or --length must be provided.")
        sys.exit(1)

    # ---- Save ----
    fasta_dict = {
        f"design_{rank:03d}_r{r:.4f}": seq
        for rank, (seq, r, _) in enumerate(results, start=1)
    }
    fasta_path = str(out_dir / "designs.fasta")
    write_fasta(fasta_dict, fasta_path)
    logger.info(f"Saved {len(results)} sequences to {fasta_path}")

    summary = [
        {
            "rank": rank,
            "sequence": seq,
            "reward": r,
            "tm": info.get("tm"),
            "rmsd": info.get("rmsd"),
            "plddt": info.get("plddt"),
        }
        for rank, (seq, r, info) in enumerate(results, start=1)
    ]
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary written to {out_dir / 'summary.json'}")

    if results:
        best_seq, best_r, best_info = results[0]
        print(f"\nBest design  reward={best_r:.4f}")
        for k in ("tm", "rmsd", "plddt"):
            v = best_info.get(k)
            if v is not None:
                print(f"  {k:8s}: {v:.4f}")
        print(f"  Seq      : {best_seq[:80]}{'…' if len(best_seq) > 80 else ''}")


if __name__ == "__main__":
    main()
