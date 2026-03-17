#!/usr/bin/env python3
"""
Inverse Folding with MCTD-ME.

Given a target protein backbone (PDB file), design amino-acid sequences
that fold into that backbone by maximising the inverse-folding reward:

    R_inv = 0.60 * AAR + 0.35 * scTM + 0.05 * B

Usage
-----
python run_inverse_folding.py \\
    --pdb /path/to/target.pdb \\
    --chain A \\
    --output ./results/ \\
    [--experts dplm_150m dplm_650m dplm_3b] \\
    [--num_iters 100] \\
    [--return_top_k 10]

See --help for full option list.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mctd_me.config import MCTDMEConfig
from mctd_me.critics import CompositeReward, ESMFoldCritic
from mctd_me.experts import build_experts
from mctd_me.mcts import MCTDME
from mctd_me.utils import (
    parse_pdb_ca,
    setup_logging,
    write_fasta,
    write_pdb_ca,
    EVODIFF_MOTIF_IDS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCTD-ME inverse folding: design sequences for a target backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdb", required=True, help="Target backbone PDB file")
    p.add_argument("--chain", default="A", help="PDB chain to design for")
    p.add_argument("--output", default="./outputs/inverse_folding",
                   help="Output directory")
    p.add_argument("--experts", nargs="+",
                   default=["airkingbd/dplm_150m",
                            "airkingbd/dplm_650m",
                            "airkingbd/dplm_3b"],
                   help="DPLM-2 model IDs (HuggingFace)")
    p.add_argument("--use_proteinmpnn", action="store_true",
                   help="Add ProteinMPNN as an extra expert")
    p.add_argument("--proteinmpnn_path", default=None,
                   help="Path to ProteinMPNN repository")
    p.add_argument("--num_iters", type=int, default=100,
                   help="MCTS iterations (T)")
    p.add_argument("--num_rollouts", type=int, default=3,
                   help="Rollouts per expert per expansion (k_roll)")
    p.add_argument("--top_k_children", type=int, default=3,
                   help="Children kept per expansion (K)")
    p.add_argument("--max_depth", type=int, default=5,
                   help="Max MCTS tree depth")
    p.add_argument("--diffusion_steps", type=int, default=150,
                   help="Reverse diffusion steps")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Consensus prior temperature τ")
    p.add_argument("--return_top_k", type=int, default=10,
                   help="Top-k sequences to return")
    p.add_argument("--device", default="cuda", help="PyTorch device")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--log_level", default="INFO", help="Logging level")
    p.add_argument("--cache_dir", default=None, help="HuggingFace cache dir")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=args.log_level, log_file=str(out_dir / "run.log"))

    # ---- Load target backbone ----
    logger.info(f"Loading target backbone from {args.pdb} (chain {args.chain})")
    target_coords, native_seq = parse_pdb_ca(args.pdb, chain_id=args.chain)
    logger.info(f"Target length: {len(native_seq)} residues")

    # ---- Build config ----
    config = MCTDMEConfig(
        task="inverse_folding",
        experts=args.experts,
        use_proteinmpnn=args.use_proteinmpnn,
        proteinmpnn_path=args.proteinmpnn_path,
        num_rollouts=args.num_rollouts,
        top_k_children=args.top_k_children,
        max_depth=args.max_depth,
        num_mcts_iterations=args.num_iters,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
        output_dir=str(out_dir),
        cache_dir=args.cache_dir or "~/.cache/mctd_me",
    )

    # ---- Build MCTD-ME engine ----
    engine = MCTDME(config=config)

    # ---- Critic kwargs ----
    critic_kwargs = {
        "native_sequence": native_seq,
        "target_coords": target_coords,
    }

    # ---- Run algorithm (lead optimisation starting from native) ----
    logger.info("Starting MCTD-ME inverse folding …")
    results = engine.run_lead_optimization(
        lead_sequence=native_seq,
        critic_kwargs=critic_kwargs,
        return_top_k=args.return_top_k,
        verbose=True,
    )

    # ---- Save results ----
    fasta_dict: dict = {}
    summary_rows = []
    for rank, (seq, reward, info) in enumerate(results, start=1):
        tag = f"design_{rank:03d}_reward{reward:.4f}"
        fasta_dict[tag] = seq
        summary_rows.append({
            "rank": rank,
            "sequence": seq,
            "reward": reward,
            "aar": info.get("aar", "N/A"),
            "sc_tm": info.get("sc_tm", "N/A"),
            "plddt": info.get("plddt", "N/A"),
        })

    fasta_path = str(out_dir / "designs.fasta")
    write_fasta(fasta_dict, fasta_path)
    logger.info(f"Saved {len(results)} designed sequences to {fasta_path}")

    summary_path = str(out_dir / "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary_rows, fh, indent=2)
    logger.info(f"Summary written to {summary_path}")

    # Print top result
    if results:
        best_seq, best_reward, best_info = results[0]
        print("\n=== Best Design ===")
        print(f"  Reward : {best_reward:.4f}")
        print(f"  AAR    : {best_info.get('aar', 'N/A'):.4f}" if isinstance(best_info.get('aar'), float) else f"  AAR    : N/A")
        print(f"  scTM   : {best_info.get('sc_tm', 'N/A'):.4f}" if isinstance(best_info.get('sc_tm'), float) else f"  scTM   : N/A")
        print(f"  pLDDT  : {best_info.get('plddt', 'N/A'):.4f}" if isinstance(best_info.get('plddt'), float) else f"  pLDDT  : N/A")
        print(f"  Seq    : {best_seq[:80]}{'…' if len(best_seq) > 80 else ''}")


if __name__ == "__main__":
    main()
