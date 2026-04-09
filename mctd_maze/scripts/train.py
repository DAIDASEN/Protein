"""
Train the trajectory diffusion planner.

Usage:
    python scripts/train.py \
        --env pointmaze-medium-navigate-v0 \
        --save_dir /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium \
        --device cuda:0 \
        --train_steps 200005 \
        --batch_size 1024 \
        --lr 5e-4 \
        --precision 16-mixed
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow running as script without package install
sys.path.insert(0, str(Path(__file__).parent.parent))

from mctd_maze.config import MCTDMazeConfig, ModelConfig, TrainConfig
from mctd_maze.diffusion import TrajDiffusion
from mctd_maze.envs import make_env_and_dataset
from mctd_maze.utils import (
    dataset_to_trajectories,
    load_checkpoint,
    save_checkpoint,
    save_results,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MCTD trajectory diffusion planner")
    p.add_argument("--env", type=str, default="pointmaze-medium-navigate-v0")
    p.add_argument("--save_dir", type=str,
                   default="/root/rivermind-data/checkpoints/mctd_maze/default")
    p.add_argument("--data_dir", type=str,
                   default="/root/rivermind-data/data/ogbench")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_steps", type=int, default=200_005)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--precision", type=str, default="16-mixed",
                   choices=["32", "16-mixed"])
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n_diffusion_steps", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=512)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mctd_maze")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    import logging
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.env}-seed{args.seed}",
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.env}")
    env, dataset = make_env_and_dataset(args.env, data_dir=args.data_dir, seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    logger.info(f"obs_dim={obs_dim}  action_dim={act_dim}  horizon={args.horizon}")

    trajs = dataset_to_trajectories(dataset, horizon=args.horizon)
    loader = DataLoader(
        TensorDataset(trajs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Dataset: {len(trajs)} trajectories")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TrajDiffusion(
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_diffusion_steps=args.n_diffusion_steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "16-mixed"))

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, args.resume, optimizer, str(device))
        logger.info(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    data_iter = iter(loader)
    best_loss = float("inf")

    for step in range(start_step, args.train_steps):
        try:
            (batch,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch,) = next(data_iter)

        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(args.precision == "16-mixed")):
            loss = model.loss(batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 500 == 0:
            logger.info(f"Step {step:7d} | loss={loss.item():.6f}")
            if not args.no_wandb:
                import wandb
                wandb.log({"train/loss": loss.item(), "step": step})

        if step % 10_000 == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.save_dir)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = os.path.join(args.save_dir, "best.pt")
                torch.save({"model": model.state_dict(), "step": step}, best_path)
                logger.info(f"New best saved: {best_path}")

    # Final save
    save_checkpoint(model, optimizer, args.train_steps, args.save_dir)
    best_path = os.path.join(args.save_dir, "best.pt")
    torch.save({"model": model.state_dict(), "step": args.train_steps}, best_path)
    logger.info("Training complete.")

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
