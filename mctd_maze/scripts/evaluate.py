"""
Evaluate MCTD planning on OGBench environments.

Usage:
    python scripts/evaluate.py \
        --env pointmaze-medium-navigate-v0 \
        --checkpoint /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium/best.pt \
        --output /root/rivermind-data/outputs/mctd_maze/pointmaze_medium/ \
        --device cuda:0 \
        --num_eval_episodes 100 \
        --max_search 500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mctd_maze.config import MCTDConfig, MCTDMazeConfig
from mctd_maze.diffusion import TrajDiffusion
from mctd_maze.envs import HeuristicController, goal_reaching_reward, make_env_and_dataset
from mctd_maze.planner import MCTD
from mctd_maze.utils import load_checkpoint, save_results, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCTD evaluation on OGBench")
    p.add_argument("--env", type=str, default="pointmaze-medium-navigate-v0")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str,
                   default="/root/rivermind-data/outputs/mctd_maze/default/")
    p.add_argument("--data_dir", type=str,
                   default="/root/rivermind-data/data/ogbench")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_eval_episodes", type=int, default=100)
    # MCTD search parameters (paper Table 10)
    p.add_argument("--max_search", type=int, default=500)
    p.add_argument("--num_subplans", type=int, default=5)
    p.add_argument("--partial_denoise_steps", type=int, default=20)
    p.add_argument("--jumpy_interval", type=int, default=10)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    # Model
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n_diffusion_steps", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=512)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mctd_maze")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    device = torch.device(args.device)

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"eval-{args.env}-seed{args.seed}",
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    logger.info(f"Loading env: {args.env}")
    env, _ = make_env_and_dataset(args.env, data_dir=args.data_dir, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

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
    )
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------
    # MCTD config
    # ------------------------------------------------------------------
    mctd_cfg = MCTDConfig(
        num_subplans=args.num_subplans,
        max_search=args.max_search,
        partial_denoise_steps=args.partial_denoise_steps,
        jumpy_interval=args.jumpy_interval,
        ddim_eta=args.ddim_eta,
    )
    controller = HeuristicController(obs_dim, act_dim)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    successes = []
    returns = []

    for ep in range(args.num_eval_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        goal_obs = info.get("goal", None)
        goal_tensor = torch.tensor(goal_obs, dtype=torch.float32) if goal_obs is not None else None

        def reward_fn(traj: torch.Tensor) -> float:
            if goal_tensor is None:
                return 0.0
            return goal_reaching_reward(traj, goal_tensor, obs_dim)

        planner = MCTD(model, reward_fn, mctd_cfg, goal=goal_tensor, device=str(device))

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        planned_traj = planner.plan(obs_tensor)

        ep_return = 0.0
        terminated = False
        for t in range(args.horizon):
            action = controller.get_action(obs, planned_traj, t)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break

        success = float(terminated)
        successes.append(success)
        returns.append(ep_return)

        if (ep + 1) % 10 == 0:
            sr = np.mean(successes) * 100
            logger.info(
                f"Episode {ep+1:4d}/{args.num_eval_episodes} | "
                f"SR={sr:.1f}%  mean_return={np.mean(returns):.3f}"
            )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    results = {
        "env": args.env,
        "success_rate": float(np.mean(successes)) * 100,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "num_episodes": args.num_eval_episodes,
        "checkpoint": args.checkpoint,
        "max_search": args.max_search,
    }
    logger.info(
        f"\n=== Results ===\n"
        f"  Success Rate : {results['success_rate']:.1f}%\n"
        f"  Mean Return  : {results['mean_return']:.3f}\n"
    )
    save_results(results, args.output)

    if not args.no_wandb:
        import wandb
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()
