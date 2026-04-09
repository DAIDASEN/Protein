"""
MCTD-Maze: Monte Carlo Tree Diffusion for System 2 Planning.

arXiv:2502.07202 — ICML 2025 Spotlight
Official code: https://github.com/ahn-ml/mctd
"""

from mctd_maze.config import MCTDMazeConfig
from mctd_maze.planner import MCTD
from mctd_maze.tree import MCTSNode

__version__ = "0.1.0"
__all__ = ["MCTDMazeConfig", "MCTD", "MCTSNode"]
