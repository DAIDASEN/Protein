"""Tests for MCTS tree node."""
import math

import torch
import pytest

from mctd_maze.tree import GuidanceLevel, MCTSNode


def make_traj():
    return torch.zeros(1, 10, 6)


def test_node_init():
    node = MCTSNode(subplan_idx=0, partial_traj=make_traj())
    assert node.visit_count == 0
    assert node.q_value == 0.0
    assert node.is_leaf()


def test_backpropagate():
    root = MCTSNode(0, make_traj())
    child = MCTSNode(1, make_traj(), parent=root)
    root.children.append(child)

    child.backpropagate(1.0)

    assert child.visit_count == 1
    assert root.visit_count == 1
    assert child.q_value == pytest.approx(1.0)


def test_ucb_score_unvisited_child():
    root = MCTSNode(0, make_traj())
    root.visit_count = 5
    child = MCTSNode(1, make_traj(), parent=root)
    root.children.append(child)
    # Unvisited child should have high (inf) UCB
    assert math.isinf(child.ucb_score())


def test_best_child():
    root = MCTSNode(0, make_traj())
    root.visit_count = 10
    for i, val in enumerate([0.3, 0.8, 0.5]):
        c = MCTSNode(1, make_traj(), parent=root)
        c.visit_count = 3
        c.total_value = val * 3
        root.children.append(c)
    best = root.best_child()
    assert best.q_value == pytest.approx(0.8)
