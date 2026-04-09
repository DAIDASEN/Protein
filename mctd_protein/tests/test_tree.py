"""
Unit tests for mctd_me.tree – MCTSNode.
"""

import pytest
from mctd_me.tree import MCTSNode


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMCTSNodeConstruction:
    def test_basic_attributes(self):
        node = MCTSNode(sequence="ACDEFGHIK", depth=0, reward=0.5)
        assert node.sequence == "ACDEFGHIK"
        assert node.depth == 0
        assert node.reward == 0.5
        assert node.visit_count == 0
        assert node.q_value == 0.0
        assert node.is_leaf
        assert node.is_root

    def test_default_values(self):
        node = MCTSNode(sequence="AAA")
        assert node.parent is None
        assert node.depth == 0
        assert node.reward == 0.0
        assert node.u_ent == 0.0
        assert node.u_div == 0.0
        assert not node.is_expanded


# ---------------------------------------------------------------------------
# Child management
# ---------------------------------------------------------------------------

class TestMCTSNodeChildren:
    def test_add_child(self):
        root = MCTSNode(sequence="AAAA", depth=0)
        child = root.add_child(sequence="CCCC", reward=0.7, u_ent=0.1, u_div=0.2)

        assert child.parent is root
        assert child.depth == 1
        assert child.sequence == "CCCC"
        assert child.reward == 0.7
        assert child.u_ent == 0.1
        assert child.u_div == 0.2
        assert child in root.children
        assert not root.is_leaf

    def test_multiple_children(self):
        root = MCTSNode(sequence="AAAA")
        for i in range(3):
            root.add_child(sequence=f"SEQ{i}", reward=float(i) * 0.1)
        assert len(root.children) == 3

    def test_best_child(self):
        root = MCTSNode(sequence="ROOT")
        c1 = root.add_child("C1", reward=0.1)
        c2 = root.add_child("C2", reward=0.9)
        c3 = root.add_child("C3", reward=0.5)
        # Before any updates, Q=0 for all; best_child returns first
        # After update:
        c2.update(0.9)
        assert root.best_child() is c2


# ---------------------------------------------------------------------------
# Update and backup
# ---------------------------------------------------------------------------

class TestMCTSNodeUpdate:
    def test_update_increments_visit(self):
        node = MCTSNode(sequence="AAA")
        node.update(0.5)
        assert node.visit_count == 1
        node.update(0.3)
        assert node.visit_count == 2

    def test_max_q_value(self):
        node = MCTSNode(sequence="AAA")
        node.update(0.3, rule="max")
        node.update(0.7, rule="max")
        node.update(0.5, rule="max")
        assert node.q_value == pytest.approx(0.7)

    def test_mean_q_value(self):
        node = MCTSNode(sequence="AAA")
        node.update(0.3)
        node.update(0.7)
        assert node.q_mean == pytest.approx(0.5)

    def test_backpropagate(self):
        root = MCTSNode(sequence="ROOT", depth=0)
        child = root.add_child("CHILD", reward=0.8)
        grandchild = child.add_child("GRANDCHILD", reward=0.9)

        grandchild.backpropagate(0.9, rule="max")

        assert grandchild.visit_count == 1
        assert child.visit_count == 1
        assert root.visit_count == 1
        assert root.q_value == pytest.approx(0.9)

    def test_backpropagate_sum(self):
        root = MCTSNode(sequence="ROOT")
        child = root.add_child("CHILD")
        grandchild = child.add_child("GRANDCHILD")
        grandchild.backpropagate(0.6, rule="sum")
        grandchild.backpropagate(0.4, rule="sum")
        assert root.visit_count == 2
        assert root.q_mean == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestMCTSNodeUtility:
    def test_subtree_size(self):
        root = MCTSNode(sequence="ROOT")
        c1 = root.add_child("C1")
        c2 = root.add_child("C2")
        c1.add_child("C1A")
        assert root.subtree_size() == 4

    def test_all_sequences(self):
        root = MCTSNode(sequence="ROOT")
        root.add_child("C1")
        root.add_child("C2")
        seqs = root.all_sequences()
        assert "ROOT" in seqs
        assert "C1" in seqs
        assert "C2" in seqs

    def test_path_to_root(self):
        root = MCTSNode(sequence="ROOT", depth=0)
        child = root.add_child("CHILD")
        grandchild = child.add_child("GRAND")
        path = grandchild.path_to_root()
        assert len(path) == 3
        assert path[0] is root
        assert path[-1] is grandchild

    def test_repr(self):
        node = MCTSNode(sequence="ACDEFG", depth=1, reward=0.42)
        r = repr(node)
        assert "MCTSNode" in r
        assert "depth=1" in r
