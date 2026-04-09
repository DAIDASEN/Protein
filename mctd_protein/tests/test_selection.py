"""
Unit tests for mctd_me.selection – PH-UCT-ME selection rule.
"""

import math

import pytest
import torch

from mctd_me.selection import (
    _entropy,
    compute_consensus_prior,
    compute_u_div,
    compute_u_ent_multi,
    compute_u_ent_single,
    ph_ucb_me_score,
    select_child_ph_uct_me,
)
from mctd_me.tree import MCTSNode


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform(self):
        probs = torch.ones(20) / 20.0
        h = _entropy(probs)
        expected = math.log(20)
        assert float(h) == pytest.approx(expected, rel=1e-4)

    def test_deterministic(self):
        probs = torch.zeros(20)
        probs[5] = 1.0
        h = _entropy(probs)
        assert float(h) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Consensus prior
# ---------------------------------------------------------------------------

class TestConsensusPrior:
    def test_single_expert_identity(self):
        """With one expert the consensus prior should match the expert distribution."""
        lp = torch.log_softmax(torch.randn(20), dim=-1)
        pi = compute_consensus_prior([lp], temperature=1.0)
        expected = torch.softmax(lp, dim=-1)
        assert torch.allclose(pi, expected, atol=1e-5)

    def test_sum_to_one(self):
        lp1 = torch.log_softmax(torch.randn(20), dim=-1)
        lp2 = torch.log_softmax(torch.randn(20), dim=-1)
        pi = compute_consensus_prior([lp1, lp2], temperature=1.0)
        assert float(pi.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_temperature_effect(self):
        """Higher temperature → flatter distribution."""
        lp1 = torch.log_softmax(torch.randn(20), dim=-1)
        lp2 = torch.log_softmax(torch.randn(20), dim=-1)
        pi_cold = compute_consensus_prior([lp1, lp2], temperature=0.1)
        pi_hot = compute_consensus_prior([lp1, lp2], temperature=10.0)
        # Entropy of hot distribution should be higher
        h_cold = float(_entropy(pi_cold))
        h_hot = float(_entropy(pi_hot))
        assert h_hot > h_cold


# ---------------------------------------------------------------------------
# Epistemic uncertainty
# ---------------------------------------------------------------------------

class TestUEnt:
    def test_identical_experts_zero(self):
        """When both experts agree perfectly, epistemic uncertainty → 0."""
        probs = torch.softmax(torch.randn(5, 20), dim=-1)  # (|M|=5, V=20)
        u = compute_u_ent_multi([probs, probs.clone()])
        assert u == pytest.approx(0.0, abs=1e-5)

    def test_single_expert(self):
        """Single-expert returns Shannon entropy over masked positions."""
        probs = torch.ones(5, 20) / 20.0  # (5, 20)
        u = compute_u_ent_single(probs)
        assert u == pytest.approx(math.log(20), rel=1e-4)

    def test_multi_expert_positive(self):
        """Different experts should produce positive BALD uncertainty."""
        p1 = torch.zeros(3, 20)
        p2 = torch.zeros(3, 20)
        p1[:, 0] = 1.0   # expert 1 always picks AA 0
        p2[:, 1] = 1.0   # expert 2 always picks AA 1
        u = compute_u_ent_multi([p1, p2])
        assert u > 0.0


# ---------------------------------------------------------------------------
# Diversity bonus
# ---------------------------------------------------------------------------

class TestUDiv:
    def test_identical(self):
        assert compute_u_div("ACDEF", "ACDEF") == pytest.approx(0.0)

    def test_all_different(self):
        assert compute_u_div("AAAAA", "CCCCC") == pytest.approx(1.0)

    def test_partial(self):
        # 2 out of 4 positions differ
        result = compute_u_div("ACDE", "ACFF")
        assert result == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# PH-UCB-ME score
# ---------------------------------------------------------------------------

class TestPHUCBScore:
    def test_unvisited_child_infinite(self):
        """Unvisited child with parent visits should get high score."""
        node = MCTSNode(sequence="TEST", u_ent=0.5, u_div=0.5)
        # parent has 1 visit, child has 0
        score = ph_ucb_me_score(node, parent_visit_count=1, pi_cons=1.0)
        assert score > 1e6  # exploration bonus dominates

    def test_zero_parent_visits(self):
        """If parent has 0 visits, exploration term should be infinite."""
        node = MCTSNode(sequence="TEST")
        score = ph_ucb_me_score(node, parent_visit_count=0)
        assert score == float("inf")

    def test_high_q_dominates(self):
        """Well-visited child with high Q should score well."""
        node = MCTSNode(sequence="TEST", u_ent=0.1, u_div=0.1)
        for _ in range(100):
            node.update(0.95, rule="max")
        score = ph_ucb_me_score(node, parent_visit_count=200, pi_cons=1.0)
        assert score >= 0.9


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelectChild:
    def test_selects_best_unvisited(self):
        """Selection should prefer unvisited children (exploration)."""
        parent = MCTSNode(sequence="ROOT")
        parent.update(0.5)  # parent has 1 visit
        c1 = parent.add_child("C1", u_ent=0.5, u_div=0.5)
        c2 = parent.add_child("C2", u_ent=0.1, u_div=0.1)
        # c1 has higher uncertainty bonus; both unvisited
        selected = select_child_ph_uct_me(parent)
        assert selected is c1

    def test_no_children_raises(self):
        parent = MCTSNode(sequence="ROOT")
        with pytest.raises(ValueError):
            select_child_ph_uct_me(parent)

    def test_exploitation_after_visits(self):
        """After many visits, should prefer high-Q child."""
        parent = MCTSNode(sequence="ROOT")
        for _ in range(100):
            parent.update(0.5)

        c_high = parent.add_child("HIGH", u_ent=0.0, u_div=0.0)
        c_low  = parent.add_child("LOW",  u_ent=0.0, u_div=0.0)
        # Give c_high very high Q
        for _ in range(50):
            c_high.update(0.95, rule="max")
        # c_low has bad Q
        for _ in range(50):
            c_low.update(0.1, rule="max")

        selected = select_child_ph_uct_me(parent, exploration_constant=0.01)
        assert selected is c_high
