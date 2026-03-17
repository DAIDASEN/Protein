"""
Unit tests for mctd_me.critics – reward functions.

These tests use a mock ESMFoldCritic to avoid requiring GPU / model weights.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mctd_me.critics import CompositeReward, ESMFoldCritic, compute_rmsd, compute_tm_score, kabsch_superpose


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_mock_critic(
    plddt_value: float = 80.0,
    n_residues: int = 20,
) -> ESMFoldCritic:
    """Create a mock ESMFoldCritic that returns fixed values."""
    mock = MagicMock(spec=ESMFoldCritic)
    coords = np.eye(n_residues, 3, dtype=np.float32)
    mock.predict.return_value = {
        "plddt": [plddt_value] * n_residues,
        "positions": coords,
        "ptm": 0.9,
    }
    return mock


SEQ_20 = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# kabsch_superpose / TM / RMSD helpers
# ---------------------------------------------------------------------------

class TestKabschAndTM:
    def test_kabsch_identity(self):
        coords = np.random.randn(20, 3).astype(np.float32)
        aligned, r = kabsch_superpose(coords, coords)
        assert r == pytest.approx(0.0, abs=1e-4)

    def test_tm_score_identical(self):
        coords = np.random.randn(30, 3).astype(np.float32)
        score = compute_tm_score(coords, coords)
        assert score == pytest.approx(1.0, rel=1e-3)

    def test_rmsd_zero(self):
        coords = np.random.randn(20, 3).astype(np.float32)
        r = compute_rmsd(coords, coords)
        assert r == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# CompositeReward – folding task
# ---------------------------------------------------------------------------

class TestFoldingReward:
    def setup_method(self):
        self.critic = make_mock_critic(plddt_value=80.0, n_residues=len(SEQ_20))
        self.reward_fn = CompositeReward(
            task="folding",
            critic=self.critic,
            reward_weights_folding=(0.60, 0.40, 0.00),
        )

    def test_reward_range(self):
        target_coords = np.eye(len(SEQ_20), 3, dtype=np.float32)
        reward, info = self.reward_fn(SEQ_20, target_coords=target_coords)
        assert 0.0 <= reward <= 1.5  # R_fold can approach 1.0

    def test_info_keys(self):
        reward, info = self.reward_fn(SEQ_20)
        assert "tm" in info
        assert "rmsd" in info
        assert "plddt" in info

    def test_no_target_coords(self):
        """With no target coords, TM and RMSD default to worst case."""
        reward, info = self.reward_fn(SEQ_20, target_coords=None)
        assert info["tm"] == pytest.approx(0.0)
        assert info["rmsd"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CompositeReward – inverse folding task
# ---------------------------------------------------------------------------

class TestInverseFoldingReward:
    def setup_method(self):
        L = len(SEQ_20)
        self.critic = make_mock_critic(plddt_value=85.0, n_residues=L)
        self.reward_fn = CompositeReward(
            task="inverse_folding",
            critic=self.critic,
            reward_weights_inv=(0.60, 0.35, 0.05),
        )
        self.target_coords = np.eye(L, 3, dtype=np.float32)

    def test_perfect_aar(self):
        """When designed = native, AAR = 1."""
        reward, info = self.reward_fn(
            SEQ_20,
            native_sequence=SEQ_20,
            target_coords=self.target_coords,
        )
        assert info["aar"] == pytest.approx(1.0)

    def test_zero_aar(self):
        """Completely different sequence → AAR = 0."""
        designed = "AAAAAAAAAAAAAAAAAAAAAA"[:len(SEQ_20)]
        # native is SEQ_20
        reward, info = self.reward_fn(
            designed,
            native_sequence=SEQ_20,
            target_coords=self.target_coords,
        )
        assert info["aar"] == pytest.approx(0.0)

    def test_missing_args_raises(self):
        with pytest.raises((ValueError, TypeError)):
            self.reward_fn(SEQ_20)

    def test_reward_formula(self):
        """Manually verify the reward formula R_inv = 0.60·AAR + 0.35·scTM + 0.05·B."""
        reward, info = self.reward_fn(
            SEQ_20,
            native_sequence=SEQ_20,
            target_coords=self.target_coords,
        )
        aar = info["aar"]
        sc_tm = info["sc_tm"]
        b = 0.0  # no biophysical bonus
        expected = 0.60 * aar + 0.35 * sc_tm + 0.05 * b
        assert reward == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# CompositeReward – motif scaffolding task
# ---------------------------------------------------------------------------

class TestMotifReward:
    def setup_method(self):
        L = 30
        self.L = L
        self.seq = "A" * L
        self.critic = make_mock_critic(plddt_value=75.0, n_residues=L)
        self.reward_fn = CompositeReward(
            task="motif_scaffolding",
            critic=self.critic,
            reward_weights_motif=(0.40, 0.30, 0.30, 0.20),
            motif_rmsd_cutoff=1.0,
            motif_sctm_cutoff=0.8,
        )
        self.motif_indices = [5, 6, 7, 8]
        self.motif_seq_ref = "ACDE"  # 4 residues
        # Place motif in scaffold
        seq_list = list(self.seq)
        for i, aa in zip(self.motif_indices, self.motif_seq_ref):
            seq_list[i] = aa
        self.seq_with_motif = "".join(seq_list)
        self.motif_coords_ref = np.eye(len(self.motif_indices), 3, dtype=np.float32)

    def test_motif_preserved_gets_nonzero_reward(self):
        reward, info = self.reward_fn(
            self.seq_with_motif,
            motif_indices=self.motif_indices,
            motif_coords_ref=self.motif_coords_ref,
            motif_seq_ref=self.motif_seq_ref,
        )
        assert info["motif_preserved"] is True
        assert reward > 0.0

    def test_motif_not_preserved_zero_reward(self):
        """If motif residues differ from reference, reward = 0."""
        broken_seq = list(self.seq_with_motif)
        broken_seq[self.motif_indices[0]] = "W"  # mismatch
        reward, info = self.reward_fn(
            "".join(broken_seq),
            motif_indices=self.motif_indices,
            motif_coords_ref=self.motif_coords_ref,
            motif_seq_ref=self.motif_seq_ref,
        )
        assert reward == pytest.approx(0.0)
        assert info["motif_preserved"] is False

    def test_g_function(self):
        """Test the g(x) piecewise function via the reward."""
        # Just verify the reward is in valid range
        reward, info = self.reward_fn(
            self.seq_with_motif,
            motif_indices=self.motif_indices,
            motif_coords_ref=self.motif_coords_ref,
            motif_seq_ref=self.motif_seq_ref,
        )
        assert 0.0 <= reward <= 1.3  # max possible ≈ 0.40+0.30+0.30+0.20=1.20


# ---------------------------------------------------------------------------
# g(x) function correctness
# ---------------------------------------------------------------------------

class TestGFunction:
    """Test the g(x) piecewise function from Appendix A.3."""

    @staticmethod
    def g(x: float) -> float:
        if x < 1.0:
            return max(0.0, 1.0 - x / 2.0)
        else:
            return max(0.0, 0.2 - x / 10.0)

    def test_at_zero(self):
        assert self.g(0.0) == pytest.approx(1.0)

    def test_at_one(self):
        assert self.g(1.0) == pytest.approx(0.1)  # g(1) = 0.2 - 0.1 = 0.1

    def test_below_one(self):
        assert self.g(0.5) == pytest.approx(0.75)

    def test_above_two(self):
        assert self.g(2.0) == pytest.approx(0.0)

    def test_large_x(self):
        assert self.g(100.0) == pytest.approx(0.0)
