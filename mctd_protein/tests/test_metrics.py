"""
Unit tests for mctd_me.metrics – evaluation metrics.
"""

import math

import numpy as np
import pytest

from mctd_me.metrics import (
    amino_acid_recovery,
    compute_all_metrics,
    kabsch_superpose,
    mean_plddt,
    motif_rmsd,
    normalise_plddt,
    rmsd,
    sctm_score,
    tm_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_coords(n: int, scale: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, 3).astype(np.float32) * scale


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------

class TestTMScore:
    def test_identical(self):
        coords = random_coords(50)
        assert tm_score(coords, coords) == pytest.approx(1.0, rel=1e-4)

    def test_range(self):
        pred = random_coords(50, seed=0)
        ref  = random_coords(50, seed=1)
        score = tm_score(pred, ref)
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert tm_score(np.zeros((0, 3)), np.zeros((0, 3))) == 0.0

    def test_short_protein(self):
        """For short proteins d0 is capped at 0.5."""
        coords = random_coords(10)
        score = tm_score(coords, coords)
        assert score == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Kabsch superposition
# ---------------------------------------------------------------------------

class TestKabschSuperpose:
    def test_identity(self):
        coords = random_coords(30)
        aligned, rmsd_val = kabsch_superpose(coords, coords)
        assert rmsd_val == pytest.approx(0.0, abs=1e-4)
        assert np.allclose(aligned, coords, atol=1e-4)

    def test_pure_translation(self):
        coords = random_coords(20)
        shifted = coords + np.array([5.0, 3.0, -2.0])
        aligned, rmsd_val = kabsch_superpose(shifted, coords)
        assert rmsd_val == pytest.approx(0.0, abs=1e-4)

    def test_pure_rotation(self):
        coords = random_coords(20)
        # Rotate 90° around z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        rotated = coords @ R.T
        aligned, rmsd_val = kabsch_superpose(rotated, coords)
        assert rmsd_val == pytest.approx(0.0, abs=1e-4)

    def test_mismatched_length_raises(self):
        a = random_coords(10)
        b = random_coords(15)
        with pytest.raises(AssertionError):
            kabsch_superpose(a, b)


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

class TestRMSD:
    def test_zero_rmsd(self):
        coords = random_coords(20)
        assert rmsd(coords, coords) == pytest.approx(0.0, abs=1e-4)

    def test_positive(self):
        a = random_coords(20, seed=0)
        b = random_coords(20, seed=1)
        val = rmsd(a, b, superpose=True)
        assert val >= 0.0

    def test_no_superpose(self):
        coords = random_coords(10)
        shifted = coords + 5.0
        val = rmsd(coords, shifted, superpose=False)
        assert val == pytest.approx(5.0 * math.sqrt(3), rel=1e-4)


# ---------------------------------------------------------------------------
# Motif RMSD
# ---------------------------------------------------------------------------

class TestMotifRMSD:
    def test_identical(self):
        coords = random_coords(50)
        motif_idx = [5, 10, 15, 20]
        assert motif_rmsd(coords, coords, motif_idx) == pytest.approx(0.0, abs=1e-4)

    def test_partial_match(self):
        pred = random_coords(50, seed=0)
        ref  = random_coords(50, seed=1)
        motif_idx = list(range(10))
        val = motif_rmsd(pred, ref, motif_idx)
        assert val >= 0.0

    def test_empty_motif(self):
        coords = random_coords(10)
        val = motif_rmsd(coords, coords, [])
        assert math.isinf(val)


# ---------------------------------------------------------------------------
# scTM
# ---------------------------------------------------------------------------

class TestSCTM:
    def test_identical(self):
        coords = random_coords(40)
        score = sctm_score(coords, coords)
        assert score == pytest.approx(1.0, rel=1e-3)

    def test_range(self):
        pred = random_coords(40, seed=0)
        tgt  = random_coords(40, seed=1)
        score = sctm_score(pred, tgt)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# pLDDT
# ---------------------------------------------------------------------------

class TestPLDDT:
    def test_mean_plddt_normalisation(self):
        plddt = [70.0, 80.0, 90.0]
        val = mean_plddt(plddt, scale=100.0)
        assert val == pytest.approx(0.8)

    def test_empty(self):
        assert mean_plddt([]) == 0.0

    def test_normalise_list(self):
        normalised = normalise_plddt([50.0, 75.0, 100.0])
        assert normalised == pytest.approx([0.5, 0.75, 1.0])


# ---------------------------------------------------------------------------
# AAR
# ---------------------------------------------------------------------------

class TestAAR:
    def test_perfect_recovery(self):
        assert amino_acid_recovery("ACDEF", "ACDEF") == pytest.approx(1.0)

    def test_zero_recovery(self):
        assert amino_acid_recovery("AAAAA", "CCCCC") == pytest.approx(0.0)

    def test_partial(self):
        # 3 out of 5 match
        assert amino_acid_recovery("ACDEF", "ACCCC") == pytest.approx(2 / 5)

    def test_empty(self):
        assert amino_acid_recovery("", "") == 0.0

    def test_mask_only(self):
        # Only check positions 1, 3
        result = amino_acid_recovery("ACDEF", "ACCCF", mask_only=[1, 3])
        # pos 1: C==C ✓; pos 3: E!=C ✗  → 1/2
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_runs_without_error(self):
        L = 30
        seq = "A" * L
        coords = random_coords(L)
        plddt_raw = [70.0] * L

        metrics = compute_all_metrics(
            designed_sequence=seq,
            native_sequence=seq,
            coords_pred=coords,
            coords_target=coords,
            plddt_raw=plddt_raw,
        )
        assert "tm" in metrics
        assert "sc_tm" in metrics
        assert "rmsd_global" in metrics
        assert "plddt_mean" in metrics
        assert "aar" in metrics
        assert metrics["tm"] == pytest.approx(1.0, rel=1e-3)
        assert metrics["aar"] == pytest.approx(1.0)

    def test_with_motif(self):
        L = 30
        seq = "A" * L
        coords = random_coords(L)
        plddt_raw = [80.0] * L
        motif_idx = [5, 10, 15]
        motif_ref = coords[motif_idx]

        metrics = compute_all_metrics(
            designed_sequence=seq,
            native_sequence=seq,
            coords_pred=coords,
            coords_target=coords,
            plddt_raw=plddt_raw,
            motif_indices=motif_idx,
            motif_coords_ref=motif_ref,
        )
        assert "rmsd_motif" in metrics
        assert metrics["rmsd_motif"] == pytest.approx(0.0, abs=1e-4)
