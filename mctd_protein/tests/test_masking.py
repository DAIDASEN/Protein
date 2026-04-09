"""
Unit tests for mctd_me.masking – pLDDT-guided progressive masking.
"""

import pytest

from mctd_me.masking import (
    apply_mask,
    apply_mask_str,
    compute_plddt_threshold,
    get_mask_indices,
    get_mask_set_for_node,
    make_all_masked_sequence,
    MASK_TOKEN,
)


# ---------------------------------------------------------------------------
# Threshold schedule
# ---------------------------------------------------------------------------

class TestPLDDTThreshold:
    def test_no_progressive(self):
        t = compute_plddt_threshold(depth=3, max_depth=5, progressive=False)
        assert t == 0.7

    def test_progressive_root(self):
        t = compute_plddt_threshold(depth=0, max_depth=5, progressive=True)
        assert t == pytest.approx(0.7)

    def test_progressive_max_depth(self):
        t = compute_plddt_threshold(depth=5, max_depth=5, progressive=True)
        assert t == pytest.approx(0.4)

    def test_progressive_midpoint(self):
        t = compute_plddt_threshold(depth=5, max_depth=10, progressive=True)
        assert t == pytest.approx(0.55)  # halfway between 0.7 and 0.4

    def test_zero_max_depth(self):
        t = compute_plddt_threshold(depth=0, max_depth=0, progressive=True)
        assert t == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Mask index computation
# ---------------------------------------------------------------------------

class TestGetMaskIndices:
    def test_below_threshold(self):
        plddt = [0.9, 0.3, 0.85, 0.2, 0.95]
        indices = get_mask_indices(plddt, threshold=0.5)
        assert 1 in indices
        assert 3 in indices
        assert 0 not in indices

    def test_minimum_masking_enforced(self):
        """Even if nothing is below threshold, min masking fraction applies."""
        plddt = [0.95, 0.95, 0.95, 0.95, 0.95]  # all high
        indices = get_mask_indices(plddt, threshold=0.5, min_mask_frac=0.2)
        assert len(indices) >= 1  # at least 1 = 20% of 5

    def test_maximum_masking_enforced(self):
        plddt = [0.1] * 20  # all very low
        indices = get_mask_indices(plddt, threshold=0.9, max_mask_frac=0.5)
        assert len(indices) <= 10  # 50% of 20

    def test_sorted_output(self):
        plddt = [0.1, 0.9, 0.2, 0.9, 0.3]
        indices = get_mask_indices(plddt, threshold=0.5)
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------

class TestApplyMask:
    def test_apply_mask(self):
        tokens = apply_mask("ACDEF", [1, 3])
        assert tokens[1] == MASK_TOKEN
        assert tokens[3] == MASK_TOKEN
        assert tokens[0] == "A"
        assert tokens[4] == "F"
        assert len(tokens) == 5

    def test_no_mask(self):
        tokens = apply_mask("ACDEF", [])
        assert tokens == list("ACDEF")


class TestApplyMaskStr:
    def test_mask_str(self):
        masked = apply_mask_str("ACDEF", [1, 3])
        assert masked[1] == "X"
        assert masked[3] == "X"
        assert masked == "AXDXF"


# ---------------------------------------------------------------------------
# make_all_masked_sequence
# ---------------------------------------------------------------------------

class TestAllMasked:
    def test_length(self):
        tokens = make_all_masked_sequence(10)
        assert len(tokens) == 10

    def test_all_mask_tokens(self):
        tokens = make_all_masked_sequence(5)
        assert all(t == MASK_TOKEN for t in tokens)


# ---------------------------------------------------------------------------
# get_mask_set_for_node
# ---------------------------------------------------------------------------

class TestGetMaskSetForNode:
    def test_returns_tuple(self):
        plddt = [0.9, 0.3, 0.8, 0.2, 0.7]
        mask_indices, threshold = get_mask_set_for_node(
            sequence="ACDEF",
            plddt_scores=plddt,
            depth=0,
            max_depth=5,
        )
        assert isinstance(mask_indices, list)
        assert isinstance(threshold, float)
        assert 1 in mask_indices or 3 in mask_indices

    def test_progressive_increases_masking(self):
        """Deeper nodes should mask more residues (lower threshold)."""
        plddt = [0.6] * 10  # moderate pLDDT
        seq = "ACDEFGHIKL"

        _, threshold_shallow = get_mask_set_for_node(
            seq, plddt, depth=0, max_depth=5
        )
        _, threshold_deep = get_mask_set_for_node(
            seq, plddt, depth=5, max_depth=5
        )
        assert threshold_deep <= threshold_shallow
