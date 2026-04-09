"""
Unit tests for mctd_me.utils – utility functions.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mctd_me.utils import (
    SequenceCache,
    amino_acid_recovery,
    clean_sequence,
    hamming_distance,
    identity,
    parse_motif_spec,
    parse_pdb_ca,
    read_fasta,
    sequence_hash,
    validate_sequence,
    write_fasta,
    write_pdb_ca,
    EVODIFF_MOTIF_IDS,
)

# Import amino_acid_recovery from metrics (not utils, but re-exported via utils)
from mctd_me.metrics import amino_acid_recovery


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

class TestSequenceValidation:
    def test_valid(self):
        assert validate_sequence("ACDEFGHIKLMNPQRSTVWY") is True

    def test_invalid_char(self):
        assert validate_sequence("ACDB") is False  # B is not standard

    def test_empty(self):
        assert validate_sequence("") is True  # vacuously true


class TestCleanSequence:
    def test_strips_whitespace(self):
        assert clean_sequence("  ACD\n  EF  ") == "ACDEF"

    def test_uppercase(self):
        assert clean_sequence("acdef") == "ACDEF"

    def test_nonstandard_replaced(self):
        result = clean_sequence("AZBDEF")  # Z is non-standard
        assert "Z" not in result
        assert len(result) == 6


class TestSequenceHash:
    def test_deterministic(self):
        h1 = sequence_hash("ACDEF")
        h2 = sequence_hash("ACDEF")
        assert h1 == h2

    def test_different(self):
        assert sequence_hash("ACDEF") != sequence_hash("AAAAA")

    def test_length_16(self):
        assert len(sequence_hash("ACDEF")) == 16


class TestHamming:
    def test_zero(self):
        assert hamming_distance("ACDEF", "ACDEF") == 0

    def test_full(self):
        assert hamming_distance("AAAAA", "CCCCC") == 5

    def test_partial(self):
        assert hamming_distance("ACDEF", "ACCCF") == 2


class TestIdentity:
    def test_identical(self):
        assert identity("ACDEF", "ACDEF") == pytest.approx(1.0)

    def test_none(self):
        assert identity("AAAAA", "CCCCC") == pytest.approx(0.0)

    def test_empty(self):
        assert identity("", "") == 0.0


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

class TestFASTA:
    def test_roundtrip(self):
        seqs = {"seq1 description": "ACDEF", "seq2": "GHIKL"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as fh:
            path = fh.name
        try:
            write_fasta(seqs, path)
            loaded = read_fasta(path)
            assert loaded == seqs
        finally:
            os.unlink(path)

    def test_long_sequences_wrapped(self):
        long_seq = "A" * 200
        seqs = {"long": long_seq}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as fh:
            path = fh.name
        try:
            write_fasta(seqs, path, line_width=60)
            loaded = read_fasta(path)
            assert loaded["long"] == long_seq
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# PDB I/O
# ---------------------------------------------------------------------------

class TestPDBIO:
    def test_write_read_roundtrip(self):
        """Write a synthetic Cα PDB and read it back."""
        L = 10
        coords = np.random.randn(L, 3).astype(np.float32)
        seq = "ACDEFGHIKL"

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fh:
            path = fh.name
        try:
            write_pdb_ca(coords, seq, path)
            ca_read, seq_read = parse_pdb_ca(path, chain_id="A")
            assert len(seq_read) == L
            assert np.allclose(ca_read, coords, atol=0.01)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# SequenceCache
# ---------------------------------------------------------------------------

class TestSequenceCache:
    def test_basic_set_get(self):
        cache = SequenceCache()
        cache["ACDEF"] = (0.8, {"tm": 0.8})
        assert "ACDEF" in cache
        reward, info = cache["ACDEF"]
        assert reward == 0.8
        assert info["tm"] == 0.8

    def test_not_in(self):
        cache = SequenceCache()
        assert "ZZZZZ" not in cache

    def test_len(self):
        cache = SequenceCache()
        cache["A"] = (0.1, {})
        cache["B"] = (0.5, {})
        assert len(cache) == 2

    def test_top_k(self):
        cache = SequenceCache()
        cache["LOW"]  = (0.1, {})
        cache["HIGH"] = (0.9, {})
        cache["MID"]  = (0.5, {})
        top = cache.top_k(2)
        assert top[0][0] == "HIGH"
        assert top[1][0] == "MID"

    def test_get_default(self):
        cache = SequenceCache()
        assert cache.get("MISSING", None) is None

    def test_disk_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            path = fh.name
        try:
            cache = SequenceCache(cache_path=path)
            cache["ACDEF"] = (0.75, {"test": True})
            cache.save()

            cache2 = SequenceCache(cache_path=path)
            assert "ACDEF" in cache2
            assert cache2["ACDEF"][0] == 0.75
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Motif utilities
# ---------------------------------------------------------------------------

class TestMotifUtils:
    def test_parse_motif_spec_3parts(self):
        name, indices = parse_motif_spec("1BCF:A:10-15")
        assert name == "1BCF_A"
        assert indices == [9, 10, 11, 12, 13, 14]  # 0-based

    def test_parse_motif_spec_2parts(self):
        name, indices = parse_motif_spec("1BCF:10-12")
        assert name == "1BCF"
        assert indices == [9, 10, 11]

    def test_invalid_spec(self):
        with pytest.raises(ValueError):
            parse_motif_spec("INVALID")


class TestEvoDiffMotifIDs:
    def test_count(self):
        """Verify the EvoDiff motif set has the expected motifs."""
        assert len(EVODIFF_MOTIF_IDS) >= 20

    def test_known_ids(self):
        known = ["1BCF", "1PRW", "5TRV_long", "6EXZ_med"]
        for m in known:
            assert m in EVODIFF_MOTIF_IDS
