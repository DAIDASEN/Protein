"""
Integration-level unit tests for mctd_me.mcts – MCTDME algorithm.

These tests mock the heavy models (ESMFold, DPLM-2) so no GPU or downloaded
weights are needed.  They verify the algorithm's control flow and statistics.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mctd_me.config import MCTDMEConfig
from mctd_me.critics import CompositeReward, ESMFoldCritic
from mctd_me.experts import BaseExpert
from mctd_me.mcts import MCTDME
from mctd_me.tree import MCTSNode


# ---------------------------------------------------------------------------
# Dummy expert
# ---------------------------------------------------------------------------

class DummyExpert(BaseExpert):
    """Returns random logits; always fills masks with 'A'."""

    def get_logprobs(self, sequence, mask_indices, structure=None):
        n = len(mask_indices)
        if n == 0:
            return torch.zeros(0, 20)
        return torch.log_softmax(torch.randn(n, 20), dim=-1)

    def sample(self, sequence, mask_indices, temperature=1.0, structure=None):
        result = list(sequence)
        for idx in mask_indices:
            result[idx] = "A"
        return "".join(result)

    @property
    def name(self):
        return "DummyExpert"


# ---------------------------------------------------------------------------
# Dummy ESMFold critic
# ---------------------------------------------------------------------------

def make_dummy_esmfold(seq_len: int = 20) -> MagicMock:
    mock = MagicMock(spec=ESMFoldCritic)
    mock.predict.return_value = {
        "plddt": [75.0] * seq_len,
        "positions": np.eye(seq_len, 3, dtype=np.float32),
        "ptm": 0.85,
    }
    return mock


# ---------------------------------------------------------------------------
# Build a test MCTDME engine
# ---------------------------------------------------------------------------

def build_engine(
    task: str = "inverse_folding",
    seq_len: int = 20,
    num_iters: int = 5,
    max_depth: int = 2,
) -> MCTDME:
    seq = "A" * seq_len
    native = "C" * seq_len
    target_coords = np.eye(seq_len, 3, dtype=np.float32)

    config = MCTDMEConfig(
        task=task,
        experts=["dummy"],
        num_rollouts=2,
        top_k_children=2,
        max_depth=max_depth,
        num_mcts_iterations=num_iters,
        diffusion_steps=2,
        temperature=1.0,
        device="cpu",
        seed=0,
    )

    dummy_expert = DummyExpert()
    dummy_esmfold = make_dummy_esmfold(seq_len)

    # Build a real CompositeReward backed by the dummy ESMFold
    critic = CompositeReward(
        task=task,
        critic=dummy_esmfold,
        reward_weights_inv=(0.60, 0.35, 0.05),
        reward_weights_folding=(0.60, 0.40, 0.00),
        reward_weights_motif=(0.40, 0.30, 0.30, 0.20),
    )

    engine = MCTDME(
        config=config,
        experts=[dummy_expert],
        critic=critic,
        esmfold_critic=dummy_esmfold,
    )
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMCTDMEBasic:
    def test_run_returns_list(self):
        engine = build_engine(task="inverse_folding", seq_len=20, num_iters=3)
        seq = "A" * 20
        target_coords = np.eye(20, 3, dtype=np.float32)
        results = engine.run(
            initial_sequence=seq,
            critic_kwargs={
                "native_sequence": "C" * 20,
                "target_coords": target_coords,
            },
            return_top_k=5,
            verbose=False,
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        seq_r, reward, info = results[0]
        assert isinstance(seq_r, str)
        assert isinstance(reward, float)
        assert isinstance(info, dict)

    def test_rewards_sorted_descending(self):
        engine = build_engine(task="inverse_folding", seq_len=20, num_iters=5)
        seq = "A" * 20
        results = engine.run(
            initial_sequence=seq,
            critic_kwargs={
                "native_sequence": "C" * 20,
                "target_coords": np.eye(20, 3, dtype=np.float32),
            },
            return_top_k=10,
            verbose=False,
        )
        rewards = [r for _, r, _ in results]
        assert rewards == sorted(rewards, reverse=True)

    def test_denovo_runs(self):
        engine = build_engine(task="folding", seq_len=15, num_iters=3)
        results = engine.run_denovo(
            sequence_length=15,
            critic_kwargs={"target_coords": None},
            return_top_k=3,
            verbose=False,
        )
        assert len(results) >= 1

    def test_lead_optimization_runs(self):
        engine = build_engine(task="inverse_folding", seq_len=20, num_iters=3)
        results = engine.run_lead_optimization(
            lead_sequence="A" * 20,
            critic_kwargs={
                "native_sequence": "C" * 20,
                "target_coords": np.eye(20, 3, dtype=np.float32),
            },
            return_top_k=5,
            verbose=False,
        )
        assert len(results) >= 1

    def test_cache_grows(self):
        engine = build_engine(task="folding", seq_len=20, num_iters=5)
        assert len(engine.cache) == 0
        engine.run(
            initial_sequence="A" * 20,
            critic_kwargs={},
            return_top_k=5,
            verbose=False,
        )
        assert len(engine.cache) >= 1

    def test_no_model_download_needed(self):
        """Verify that with dummy experts no real model loading occurs."""
        engine = build_engine(task="folding", seq_len=10, num_iters=2)
        # If this runs without ImportError or network calls, the test passes
        results = engine.run(
            initial_sequence="A" * 10,
            critic_kwargs={},
            return_top_k=3,
            verbose=False,
        )
        assert len(results) >= 1


class TestMCTDMEMotif:
    def test_motif_scaffolding_runs(self):
        seq_len = 20
        engine = build_engine(task="motif_scaffolding", seq_len=seq_len, num_iters=3)

        motif_idx = [5, 6, 7]
        motif_seq_ref = "ACD"
        motif_coords_ref = np.eye(len(motif_idx), 3, dtype=np.float32)

        seq = list("A" * seq_len)
        for i, aa in zip(motif_idx, motif_seq_ref):
            seq[i] = aa
        initial_seq = "".join(seq)

        results = engine.run(
            initial_sequence=initial_seq,
            critic_kwargs={
                "motif_indices": motif_idx,
                "motif_coords_ref": motif_coords_ref,
                "motif_seq_ref": motif_seq_ref,
            },
            return_top_k=5,
            verbose=False,
        )
        assert len(results) >= 1


class TestMCTDMEStatistics:
    def test_evaluations_tracked(self):
        engine = build_engine(num_iters=5)
        engine.run(
            initial_sequence="A" * 20,
            critic_kwargs={
                "native_sequence": "C" * 20,
                "target_coords": np.eye(20, 3, dtype=np.float32),
            },
            return_top_k=5,
            verbose=False,
        )
        assert engine._n_evaluations >= 1

    def test_cache_hits_tracked(self):
        engine = build_engine(num_iters=10)
        # Run twice to trigger cache hits on repeated sequences
        for _ in range(2):
            engine.run(
                initial_sequence="A" * 20,
                critic_kwargs={
                    "native_sequence": "C" * 20,
                    "target_coords": np.eye(20, 3, dtype=np.float32),
                },
                return_top_k=5,
                verbose=False,
            )
        # The root sequence "A"*20 is evaluated once, then cached
        assert engine._n_cache_hits >= 1
