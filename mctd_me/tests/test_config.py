"""
Unit tests for mctd_me.config – MCTDMEConfig.
"""

import pytest
from mctd_me.config import MCTDMEConfig


class TestMCTDMEConfig:
    def test_default_construction(self):
        cfg = MCTDMEConfig()
        assert cfg.task == "inverse_folding"
        assert len(cfg.experts) == 3
        assert cfg.num_rollouts == 3
        assert cfg.top_k_children == 3
        assert cfg.max_depth == 5
        assert cfg.exploration_constant == pytest.approx(1.414)
        assert cfg.diffusion_steps == 150
        assert cfg.temperature == pytest.approx(1.0)
        assert cfg.w_ent == pytest.approx(1.0)
        assert cfg.w_div == pytest.approx(1.0)
        assert cfg.backup_rule == "max"

    def test_num_experts_property(self):
        cfg = MCTDMEConfig(experts=["airkingbd/dplm_150m", "airkingbd/dplm_650m"])
        assert cfg.num_experts == 2

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task must be one of"):
            MCTDMEConfig(task="unknown")

    def test_invalid_backup_rule_raises(self):
        with pytest.raises(ValueError, match="backup_rule"):
            MCTDMEConfig(backup_rule="invalid")

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            MCTDMEConfig(temperature=-1.0)

    def test_empty_experts_raises(self):
        with pytest.raises(ValueError, match="expert"):
            MCTDMEConfig(experts=[])

    def test_reward_weights_folding(self):
        cfg = MCTDMEConfig()
        alpha, beta, gamma = cfg.reward_weights_folding
        assert alpha == pytest.approx(0.60)
        assert beta  == pytest.approx(0.40)
        assert gamma == pytest.approx(0.00)

    def test_reward_weights_inv(self):
        cfg = MCTDMEConfig()
        w_aar, w_sctm, w_b = cfg.reward_weights_inv
        assert w_aar  == pytest.approx(0.60)
        assert w_sctm == pytest.approx(0.35)
        assert w_b    == pytest.approx(0.05)

    def test_reward_weights_motif(self):
        cfg = MCTDMEConfig()
        w1, w2, w3, w4 = cfg.reward_weights_motif
        assert w1 == pytest.approx(0.40)
        assert w2 == pytest.approx(0.30)
        assert w3 == pytest.approx(0.30)
        assert w4 == pytest.approx(0.20)

    def test_valid_tasks(self):
        for task in ("folding", "inverse_folding", "motif_scaffolding"):
            cfg = MCTDMEConfig(task=task)
            assert cfg.task == task

    def test_custom_experts(self):
        cfg = MCTDMEConfig(experts=["airkingbd/dplm_3b"])
        assert cfg.num_experts == 1
        assert cfg.experts[0] == "airkingbd/dplm_3b"

    def test_plddt_thresholds(self):
        cfg = MCTDMEConfig()
        assert cfg.plddt_mask_threshold == pytest.approx(0.7)
        assert cfg.plddt_threshold_min  == pytest.approx(0.4)

    def test_motif_cutoffs(self):
        cfg = MCTDMEConfig()
        assert cfg.motif_rmsd_hard_cutoff  == pytest.approx(1.0)
        assert cfg.motif_sctm_hard_cutoff  == pytest.approx(0.8)
