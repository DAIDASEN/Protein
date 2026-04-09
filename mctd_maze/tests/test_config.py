"""Tests for MCTDMazeConfig defaults."""

from mctd_maze.config import MCTDConfig, MCTDMazeConfig, ModelConfig, TrainConfig


def test_defaults_match_paper():
    cfg = MCTDMazeConfig()
    assert cfg.model.hidden_dim == 128
    assert cfg.model.n_layers == 12
    assert cfg.model.n_heads == 4
    assert cfg.model.ffn_dim == 512
    assert cfg.train.lr == 5e-4
    assert cfg.train.batch_size == 1024
    assert cfg.train.train_steps == 200_005
    assert cfg.mctd.num_subplans == 5
    assert cfg.mctd.max_search == 500
    assert cfg.mctd.partial_denoise_steps == 20
    assert cfg.mctd.jumpy_interval == 10
    assert cfg.mctd.ddim_eta == 0.0
