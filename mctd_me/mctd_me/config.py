"""
Configuration dataclass for MCTD-ME.

All hyperparameters are taken from Table 10 and Appendix A of the paper:
  "Monte Carlo Tree Diffusion with Multiple Experts for Protein Design"
  arXiv:2509.15796
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MCTDMEConfig:
    """
    Central configuration for the MCTD-ME algorithm.

    Attributes
    ----------
    task : str
        One of {"folding", "inverse_folding", "motif_scaffolding"}.
    experts : List[str]
        HuggingFace model IDs for DPLM-2 experts.
    use_proteinmpnn : bool
        Whether to include ProteinMPNN as an additional inverse-folding expert.
    num_rollouts : int
        Number of diffusion rollouts per expert per expansion (k_roll).
    top_k_children : int
        Number of children kept after expansion (K).
    max_depth : int
        Maximum MCTS tree depth.
    exploration_constant : float
        UCB1 exploration constant c_p (√2 ≈ 1.414 from paper).
    diffusion_steps : int
        Number of reverse diffusion steps per rollout.
    temperature : float
        Temperature τ for the consensus prior (Eq. 3).
    w_ent : float
        Weight for epistemic-uncertainty bonus U_ent (Eq. 5).
    w_div : float
        Weight for diversity bonus U_div (Eq. 6).
    backup_rule : str
        One of {"max", "sum"} – value backup strategy.
    plddt_mask_threshold : float
        Initial pLDDT threshold; residues below this are masked for expansion.
    progressive_mask : bool
        Whether to lower the pLDDT threshold progressively over tree depth.
    plddt_threshold_min : float
        Minimum pLDDT threshold reached at maximum depth.
    reward_weights_folding : Tuple[float, float, float]
        (alpha, beta, gamma) for the folding reward R_fold.
    reward_weights_inv : Tuple[float, float, float]
        (w_aar, w_sctm, w_b) for the inverse-folding reward R_inv.
    reward_weights_motif : Tuple[float, float, float, float]
        (w_plddt, w_rmsd, w_sctm, w_bonus) for R_motif.
    motif_rmsd_hard_cutoff : float
        Hard RMSD cutoff (Å) below which the motif is considered preserved.
    motif_sctm_hard_cutoff : float
        Hard scTM cutoff for the motif bonus indicator.
    device : str
        PyTorch device string ("cuda", "cpu", "cuda:0", …).
    seed : int
        Random seed.
    cache_dir : str
        Directory for caching downloaded models.
    output_dir : str
        Directory to write output sequences and metrics.
    proteinmpnn_path : Optional[str]
        Filesystem path to the ProteinMPNN repository (cloned from GitHub).
    esmfold_model : str
        HuggingFace model ID for ESMFold.
    max_sequence_length : int
        Maximum sequence length accepted by the pipeline.
    num_mcts_iterations : int
        Total number of MCTS selection-expansion-backup iterations (T).
    """

    # --- Task ---
    task: str = "inverse_folding"

    # --- Expert models ---
    experts: List[str] = field(
        default_factory=lambda: [
            "airkingbd/dplm_150m",
            "airkingbd/dplm_650m",
            "airkingbd/dplm_3b",
        ]
    )
    use_proteinmpnn: bool = False
    proteinmpnn_path: Optional[str] = None

    # --- MCTS hyperparameters (Table 10) ---
    num_rollouts: int = 3          # k_roll
    top_k_children: int = 3        # K
    max_depth: int = 5
    num_mcts_iterations: int = 100  # T
    exploration_constant: float = 1.414  # c_p = √2

    # --- Diffusion ---
    diffusion_steps: int = 150
    temperature: float = 1.0

    # --- Uncertainty / diversity bonuses ---
    w_ent: float = 1.0
    w_div: float = 1.0

    # --- Backup ---
    backup_rule: str = "max"  # "max" or "sum"

    # --- Progressive masking ---
    plddt_mask_threshold: float = 0.7  # residues below this get masked
    progressive_mask: bool = True
    plddt_threshold_min: float = 0.4

    # --- Reward weights ---
    # Folding: R_fold = α·TM + β·(1 - min(RMSD/10,1)) + γ·pLDDT
    reward_weights_folding: Tuple[float, float, float] = (0.60, 0.40, 0.00)
    # If pLDDT available: γ=0.05, renormalize so α+β+γ=1
    reward_folding_plddt_bonus: float = 0.05

    # Inverse folding: R_inv = 0.60·AAR + 0.35·scTM + 0.05·B
    reward_weights_inv: Tuple[float, float, float] = (0.60, 0.35, 0.05)

    # Motif scaffolding
    reward_weights_motif: Tuple[float, float, float, float] = (0.40, 0.30, 0.30, 0.20)
    motif_rmsd_hard_cutoff: float = 1.0   # Å
    motif_sctm_hard_cutoff: float = 0.8

    # --- Infrastructure ---
    device: str = "cuda"
    seed: int = 42
    cache_dir: str = "~/.cache/mctd_me"
    output_dir: str = "./outputs"
    esmfold_model: str = "facebook/esmfold_v1"
    max_sequence_length: int = 1024

    def __post_init__(self) -> None:
        """Validate config values."""
        valid_tasks = {"folding", "inverse_folding", "motif_scaffolding"}
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{self.task}'")
        valid_backup = {"max", "sum"}
        if self.backup_rule not in valid_backup:
            raise ValueError(f"backup_rule must be one of {valid_backup}")
        if not (0.0 < self.temperature):
            raise ValueError("temperature must be > 0")
        if not self.experts:
            raise ValueError("At least one expert model must be specified")

    @property
    def num_experts(self) -> int:
        """Total number of DPLM-2 experts."""
        return len(self.experts)
