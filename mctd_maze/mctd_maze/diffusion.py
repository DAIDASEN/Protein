"""
Transformer-based trajectory diffusion model.

Architecture from paper §3 / Appendix B:
  - x0-parameterisation: predicts fully-denoised data from noisy tokens
  - Causal noise schedule: earlier sub-plans denoise faster
  - Training loss: MSE between predicted x0 and true x0
  - Expandable token sequences (no hard horizon limit)

Hidden=128, 12 layers, 4 heads, FFN=512  (Table 10)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def linear_beta_schedule(n_steps: int) -> torch.Tensor:
    """Linear beta schedule β_1 … β_T."""
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, n_steps)


def get_alphas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


# ---------------------------------------------------------------------------
# Sinusoidal positional / timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float()[:, None] * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Transformer backbone
# ---------------------------------------------------------------------------

class TrajectoryTransformer(nn.Module):
    """
    Transformer that processes a noisy trajectory token sequence and
    predicts the denoised x0 (x0-parameterisation).

    Input  : noisy trajectory [B, L, obs+act_dim] + timestep t [B]
    Output : denoised x0      [B, L, obs+act_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 12,
        n_heads: int = 4,
        ffn_dim: int = 512,
        n_diffusion_steps: int = 200,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.token_dim = obs_dim + act_dim

        # Input projection
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)

        # Timestep embedding
        self.time_emb = SinusoidalEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection (predict x0)
        self.output_proj = nn.Linear(hidden_dim, self.token_dim)

    def forward(
        self,
        x_noisy: torch.Tensor,   # [B, L, token_dim]
        t: torch.Tensor,          # [B]  integer timestep
        goal: Optional[torch.Tensor] = None,  # [B, obs_dim] optional goal token
    ) -> torch.Tensor:
        """Return predicted x0, same shape as x_noisy."""
        B, L, _ = x_noisy.shape

        # Token embedding + timestep conditioning
        h = self.input_proj(x_noisy)                     # [B, L, D]
        t_emb = self.time_proj(self.time_emb(t))         # [B, D]
        h = h + t_emb.unsqueeze(1)                       # broadcast over L

        # Optional goal conditioning: prepend goal as extra token
        if goal is not None:
            goal_tok = self.input_proj(
                F.pad(goal, (0, self.act_dim))            # pad action dims with 0
            ).unsqueeze(1)                                # [B, 1, D]
            h = torch.cat([goal_tok, h], dim=1)          # [B, L+1, D]

        h = self.transformer(h)

        # Remove goal token if prepended
        if goal is not None:
            h = h[:, 1:, :]

        return self.output_proj(h)                        # [B, L, token_dim]


# ---------------------------------------------------------------------------
# Diffusion wrapper with DDIM sampling
# ---------------------------------------------------------------------------

class TrajDiffusion(nn.Module):
    """
    Wraps TrajectoryTransformer with a forward/reverse diffusion process.

    Supports:
      - Training (noise + x0 prediction loss)
      - Full DDIM denoising
      - Partial denoising  (denoise from t_start → t_end, paper §3.1)
      - Jumpy DDIM         (skip every C steps, paper §3.3)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_diffusion_steps: int = 200,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.n_steps = n_diffusion_steps
        self.model = TrajectoryTransformer(obs_dim, act_dim, **model_kwargs)

        betas = linear_beta_schedule(n_diffusion_steps)
        alphas, alphas_cumprod = get_alphas(betas)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def loss(self, x0: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MSE x0-prediction loss over random timesteps."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        ac = self.alphas_cumprod[t].view(B, 1, 1)
        x_noisy = ac.sqrt() * x0 + (1 - ac).sqrt() * noise
        x0_pred = self.model(x_noisy, t, goal)
        return F.mse_loss(x0_pred, x0)

    # -----------------------------------------------------------------------
    # Sampling helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        goal: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM step from t → t_prev."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        x0_pred = self.model(x_t, t_tensor, goal)

        ac_t = self.alphas_cumprod[t]
        ac_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        sigma = eta * ((1 - ac_prev) / (1 - ac_t) * (1 - ac_t / ac_prev)).sqrt()
        pred_dir = (1 - ac_prev - sigma ** 2).sqrt() * (
            (x_t - ac_t.sqrt() * x0_pred) / (1 - ac_t).sqrt()
        )
        x_prev = ac_prev.sqrt() * x0_pred + pred_dir
        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_prev)
        return x_prev

    @torch.no_grad()
    def partial_denoise(
        self,
        x_t: torch.Tensor,
        t_start: int,
        n_steps: int,
        goal: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Denoise x_t for n_steps steps starting from t_start."""
        x = x_t
        step_size = t_start // max(n_steps, 1)
        times = list(range(t_start, 0, -step_size))[:n_steps]
        for i, t in enumerate(times):
            t_prev = times[i + 1] if i + 1 < len(times) else 0
            x = self.ddim_step(x, t, t_prev, goal, eta)
        return x

    @torch.no_grad()
    def jumpy_denoise(
        self,
        x_t: torch.Tensor,
        t_start: int,
        jumpy_interval: int = 10,
        goal: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Fast simulation: DDIM with skip-C steps (paper §3.3)."""
        x = x_t
        times = list(range(t_start, 0, -jumpy_interval))
        for i, t in enumerate(times):
            t_prev = times[i + 1] if i + 1 < len(times) else 0
            x = self.ddim_step(x, t, t_prev, goal, eta)
        return x
