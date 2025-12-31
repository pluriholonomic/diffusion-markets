"""
Legacy-compatible C_t loader for checkpoints saved with simpler architecture.

This handles checkpoints where the model was saved with:
- Simple Sequential time_embed (no LayerNorm)
- Simple Sequential net (no residual blocks)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DiffusionSchedule:
    T: int = 64
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass(frozen=True)
class LegacyLogitSpec:
    """Spec matching the saved checkpoint structure."""
    out_dim: int = 1
    cond_dim: int = 384
    time_dim: int = 64
    hidden_dim: int = 256
    depth: int = 3
    schedule: DiffusionSchedule = field(default_factory=DiffusionSchedule)


class LegacyDenoiserMLP(nn.Module):
    """Simple MLP matching the saved checkpoint structure."""
    
    def __init__(self, spec: LegacyLogitSpec):
        super().__init__()
        self.spec = spec
        
        # Time embedding: Linear -> SiLU -> Linear (no LayerNorm)
        self.time_embed = nn.Sequential(
            nn.Linear(spec.time_dim, spec.hidden_dim),
            nn.SiLU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
        )
        
        # Net: input_dim -> hidden -> ... -> out
        in_dim = spec.out_dim + spec.hidden_dim + spec.cond_dim
        h = spec.hidden_dim
        
        layers = []
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.SiLU())
        for _ in range(spec.depth - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(h, spec.out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h_t = self.time_embed(t_embed)
        inp = torch.cat([x_t, h_t, cond], dim=-1)
        return self.net(inp)


def sinusoidal_time_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """Standard sinusoidal embedding for timesteps."""
    t = t.astype(np.float32).reshape(-1, 1)
    half = dim // 2
    freqs = np.exp(-np.log(10_000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1))
    ang = t * freqs.reshape(1, -1)
    emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    if dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)))
    return emb.astype(np.float32)


class LegacyDiffusionForecaster:
    """
    Diffusion forecaster compatible with legacy checkpoints.
    """
    
    def __init__(self, spec: LegacyLogitSpec, device: str = "cpu"):
        self.spec = spec
        self.device = device
        self._init_schedule()
        self.model = LegacyDenoiserMLP(spec).to(device)
    
    def _init_schedule(self):
        s = self.spec.schedule
        betas = np.linspace(s.beta_start, s.beta_end, s.T, dtype=np.float32)
        alphas = 1.0 - betas
        self.alpha_bar = np.cumprod(alphas, axis=0)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LegacyDiffusionForecaster":
        """Load a legacy checkpoint."""
        from forecastbench.models.diffusion_logit import LogitDiffusionSpec
        torch.serialization.add_safe_globals([LogitDiffusionSpec])
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        old_spec = checkpoint['spec']
        
        # Convert to legacy spec
        spec = LegacyLogitSpec(
            out_dim=old_spec.out_dim,
            cond_dim=old_spec.cond_dim,
            time_dim=old_spec.time_dim,
            hidden_dim=old_spec.hidden_dim,
            depth=old_spec.depth,
            schedule=DiffusionSchedule(
                T=old_spec.schedule.T,
                beta_start=old_spec.schedule.beta_start,
                beta_end=old_spec.schedule.beta_end,
            ),
        )
        
        obj = cls(spec, device=device)
        
        # Load state dict
        state = checkpoint['state_dict']
        obj.model.time_embed.load_state_dict(state['time_embed'])
        obj.model.net.load_state_dict(state['net'])
        obj.model.eval()
        
        return obj
    
    def sample_x(
        self,
        *,
        cond: np.ndarray,
        n_steps: int = 32,
        seed: int = 0,
        eta: float = 0.0,
    ) -> np.ndarray:
        """
        DDIM-style sampler.
        
        Args:
            cond: Conditioning input (batch, cond_dim)
            n_steps: Number of DDIM steps
            seed: Random seed
            eta: DDIM stochasticity (0 = deterministic)
            
        Returns:
            Samples (batch, out_dim) as logits
        """
        T = min(n_steps, self.spec.schedule.T)
        rng = np.random.default_rng(seed)
        bsz = cond.shape[0]
        
        # Start from noise
        x = rng.standard_normal(size=(bsz, self.spec.out_dim)).astype(np.float32)
        x_t = torch.from_numpy(x).to(self.device)
        c_t = torch.from_numpy(cond.astype(np.float32)).to(self.device)
        
        # Uniform stride through timesteps
        t_idx = np.linspace(self.spec.schedule.T - 1, 0, T, dtype=np.int64)
        
        for t in t_idx:
            t_arr = np.full((bsz,), int(t), dtype=np.int64)
            t_emb = sinusoidal_time_embedding(t_arr, self.spec.time_dim)
            t_t = torch.from_numpy(t_emb).to(self.device)
            
            with torch.no_grad():
                eps = self.model(x_t, t_t, c_t)
            
            ab_t = float(self.alpha_bar[t])
            # Predict x0
            x0 = (x_t - (1.0 - ab_t) ** 0.5 * eps) / (ab_t ** 0.5)
            
            if t == 0:
                x_t = x0
                continue
            
            ab_prev = float(self.alpha_bar[t - 1])
            # DDIM update
            sigma = eta * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)) ** 0.5
            noise = torch.randn_like(x_t) if sigma > 0 else 0.0
            x_t = (
                (ab_prev ** 0.5) * x0
                + ((1 - ab_prev - sigma ** 2) ** 0.5) * eps
                + sigma * noise
            )
        
        return x_t.detach().cpu().numpy().astype(np.float32)
    
    def sample_proba(
        self,
        cond: np.ndarray,
        n_samples: int = 64,
        n_steps: int = 32,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Sample multiple probability predictions.
        
        Args:
            cond: Conditioning input (batch, cond_dim)
            n_samples: Number of samples to generate
            n_steps: DDIM steps per sample
            seed: Base random seed
            
        Returns:
            Samples (n_samples, batch) as probabilities in [0, 1]
        """
        samples = []
        for i in range(n_samples):
            logits = self.sample_x(cond=cond, n_steps=n_steps, seed=seed + i * 1000)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits.flatten(), -20, 20)))
            samples.append(probs)
        return np.array(samples)



