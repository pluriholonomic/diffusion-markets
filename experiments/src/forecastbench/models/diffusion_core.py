from __future__ import annotations


from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DiffusionSchedule:
    T: int = 64
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass(frozen=True)
class DiffusionModelSpec:
    """
    Generic conditional diffusion model spec for continuous targets x0 ∈ R^{out_dim}.
    """

    out_dim: int
    cond_dim: int
    time_dim: int = 64
    hidden_dim: int = 256
    depth: int = 3
    schedule: DiffusionSchedule = field(default_factory=DiffusionSchedule)


class _ResidualBlock:
    """A residual block with LayerNorm and SiLU activation."""
    def __init__(self, dim: int, dropout: float = 0.1):
        import torch.nn as nn
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def to(self, device):
        self.norm.to(device)
        self.linear1.to(device)
        self.linear2.to(device)
        self.dropout.to(device)
        return self
    
    def train(self):
        self.norm.train()
        self.linear1.train()
        self.linear2.train()
        self.dropout.train()
    
    def eval(self):
        self.norm.eval()
        self.linear1.eval()
        self.linear2.eval()
        self.dropout.eval()
    
    def parameters(self):
        return list(self.norm.parameters()) + list(self.linear1.parameters()) + list(self.linear2.parameters())
    
    def state_dict(self):
        return {"norm": self.norm.state_dict(), "linear1": self.linear1.state_dict(), "linear2": self.linear2.state_dict()}
    
    def load_state_dict(self, sd):
        self.norm.load_state_dict(sd["norm"])
        self.linear1.load_state_dict(sd["linear1"])
        self.linear2.load_state_dict(sd["linear2"])
    
    def __call__(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class _TorchDenoiserMLP:
    def __init__(self, spec: DiffusionModelSpec, dropout: float = 0.1):
        import torch.nn as nn

        self.spec = spec
        self.dropout_rate = dropout
        
        # Time embedding with LayerNorm
        self.time_embed = nn.Sequential(
            nn.Linear(spec.time_dim, spec.hidden_dim),
            nn.SiLU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.LayerNorm(spec.hidden_dim),
        )

        # Input projection
        in_dim = spec.out_dim + spec.hidden_dim + spec.cond_dim
        self.input_proj = nn.Linear(in_dim, spec.hidden_dim)
        self.input_norm = nn.LayerNorm(spec.hidden_dim)
        
        # Residual blocks
        self.blocks = [_ResidualBlock(spec.hidden_dim, dropout) for _ in range(spec.depth)]
        
        # Output projection
        self.output_norm = nn.LayerNorm(spec.hidden_dim)
        self.output_proj = nn.Linear(spec.hidden_dim, spec.out_dim)
        
        # For backward compatibility, create a dummy net
        h = spec.hidden_dim
        layers = []
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.SiLU())
        for _ in range(spec.depth - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(h, spec.out_dim))
        self.net = nn.Sequential(*layers)  # kept for state_dict compat but unused

    def parameters(self):
        params = list(self.time_embed.parameters())
        params += list(self.input_proj.parameters())
        params += list(self.input_norm.parameters())
        for block in self.blocks:
            params += block.parameters()
        params += list(self.output_norm.parameters())
        params += list(self.output_proj.parameters())
        return params

    def to(self, device):
        self.time_embed.to(device)
        self.input_proj.to(device)
        self.input_norm.to(device)
        for block in self.blocks:
            block.to(device)
        self.output_norm.to(device)
        self.output_proj.to(device)
        self.net.to(device)
        return self

    def eval(self):
        self.time_embed.eval()
        self.input_proj.eval()
        self.input_norm.eval()
        for block in self.blocks:
            block.eval()
        self.output_norm.eval()
        self.output_proj.eval()
        self.net.eval()
        return self

    def train(self):
        self.time_embed.train()
        self.input_proj.train()
        self.input_norm.train()
        for block in self.blocks:
            block.train()
        self.output_norm.train()
        self.output_proj.train()
        self.net.train()
        return self

    def state_dict(self):
        return {
            "time_embed": self.time_embed.state_dict(),
            "input_proj": self.input_proj.state_dict(),
            "input_norm": self.input_norm.state_dict(),
            "blocks": [b.state_dict() for b in self.blocks],
            "output_norm": self.output_norm.state_dict(),
            "output_proj": self.output_proj.state_dict(),
        }

    def load_state_dict(self, sd):
        self.time_embed.load_state_dict(sd["time_embed"])
        # Handle legacy checkpoints without new layers
        if "input_proj" in sd:
            self.input_proj.load_state_dict(sd["input_proj"])
            self.input_norm.load_state_dict(sd["input_norm"])
            for i, b in enumerate(self.blocks):
                b.load_state_dict(sd["blocks"][i])
            self.output_norm.load_state_dict(sd["output_norm"])
            self.output_proj.load_state_dict(sd["output_proj"])
        else:
            # Legacy: load old net weights
            self.net.load_state_dict(sd["net"])

    def __call__(self, x_t, t_embed, cond):
        import torch

        h_t = self.time_embed(t_embed)
        inp = torch.cat([x_t, h_t, cond], dim=-1)
        
        # New architecture with residual blocks
        x = self.input_proj(inp)
        x = self.input_norm(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        return self.output_proj(x)


def sinusoidal_time_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """
    Standard sinusoidal embedding for integer timesteps t in [0, T).
    Returns float32 array of shape (batch, dim).
    """
    t = t.astype(np.float32).reshape(-1, 1)
    half = dim // 2
    freqs = np.exp(-np.log(10_000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1))
    ang = t * freqs.reshape(1, -1)
    emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    if dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)))
    return emb.astype(np.float32)


class ContinuousDiffusionForecaster:
    """
    A tiny conditional DDPM/DDIM forecaster for continuous targets x0 ∈ R^{out_dim}.
    """

    def __init__(self, spec: DiffusionModelSpec, *, device: str = "auto"):
        import torch

        self.spec = spec
        self.device = device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self._init_schedule()
        self.model = _TorchDenoiserMLP(spec).to(self.device)

    def _init_schedule(self):
        s = self.spec.schedule
        betas = np.linspace(s.beta_start, s.beta_end, s.T, dtype=np.float32)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas, axis=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

    def save(self, path: str):
        import torch

        payload = {"spec": self.spec, "state_dict": self.model.state_dict()}
        torch.save(payload, path)

    @staticmethod
    def load(path: str, *, device: str = "auto") -> "ContinuousDiffusionForecaster":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec = payload["spec"]
        obj = ContinuousDiffusionForecaster(spec, device=device)
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        return obj

    def _q_sample(self, x0: np.ndarray, t: np.ndarray, noise: np.ndarray) -> np.ndarray:
        # x_t = sqrt(alpha_bar[t]) x0 + sqrt(1-alpha_bar[t]) eps
        a = self.alpha_bar[t].reshape(-1, 1)
        return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise

    def train_mse_eps(
        self,
        *,
        x0: np.ndarray,
        cond: np.ndarray,
        steps: int = 2000,
        batch_size: int = 256,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        seed: int = 0,
        grad_clip: float = 1.0,
        log_every: int = 200,
    ) -> dict:
        """
        Improved DDPM training loop with cosine LR schedule and weight decay.

        x0: (N, out_dim) in R
        cond: (N, cond_dim) float
        """
        import torch
        import math

        rng = np.random.default_rng(seed)
        N = x0.shape[0]
        x0 = x0.astype(np.float32)
        cond = cond.astype(np.float32)

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        losses = []
        for step in range(steps):
            idx = rng.integers(0, N, size=(batch_size,))
            t = rng.integers(0, self.spec.schedule.T, size=(batch_size,))
            eps = rng.standard_normal(size=(batch_size, self.spec.out_dim)).astype(np.float32)
            x_t = self._q_sample(x0[idx], t, eps)
            t_emb = sinusoidal_time_embedding(t, self.spec.time_dim)

            x_t_t = torch.from_numpy(x_t).to(self.device)
            eps_t = torch.from_numpy(eps).to(self.device)
            t_t = torch.from_numpy(t_emb).to(self.device)
            c_t = torch.from_numpy(cond[idx]).to(self.device)

            pred = self.model(x_t_t, t_t, c_t)
            loss = torch.mean((pred - eps_t) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            opt.step()
            scheduler.step()

            losses.append(float(loss.detach().cpu().item()))
            if log_every and (step + 1) % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[diffusion] step {step+1}/{steps}  loss={np.mean(losses[-log_every:]):.6f}  lr={current_lr:.2e}"
                )

        self.model.eval()
        tail = losses[-min(200, len(losses)) :] if losses else [float("nan")]
        return {"steps": int(steps), "final_loss": float(np.mean(tail))}

    def sample_x(
        self,
        *,
        cond: np.ndarray,
        n_steps: Optional[int] = None,
        seed: int = 0,
        eta: float = 0.0,
    ) -> np.ndarray:
        """
        DDIM-style sampler (eta=0 => deterministic).
        Returns samples in R^{out_dim}.
        """
        import torch

        T = self.spec.schedule.T if n_steps is None else int(n_steps)
        if T > self.spec.schedule.T:
            raise ValueError("n_steps cannot exceed training T")

        rng = np.random.default_rng(seed)
        bsz = cond.shape[0]

        x = rng.standard_normal(size=(bsz, self.spec.out_dim)).astype(np.float32)
        x_t = torch.from_numpy(x).to(self.device)
        c_t = torch.from_numpy(cond.astype(np.float32)).to(self.device)

        # Use a simple uniform stride if n_steps < T_train.
        t_idx = np.linspace(self.spec.schedule.T - 1, 0, T, dtype=np.int64)

        for t in t_idx:
            t_arr = np.full((bsz,), int(t), dtype=np.int64)
            t_emb = sinusoidal_time_embedding(t_arr, self.spec.time_dim)
            t_t = torch.from_numpy(t_emb).to(self.device)

            with torch.no_grad():
                eps = self.model(x_t, t_t, c_t)

            ab_t = float(self.alpha_bar[t])
            # Predict x0
            x0 = (x_t - (1.0 - ab_t) ** 0.5 * eps) / (ab_t**0.5)

            if t == 0:
                x_t = x0
                continue

            ab_prev = float(self.alpha_bar[t - 1])
            # DDIM update
            sigma = eta * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)) ** 0.5
            noise = torch.randn_like(x_t) if sigma > 0 else 0.0
            x_t = (
                (ab_prev**0.5) * x0
                + ((1 - ab_prev - sigma**2) ** 0.5) * eps
                + sigma * noise
            )

        return x_t.detach().cpu().numpy().astype(np.float32)


