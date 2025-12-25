from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from forecastbench.models.diffusion_core import DiffusionSchedule, sinusoidal_time_embedding


@dataclass(frozen=True)
class BundleLogitDiffusionSpec:
    """
    Joint diffusion forecaster for *bundles* of binary markets.

    We model a bundle of size B as a logit vector x0 ∈ R^B, and learn a denoiser
    ε_θ(x_t, t, cond) where cond contains per-market text embeddings.

    This is intentionally lightweight: the text encoder is external (e.g. Qwen3 as embedder),
    and we train only the diffusion denoiser.
    """

    bundle_size: int
    embed_dim: int  # per-market embedding dimension (e.g. Qwen hidden size)

    # internal model width
    model_dim: int = 256  # token/channel dim for the denoiser transformer
    time_dim: int = 64  # sinusoidal timestep embedding dim
    depth: int = 2  # transformer layers
    n_heads: int = 4
    dropout: float = 0.0

    schedule: DiffusionSchedule = field(default_factory=DiffusionSchedule)


class _TorchBundleDenoiser:
    def __init__(self, spec: BundleLogitDiffusionSpec):
        import torch.nn as nn

        self.spec = spec

        self.time_proj = nn.Sequential(
            nn.Linear(int(spec.time_dim), int(spec.model_dim)),
            nn.SiLU(),
            nn.Linear(int(spec.model_dim), int(spec.model_dim)),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(int(spec.embed_dim), int(spec.model_dim)),
            nn.LayerNorm(int(spec.model_dim)),
        )
        self.x_proj = nn.Linear(1, int(spec.model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(spec.model_dim),
            nhead=int(spec.n_heads),
            dim_feedforward=int(4 * spec.model_dim),
            dropout=float(spec.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(spec.depth))
        self.out = nn.Linear(int(spec.model_dim), 1)

    def parameters(self):
        import itertools

        return list(
            itertools.chain(
                self.time_proj.parameters(),
                self.cond_proj.parameters(),
                self.x_proj.parameters(),
                self.encoder.parameters(),
                self.out.parameters(),
            )
        )

    def to(self, device):
        self.time_proj.to(device)
        self.cond_proj.to(device)
        self.x_proj.to(device)
        self.encoder.to(device)
        self.out.to(device)
        return self

    def eval(self):
        self.time_proj.eval()
        self.cond_proj.eval()
        self.x_proj.eval()
        self.encoder.eval()
        self.out.eval()
        return self

    def train(self):
        self.time_proj.train()
        self.cond_proj.train()
        self.x_proj.train()
        self.encoder.train()
        self.out.train()
        return self

    def state_dict(self):
        return {
            "time_proj": self.time_proj.state_dict(),
            "cond_proj": self.cond_proj.state_dict(),
            "x_proj": self.x_proj.state_dict(),
            "encoder": self.encoder.state_dict(),
            "out": self.out.state_dict(),
        }

    def load_state_dict(self, sd):
        self.time_proj.load_state_dict(sd["time_proj"])
        self.cond_proj.load_state_dict(sd["cond_proj"])
        self.x_proj.load_state_dict(sd["x_proj"])
        self.encoder.load_state_dict(sd["encoder"])
        self.out.load_state_dict(sd["out"])

    def __call__(self, x_t, t_embed, cond, mask):
        """
        Args:
          x_t: (Bsz, B) float
          t_embed: (Bsz, time_dim) float
          cond: (Bsz, B, embed_dim) float
          mask: (Bsz, B) bool (True = valid token)
        Returns:
          eps_pred: (Bsz, B) float
        """
        import torch

        bsz, B = x_t.shape
        if B != int(self.spec.bundle_size):
            raise ValueError(f"x_t has B={B} but spec.bundle_size={self.spec.bundle_size}")

        # Project time to token dim and broadcast to all positions.
        t_h = self.time_proj(t_embed).unsqueeze(1)  # (Bsz, 1, D)

        # Project condition and x.
        c_h = self.cond_proj(cond)  # (Bsz, B, D)
        x_h = self.x_proj(x_t.unsqueeze(-1))  # (Bsz, B, D)

        tok = c_h + x_h + t_h  # (Bsz, B, D)

        # Transformer expects src_key_padding_mask=True for PAD positions.
        if mask is None:
            pad = None
        else:
            pad = ~mask.to(torch.bool)

        h = self.encoder(tok, src_key_padding_mask=pad)  # (Bsz, B, D)
        out = self.out(h).squeeze(-1)  # (Bsz, B)

        if mask is not None:
            out = out * mask.to(out.dtype)
        return out


class BundleLogitDiffusionForecaster:
    """
    Conditional diffusion forecaster producing a *joint* probability vector over a fixed-size bundle.
    """

    def __init__(self, spec: BundleLogitDiffusionSpec, *, device: str = "auto"):
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
        self.model = _TorchBundleDenoiser(spec).to(self.device)

    def _init_schedule(self):
        s = self.spec.schedule
        betas = np.linspace(s.beta_start, s.beta_end, s.T, dtype=np.float32)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas, axis=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

    def save(self, path: str) -> None:
        import torch

        payload = {
            "kind": "bundle_logit_diffusion",
            "spec": self.spec,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, *, device: str = "auto") -> "BundleLogitDiffusionForecaster":
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        spec = payload.get("spec")
        if spec is None:
            raise ValueError("Missing spec in checkpoint.")
        obj = BundleLogitDiffusionForecaster(spec, device=device)
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
        mask: np.ndarray,
        steps: int = 2000,
        batch_size: int = 128,
        lr: float = 2e-4,
        seed: int = 0,
        grad_clip: float = 1.0,
        log_every: int = 200,
    ) -> dict:
        """
        Minimal DDPM training loop (predict epsilon) for bundle logits.

        x0: (N, B) float32 logits
        cond: (N, B, embed_dim) float32
        mask: (N, B) bool
        """
        import torch

        rng = np.random.default_rng(int(seed))
        N, B = x0.shape
        if B != int(self.spec.bundle_size):
            raise ValueError(f"x0 has B={B} but spec.bundle_size={self.spec.bundle_size}")
        if cond.shape[:2] != (N, B):
            raise ValueError(f"cond shape {cond.shape} incompatible with x0 shape {x0.shape}")
        if mask.shape != (N, B):
            raise ValueError(f"mask shape {mask.shape} incompatible with x0 shape {x0.shape}")

        x0 = x0.astype(np.float32, copy=False)
        cond = cond.astype(np.float32, copy=False)
        mask = mask.astype(bool, copy=False)

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(lr))

        losses = []
        for step in range(int(steps)):
            idx = rng.integers(0, N, size=(int(batch_size),))
            t = rng.integers(0, int(self.spec.schedule.T), size=(int(batch_size),))
            eps = rng.standard_normal(size=(int(batch_size), B)).astype(np.float32)
            x_t = self._q_sample(x0[idx], t, eps)
            t_emb = sinusoidal_time_embedding(t, int(self.spec.time_dim))

            x_t_t = torch.from_numpy(x_t).to(self.device)
            eps_t = torch.from_numpy(eps).to(self.device)
            t_t = torch.from_numpy(t_emb).to(self.device)
            c_t = torch.from_numpy(cond[idx]).to(self.device)
            m_t = torch.from_numpy(mask[idx]).to(self.device)

            pred = self.model(x_t_t, t_t, c_t, m_t)
            err = (pred - eps_t) ** 2
            w = m_t.to(err.dtype)
            denom = torch.clamp(torch.sum(w), min=1.0)
            loss = torch.sum(err * w) / denom

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            if log_every and (step + 1) % int(log_every) == 0:
                print(f"[bundle_diffusion] step {step+1}/{steps}  loss={np.mean(losses[-int(log_every):]):.6f}")

        self.model.eval()
        tail = losses[-min(200, len(losses)) :] if losses else [float("nan")]
        return {"steps": int(steps), "final_loss": float(np.mean(tail))}

    def sample_x(
        self,
        *,
        cond: np.ndarray,
        mask: np.ndarray,
        n_steps: Optional[int] = None,
        seed: int = 0,
        eta: float = 0.0,
    ) -> np.ndarray:
        """
        DDIM-style sampler (eta=0 => deterministic).

        Returns:
          logits: (N, B) float32
        """
        import torch

        T_train = int(self.spec.schedule.T)
        T = T_train if n_steps is None else int(n_steps)
        if T > T_train:
            raise ValueError("n_steps cannot exceed training T")

        rng = np.random.default_rng(int(seed))
        bsz = int(cond.shape[0])
        B = int(self.spec.bundle_size)

        if cond.shape[1] != B or mask.shape != (bsz, B):
            raise ValueError(f"cond/mask shapes incompatible: cond={cond.shape} mask={mask.shape} B={B}")

        x = rng.standard_normal(size=(bsz, B)).astype(np.float32)
        x_t = torch.from_numpy(x).to(self.device)
        c_t = torch.from_numpy(cond.astype(np.float32)).to(self.device)
        m_t = torch.from_numpy(mask.astype(bool)).to(self.device)

        # Use a simple uniform stride if n_steps < T_train.
        t_idx = np.linspace(T_train - 1, 0, T, dtype=np.int64)

        for t in t_idx:
            t_arr = np.full((bsz,), int(t), dtype=np.int64)
            t_emb = sinusoidal_time_embedding(t_arr, int(self.spec.time_dim))
            t_t = torch.from_numpy(t_emb).to(self.device)

            with torch.no_grad():
                eps = self.model(x_t, t_t, c_t, m_t)

            ab_t = float(self.alpha_bar[t])
            x0 = (x_t - (1.0 - ab_t) ** 0.5 * eps) / (ab_t**0.5)

            if t == 0:
                x_t = x0
                continue

            ab_prev = float(self.alpha_bar[t - 1])
            sigma = float(eta) * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)) ** 0.5
            noise = torch.randn_like(x_t) if sigma > 0 else 0.0
            x_t = (
                (ab_prev**0.5) * x0
                + ((1 - ab_prev - sigma**2) ** 0.5) * eps
                + sigma * noise
            )

        return x_t.detach().cpu().numpy().astype(np.float32)


