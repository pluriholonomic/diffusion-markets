from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from forecastbench.models.diffusion_core import (
    ContinuousDiffusionForecaster,
    DiffusionModelSpec,
    DiffusionSchedule,
)
from forecastbench.utils.logits import alr_to_simplex, simplex_to_alr


@dataclass(frozen=True)
class SimplexDiffusionSpec:
    """
    Diffusion spec for n-outcome categorical markets (simplex targets).

    We represent probabilities p ∈ Δ^{n-1} using an ALR transform:
      u = alr(p) ∈ R^{n-1}
    and run diffusion in u-space.
    """

    n_outcomes: int
    cond_dim: int
    time_dim: int = 64
    hidden_dim: int = 256
    depth: int = 3
    schedule: DiffusionSchedule = field(default_factory=DiffusionSchedule)

    def to_model_spec(self) -> DiffusionModelSpec:
        if self.n_outcomes < 2:
            raise ValueError("n_outcomes must be >= 2")
        return DiffusionModelSpec(
            out_dim=self.n_outcomes - 1,
            cond_dim=self.cond_dim,
            time_dim=self.time_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            schedule=self.schedule,
        )


class SimplexALRDiffusionForecaster:
    """
    Conditional diffusion forecaster for simplex outputs.
    """

    def __init__(self, spec: SimplexDiffusionSpec, *, device: str = "auto"):
        self.spec = spec
        self.core = ContinuousDiffusionForecaster(spec.to_model_spec(), device=device)

    @property
    def device(self) -> str:
        return self.core.device

    def save(self, path: str):
        import torch

        payload = {"spec": self.spec, "core_state_dict": self.core.model.state_dict()}
        torch.save(payload, path)

    @staticmethod
    def load(path: str, *, device: str = "auto") -> "SimplexALRDiffusionForecaster":
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        spec = payload["spec"]
        obj = SimplexALRDiffusionForecaster(spec, device=device)
        obj.core.model.load_state_dict(payload["core_state_dict"])
        obj.core.model.eval()
        return obj

    def train_mse_eps(
        self,
        *,
        p0: np.ndarray,
        cond: np.ndarray,
        steps: int = 2000,
        batch_size: int = 256,
        lr: float = 2e-4,
        seed: int = 0,
        grad_clip: float = 1.0,
        log_every: int = 200,
    ) -> dict:
        """
        Train diffusion in ALR space.

        p0: (N, n_outcomes) on the simplex
        cond: (N, cond_dim)
        """
        u0 = simplex_to_alr(p0).astype(np.float32)
        return self.core.train_mse_eps(
            x0=u0,
            cond=cond,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            grad_clip=grad_clip,
            log_every=log_every,
        )

    def sample_alr(
        self,
        *,
        cond: np.ndarray,
        n_steps: Optional[int] = None,
        seed: int = 0,
        eta: float = 0.0,
    ) -> np.ndarray:
        return self.core.sample_x(cond=cond, n_steps=n_steps, seed=seed, eta=eta)

    def predict_simplex_from_cond(
        self, cond: np.ndarray, *, n_steps: Optional[int] = None, seed: int = 0, eta: float = 0.0
    ) -> np.ndarray:
        u = self.sample_alr(cond=cond, n_steps=n_steps, seed=seed, eta=eta)
        p = alr_to_simplex(u)
        return p.astype(np.float32)


