from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecastbench.models.diffusion_core import (
    ContinuousDiffusionForecaster,
    DiffusionModelSpec,
    DiffusionSchedule,
)
from forecastbench.utils.logits import logit_to_prob


@dataclass(frozen=True)
class LogitDiffusionSpec(DiffusionModelSpec):
    """
    Diffusion model spec for binary logit targets (out_dim=1 by default).
    """

    out_dim: int = 1
    cond_dim: int = 16


class LogitDiffusionForecaster(ContinuousDiffusionForecaster):
    """
    Conditional diffusion forecaster that returns probabilities via sigmoid(logits).
    """

    def predict_proba_from_cond(self, cond: np.ndarray, *, seed: int = 0) -> np.ndarray:
        logits = self.sample_x(cond=cond, seed=seed)
        return logit_to_prob(logits).astype(np.float32)


