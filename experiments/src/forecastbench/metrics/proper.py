from __future__ import annotations

import numpy as np


def _clip_prob(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(q, eps, 1.0 - eps)


def brier_loss(q: np.ndarray, y: np.ndarray) -> float:
    q = q.astype(np.float64)
    y = y.astype(np.float64)
    return float(np.mean((q - y) ** 2))


def log_loss(q: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    q = _clip_prob(q.astype(np.float64), eps=eps)
    y = y.astype(np.float64)
    return float(np.mean(-(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def squared_calibration_error(q: np.ndarray, p_true: np.ndarray) -> float:
    q = q.astype(np.float64)
    p_true = p_true.astype(np.float64)
    return float(np.mean((q - p_true) ** 2))




