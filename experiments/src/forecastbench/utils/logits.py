from __future__ import annotations

import numpy as np


def clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Clip probabilities away from {0,1} for numerical stability.
    """
    return np.clip(p, eps, 1.0 - eps)


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Binary logit transform: log(p/(1-p)).
    """
    p = clip_probs(np.asarray(p, dtype=np.float64), eps=eps)
    return np.log(p / (1.0 - p))


def logit_to_prob(u: np.ndarray) -> np.ndarray:
    """
    Binary inverse-logit: sigmoid(u).
    """
    u = np.asarray(u, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-u))


def simplex_to_alr(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Additive log-ratio (ALR) transform mapping the simplex Δ^{n-1} to R^{n-1}:
      alr_i = log(p_i / p_n) for i=1..n-1

    Input:
      p: (..., n) with p>=0 and sum(p)=1 (approximately).
    Output:
      alr: (..., n-1)

    This is useful for diffusion over multi-outcome markets where the target is a simplex.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] < 2:
        raise ValueError("simplex_to_alr requires last dimension n>=2")
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    denom = p[..., [-1]]
    return np.log(p[..., :-1] / denom)


def alr_to_simplex(alr: np.ndarray) -> np.ndarray:
    """
    Inverse ALR transform:
      p_i = exp(alr_i) / (1 + sum_j exp(alr_j)), i=1..n-1
      p_n = 1 / (1 + sum_j exp(alr_j))

    Input:
      alr: (..., n-1)
    Output:
      p: (..., n) on the simplex.
    """
    alr = np.asarray(alr, dtype=np.float64)
    expu = np.exp(alr)
    denom = 1.0 + np.sum(expu, axis=-1, keepdims=True)
    p_front = expu / denom
    p_last = 1.0 / denom
    return np.concatenate([p_front, p_last], axis=-1)




