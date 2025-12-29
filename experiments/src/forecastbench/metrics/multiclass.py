from __future__ import annotations

import numpy as np


def multiclass_brier_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Multi-class Brier score:
      E[ ||q - e_y||_2^2 ]  where e_y is one-hot.

    probs: (N, C) probabilities on simplex
    y: (N,) integer labels in {0,...,C-1}
    """
    probs = probs.astype(np.float64)
    y = y.astype(np.int64)
    N, C = probs.shape
    onehot = np.zeros((N, C), dtype=np.float64)
    onehot[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def multiclass_log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """
    Cross-entropy / log loss:
      E[ -log q_y ].
    """
    probs = probs.astype(np.float64)
    y = y.astype(np.int64)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    p = probs[np.arange(len(y)), y]
    return float(np.mean(-np.log(p)))


def multiclass_sce(probs: np.ndarray, p_true: np.ndarray) -> float:
    """
    Vector analogue of SCE:
      E[ ||q - p_true||_2^2 ].
    """
    probs = probs.astype(np.float64)
    p_true = p_true.astype(np.float64)
    return float(np.mean(np.sum((probs - p_true) ** 2, axis=1)))


def top_label_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    Classification-style ECE:
      bin by confidence max_j q_j, compare confidence vs accuracy.
    """
    probs = probs.astype(np.float64)
    y = y.astype(np.int64)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    N = len(y)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        ece += (mask.sum() / N) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)



