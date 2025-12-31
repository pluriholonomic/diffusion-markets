from __future__ import annotations

import numpy as np


def expected_calibration_error(q: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """
    Standard ECE with equal-width bins on [0,1].
    """
    q = q.astype(np.float64)
    y = y.astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize gives 1..n_bins; put q=1.0 into last bin
    idx = np.digitize(q, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    n = len(q)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        qb = float(np.mean(q[mask]))
        yb = float(np.mean(y[mask]))
        ece += (mask.sum() / n) * abs(yb - qb)
    return float(ece)




