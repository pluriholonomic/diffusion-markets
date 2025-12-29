from __future__ import annotations

import numpy as np


def best_bounded_trader_profit(
    p_true: np.ndarray, q: np.ndarray, *, B: float = 1.0, transaction_cost: float = 0.0
) -> float:
    """
    Best expected profit of a bounded *static* trader with linear payoff:
      sup_{b in [-B,B]} E[ b (Y - q) ] = B * E[ |p - q| ]

    With transaction cost c (per unit position):
      profit = B * E[ (|p - q| - c)_+ ].
    """
    p_true = p_true.astype(np.float64)
    q = q.astype(np.float64)
    gap = np.abs(p_true - q)
    if transaction_cost > 0:
        gap = np.maximum(gap - transaction_cost, 0.0)
    return float(B * np.mean(gap))



