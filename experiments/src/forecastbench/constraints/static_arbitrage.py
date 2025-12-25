from __future__ import annotations

from typing import Optional

import numpy as np


def frechet_violations(
    *,
    p_a: float,
    p_b: float,
    p_ab: float,
    include_box: bool = True,
) -> np.ndarray:
    """
    Return a vector v(p) of Fréchet (A,B, A∧B) no-arbitrage inequality residuals such that
    no-arbitrage is encoded by v(p) <= 0 coordinatewise.

    Contracts:
      - S_A pays A
      - S_B pays B
      - S_AB pays (A ∧ B)

    Inequalities:
      - p_ab >= 0
      - p_ab <= p_a
      - p_ab <= p_b
      - p_ab >= p_a + p_b - 1

    If include_box=True, we also include 0<=p<=1 box constraints for each coordinate.
    """
    p_a = float(p_a)
    p_b = float(p_b)
    p_ab = float(p_ab)

    vals = [
        -p_ab,  # p_ab >= 0
        p_ab - p_a,  # p_ab <= p_a
        p_ab - p_b,  # p_ab <= p_b
        (p_a + p_b - 1.0) - p_ab,  # p_ab >= p_a + p_b - 1
    ]
    if include_box:
        vals += [
            -p_a,
            p_a - 1.0,
            -p_b,
            p_b - 1.0,
            -p_ab,
            p_ab - 1.0,
        ]
    return np.asarray(vals, dtype=np.float64)


def implication_violations(*, p_a: float, p_b: float, include_box: bool = True) -> np.ndarray:
    """
    Inequality residuals for implication A => B:
      p_a <= p_b  <=>  p_a - p_b <= 0.

    If include_box=True, also include 0<=p<=1 box constraints for each coordinate.
    """
    p_a = float(p_a)
    p_b = float(p_b)
    vals = [p_a - p_b]
    if include_box:
        vals += [-p_a, p_a - 1.0, -p_b, p_b - 1.0]
    return np.asarray(vals, dtype=np.float64)


def mutual_exclusion_violation(*, p: np.ndarray) -> np.ndarray:
    """
    Mutual-exclusion inequality for a set of events:
      sum_i p_i <= 1  <=>  (sum_i p_i - 1) <= 0.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return np.asarray([float(np.sum(p) - 1.0)], dtype=np.float64)


def simplex_sum_to_one_violations(*, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Two-sided equality residuals for sum-to-one on a simplex:
      sum_i p_i == 1  encoded as two inequalities:
        sum_i p_i - 1 <= 0
        1 - sum_i p_i <= 0
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    s = float(np.sum(p))
    return np.asarray([s - 1.0, 1.0 - s], dtype=np.float64)


