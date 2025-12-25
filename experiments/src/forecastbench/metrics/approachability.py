from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def distance_to_box_linf(v: np.ndarray, *, eps: float) -> float:
    """
    Distance in ℓ∞ to the box [-eps, eps]^M.

      d∞(v, [-eps,eps]^M) = max_i (|v_i| - eps)_+.
    """
    v = np.asarray(v, dtype=np.float64)
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if v.size == 0:
        return 0.0
    return float(np.maximum(np.abs(v) - float(eps), 0.0).max())


def distance_to_upper_orthant_linf(v: np.ndarray, *, ub: float = 0.0) -> float:
    """
    Distance in ℓ∞ to the (buffered) upper orthant {x : x <= ub} coordinatewise.

      d∞(v, {x: x<=ub}) = max_i (v_i - ub)_+.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return 0.0
    return float(np.maximum(v - float(ub), 0.0).max())


@dataclass(frozen=True)
class AppErrCurve:
    """
    A sparse-sampled approachability curve.
    """

    t: np.ndarray  # (m,) int64, cumulative step count
    app_err: np.ndarray  # (m,) float64

    def to_jsonable(self) -> Dict[str, Any]:
        return {"t": self.t.tolist(), "app_err": self.app_err.tolist()}


def app_err_curve_upper_orthant_dense(
    *,
    v: np.ndarray,
    ub: float = 0.0,
    every: int = 1,
) -> AppErrCurve:
    """
    Approachability curve for a dense violation sequence v_t in R^M with target {x: x<=ub}.

    Uses:
      AppErr_t = d∞( (1/t)∑_{s<=t} v_s, {x: x<=ub} ) = max_i (mean_i - ub)_+.

    Args:
      v: (T,M) array
      ub: orthant upper bound (0 for standard no-arbitrage inequalities)
      every: record every N steps (>=1)
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("v must be a 2D array of shape (T,M)")
    if every <= 0:
        raise ValueError("every must be >= 1")

    T = int(v.shape[0])
    cum = np.zeros((int(v.shape[1]),), dtype=np.float64)

    ts: List[int] = []
    errs: List[float] = []

    for t in range(T):
        cum += v[t]
        step = t + 1
        if (step % every) == 0 or t == (T - 1):
            mean = cum / float(step)
            errs.append(distance_to_upper_orthant_linf(mean, ub=float(ub)))
            ts.append(step)

    return AppErrCurve(t=np.asarray(ts, dtype=np.int64), app_err=np.asarray(errs, dtype=np.float64))


def app_err_curve_single_coordinate(
    *,
    coord: np.ndarray,
    value: np.ndarray,
    M: int,
    eps: float,
    every: int = 1,
) -> AppErrCurve:
    """
    Compute AppErr curve for a sequence where each round updates exactly one coordinate:

      g_t(i) = value_t if i == coord_t else 0.

    Args:
      coord: (T,) int64 coordinate indices in [0, M)
      value: (T,) float values to add to that coordinate
      M: number of coordinates
      eps: box half-width for target [-eps,eps]^M
      every: record curve every `every` steps (>=1)
    """
    coord = np.asarray(coord, dtype=np.int64)
    value = np.asarray(value, dtype=np.float64)
    if coord.shape != value.shape:
        raise ValueError("coord and value must have the same shape")
    if M <= 0:
        raise ValueError("M must be positive")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if every <= 0:
        raise ValueError("every must be >= 1")

    T = int(coord.shape[0])
    cum = np.zeros((int(M),), dtype=np.float64)

    ts: List[int] = []
    errs: List[float] = []

    for t in range(T):
        i = int(coord[t])
        if i < 0 or i >= M:
            raise ValueError(f"coord[{t}]={i} out of bounds for M={M}")
        cum[i] += float(value[t])

        step = t + 1
        if (step % every) == 0 or t == (T - 1):
            mean = cum / float(step)
            errs.append(distance_to_box_linf(mean, eps=float(eps)))
            ts.append(step)

    return AppErrCurve(t=np.asarray(ts, dtype=np.int64), app_err=np.asarray(errs, dtype=np.float64))


def summarize_top_box_violations(
    *,
    mean_vec: np.ndarray,
    eps: float,
    topk: int = 10,
) -> List[Tuple[int, float, float]]:
    """
    Return top-k coordinates by box-violation magnitude.

    Returns list of (coord_idx, mean_value, violation) where
      violation = (|mean_value| - eps)_+.
    """
    mean_vec = np.asarray(mean_vec, dtype=np.float64)
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if topk <= 0:
        return []
    if mean_vec.size == 0:
        return []

    viol = np.maximum(np.abs(mean_vec) - float(eps), 0.0)
    if not np.any(viol > 0):
        return []

    k = min(int(topk), int(mean_vec.size))
    # partial sort, then sort those k
    idx = np.argpartition(-viol, kth=k - 1)[:k]
    idx = idx[np.argsort(-viol[idx])]
    return [(int(i), float(mean_vec[i]), float(viol[i])) for i in idx if viol[i] > 0]


