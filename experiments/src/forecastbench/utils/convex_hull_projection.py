from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def project_to_simplex(v: np.ndarray, *, z: float = 1.0) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex:
      Δ(z) = {w : w_i >= 0, sum_i w_i = z}.

    Implements the O(n log n) algorithm from:
      Duchi, Shalev-Shwartz, Singer, Chandra (2008).
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return v
    z = float(z)
    if z <= 0:
        raise ValueError("z must be > 0")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u - (cssv - z) / (np.arange(len(u)) + 1) > 0)[0]
    if rho.size == 0:
        # All projected to uniform.
        return np.full_like(v, z / float(len(v)), dtype=np.float64)
    rho = int(rho[-1])
    theta = (cssv[rho] - z) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    # numerical fixup to ensure exact sum in float64
    s = float(w.sum())
    if s != 0:
        w *= z / s
    return w.astype(np.float64)


@dataclass(frozen=True)
class ConvexHullProjectionResult:
    proj: np.ndarray  # (d,)
    weights: np.ndarray  # (n,)
    dist_l2: float
    dist_linf: float
    obj: float
    n_iter: int
    converged: bool


def _lipschitz_bound(points: np.ndarray) -> float:
    """
    Upper bound on Lipschitz constant of grad_w 0.5||P^T w - x||^2.

    With A = P^T, L = ||A||_2^2 = ||P||_2^2.
    We compute ||P||_2 via SVD for stability; intended for small n (MC samples).
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("points must be 2D")
    if P.shape[0] == 0:
        return 1.0
    # numpy SVD is fine for n<=~256, d<=~128 typical here
    s = np.linalg.svd(P, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return 1.0
    return float(np.max(s) ** 2)


def project_point_to_convex_hull(
    x: np.ndarray,
    points: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    step: Optional[float] = None,
    init: Optional[np.ndarray] = None,
) -> ConvexHullProjectionResult:
    """
    Project x onto conv(points) under Euclidean distance.

    Solves:
      min_{w in Δ} 0.5 || P^T w - x ||^2
    where P is (n, d).

    Uses projected gradient descent on w with simplex projection.
    Intended for small n (MC samples).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("points must be a 2D array (n,d)")
    n, d = P.shape
    if x.shape != (d,):
        raise ValueError(f"x must have shape ({d},) but got {x.shape}")
    if n <= 0:
        raise ValueError("points must have n>=1")
    if n == 1:
        proj = P[0].copy()
        delta = proj - x
        dist_l2 = float(np.linalg.norm(delta))
        dist_linf = float(np.max(np.abs(delta))) if d else 0.0
        obj = 0.5 * float(dist_l2**2)
        return ConvexHullProjectionResult(
            proj=proj,
            weights=np.asarray([1.0], dtype=np.float64),
            dist_l2=dist_l2,
            dist_linf=dist_linf,
            obj=obj,
            n_iter=0,
            converged=True,
        )

    if init is None:
        w = np.full((n,), 1.0 / float(n), dtype=np.float64)
    else:
        w = project_to_simplex(np.asarray(init, dtype=np.float64).reshape(-1), z=1.0)
        if w.shape != (n,):
            raise ValueError(f"init must have shape ({n},) but got {w.shape}")

    if step is None:
        L = _lipschitz_bound(P)
        step = 1.0 / max(L, 1e-12)
        # keep things conservative in case L bound is loose
        step = float(min(step, 1.0))
    else:
        step = float(step)
        if step <= 0:
            raise ValueError("step must be > 0")

    # iterate
    prev_obj = None
    converged = False
    for it in range(int(max_iter)):
        # current projection
        proj = w @ P  # (d,)
        r = proj - x  # (d,)
        obj = 0.5 * float(np.dot(r, r))

        if prev_obj is not None:
            if abs(prev_obj - obj) <= float(tol) * max(1.0, prev_obj):
                converged = True
                break
        prev_obj = obj

        # grad in w: P @ (proj - x)
        grad = P @ r  # (n,)
        w = project_to_simplex(w - step * grad, z=1.0)

    proj = w @ P
    delta = proj - x
    dist_l2 = float(np.linalg.norm(delta))
    dist_linf = float(np.max(np.abs(delta))) if d else 0.0
    obj = 0.5 * float(dist_l2**2)
    return ConvexHullProjectionResult(
        proj=proj.astype(np.float64),
        weights=w.astype(np.float64),
        dist_l2=dist_l2,
        dist_linf=dist_linf,
        obj=obj,
        n_iter=int(it + (0 if converged else 1)),
        converged=bool(converged),
    )


def ct_projection_features(
    *,
    x: np.ndarray,
    samples: np.ndarray,
    max_iter: int = 200,
) -> Tuple[np.ndarray, dict]:
    """
    Convenience wrapper: project x to conv(samples) and return (proj, features).

    features includes residual and distances, suitable for downstream arbitrage models.
    """
    res = project_point_to_convex_hull(x=x, points=samples, max_iter=int(max_iter))
    residual = (np.asarray(x, dtype=np.float64).reshape(-1) - res.proj).astype(np.float64)
    norm = float(np.linalg.norm(residual))
    direction = residual / (norm + 1e-12)
    feats = {
        "dist_l2": float(res.dist_l2),
        "dist_linf": float(res.dist_linf),
        "obj": float(res.obj),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "residual": residual.astype(np.float32),
        "direction": direction.astype(np.float32),
    }
    return res.proj.astype(np.float32), feats


def summarize_ct_samples(samples: np.ndarray) -> dict:
    """
    Lightweight diagnostics for a sample cloud representing a learned feasible set.
    """
    P = np.asarray(samples, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("samples must be 2D")
    n, d = P.shape
    if n == 0:
        return {"n": 0, "d": int(d)}
    lo = np.min(P, axis=0)
    hi = np.max(P, axis=0)
    mean = np.mean(P, axis=0)
    std = np.std(P, axis=0)
    # cheap notion of size: average squared distance to mean
    rad2 = float(np.mean(np.sum((P - mean) ** 2, axis=1)))
    return {
        "n": int(n),
        "d": int(d),
        "box": {"min": lo.astype(np.float32), "max": hi.astype(np.float32)},
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "radius2": float(rad2),
    }



