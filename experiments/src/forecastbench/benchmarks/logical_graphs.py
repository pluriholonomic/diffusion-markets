from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from forecastbench.metrics.approachability import (
    app_err_curve_upper_orthant_dense,
    distance_to_upper_orthant_linf,
)
from forecastbench.utils.logits import clip_probs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class LogicalGraphSpec:
    """
    Benchmark for 'Family S2': scalable bundles with sparse logical graphs.
    
    Generates m binary markets with a sparse implication graph.
    The Truth function maps context x -> p in [0,1]^m such that the implication
    constraints are satisfied by construction.
    """

    d: int = 16
    m: int = 10
    structure: Literal["chain", "star_in", "star_out"] = "chain"
    # chain: 0 -> 1 -> ... -> m-1  => p[0] <= p[1] <= ... <= p[m-1]
    # star_in: i -> root (last)    => p[i] <= p[m-1] for i < m-1
    # star_out: root (0) -> i      => p[0] <= p[i] for i > 0
    seed: int = 0
    noise: float = 0.25


def sample_truth_prices(spec: LogicalGraphSpec, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n, d) float32
      P: (n, m) float32 satisfying the graph constraints.
    """
    rng = np.random.default_rng(spec.seed)
    
    # 1. Generate context X
    X = rng.standard_normal(size=(n, spec.d)).astype(np.float32)
    
    # 2. Generate m latent scores based on X
    # We use a random linear projection + noise
    W = rng.standard_normal(size=(spec.d, spec.m)).astype(np.float32) / np.sqrt(spec.d)
    logits = X @ W + spec.noise * rng.standard_normal(size=(n, spec.m)).astype(np.float32)
    
    # 3. Apply structure to enforce constraints
    # We transform the raw logits/probs to satisfy inequalities by construction.
    # Using 'sorting' or 'cumulative' approaches preserves the dependence on X.
    
    raw_p = _sigmoid(logits)
    P = np.zeros_like(raw_p)

    if spec.structure == "chain":
        # Enforce p[0] <= p[1] <= ... <= p[m-1]
        # We can just sort the raw probabilities for each example.
        # This defines a valid map f(x) = sort(sigmoid(Wx + noise)).
        P = np.sort(raw_p, axis=1)
        
    elif spec.structure == "star_in":
        # Leaves i=0..m-2 imply Root m-1.  p[i] <= p[root].
        # Let p[root] be max(raw_p). Let p[i] be min(raw_p[i], p[root]).
        # Or simpler: p[root] = max(p_0, ..., p_{m-1}).
        # To make it distributed:
        # Set p[root] = raw_p[:, -1]
        # Set p[i] = raw_p[:, i] * p[root]  => implies p[i] <= p[root]
        root = raw_p[:, -1:]
        leaves = raw_p[:, :-1] * root
        P[:, :-1] = leaves
        P[:, -1:] = root
        
    elif spec.structure == "star_out":
        # Root 0 implies leaves i=1..m-1. p[0] <= p[i].
        # Set p[0] = raw_p[:, 0]
        # Set p[i] = p[0] + (1 - p[0]) * raw_p[:, i]
        root = raw_p[:, 0:1]
        leaves = root + (1 - root) * raw_p[:, 1:]
        P[:, 0:1] = root
        P[:, 1:] = leaves
        
    else:
        raise ValueError(f"Unknown structure {spec.structure}")

    return X, P.astype(np.float32)


def make_graph_cond(X: np.ndarray, m: int) -> np.ndarray:
    """
    Build per-market conditioning embeddings for bundle diffusion.
    (n, m, d + m) -- one-hot encoding of market index.
    """
    n, d = X.shape
    cond = np.zeros((n, m, d + m), dtype=np.float32)
    
    # Broadcast X
    cond[:, :, :d] = X[:, None, :]
    
    # One-hot market indices
    eye = np.eye(m, dtype=np.float32)
    cond[:, :, d:] = eye[None, :, :]
    
    return cond


def graph_violation_matrix(pred: np.ndarray, structure: str) -> np.ndarray:
    """
    Compute constraint violations.
    Returns (n, n_constraints).
    No-arbitrage means V <= 0.
    """
    n, m = pred.shape
    violations = []
    
    if structure == "chain":
        # p[i] <= p[i+1]  =>  p[i] - p[i+1] <= 0
        # Constraints: p[0]-p[1], p[1]-p[2], ...
        for i in range(m - 1):
            # violation = p[i] - p[i+1]
            v = pred[:, i] - pred[:, i+1]
            violations.append(v)
            
    elif structure == "star_in":
        # p[i] <= p[m-1] for i < m-1
        root = pred[:, -1]
        for i in range(m - 1):
            v = pred[:, i] - root
            violations.append(v)
            
    elif structure == "star_out":
        # p[0] <= p[i] for i > 0
        root = pred[:, 0]
        for i in range(1, m):
            v = root - pred[:, i]
            violations.append(v)
            
    else:
        # fallback, no constraints
        return np.zeros((n, 0), dtype=np.float32)
        
    return np.stack(violations, axis=1)


def summarize_logical_graph_arbitrage(
    *,
    pred: np.ndarray,
    p_true: np.ndarray,
    structure: str,
    curve_every: int = 200,
    include_box: bool = True,
) -> Dict:
    """
    Evaluate Blackwell approachability for the graph constraints.
    """
    pred = clip_probs(np.asarray(pred, dtype=np.float64), eps=1e-6)
    p_true = np.asarray(p_true, dtype=np.float64)
    
    # 1. Compute violations matrix V
    V = graph_violation_matrix(pred, structure)
    
    # 2. Approachability of Upper Orthant (V <= 0)
    # We track distance of mean(max(V, 0)) to 0.
    pos = np.maximum(V, 0.0)
    curve = app_err_curve_upper_orthant_dense(v=pos, ub=0.0, every=int(curve_every))
    
    mean_pos_vec = np.mean(pos, axis=0) if len(pos) else np.zeros((pos.shape[1],), dtype=np.float64)
    app_final = distance_to_upper_orthant_linf(mean_pos_vec, ub=0.0)
    
    # Stats
    frac_any = float(np.mean(np.any(pos > 0, axis=1))) if len(V) else 0.0
    mean_pos = float(np.mean(pos)) if len(V) else 0.0
    mse = float(np.mean((pred - p_true) ** 2))
    
    return {
        "n": int(len(pred)),
        "mse": float(mse),
        "static_arbitrage": {
            "structure": structure,
            "M": int(V.shape[1]),
            "mean_pos": float(mean_pos),
            "frac_any_violated": float(frac_any),
            "final_app_err": float(app_final),
            "curve": curve.to_jsonable(),
        }
    }
