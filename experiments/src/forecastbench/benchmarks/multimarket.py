from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np

from forecastbench.constraints import equality_violations_3, frechet_violations
from forecastbench.metrics.approachability import app_err_curve_upper_orthant_dense, distance_to_upper_orthant_linf
from forecastbench.utils.logits import clip_probs, prob_to_logit


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class MultiMarketSpec:
    """
    Synthetic multi-market benchmark aligned with main.tex "static-arbitrage" Blackwell diagnostics.

    We generate a context vector x, and three related binary markets:
      A, B, and (A ∧ B).

    The task is to output a *joint* price vector p=(p_A,p_B,p_AB) that satisfies
    the chosen no-arbitrage constraint family.
    """

    d: int = 16
    structure: Literal["frechet", "equal"] = "frechet"  # unknown correlation vs known A=B
    seed: int = 0

    # how "hard" the regression is
    noise: float = 0.25  # latent noise in logits


def sample_truth_prices(spec: MultiMarketSpec, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n,d) float32 contexts
      P: (n,3) float32 prices [p_A,p_B,p_AB] that satisfy the structure's constraints.
    """
    rng = np.random.default_rng(int(spec.seed))
    d = int(spec.d)
    X = rng.standard_normal(size=(n, d)).astype(np.float32)

    # Fixed random weights define the realizable Truth mapping.
    w_a = rng.standard_normal(size=(d,)).astype(np.float64) / np.sqrt(d)
    w_b = rng.standard_normal(size=(d,)).astype(np.float64) / np.sqrt(d)
    w_m = rng.standard_normal(size=(d,)).astype(np.float64) / np.sqrt(d)

    xa = X.astype(np.float64) @ w_a + float(spec.noise) * rng.standard_normal(size=(n,))
    xb = X.astype(np.float64) @ w_b + float(spec.noise) * rng.standard_normal(size=(n,))
    p_a = _sigmoid(xa)

    if spec.structure == "equal":
        # Known perfect correlation: A=B almost surely => p_A=p_B=p_AB.
        p_b = p_a.copy()
        p_ab = p_a.copy()
    elif spec.structure == "frechet":
        p_b = _sigmoid(xb)
        # Sample p_AB inside the Fréchet interval via a learned-ish mixing variable.
        lo = np.maximum(0.0, p_a + p_b - 1.0)
        hi = np.minimum(p_a, p_b)
        mix = _sigmoid(X.astype(np.float64) @ w_m)
        p_ab = lo + mix * (hi - lo)
    else:
        raise ValueError(f"Unknown structure={spec.structure!r}")

    P = np.stack([p_a, p_b, p_ab], axis=1).astype(np.float32)
    return X.astype(np.float32), P.astype(np.float32)


def make_bundle_cond(X: np.ndarray) -> np.ndarray:
    """
    Build per-market conditioning embeddings for bundle diffusion.

    For each context x, create 3 tokens corresponding to markets A,B,AB:
      cond_i = concat(x, one_hot(i))
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    one_hot = np.eye(3, dtype=np.float32)  # (3,3)
    cond = np.zeros((n, 3, d + 3), dtype=np.float32)
    for i in range(3):
        cond[:, i, :d] = X
        cond[:, i, d:] = one_hot[i]
    return cond


def violation_matrix(pred: np.ndarray, *, structure: str, include_box: bool = True) -> np.ndarray:
    """
    Convert predicted prices (n,3) to a violation matrix V (n,M) where no-arb is V<=0.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("pred must have shape (n,3)")
    out = []
    for p_a, p_b, p_ab in pred.tolist():
        if structure == "frechet":
            out.append(frechet_violations(p_a=p_a, p_b=p_b, p_ab=p_ab, include_box=include_box))
        elif structure == "equal":
            out.append(equality_violations_3(p_a=p_a, p_b=p_b, p_c=p_ab, include_box=include_box))
        else:
            raise ValueError(f"Unknown structure={structure!r}")
    return np.stack(out, axis=0).astype(np.float64)


def summarize_static_arbitrage(
    *,
    pred: np.ndarray,
    p_true: np.ndarray,
    structure: str,
    curve_every: int = 200,
    include_box: bool = True,
) -> Dict:
    """
    Summarize static-arbitrage approachability for a prediction sequence.
    """
    pred = clip_probs(np.asarray(pred, dtype=np.float64), eps=1e-6)
    p_true = np.asarray(p_true, dtype=np.float64)

    V = violation_matrix(pred, structure=str(structure), include_box=bool(include_box))  # (n,M)
    # For one-sided inequality constraints, negative residuals do not “cancel” positive violations:
    # a trader can simply choose to act only when a constraint is violated.
    # Therefore we compute approachability on the *positive part* V_+.
    pos = np.maximum(V, 0.0)
    curve = app_err_curve_upper_orthant_dense(v=pos, ub=0.0, every=int(curve_every))

    mean_pos_vec = np.mean(pos, axis=0) if len(pos) else np.zeros((pos.shape[1],), dtype=np.float64)
    app_final = distance_to_upper_orthant_linf(mean_pos_vec, ub=0.0)
    frac_any = float(np.mean(np.any(pos > 0, axis=1))) if len(V) else float("nan")
    mean_pos = float(np.mean(pos)) if len(V) else float("nan")

    mse = float(np.mean((pred - p_true) ** 2))

    return {
        "n": int(len(pred)),
        "mse": float(mse),
        "static_arbitrage": {
            "structure": str(structure),
            "include_box": bool(include_box),
            "M": int(V.shape[1]),
            "mean_pos": float(mean_pos),
            "frac_any_violated": float(frac_any),
            "final_app_err": float(app_final),
            "curve_every": int(curve_every),
            "curve": curve.to_jsonable(),
        },
    }


def logits_from_prices(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    p = clip_probs(np.asarray(p, dtype=np.float64), eps=float(eps))
    return prob_to_logit(p, eps=1e-12).astype(np.float32)


