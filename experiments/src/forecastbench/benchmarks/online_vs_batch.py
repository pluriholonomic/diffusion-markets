"""
Online vs Batch Benchmark: How do AR and Diffusion compare under different evaluation regimes?

Key distinction:
- BATCH: Train on history, evaluate on held-out test (iid assumption)
- ONLINE: Sequential predictions, possible distribution shift, regret matters

Questions:
1. Does diffusion's smoothness help avoid catastrophic online mistakes?
2. Does AR's reasoning help when distribution shifts?
3. How does calibration evolve over time vs fixed test set?

Metrics:
- Batch: Brier, ECE, LogLoss on test set
- Online: Cumulative regret, time-averaged calibration, approachability curve
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OnlineBatchSpec:
    """Specification for online vs batch comparison."""
    
    T: int = 5000  # Total time steps
    d: int = 32    # Context dimension
    
    # Distribution shift
    shift_type: Literal["none", "gradual", "sudden", "periodic"] = "gradual"
    n_regimes: int = 3  # For sudden/periodic shifts
    
    # Noise
    noise: float = 0.2
    seed: int = 0
    
    # Batch split
    train_frac: float = 0.7


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_stream(spec: OnlineBatchSpec) -> Dict:
    """
    Generate a data stream with optional distribution shift.
    
    Returns:
    - X: (T, d) contexts
    - p_true: (T,) true probabilities
    - y: (T,) outcomes
    - regime: (T,) regime indicator
    """
    rng = np.random.default_rng(spec.seed)
    T, d = spec.T, spec.d
    
    X = rng.standard_normal((T, d)).astype(np.float32)
    
    # Generate weights for each regime
    n_regimes = max(1, spec.n_regimes)
    regime_weights = [
        rng.standard_normal(d).astype(np.float32) / np.sqrt(d)
        for _ in range(n_regimes)
    ]
    
    # Assign regimes based on shift type
    regime = np.zeros(T, dtype=np.int32)
    
    if spec.shift_type == "none":
        regime[:] = 0
        
    elif spec.shift_type == "sudden":
        # Sudden shifts at evenly spaced points
        boundaries = np.linspace(0, T, n_regimes + 1).astype(int)
        for r in range(n_regimes):
            regime[boundaries[r]:boundaries[r+1]] = r
            
    elif spec.shift_type == "gradual":
        # Smooth transition between regimes
        # Use time as mixing coefficient
        t_norm = np.arange(T) / T
        # At t=0, regime 0 dominates; at t=1, regime n_regimes-1 dominates
        regime = (t_norm * (n_regimes - 0.01)).astype(np.int32)
        
    elif spec.shift_type == "periodic":
        # Periodic cycling through regimes
        period = T // (n_regimes * 2)  # Complete ~2 cycles
        regime = (np.arange(T) // period) % n_regimes
    
    # Generate probabilities based on regime
    p_true = np.zeros(T, dtype=np.float32)
    
    for r in range(n_regimes):
        mask = regime == r
        if mask.sum() > 0:
            logits = X[mask] @ regime_weights[r] + spec.noise * rng.standard_normal(mask.sum())
            p_true[mask] = _sigmoid(logits)
    
    # Sample outcomes
    y = (rng.uniform(size=T) < p_true).astype(np.float32)
    
    return {
        "X": X,
        "p_true": p_true,
        "y": y,
        "regime": regime,
        "n_regimes": n_regimes,
    }


def compute_batch_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    p_true: Optional[np.ndarray] = None,
) -> Dict:
    """Standard batch evaluation metrics."""
    p_pred = np.clip(p_pred, 1e-6, 1 - 1e-6)
    
    brier = float(np.mean((p_pred - y_true) ** 2))
    logloss = float(-np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred)))
    
    # ECE
    n_bins = 10
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_true[mask].mean() - p_pred[mask].mean())
    ece = float(ece / len(y_true))
    
    metrics = {"brier": brier, "logloss": logloss, "ece": ece, "n": len(y_true)}
    
    if p_true is not None:
        metrics["mse_to_truth"] = float(np.mean((p_pred - p_true) ** 2))
    
    return metrics


def compute_online_metrics(
    y_seq: np.ndarray,
    p_pred_seq: np.ndarray,
    p_true_seq: Optional[np.ndarray] = None,
    window_size: int = 100,
) -> Dict:
    """
    Online evaluation metrics.
    
    Key differences from batch:
    - Cumulative regret over time
    - Rolling calibration error
    - Time to recover from distribution shift
    """
    T = len(y_seq)
    p_pred_seq = np.clip(p_pred_seq, 1e-6, 1 - 1e-6)
    
    # Cumulative squared error (Brier regret)
    sq_errors = (p_pred_seq - y_seq) ** 2
    cum_brier = np.cumsum(sq_errors)
    
    # Rolling ECE (within windows)
    rolling_ece = []
    for t in range(window_size, T, window_size):
        window_p = p_pred_seq[t-window_size:t]
        window_y = y_seq[t-window_size:t]
        
        ece = 0.0
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            mask = (window_p >= lo) & (window_p < hi)
            if mask.sum() > 0:
                ece += mask.sum() * abs(window_y[mask].mean() - window_p[mask].mean())
        ece /= window_size
        rolling_ece.append({"t": t, "ece": float(ece)})
    
    # Time-averaged error (approachability-style)
    time_avg_error = cum_brier / (np.arange(T) + 1)
    
    metrics = {
        "final_cum_brier": float(cum_brier[-1]),
        "final_time_avg_brier": float(time_avg_error[-1]),
        "max_cum_brier": float(np.max(cum_brier)),
        "rolling_ece": rolling_ece,
        "time_avg_error_curve": [
            {"t": int(t), "err": float(time_avg_error[t])}
            for t in range(0, T, max(1, T // 50))
        ],
    }
    
    if p_true_seq is not None:
        mse_to_truth = (p_pred_seq - p_true_seq) ** 2
        cum_mse = np.cumsum(mse_to_truth)
        metrics["final_cum_mse_to_truth"] = float(cum_mse[-1])
        metrics["final_time_avg_mse_to_truth"] = float(cum_mse[-1] / T)
    
    return metrics


def compute_regime_breakdown(
    y: np.ndarray,
    p_pred: np.ndarray,
    regime: np.ndarray,
) -> Dict:
    """Break down performance by regime (pre/post shift)."""
    results = {}
    
    for r in np.unique(regime):
        mask = regime == r
        if mask.sum() > 0:
            results[f"regime_{r}"] = compute_batch_metrics(
                y[mask], p_pred[mask]
            )
    
    return results


def simulate_ar_predictor(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    L: int = 8,
) -> np.ndarray:
    """
    Simulate AR predictor in batch vs online settings.
    
    Batch: Train on first train_frac, predict rest
    """
    from sklearn.linear_model import LogisticRegression
    
    T = len(y)
    n_train = int(T * train_frac)
    
    # Use only L coordinates (simulating L-query limitation)
    X_limited = X[:, :min(L, X.shape[1])]
    
    # Train on first n_train
    model = LogisticRegression(max_iter=500, solver='lbfgs')
    model.fit(X_limited[:n_train], y[:n_train].astype(int))
    
    # Predict all (but only test portion matters for batch eval)
    p_pred = model.predict_proba(X_limited)[:, 1]
    
    return p_pred.astype(np.float32)


def simulate_diffusion_predictor(
    X: np.ndarray,
    p_true: np.ndarray,
    rho: float = 0.9,
) -> np.ndarray:
    """
    Simulate diffusion predictor with smoothing parameter rho.
    """
    # Diffusion smoothly interpolates toward truth
    p_pred = rho * p_true + (1 - rho) * 0.5
    return p_pred.astype(np.float32)


def simulate_online_ar_predictor(
    X: np.ndarray,
    y: np.ndarray,
    L: int = 8,
    update_every: int = 100,
) -> np.ndarray:
    """
    Online AR with periodic retraining.
    
    Retrain every `update_every` steps on all data seen so far.
    """
    from sklearn.linear_model import LogisticRegression
    
    T = len(y)
    X_limited = X[:, :min(L, X.shape[1])]
    p_pred = np.full(T, 0.5, dtype=np.float32)
    
    model = None
    
    for t in range(T):
        # Predict
        if model is not None and t > 0:
            p_pred[t] = model.predict_proba(X_limited[t:t+1])[0, 1]
        
        # Periodically retrain
        if (t + 1) % update_every == 0 and t > 10:
            model = LogisticRegression(max_iter=200, solver='lbfgs')
            try:
                model.fit(X_limited[:t+1], y[:t+1].astype(int))
            except:
                pass  # May fail if all same class
    
    return p_pred


def run_online_batch_comparison(spec: OnlineBatchSpec) -> Dict:
    """
    Run full online vs batch comparison.
    """
    data = generate_stream(spec)
    X, p_true, y, regime = data["X"], data["p_true"], data["y"], data["regime"]
    T = len(y)
    n_train = int(T * spec.train_frac)
    
    results = {
        "spec": {
            "T": spec.T,
            "d": spec.d,
            "shift_type": spec.shift_type,
            "n_regimes": spec.n_regimes,
            "noise": spec.noise,
            "train_frac": spec.train_frac,
        },
        "batch": {},
        "online": {},
    }
    
    # === BATCH EVALUATION ===
    # Train on first n_train, test on rest
    y_test = y[n_train:]
    p_true_test = p_true[n_train:]
    regime_test = regime[n_train:]
    
    # AR (batch)
    p_ar_batch = simulate_ar_predictor(X, y, spec.train_frac, L=8)
    results["batch"]["ar"] = compute_batch_metrics(y_test, p_ar_batch[n_train:], p_true_test)
    results["batch"]["ar"]["regime_breakdown"] = compute_regime_breakdown(
        y_test, p_ar_batch[n_train:], regime_test
    )
    
    # Diffusion (batch) - simulated with rho
    for rho in [0.8, 0.9, 0.95]:
        p_diff = simulate_diffusion_predictor(X, p_true, rho)
        results["batch"][f"diffusion_rho{rho}"] = compute_batch_metrics(
            y_test, p_diff[n_train:], p_true_test
        )
    
    # Baselines
    results["batch"]["constant_0.5"] = compute_batch_metrics(
        y_test, np.full(len(y_test), 0.5), p_true_test
    )
    results["batch"]["oracle"] = compute_batch_metrics(y_test, p_true_test, p_true_test)
    
    # === ONLINE EVALUATION ===
    # Full sequence, measure regret over time
    
    # AR (online with periodic updates)
    p_ar_online = simulate_online_ar_predictor(X, y, L=8, update_every=100)
    results["online"]["ar"] = compute_online_metrics(y, p_ar_online, p_true)
    
    # Diffusion (doesn't update, same as batch)
    for rho in [0.9]:
        p_diff = simulate_diffusion_predictor(X, p_true, rho)
        results["online"][f"diffusion_rho{rho}"] = compute_online_metrics(y, p_diff, p_true)
    
    # Oracle
    results["online"]["oracle"] = compute_online_metrics(y, p_true, p_true)
    
    return results


def create_online_batch_summary(results: Dict) -> str:
    """Create markdown summary of results."""
    lines = []
    lines.append("# Online vs Batch Comparison")
    lines.append("")
    lines.append(f"Shift type: {results['spec']['shift_type']}, T={results['spec']['T']}")
    lines.append("")
    
    lines.append("## Batch Metrics (held-out test set)")
    lines.append("")
    lines.append("| Model | Brier | ECE | MSE to Truth |")
    lines.append("|-------|-------|-----|--------------|")
    
    for name, metrics in results["batch"].items():
        if isinstance(metrics, dict) and "brier" in metrics:
            mse = metrics.get("mse_to_truth", "N/A")
            if isinstance(mse, float):
                mse = f"{mse:.4f}"
            lines.append(f"| {name} | {metrics['brier']:.4f} | {metrics['ece']:.4f} | {mse} |")
    
    lines.append("")
    lines.append("## Online Metrics (sequential evaluation)")
    lines.append("")
    lines.append("| Model | Final Time-Avg Brier | Final Cum Brier |")
    lines.append("|-------|---------------------|-----------------|")
    
    for name, metrics in results["online"].items():
        if isinstance(metrics, dict) and "final_time_avg_brier" in metrics:
            lines.append(
                f"| {name} | {metrics['final_time_avg_brier']:.4f} | "
                f"{metrics['final_cum_brier']:.1f} |"
            )
    
    return "\n".join(lines)


