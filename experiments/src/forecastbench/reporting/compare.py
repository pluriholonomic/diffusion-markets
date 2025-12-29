from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from forecastbench.benchmarks.polymarket_eval import realized_trading_pnl
from forecastbench.metrics import brier_loss, expected_calibration_error, log_loss
from forecastbench.metrics.trading_sim import KellySimConfig, simulate_kelly_roi


@dataclass(frozen=True)
class ModelRunSpec:
    name: str
    run_dir: Path
    pred_col: Optional[str] = None


def _infer_pred_col_from_config(cfg: dict) -> Optional[str]:
    if isinstance(cfg.get("pred_col"), str):
        return str(cfg["pred_col"])
    ev = cfg.get("eval")
    if isinstance(ev, dict) and isinstance(ev.get("pred_col"), str):
        return str(ev["pred_col"])
    return None


def load_run_predictions(spec: ModelRunSpec) -> Tuple[pd.DataFrame, dict]:
    run_dir = Path(spec.run_dir)
    cfg_path = run_dir / "config.json"
    pred_path = run_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.parquet in {run_dir}")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    pred_col = spec.pred_col or _infer_pred_col_from_config(cfg)
    if not pred_col:
        raise ValueError(f"Could not infer pred_col for run_dir={run_dir}; pass pred_col explicitly.")
    df = pd.read_parquet(pred_path)
    required = {"id", "y", "market_prob"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Run {spec.name} missing columns {missing} in predictions.parquet")

    if pred_col == "market_prob":
        # Special case: market baseline uses market_prob as the predictor. Keep market_prob for merging
        # and create a separate prediction column.
        df = df[["id", "y", "market_prob"]].copy()
        df[f"pred__{spec.name}"] = df["market_prob"].astype(float)
        return df, cfg

    if pred_col not in df.columns:
        raise ValueError(f"Run {spec.name} missing pred_col={pred_col!r} in predictions.parquet")

    df = df[["id", "y", "market_prob", pred_col]].copy()
    df = df.rename(columns={pred_col: f"pred__{spec.name}"})
    return df, cfg


def compute_point_metrics(
    *,
    y: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    bins: int,
    B: float,
    transaction_cost: float,
    trading_mode: str,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out: Dict[str, float] = {
        "brier": float(brier_loss(p.astype(np.float32), y)),
        "logloss": float(log_loss(p.astype(np.float32), y)),
        "ece": float(expected_calibration_error(p.astype(np.float32), y, n_bins=int(bins))),
        "pnl_per_event": float(
            realized_trading_pnl(
                y=y.astype(np.float64),
                market_prob=q.astype(np.float64),
                pred_prob=p.astype(np.float64),
                B=float(B),
                transaction_cost=float(transaction_cost),
                mode=str(trading_mode),
            )
        ),
    }
    kelly = simulate_kelly_roi(
        p=p,
        q=q,
        y=y,
        cfg=KellySimConfig(
            initial_bankroll=1.0,
            scale=1.0,
            frac_cap=0.25,
            fee=float(transaction_cost),
        ),
        return_curve=False,
    )
    out["kelly_roi"] = float(kelly["roi"])
    return out


def bootstrap_ci(
    *,
    metric_fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    vals = []
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=(n,))
        vals.append(float(metric_fn(idx)))
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return lo, hi


def compare_polymarket_runs(
    *,
    specs: Sequence[ModelRunSpec],
    bins: int = 20,
    B: float = 1.0,
    transaction_cost: float = 0.0,
    trading_mode: str = "sign",
    n_boot: int = 200,
    seed: int = 0,
    baseline: Optional[str] = None,
) -> Dict[str, object]:
    if not specs:
        raise ValueError("No model specs provided")

    # Load and inner-join on id.
    base_df = None
    cfgs: Dict[str, dict] = {}
    for s in specs:
        df_i, cfg_i = load_run_predictions(s)
        cfgs[str(s.name)] = cfg_i
        if base_df is None:
            base_df = df_i
        else:
            base_df = base_df.merge(df_i, on=["id", "y", "market_prob"], how="inner")
    assert base_df is not None
    df = base_df
    n = int(len(df))
    if n <= 0:
        raise ValueError("No overlapping rows across runs after merge")

    y = df["y"].to_numpy(dtype=np.float64)
    q = df["market_prob"].to_numpy(dtype=np.float64)

    model_names = [str(s.name) for s in specs]
    pred_cols = [f"pred__{name}" for name in model_names]

    # Point estimates
    point = {}
    for name, col in zip(model_names, pred_cols):
        p = df[col].to_numpy(dtype=np.float64)
        point[name] = compute_point_metrics(
            y=y, q=q, p=p, bins=int(bins), B=float(B), transaction_cost=float(transaction_cost), trading_mode=str(trading_mode)
        )

    # Bootstrap CIs (paired, using same resample indices for all models)
    ci: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in model_names}
    diffs: Dict[str, Dict[str, Dict[str, float]]] = {}

    for metric in ["brier", "logloss", "ece", "pnl_per_event", "kelly_roi"]:
        # pre-cache per-model predictions
        preds = {name: df[f"pred__{name}"].to_numpy(dtype=np.float64) for name in model_names}

        def _metric_for(name: str, idx: np.ndarray) -> float:
            return float(
                compute_point_metrics(
                    y=y[idx],
                    q=q[idx],
                    p=preds[name][idx],
                    bins=int(bins),
                    B=float(B),
                    transaction_cost=float(transaction_cost),
                    trading_mode=str(trading_mode),
                )[metric]
            )

        for name in model_names:
            lo, hi = bootstrap_ci(metric_fn=lambda idx, nm=name: _metric_for(nm, idx), n=n, n_boot=int(n_boot), seed=int(seed))
            ci[name][metric] = {"lo": float(lo), "hi": float(hi)}

        if baseline is not None and baseline in model_names:
            diffs[metric] = {}
            for name in model_names:
                if name == baseline:
                    continue

                def _diff(idx: np.ndarray, nm=name) -> float:
                    return float(_metric_for(nm, idx) - _metric_for(baseline, idx))

                lo, hi = bootstrap_ci(metric_fn=_diff, n=n, n_boot=int(n_boot), seed=int(seed))
                diffs[metric][name] = {"lo": float(lo), "hi": float(hi), "diff": float(point[name][metric] - point[baseline][metric])}

    return {
        "n": int(n),
        "specs": [{"name": s.name, "run_dir": str(s.run_dir), "pred_col": s.pred_col} for s in specs],
        "params": {
            "bins": int(bins),
            "B": float(B),
            "transaction_cost": float(transaction_cost),
            "trading_mode": str(trading_mode),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "baseline": baseline,
        },
        "point": point,
        "ci95": ci,
        "diffs_vs_baseline": diffs if diffs else None,
        "configs": cfgs,
    }


