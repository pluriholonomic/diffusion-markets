from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecastbench.metrics import (
    app_err_curve_single_coordinate,
    best_bounded_trader_profit,
    brier_loss,
    expected_calibration_error,
    log_loss,
    summarize_top_box_violations,
)
from forecastbench.metrics.trading_sim import KellySimConfig, simulate_kelly_roi
from forecastbench.metrics.approachability import distance_to_box_linf
from forecastbench.utils.logits import clip_probs


def realized_trading_pnl(
    *,
    y: np.ndarray,
    market_prob: np.ndarray,
    pred_prob: np.ndarray,
    B: float = 1.0,
    transaction_cost: float = 0.0,
    mode: str = "sign",
) -> float:
    """
    Simple realized PnL proxy for Polymarket-style evaluation.

    We treat `market_prob` as the traded price and `pred_prob` as our estimate.
    We take a position b in [-B,B] and realize:
      pnl = b * (y - market_prob) - c * |b|

    Modes:
    - sign: b = B * sign(pred - market)
    - linear: b = clip(B * (pred - market), -B, B)
    """
    y = y.astype(np.float64)
    q = market_prob.astype(np.float64)
    p = pred_prob.astype(np.float64)

    if mode == "sign":
        b = B * np.sign(p - q)
    elif mode == "linear":
        b = np.clip(B * (p - q), -B, B)
    else:
        raise ValueError(f"unknown mode: {mode}")

    pnl = b * (y - q) - transaction_cost * np.abs(b)
    return float(np.mean(pnl))


def group_calibration_bias(df: pd.DataFrame, *, group_col: str, y_col: str, pred_col: str) -> Dict[str, float]:
    """
    Returns:
      - worst_abs_bias: max_g | E[y - pred | g] |
      - avg_abs_bias: E_g |E[y - pred | g]| weighted by group size
    """
    g = df[group_col]
    y = df[y_col].astype(float)
    p = df[pred_col].astype(float)
    res = y - p

    by = res.groupby(g)
    means = by.mean()
    counts = by.size()
    worst = float(np.max(np.abs(means.values))) if len(means) else float("nan")
    avg = float(np.sum(np.abs(means.values) * (counts.values / counts.values.sum()))) if len(means) else float("nan")
    return {"worst_abs_bias": worst, "avg_abs_bias": avg, "n_groups": int(len(means))}


def evaluate_polymarket_dataset(
    df: pd.DataFrame,
    *,
    pred_col: str,
    y_col: str = "y",
    market_prob_col: Optional[str] = "market_prob",
    bins: int = 20,
    transaction_cost: float = 0.0,
    B: float = 1.0,
    trading_mode: str = "sign",
    group_cols: Optional[List[str]] = None,
) -> Dict:
    y = df[y_col].to_numpy().astype(np.float64)
    pred = df[pred_col].to_numpy().astype(np.float32)

    out = {
        "n": int(len(df)),
        "brier": brier_loss(pred, y),
        "logloss": log_loss(pred, y),
        "ece": expected_calibration_error(pred, y, n_bins=bins),
    }

    if market_prob_col is not None and market_prob_col in df.columns:
        mkt = df[market_prob_col].to_numpy().astype(np.float32)
        out["trading"] = {
            "mode": trading_mode,
            "B": float(B),
            "transaction_cost": float(transaction_cost),
            "pnl_per_event": realized_trading_pnl(
                y=y, market_prob=mkt, pred_prob=pred, B=B, transaction_cost=transaction_cost, mode=trading_mode
            ),
            "kelly": simulate_kelly_roi(
                p=pred,
                q=mkt,
                y=y,
                cfg=KellySimConfig(
                    initial_bankroll=1.0,
                    scale=1.0,
                    frac_cap=0.25,
                    fee=float(transaction_cost),
                ),
                return_curve=False,
            ),
        }

    if group_cols:
        out["groups"] = {gc: group_calibration_bias(df, group_col=gc, y_col=y_col, pred_col=pred_col) for gc in group_cols}

    return out


def _series_to_unix_seconds(s: pd.Series) -> np.ndarray:
    """
    Best-effort conversion of a pandas Series to unix seconds (float64), with NaN for missing.
    Accepts:
      - numeric unix timestamps (seconds)
      - ISO datetime strings
      - pandas datetime types
    """
    if pd.api.types.is_numeric_dtype(s):
        x = s.to_numpy(dtype=np.float64, copy=False)
        return x
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    # avoid Series.view FutureWarning; keep NaT -> min int64 sentinel
    ns = dt.astype("int64")
    out = ns.astype(np.float64) / 1e9
    out[ns == np.iinfo(np.int64).min] = np.nan
    return out


def _infer_time_cols_for_repair(
    df: pd.DataFrame,
    *,
    forecast_time_col: Optional[str],
    event_time_col: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer reasonable defaults for a "forecast timestamp" and a "resolution timestamp".
    """
    fc = forecast_time_col if (forecast_time_col and forecast_time_col in df.columns) else None
    ec = event_time_col if (event_time_col and event_time_col in df.columns) else None

    if fc is None:
        if "market_prob_target_ts" in df.columns:
            fc = "market_prob_target_ts"
        elif "createdAt" in df.columns:
            fc = "createdAt"

    if ec is None:
        if "market_event_ts" in df.columns:
            ec = "market_event_ts"
        elif "endDate" in df.columns:
            ec = "endDate"
        elif "closedTime" in df.columns:
            ec = "closedTime"

    return fc, ec


@dataclass
class _OnlineGroupBinCalibrator:
    """
    Minimal online group×bin calibrator with Beta-style prior smoothing.

    Intended as a lightweight "repair map" σ_t(q, X) updated only when outcomes are revealed.
    """

    n_bins: int
    prior_strength: float
    clip_eps: float

    global_count: np.ndarray  # (n_bins,) int64
    global_sum: np.ndarray  # (n_bins,) float64
    group_count: Dict[str, np.ndarray]  # group -> (n_bins,) int64
    group_sum: Dict[str, np.ndarray]  # group -> (n_bins,) float64

    @staticmethod
    def create(*, n_bins: int, prior_strength: float, clip_eps: float) -> "_OnlineGroupBinCalibrator":
        return _OnlineGroupBinCalibrator(
            n_bins=int(n_bins),
            prior_strength=float(prior_strength),
            clip_eps=float(clip_eps),
            global_count=np.zeros((int(n_bins),), dtype=np.int64),
            global_sum=np.zeros((int(n_bins),), dtype=np.float64),
            group_count={},
            group_sum={},
        )

    def _bin(self, q: float) -> int:
        q = float(np.clip(q, float(self.clip_eps), 1.0 - float(self.clip_eps)))
        # equal-width bins on [0,1]
        b = int(np.floor(q * self.n_bins))
        if b >= self.n_bins:
            b = self.n_bins - 1
        if b < 0:
            b = 0
        return b

    def update(self, *, group: str, q: float, y: float) -> None:
        b = self._bin(q)
        self.global_count[b] += 1
        self.global_sum[b] += float(y)

        if group not in self.group_count:
            self.group_count[group] = np.zeros((self.n_bins,), dtype=np.int64)
            self.group_sum[group] = np.zeros((self.n_bins,), dtype=np.float64)
        self.group_count[group][b] += 1
        self.group_sum[group][b] += float(y)

    def predict(self, *, group: str, q: float) -> float:
        q = float(np.clip(q, float(self.clip_eps), 1.0 - float(self.clip_eps)))
        b = self._bin(q)
        prior_mean = (b + 0.5) / float(self.n_bins)  # identity-ish prior per bin
        ps = float(self.prior_strength)

        if group in self.group_count and self.group_count[group][b] > 0:
            c = float(self.group_count[group][b])
            s = float(self.group_sum[group][b])
            return float(np.clip((s + ps * prior_mean) / (c + ps), 0.0, 1.0))

        # global fallback
        if self.global_count[b] > 0:
            c = float(self.global_count[b])
            s = float(self.global_sum[b])
            return float(np.clip((s + ps * prior_mean) / (c + ps), 0.0, 1.0))

        # no data at all for this bin: identity fallback
        return q


def repair_group_bin_at_resolution(
    df: pd.DataFrame,
    *,
    pred_col: str,
    y_col: str = "y",
    group_cols: Optional[List[str]] = None,
    n_bins: int = 20,
    prior_strength: float = 5.0,
    clip_eps: float = 1e-6,
    forecast_time_col: Optional[str] = None,
    event_time_col: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Simulate "repair-at-resolution" (batch feedback) on a static dataset.

    We treat each row as a market with:
      - a forecast timestamp (when we issue q)
      - a resolution timestamp (when y is revealed)

    The repair map σ_t is updated ONLY when outcomes are revealed (resolution events), and then applied
    to subsequent forecasts.
    """
    if pred_col not in df.columns:
        raise ValueError(f"Missing pred_col {pred_col!r} in df.")
    if y_col not in df.columns:
        raise ValueError(f"Missing y_col {y_col!r} in df.")
    if n_bins <= 1:
        raise ValueError("n_bins must be >= 2")
    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")

    forecast_time_col_used, event_time_col_used = _infer_time_cols_for_repair(
        df, forecast_time_col=forecast_time_col, event_time_col=event_time_col
    )

    n = int(len(df))
    q_raw = clip_probs(df[pred_col].to_numpy().astype(np.float64), eps=float(clip_eps))
    y = df[y_col].to_numpy().astype(np.float64)

    if group_cols:
        cols = [c.strip() for c in group_cols if c.strip()]
        if not cols:
            keys = pd.Series(["__GLOBAL__"] * n, index=df.index)
        elif len(cols) == 1:
            keys = df[cols[0]].fillna("__MISSING__").astype(str)
        else:
            keys = df[cols].fillna("__MISSING__").astype(str).agg("|".join, axis=1)
    else:
        cols = []
        keys = pd.Series(["__GLOBAL__"] * n, index=df.index)

    group_key = keys.astype(str).to_numpy()

    if forecast_time_col_used is not None:
        ft = _series_to_unix_seconds(df[forecast_time_col_used])
    else:
        ft = np.arange(n, dtype=np.float64)
    if event_time_col_used is not None:
        et = _series_to_unix_seconds(df[event_time_col_used])
    else:
        et = np.full((n,), np.inf, dtype=np.float64)

    ft_sort = np.where(np.isnan(ft), np.inf, ft)
    et_sort = np.where(np.isnan(et), np.inf, et)

    order_forecast = np.argsort(ft_sort, kind="stable")
    order_event = np.argsort(et_sort, kind="stable")

    cal = _OnlineGroupBinCalibrator.create(n_bins=int(n_bins), prior_strength=float(prior_strength), clip_eps=float(clip_eps))
    out = np.empty((n,), dtype=np.float64)

    updated = np.zeros((n,), dtype=bool)
    j = 0
    n_updates = 0

    for idx_f in order_forecast.tolist():
        t = float(ft_sort[idx_f])

        # apply any resolution updates available by this forecast time
        while j < n and float(et_sort[order_event[j]]) <= t:
            idx_r = int(order_event[j])
            j += 1
            if updated[idx_r]:
                continue
            # prevent leakage if timestamps are inconsistent
            if float(ft_sort[idx_r]) > t:
                continue
            if not np.isfinite(float(et_sort[idx_r])):
                continue
            cal.update(group=str(group_key[idx_r]), q=float(q_raw[idx_r]), y=float(y[idx_r]))
            updated[idx_r] = True
            n_updates += 1

        out[idx_f] = cal.predict(group=str(group_key[idx_f]), q=float(q_raw[idx_f]))

    meta: Dict[str, object] = {
        "pred_col": str(pred_col),
        "y_col": str(y_col),
        "group_cols": cols,
        "n_bins": int(n_bins),
        "prior_strength": float(prior_strength),
        "clip_eps": float(clip_eps),
        "forecast_time_col": forecast_time_col_used,
        "event_time_col": event_time_col_used,
        "n_updates": int(n_updates),
        "n_groups_seen": int(len(cal.group_count)),
    }
    return out.astype(np.float32), meta


def evaluate_group_bin_approachability(
    df: pd.DataFrame,
    *,
    pred_col: str,
    y_col: str = "y",
    group_cols: Optional[List[str]] = None,
    n_bins: int = 20,
    eps: float = 0.0,
    time_col: Optional[str] = None,
    curve_every: int = 10,
    topk: int = 10,
    clip_eps: float = 1e-6,
) -> Dict:
    """
    Compute Blackwell-approachability diagnostics for the calibration/arbitrage payoff family

      g_t(i) = (Y_t - q_t) h^i(X_t, q_t)

    using the finite group×bin indicator family:
      h^{(g,b)}(X,q) = 1{ group(X)=g and bin(q)=b }.

    Target set is the box C_eps = [-eps, eps]^M and the metric is:
      AppErr_T = d_∞( (1/T)∑_{t<=T} g_t, C_eps ).
    """
    if pred_col not in df.columns:
        raise ValueError(f"Missing pred_col {pred_col!r} in df.")
    if y_col not in df.columns:
        raise ValueError(f"Missing y_col {y_col!r} in df.")
    if n_bins <= 1:
        raise ValueError("n_bins must be >= 2")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if curve_every <= 0:
        raise ValueError("curve_every must be >= 1")

    y = df[y_col].to_numpy().astype(np.float64)
    q = clip_probs(df[pred_col].to_numpy().astype(np.float64), eps=float(clip_eps))

    # Time ordering (optional)
    time_col_used: Optional[str] = None
    if time_col is not None and time_col in df.columns:
        time_col_used = time_col
    elif time_col is None and "market_prob_target_ts" in df.columns:
        time_col_used = "market_prob_target_ts"
    elif time_col is None and "createdAt" in df.columns:
        time_col_used = "createdAt"

    if time_col_used is not None:
        ts = _series_to_unix_seconds(df[time_col_used])
        ts_sort = np.where(np.isnan(ts), np.inf, ts)
        order = np.argsort(ts_sort, kind="stable")
        ts_sorted = ts[order]
    else:
        order = np.arange(len(df), dtype=np.int64)
        ts_sorted = None

    y = y[order]
    q = q[order]

    # Group keys
    group_cols_used = [c.strip() for c in (group_cols or []) if c.strip()]
    if group_cols_used:
        missing = [c for c in group_cols_used if c not in df.columns]
        if missing:
            raise ValueError(f"Missing group_cols={missing}; available={list(df.columns)}")
        if len(group_cols_used) == 1:
            keys = df[group_cols_used[0]].fillna("__MISSING__").astype(str)
        else:
            keys = df[group_cols_used].fillna("__MISSING__").astype(str).agg("|".join, axis=1)
    else:
        keys = pd.Series(["__GLOBAL__"] * len(df), index=df.index)

    # Factorize group keys (stable)
    codes, uniques = pd.factorize(keys, sort=True)
    codes = codes.astype(np.int64)[order]
    group_names = [str(x) for x in uniques.tolist()]

    # Bin indices on q
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    b = np.digitize(q, edges, right=True) - 1
    b = np.clip(b, 0, int(n_bins) - 1).astype(np.int64)

    # Sparse payoff updates: only the (group,bin) coordinate is non-zero each round.
    coord = codes * int(n_bins) + b
    value = (y.astype(np.float64) - q.astype(np.float64)).astype(np.float64)

    M = int(len(group_names) * int(n_bins))
    curve = app_err_curve_single_coordinate(coord=coord, value=value, M=M, eps=float(eps), every=int(curve_every))

    # Final mean vector for introspection
    sums = np.bincount(coord, weights=value, minlength=M).astype(np.float64)
    mean = sums / float(len(df)) if len(df) else np.zeros((M,), dtype=np.float64)
    app_final = distance_to_box_linf(mean, eps=float(eps))

    top = summarize_top_box_violations(mean_vec=mean, eps=float(eps), topk=int(topk))
    top_named: List[Dict[str, object]] = []
    for idx, mean_i, viol_i in top:
        gi = int(idx // int(n_bins))
        bi = int(idx % int(n_bins))
        top_named.append(
            {
                "coord": int(idx),
                "group": group_names[gi] if 0 <= gi < len(group_names) else None,
                "group_idx": gi,
                "bin_idx": bi,
                "bin_lo": float(edges[bi]),
                "bin_hi": float(edges[bi + 1]),
                "mean_payoff": float(mean_i),
                "violation": float(viol_i),
            }
        )

    # Optional: attach wall-clock times for curve samples
    curve_times = None
    if ts_sorted is not None:
        # curve.t is 1-indexed step count
        idxs = np.clip(curve.t - 1, 0, len(ts_sorted) - 1)
        curve_times = ts_sorted[idxs].astype(np.float64).tolist()

    return {
        "family": "group_bin_indicator",
        "n": int(len(df)),
        "pred_col": str(pred_col),
        "y_col": str(y_col),
        "group_cols": group_cols_used,
        "n_groups": int(len(group_names)),
        "n_bins": int(n_bins),
        "eps": float(eps),
        "clip_eps": float(clip_eps),
        "time_col": time_col_used,
        "curve_every": int(curve_every),
        "curve": curve.to_jsonable(),
        "curve_time_ts": curve_times,  # may be None
        "final": {"app_err": float(app_final), "top_violations": top_named},
    }