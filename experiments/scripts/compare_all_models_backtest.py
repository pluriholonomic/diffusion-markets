#!/usr/bin/env python3
"""
Comprehensive Backtest: Compare All C_t Models

Compares trading performance across:
1. RLCR (fine-tuned LLM)
2. AR Baseline (unfine-tuned Qwen3-14B)
3. Diffusion (standalone)
4. Market baseline (no model, just market prices)

Metrics:
- PnL, Sharpe, Win Rate
- Calibration (Brier, ECE)
- Statistical significance via bootstrap

Usage:
    python scripts/compare_all_models_backtest.py \
        --data polymarket_backups/pm_suite_derived/gamma_yesno_ready_20k.parquet \
        --rlcr-preds runs/eval_rlcr_20k/predictions.parquet \
        --ar-preds runs/eval_ar_20k/predictions.parquet \
        --output runs/model_comparison.json
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    name: str
    n_trades: int
    mean_pnl: float
    std_pnl: float
    sharpe: float
    ann_sharpe: float
    total_pnl: float
    win_rate: float
    brier: float
    ece: float
    # Bootstrap CIs
    sharpe_ci_low: float = 0.0
    sharpe_ci_high: float = 0.0


def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions parquet."""
    df = pd.read_parquet(path)
    required = ['pred_prob', 'market_prob', 'y']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    return df


def compute_pnl(
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    tc: float = 0.02,
    mode: str = "sign",
) -> np.ndarray:
    """Compute per-trade PnL."""
    if mode == "sign":
        direction = np.sign(p - q)
        pnl = direction * (y - q) - tc
    else:  # linear
        size = np.clip(np.abs(p - q) / 0.3, 0, 1)
        direction = np.sign(p - q)
        pnl = direction * size * (y - q) - tc * size
    return pnl


def compute_brier(p: np.ndarray, y: np.ndarray) -> float:
    """Compute Brier score."""
    return float(np.mean((p - y) ** 2))


def compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y[mask].mean()
            bin_conf = p[mask].mean()
            ece += mask.sum() * np.abs(bin_acc - bin_conf)
    return float(ece / len(p))


def bootstrap_sharpe(
    pnl: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap Sharpe ratio with 95% CI."""
    rng = np.random.default_rng(seed)
    sharpes = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(len(pnl), size=len(pnl), replace=True)
        sample = pnl[idx]
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std())
    
    sharpes = np.array(sharpes)
    return (
        float(np.mean(sharpes)),
        float(np.percentile(sharpes, 2.5)),
        float(np.percentile(sharpes, 97.5)),
    )


def evaluate_model(
    name: str,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    tc: float = 0.02,
    n_bootstrap: int = 1000,
) -> ModelMetrics:
    """Evaluate a model's predictions."""
    pnl = compute_pnl(p, q, y, tc)
    
    # Filter to actual trades
    traded = pnl != 0
    if traded.sum() == 0:
        return ModelMetrics(
            name=name, n_trades=0, mean_pnl=0, std_pnl=0,
            sharpe=0, ann_sharpe=0, total_pnl=0, win_rate=0,
            brier=compute_brier(p, y), ece=compute_ece(p, y),
        )
    
    pnl_traded = pnl[traded]
    
    mean_pnl = float(pnl_traded.mean())
    std_pnl = float(pnl_traded.std())
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    
    # Bootstrap
    _, ci_low, ci_high = bootstrap_sharpe(pnl_traded, n_bootstrap)
    
    return ModelMetrics(
        name=name,
        n_trades=int(traded.sum()),
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        sharpe=sharpe,
        ann_sharpe=sharpe * np.sqrt(250),
        total_pnl=float(pnl_traded.sum()),
        win_rate=float((pnl_traded > 0).mean()),
        brier=compute_brier(p, y),
        ece=compute_ece(p, y),
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
    )


def print_comparison(results: Dict[str, ModelMetrics]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    
    header = f"{'Model':<20} {'N':<8} {'PnL/Trade':<12} {'Sharpe':<10} {'Ann.Sharpe':<12} {'Win%':<8} {'Brier':<8} {'ECE':<8}"
    print(header)
    print("-" * 90)
    
    for name, m in sorted(results.items(), key=lambda x: -x[1].sharpe):
        row = f"{m.name:<20} {m.n_trades:<8} {m.mean_pnl:+.4f}{'':>6} {m.sharpe:.4f}{'':>4} {m.ann_sharpe:.2f}{'':>6} {m.win_rate:.1%}{'':>2} {m.brier:.4f}{'':>2} {m.ece:.4f}"
        print(row)
    
    print("\n" + "=" * 50)
    print("BOOTSTRAP 95% CI FOR SHARPE")
    print("=" * 50)
    
    for name, m in sorted(results.items(), key=lambda x: -x[1].sharpe):
        print(f"  {m.name:<20} [{m.sharpe_ci_low:.4f}, {m.sharpe_ci_high:.4f}]")
    
    # Statistical significance
    print("\n" + "=" * 50)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 50)
    
    models = sorted(results.items(), key=lambda x: -x[1].sharpe)
    if len(models) >= 2:
        best = models[0][1]
        for name, m in models[1:]:
            overlap = m.sharpe_ci_high > best.sharpe_ci_low and best.sharpe_ci_high > m.sharpe_ci_low
            sig = "NOT significant" if overlap else "SIGNIFICANT (p<0.05)"
            print(f"  {best.name} vs {m.name}: {sig}")


def main():
    parser = argparse.ArgumentParser(description="Compare all models")
    parser.add_argument("--data", required=True, help="Base data parquet")
    parser.add_argument("--rlcr-preds", help="RLCR predictions parquet")
    parser.add_argument("--ar-preds", help="AR baseline predictions parquet")
    parser.add_argument("--diff-preds", help="Diffusion predictions parquet")
    parser.add_argument("--max-rows", type=int, help="Max rows")
    parser.add_argument("--tc", type=float, default=0.02, help="Transaction cost")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--output", default="runs/model_comparison.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON BACKTEST")
    print("=" * 70)
    
    # Load base data
    print(f"\nLoading base data from {args.data}...")
    base_df = pd.read_parquet(args.data)
    if args.max_rows:
        base_df = base_df.head(args.max_rows)
    
    # Get market prices and outcomes
    if 'market_prob' in base_df.columns:
        q = base_df['market_prob'].values.astype(float)
    elif 'final_prob' in base_df.columns:
        q = base_df['final_prob'].values.astype(float)
    else:
        raise ValueError("No market price column found")
    
    y = base_df['y'].values.astype(float)
    n = len(y)
    print(f"  Loaded {n} markets, outcome rate: {y.mean():.1%}")
    
    results = {}
    
    # Market baseline (random predictions)
    print("\nEvaluating Market Baseline (random)...")
    rng = np.random.default_rng(args.seed)
    p_random = rng.uniform(0.3, 0.7, n)
    results["Random"] = evaluate_model("Random", p_random, q, y, args.tc, args.n_bootstrap)
    
    # Market as prediction (copy market)
    print("Evaluating Market Copy...")
    results["Market Copy"] = evaluate_model("Market Copy", q, q, y, args.tc, args.n_bootstrap)
    
    # RLCR
    if args.rlcr_preds:
        print(f"\nLoading RLCR predictions from {args.rlcr_preds}...")
        try:
            rlcr_df = pd.read_parquet(args.rlcr_preds)
            if len(rlcr_df) == n:
                p_rlcr = rlcr_df['pred_prob'].values.astype(float)
            else:
                # Align by index or market_id
                p_rlcr = rlcr_df['pred_prob'].values[:n].astype(float)
            results["RLCR"] = evaluate_model("RLCR", p_rlcr, q[:len(p_rlcr)], y[:len(p_rlcr)], args.tc, args.n_bootstrap)
            print(f"  Loaded {len(p_rlcr)} predictions")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # AR Baseline
    if args.ar_preds:
        print(f"\nLoading AR Baseline predictions from {args.ar_preds}...")
        try:
            ar_df = pd.read_parquet(args.ar_preds)
            p_ar = ar_df['pred_prob'].values[:n].astype(float)
            results["AR Baseline"] = evaluate_model("AR Baseline", p_ar, q[:len(p_ar)], y[:len(p_ar)], args.tc, args.n_bootstrap)
            print(f"  Loaded {len(p_ar)} predictions")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Diffusion
    if args.diff_preds:
        print(f"\nLoading Diffusion predictions from {args.diff_preds}...")
        try:
            diff_df = pd.read_parquet(args.diff_preds)
            p_diff = diff_df['pred_prob'].values[:n].astype(float)
            results["Diffusion"] = evaluate_model("Diffusion", p_diff, q[:len(p_diff)], y[:len(p_diff)], args.tc, args.n_bootstrap)
            print(f"  Loaded {len(p_diff)} predictions")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output = {
        "config": {
            "data": args.data,
            "n_markets": n,
            "tc": args.tc,
            "n_bootstrap": args.n_bootstrap,
        },
        "results": {name: asdict(m) for name, m in results.items()},
        "timestamp": datetime.now().isoformat(),
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best = max(results.values(), key=lambda x: x.sharpe)
    print(f"""
Best Model: {best.name}
  Sharpe: {best.sharpe:.4f} (Ann: {best.ann_sharpe:.2f})
  PnL/Trade: {best.mean_pnl:+.4f}
  Win Rate: {best.win_rate:.1%}
  Brier: {best.brier:.4f}
  
Confidence: [{best.sharpe_ci_low:.4f}, {best.sharpe_ci_high:.4f}]
""")


if __name__ == "__main__":
    main()
