#!/usr/bin/env python
"""
Validate Blackwell Calibration Strategy on Real Data.

This script runs comprehensive validation including:
1. Rolling backtest (no lookahead bias)
2. Cross-validation (k-fold)
3. Bootstrap confidence intervals
4. Statistical significance tests
5. Comparison to baseline

Usage:
    python scripts/validate_blackwell_strategy.py --data data/polymarket/pm_horizon_24h.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.strategies.blackwell_calibration import (
    BlackwellCalibrationConfig,
    BlackwellCalibrationStrategy,
)


@dataclass
class ValidationResult:
    """Results from validation."""
    
    # Rolling backtest
    rolling_n_trades: int = 0
    rolling_net_pnl: float = 0.0
    rolling_return_pct: float = 0.0
    rolling_sharpe: float = 0.0
    rolling_win_rate: float = 0.0
    
    # Cross-validation
    cv_mean_return: float = 0.0
    cv_std_return: float = 0.0
    cv_folds: List[Dict] = None
    
    # Bootstrap CIs
    bootstrap_pnl_ci: tuple = (0.0, 0.0)
    bootstrap_return_ci: tuple = (0.0, 0.0)
    bootstrap_sharpe_ci: tuple = (0.0, 0.0)
    
    # Statistical tests
    pnl_t_stat: float = 0.0
    pnl_p_value: float = 1.0
    is_significant: bool = False
    
    # Comparison to baseline
    insample_return: float = 0.0
    capture_ratio: float = 0.0
    
    def __post_init__(self):
        if self.cv_folds is None:
            self.cv_folds = []


def run_rolling_backtest(
    prices: np.ndarray,
    outcomes: np.ndarray,
    cfg: BlackwellCalibrationConfig,
    warmup_frac: float = 0.2,
    tx_cost: float = 0.02,
) -> Dict:
    """Run rolling backtest with no lookahead bias."""
    strategy = BlackwellCalibrationStrategy(cfg)
    n = len(prices)
    warmup = int(n * warmup_frac)
    
    # Warmup phase
    for i in range(warmup):
        strategy.on_resolution(f"m_{i}", float(outcomes[i]), float(prices[i]))
    strategy._recalibrate()
    
    # Trading phase
    positions = np.zeros(n)
    for i in range(warmup, n):
        pos = strategy.get_position(f"m_{i}", float(prices[i]))
        positions[i] = pos
        strategy.on_resolution(f"m_{i}", float(outcomes[i]), float(prices[i]))
    
    # Calculate metrics
    traded = positions != 0
    if traded.sum() == 0:
        return {
            "n_trades": 0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "pnl_per_trade": [],
        }
    
    pnl = positions[traded] * (outcomes[traded] - prices[traded])
    tx = np.abs(positions[traded]).sum() * tx_cost
    net = pnl.sum() - tx
    
    capital = np.where(positions > 0, prices, 1 - prices) * np.abs(positions)
    total_cap = capital[traded].sum()
    
    return {
        "n_trades": int(traded.sum()),
        "net_pnl": float(net),
        "return_pct": float(net / total_cap * 100) if total_cap > 0 else 0.0,
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(traded.sum())) if pnl.std() > 0 else 0.0,
        "win_rate": float((pnl > 0).mean()),
        "pnl_per_trade": pnl.tolist(),
    }


def run_cross_validation(
    prices: np.ndarray,
    outcomes: np.ndarray,
    cfg: BlackwellCalibrationConfig,
    k: int = 5,
    tx_cost: float = 0.02,
) -> List[Dict]:
    """Run k-fold cross-validation."""
    n = len(prices)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    results = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Train on train set
        strategy = BlackwellCalibrationStrategy(cfg)
        for idx in train_idx:
            strategy.on_resolution(f"m_{idx}", float(outcomes[idx]), float(prices[idx]))
        strategy._recalibrate()
        
        # Test on test set
        positions = np.array([
            strategy.get_position(f"m_{idx}", float(prices[idx]))
            for idx in test_idx
        ])
        
        traded = positions != 0
        if traded.sum() == 0:
            results.append({"fold": i, "n_trades": 0, "return_pct": 0.0})
            continue
        
        test_outcomes = outcomes[test_idx]
        test_prices = prices[test_idx]
        pnl = positions[traded] * (test_outcomes[traded] - test_prices[traded])
        tx = np.abs(positions[traded]).sum() * tx_cost
        net = pnl.sum() - tx
        
        capital = np.where(positions > 0, test_prices, 1 - test_prices) * np.abs(positions)
        total_cap = capital[traded].sum()
        
        results.append({
            "fold": i,
            "n_trades": int(traded.sum()),
            "return_pct": float(net / total_cap * 100) if total_cap > 0 else 0.0,
            "net_pnl": float(net),
        })
    
    return results


def run_bootstrap(
    prices: np.ndarray,
    outcomes: np.ndarray,
    cfg: BlackwellCalibrationConfig,
    n_bootstrap: int = 500,
    tx_cost: float = 0.02,
) -> Dict:
    """Run bootstrap to get confidence intervals."""
    n = len(prices)
    pnl_samples = []
    return_samples = []
    sharpe_samples = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        result = run_rolling_backtest(
            prices[idx],
            outcomes[idx],
            cfg,
            tx_cost=tx_cost,
        )
        pnl_samples.append(result["net_pnl"])
        return_samples.append(result["return_pct"])
        sharpe_samples.append(result["sharpe"])
    
    return {
        "pnl_ci": (np.percentile(pnl_samples, 2.5), np.percentile(pnl_samples, 97.5)),
        "return_ci": (np.percentile(return_samples, 2.5), np.percentile(return_samples, 97.5)),
        "sharpe_ci": (np.percentile(sharpe_samples, 2.5), np.percentile(sharpe_samples, 97.5)),
    }


def run_insample_baseline(
    prices: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    g_threshold: float = 0.05,
    tx_cost: float = 0.02,
) -> float:
    """Run in-sample baseline (with lookahead bias) for comparison."""
    bins = np.digitize(prices, np.linspace(0, 1, n_bins + 1)) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    
    # Compute g_bar for each bin (using all data = lookahead)
    bin_info = {}
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() >= 20:
            g_bar = (outcomes[mask] - prices[mask]).mean()
            if abs(g_bar) > g_threshold:
                bin_info[b] = g_bar
    
    # Trade
    positions = np.zeros(len(prices))
    for b, g_bar in bin_info.items():
        mask = bins == b
        positions[mask] = np.sign(g_bar)
    
    traded = positions != 0
    if traded.sum() == 0:
        return 0.0
    
    pnl = (positions * (outcomes - prices))[traded]
    tx = traded.sum() * tx_cost
    net = pnl.sum() - tx
    
    capital = np.where(positions > 0, prices, 1 - prices)[traded].sum()
    return float(net / capital * 100) if capital > 0 else 0.0


def validate_strategy(
    data_path: Path,
    cfg: Optional[BlackwellCalibrationConfig] = None,
    verbose: bool = True,
) -> ValidationResult:
    """
    Run full validation suite on the Blackwell calibration strategy.
    
    Args:
        data_path: Path to parquet file with market data
        cfg: Strategy configuration (uses defaults if None)
        verbose: Print progress
        
    Returns:
        ValidationResult with all metrics
    """
    if cfg is None:
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            g_bar_threshold=0.05,
            t_stat_threshold=0.0,
            lookback_trades=2000,
            recalibrate_freq=100,
            use_risk_parity=True,
            target_max_loss=0.2,
            leverage=1.5,
        )
    
    if verbose:
        print("=" * 70)
        print("BLACKWELL STRATEGY VALIDATION")
        print("=" * 70)
    
    # Load data
    df = pd.read_parquet(data_path)
    prices = df["market_prob"].values
    outcomes = df["y"].values
    n = len(prices)
    
    if verbose:
        print(f"\nData: {n} markets from {data_path}")
    
    result = ValidationResult()
    
    # 1. Rolling backtest
    if verbose:
        print("\n1. Running rolling backtest...")
    
    np.random.seed(42)
    rolling = run_rolling_backtest(prices, outcomes, cfg)
    result.rolling_n_trades = rolling["n_trades"]
    result.rolling_net_pnl = rolling["net_pnl"]
    result.rolling_return_pct = rolling["return_pct"]
    result.rolling_sharpe = rolling["sharpe"]
    result.rolling_win_rate = rolling["win_rate"]
    
    if verbose:
        print(f"   Trades: {result.rolling_n_trades}")
        print(f"   Net PnL: {result.rolling_net_pnl:.2f}")
        print(f"   Return: {result.rolling_return_pct:.1f}%")
        print(f"   Sharpe: {result.rolling_sharpe:.2f}")
    
    # 2. Cross-validation
    if verbose:
        print("\n2. Running 5-fold cross-validation...")
    
    cv_results = run_cross_validation(prices, outcomes, cfg, k=5)
    result.cv_folds = cv_results
    returns = [f["return_pct"] for f in cv_results]
    result.cv_mean_return = float(np.mean(returns))
    result.cv_std_return = float(np.std(returns))
    
    if verbose:
        print(f"   Mean return: {result.cv_mean_return:.1f}% ± {result.cv_std_return:.1f}%")
    
    # 3. Bootstrap confidence intervals
    if verbose:
        print("\n3. Running bootstrap (500 samples)...")
    
    bootstrap = run_bootstrap(prices, outcomes, cfg, n_bootstrap=500)
    result.bootstrap_pnl_ci = bootstrap["pnl_ci"]
    result.bootstrap_return_ci = bootstrap["return_ci"]
    result.bootstrap_sharpe_ci = bootstrap["sharpe_ci"]
    
    if verbose:
        print(f"   Return 95% CI: [{result.bootstrap_return_ci[0]:.1f}%, {result.bootstrap_return_ci[1]:.1f}%]")
        print(f"   Sharpe 95% CI: [{result.bootstrap_sharpe_ci[0]:.2f}, {result.bootstrap_sharpe_ci[1]:.2f}]")
    
    # 4. Statistical significance
    if verbose:
        print("\n4. Testing statistical significance...")
    
    if rolling["pnl_per_trade"]:
        pnl_per_trade = np.array(rolling["pnl_per_trade"])
        t_stat, p_value = stats.ttest_1samp(pnl_per_trade, 0)
        result.pnl_t_stat = float(t_stat)
        result.pnl_p_value = float(p_value)
        result.is_significant = p_value < 0.05
        
        if verbose:
            sig = "YES" if result.is_significant else "NO"
            print(f"   t-statistic: {result.pnl_t_stat:.2f}")
            print(f"   p-value: {result.pnl_p_value:.4f}")
            print(f"   Significant (p<0.05): {sig}")
    
    # 5. Comparison to in-sample baseline
    if verbose:
        print("\n5. Comparing to in-sample baseline...")
    
    result.insample_return = run_insample_baseline(prices, outcomes)
    if result.insample_return != 0:
        result.capture_ratio = result.rolling_return_pct / result.insample_return
    
    if verbose:
        print(f"   In-sample return: {result.insample_return:.1f}%")
        print(f"   Rolling return: {result.rolling_return_pct:.1f}%")
        print(f"   Capture ratio: {result.capture_ratio:.0%}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"""
   ✓ Rolling backtest (no lookahead):
      Return: {result.rolling_return_pct:.1f}%
      Sharpe: {result.rolling_sharpe:.2f}
      Win rate: {result.rolling_win_rate:.1%}
   
   ✓ Cross-validation:
      Mean return: {result.cv_mean_return:.1f}% ± {result.cv_std_return:.1f}%
   
   ✓ Bootstrap 95% CI:
      Return: [{result.bootstrap_return_ci[0]:.1f}%, {result.bootstrap_return_ci[1]:.1f}%]
      Sharpe: [{result.bootstrap_sharpe_ci[0]:.2f}, {result.bootstrap_sharpe_ci[1]:.2f}]
   
   ✓ Statistical significance:
      p-value: {result.pnl_p_value:.4f}
      Significant: {"YES" if result.is_significant else "NO"}
   
   ✓ Capture ratio: {result.capture_ratio:.0%} of theoretical edge
   
   VERDICT: {"VALIDATED ✓" if result.is_significant and result.rolling_return_pct > 0 else "NOT VALIDATED ✗"}
""")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Blackwell calibration strategy")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/polymarket/pm_horizon_24h.parquet"),
        help="Path to market data parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of price bins",
    )
    parser.add_argument(
        "--g-threshold",
        type=float,
        default=0.05,
        help="Minimum g-bar threshold",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=2000,
        help="Rolling lookback window",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.5,
        help="Position leverage",
    )
    
    args = parser.parse_args()
    
    cfg = BlackwellCalibrationConfig(
        n_bins=args.n_bins,
        g_bar_threshold=args.g_threshold,
        t_stat_threshold=0.0,
        lookback_trades=args.lookback,
        recalibrate_freq=100,
        use_risk_parity=True,
        target_max_loss=0.2,
        leverage=args.leverage,
    )
    
    result = validate_strategy(args.data, cfg)
    
    if args.output:
        output_dict = {
            "rolling": {
                "n_trades": int(result.rolling_n_trades),
                "net_pnl": float(result.rolling_net_pnl),
                "return_pct": float(result.rolling_return_pct),
                "sharpe": float(result.rolling_sharpe),
                "win_rate": float(result.rolling_win_rate),
            },
            "cross_validation": {
                "mean_return": float(result.cv_mean_return),
                "std_return": float(result.cv_std_return),
                "folds": result.cv_folds,
            },
            "bootstrap": {
                "pnl_ci": [float(x) for x in result.bootstrap_pnl_ci],
                "return_ci": [float(x) for x in result.bootstrap_return_ci],
                "sharpe_ci": [float(x) for x in result.bootstrap_sharpe_ci],
            },
            "significance": {
                "t_stat": float(result.pnl_t_stat),
                "p_value": float(result.pnl_p_value),
                "is_significant": bool(result.is_significant),
            },
            "comparison": {
                "insample_return": float(result.insample_return),
                "capture_ratio": float(result.capture_ratio),
            },
        }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

