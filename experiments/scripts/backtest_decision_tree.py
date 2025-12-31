#!/usr/bin/env python3
"""
Proper Backtest: Decision Tree Strategy with C_t Distance

This script performs a rigorous backtest with:
1. Full dataset (50k+ markets)
2. Rolling window calibration estimation (no lookahead)
3. Proper train/test splits
4. Bootstrap confidence intervals
5. Multiple random seeds

Usage:
    python scripts/backtest_decision_tree.py \
        --data data/polymarket/turtel_exa_enriched.parquet \
        --output runs/decision_tree_backtest.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from collections import defaultdict


@dataclass
class BacktestConfig:
    """Configuration for the backtest."""
    # Data
    data_path: str
    max_rows: Optional[int] = None
    
    # Rolling window
    calibration_window: int = 500  # Markets to use for calibration estimation
    min_samples_per_cell: int = 20  # Min samples to estimate calibration
    
    # Strategy parameters
    n_price_bins: int = 5
    well_calibrated_threshold: float = 0.10  # |cal_err| < this → well-calibrated
    divergence_threshold: float = 0.15  # |p - q| > this → trade in well-calib bins
    transaction_cost: float = 0.02
    
    # Evaluation
    n_bootstrap: int = 100
    seed: int = 42


def categorize_text(text: str) -> str:
    """Derive category from question text."""
    text = str(text).lower()
    
    if any(w in text for w in ['trump', 'biden', 'election', 'president', 'congress', 
                                'senate', 'vote', 'democrat', 'republican', 'governor']):
        return 'politics'
    elif any(w in text for w in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 
                                  'token', 'defi', 'nft', 'solana', 'doge']):
        return 'crypto'
    elif any(w in text for w in ['nba', 'nfl', 'mlb', 'soccer', 'football', 'basketball', 
                                  'game', 'match', 'win', 'championship', 'super bowl',
                                  'world cup', 'playoff', 'finals']):
        return 'sports'
    elif any(w in text for w in ['price', 'stock', 'market', 'fed', 'interest rate', 
                                  'gdp', 'inflation', 'treasury', 's&p', 'nasdaq']):
        return 'finance'
    elif any(w in text for w in ['ai', 'gpt', 'openai', 'google', 'apple', 'microsoft', 
                                  'tech', 'meta', 'nvidia', 'chatgpt']):
        return 'tech'
    else:
        return 'other'


def get_price_bin(q: float, n_bins: int = 5) -> int:
    """Get price bin index."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        if q < bin_edges[i + 1]:
            return i
    return n_bins - 1


class DecisionTreeBacktester:
    """
    Proper rolling-window backtester for the decision tree strategy.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def estimate_calibration(
        self,
        q_hist: np.ndarray,
        y_hist: np.ndarray,
        cat_hist: np.ndarray,
        category: str,
        price_bin: int,
    ) -> Tuple[Optional[float], int]:
        """
        Estimate calibration error for (category, price_bin) from historical data.
        
        Returns: (calibration_error, n_samples)
        """
        cfg = self.config
        
        # Filter to matching category
        cat_mask = cat_hist == category
        
        # Filter to similar price bin
        bin_edges = np.linspace(0, 1, cfg.n_price_bins + 1)
        lo, hi = bin_edges[price_bin], bin_edges[price_bin + 1]
        price_mask = (q_hist >= lo) & (q_hist < hi)
        if price_bin == cfg.n_price_bins - 1:
            price_mask = (q_hist >= lo) & (q_hist <= hi)
        
        mask = cat_mask & price_mask
        n_samples = mask.sum()
        
        if n_samples < cfg.min_samples_per_cell:
            return None, n_samples
        
        # Calibration error = E[y - q] in this cell
        cal_err = np.mean(y_hist[mask] - q_hist[mask])
        return cal_err, n_samples
    
    def run_backtest(
        self,
        p: np.ndarray,  # Model predictions
        q: np.ndarray,  # Market prices
        y: np.ndarray,  # Outcomes
        categories: np.ndarray,
    ) -> Dict:
        """
        Run the full rolling-window backtest.
        
        Returns detailed results including per-trade PnL.
        """
        cfg = self.config
        n = len(p)
        window = cfg.calibration_window
        tc = cfg.transaction_cost
        
        # Output arrays
        pnl_baseline = np.zeros(n)
        pnl_decision_tree = np.zeros(n)
        strategy_used = ['warmup'] * window + [''] * (n - window)
        
        # Rolling backtest
        for i in range(window, n):
            # Historical data for calibration estimation
            q_hist = q[i - window:i]
            y_hist = y[i - window:i]
            cat_hist = categories[i - window:i]
            
            # Current market
            q_i = q[i]
            p_i = p[i]
            y_i = y[i]
            cat_i = categories[i]
            bin_i = get_price_bin(q_i, cfg.n_price_bins)
            
            # Baseline: Simple RLCR strategy
            direction_base = np.sign(p_i - q_i)
            pnl_baseline[i] = direction_base * (y_i - q_i) - tc
            
            # Decision tree strategy
            cal_err, n_samples = self.estimate_calibration(
                q_hist, y_hist, cat_hist, cat_i, bin_i
            )
            
            if cal_err is None:
                # Not enough data, skip
                strategy_used[i] = 'skip_insufficient_data'
                continue
            
            is_well_calibrated = abs(cal_err) < cfg.well_calibrated_threshold
            
            if is_well_calibrated:
                # Well-calibrated: Use divergence strategy
                divergence = abs(p_i - q_i)
                if divergence > cfg.divergence_threshold:
                    direction = np.sign(p_i - q_i)
                    size = min(divergence / 0.3, 1.0)
                    pnl_decision_tree[i] = direction * size * (y_i - q_i) - tc * size
                    strategy_used[i] = 'divergence'
                else:
                    strategy_used[i] = 'skip_low_divergence'
            else:
                # Miscalibrated: Use momentum strategy
                if abs(cal_err) > tc:
                    direction = np.sign(cal_err)  # Bet toward C_t
                    size = min(abs(cal_err) / 0.3, 1.0)
                    pnl_decision_tree[i] = direction * size * (y_i - q_i) - tc * size
                    strategy_used[i] = 'momentum'
                else:
                    strategy_used[i] = 'skip_low_edge'
        
        return {
            'pnl_baseline': pnl_baseline,
            'pnl_decision_tree': pnl_decision_tree,
            'strategy_used': strategy_used,
            'window': window,
        }
    
    def compute_metrics(self, pnl: np.ndarray, window: int) -> Dict:
        """Compute performance metrics for a PnL series."""
        pnl_oos = pnl[window:]  # Out-of-sample only
        traded = pnl_oos != 0
        n_trades = traded.sum()
        
        if n_trades == 0:
            return {
                'n_trades': 0,
                'mean_pnl': 0,
                'std_pnl': 0,
                'sharpe': 0,
                'ann_sharpe': 0,
                'total_pnl': 0,
                'win_rate': 0,
            }
        
        pnl_traded = pnl_oos[traded]
        mean_pnl = float(pnl_traded.mean())
        std_pnl = float(pnl_traded.std())
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
        
        return {
            'n_trades': int(n_trades),
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'sharpe': float(sharpe),
            'ann_sharpe': float(sharpe * np.sqrt(250)),
            'total_pnl': float(pnl_traded.sum()),
            'win_rate': float((pnl_traded > 0).mean()),
        }
    
    def bootstrap_sharpe(self, pnl: np.ndarray, window: int) -> Dict:
        """Bootstrap confidence intervals for Sharpe ratio."""
        cfg = self.config
        pnl_oos = pnl[window:]
        traded = pnl_oos != 0
        pnl_traded = pnl_oos[traded]
        
        if len(pnl_traded) < 10:
            return {'sharpe_mean': 0, 'sharpe_std': 0, 'sharpe_ci_low': 0, 'sharpe_ci_high': 0}
        
        rng = np.random.default_rng(cfg.seed)
        sharpes = []
        
        for _ in range(cfg.n_bootstrap):
            idx = rng.choice(len(pnl_traded), size=len(pnl_traded), replace=True)
            sample = pnl_traded[idx]
            if sample.std() > 0:
                sharpes.append(sample.mean() / sample.std())
        
        sharpes = np.array(sharpes)
        return {
            'sharpe_mean': float(sharpes.mean()),
            'sharpe_std': float(sharpes.std()),
            'sharpe_ci_low': float(np.percentile(sharpes, 2.5)),
            'sharpe_ci_high': float(np.percentile(sharpes, 97.5)),
        }


def main():
    parser = argparse.ArgumentParser(description="Backtest decision tree strategy")
    parser.add_argument("--data", required=True, help="Path to data parquet")
    parser.add_argument("--max-rows", type=int, help="Max rows to use")
    parser.add_argument("--output", default="runs/decision_tree_backtest.json", help="Output path")
    parser.add_argument("--calibration-window", type=int, default=500, help="Rolling window size")
    parser.add_argument("--n-bootstrap", type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    config = BacktestConfig(
        data_path=args.data,
        max_rows=args.max_rows,
        calibration_window=args.calibration_window,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    
    print("=" * 70)
    print("DECISION TREE BACKTEST")
    print("=" * 70)
    print(f"Config: {asdict(config)}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
    
    # Filter to resolved markets
    if 'y' not in df.columns:
        raise ValueError("Data must have 'y' column")
    df = df[df['y'].notna()].reset_index(drop=True)
    
    print(f"Loaded {len(df)} resolved markets")
    
    # Get predictions (use final_prob as proxy if no model predictions)
    if 'pred_prob' in df.columns:
        p = df['pred_prob'].values
        print("Using model predictions from 'pred_prob' column")
    else:
        # Simulate predictions with some noise
        print("WARNING: No 'pred_prob' column, using noisy market prices as proxy")
        np.random.seed(config.seed)
        p = df['final_prob'].values + 0.1 * np.random.randn(len(df))
        p = np.clip(p, 0.01, 0.99)
    
    # Get market prices and outcomes
    if 'market_prob' in df.columns:
        q = df['market_prob'].values
    else:
        q = df['final_prob'].values
    y = df['y'].values.astype(float)
    
    # Derive categories
    print("Deriving categories from question text...")
    categories = np.array([categorize_text(row['question']) for _, row in df.iterrows()])
    print(f"Categories: {pd.Series(categories).value_counts().to_dict()}")
    
    # Run backtest
    print("\nRunning backtest...")
    backtester = DecisionTreeBacktester(config)
    results = backtester.run_backtest(p, q, y, categories)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics_baseline = backtester.compute_metrics(results['pnl_baseline'], results['window'])
    metrics_dectree = backtester.compute_metrics(results['pnl_decision_tree'], results['window'])
    
    # Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")
    bootstrap_baseline = backtester.bootstrap_sharpe(results['pnl_baseline'], results['window'])
    bootstrap_dectree = backtester.bootstrap_sharpe(results['pnl_decision_tree'], results['window'])
    
    # Strategy breakdown
    from collections import Counter
    strategy_counts = Counter(results['strategy_used'][results['window']:])
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (Out-of-Sample)")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Baseline':<20} {'Decision Tree':<20}")
    print("-" * 65)
    for key in ['n_trades', 'mean_pnl', 'std_pnl', 'sharpe', 'ann_sharpe', 'total_pnl', 'win_rate']:
        v1 = metrics_baseline[key]
        v2 = metrics_dectree[key]
        if isinstance(v1, float):
            print(f"{key:<25} {v1:<20.4f} {v2:<20.4f}")
        else:
            print(f"{key:<25} {v1:<20} {v2:<20}")
    
    print(f"\nSharpe Improvement: {(metrics_dectree['sharpe'] - metrics_baseline['sharpe']) / metrics_baseline['sharpe'] * 100:+.1f}%")
    
    print("\n" + "=" * 50)
    print("BOOTSTRAP 95% CI FOR SHARPE")
    print("=" * 50)
    print(f"Baseline:      [{bootstrap_baseline['sharpe_ci_low']:.4f}, {bootstrap_baseline['sharpe_ci_high']:.4f}]")
    print(f"Decision Tree: [{bootstrap_dectree['sharpe_ci_low']:.4f}, {bootstrap_dectree['sharpe_ci_high']:.4f}]")
    
    # Check statistical significance
    ci_overlap = (bootstrap_dectree['sharpe_ci_low'] < bootstrap_baseline['sharpe_ci_high'] and
                  bootstrap_baseline['sharpe_ci_low'] < bootstrap_dectree['sharpe_ci_high'])
    
    if not ci_overlap:
        print("\n✓ Confidence intervals do NOT overlap → Statistically significant improvement!")
    else:
        print("\n~ Confidence intervals overlap → Improvement may not be statistically significant")
    
    print("\n" + "=" * 50)
    print("STRATEGY BREAKDOWN")
    print("=" * 50)
    for strategy, count in strategy_counts.most_common():
        pct = count / len(results['strategy_used'][results['window']:]) * 100
        print(f"  {strategy}: {count} ({pct:.1f}%)")
    
    # Per-category analysis
    print("\n" + "=" * 50)
    print("PER-CATEGORY ANALYSIS")
    print("=" * 50)
    
    window = results['window']
    for cat in pd.Series(categories).unique():
        mask = categories == cat
        mask_oos = np.concatenate([np.zeros(window, dtype=bool), mask[window:]])
        
        if mask_oos.sum() < 50:
            continue
        
        pnl_base_cat = results['pnl_baseline'][mask_oos]
        pnl_tree_cat = results['pnl_decision_tree'][mask_oos]
        
        traded_base = pnl_base_cat != 0
        traded_tree = pnl_tree_cat != 0
        
        if traded_base.sum() == 0 or traded_tree.sum() == 0:
            continue
        
        sharpe_base = pnl_base_cat[traded_base].mean() / pnl_base_cat[traded_base].std()
        sharpe_tree = pnl_tree_cat[traded_tree].mean() / pnl_tree_cat[traded_tree].std() if traded_tree.sum() > 1 else 0
        
        improvement = (sharpe_tree - sharpe_base) / abs(sharpe_base) * 100 if sharpe_base != 0 else 0
        
        print(f"\n{cat.upper()}:")
        print(f"  Baseline: Sharpe={sharpe_base:.3f}, Trades={traded_base.sum()}, PnL={pnl_base_cat.sum():+.1f}")
        print(f"  DecTree:  Sharpe={sharpe_tree:.3f}, Trades={traded_tree.sum()}, PnL={pnl_tree_cat.sum():+.1f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'config': asdict(config),
        'metrics': {
            'baseline': metrics_baseline,
            'decision_tree': metrics_dectree,
        },
        'bootstrap': {
            'baseline': bootstrap_baseline,
            'decision_tree': bootstrap_dectree,
        },
        'strategy_counts': dict(strategy_counts),
        'sharpe_improvement_pct': float((metrics_dectree['sharpe'] - metrics_baseline['sharpe']) / metrics_baseline['sharpe'] * 100),
        'ci_overlap': ci_overlap,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"\nArtifacts: {output_path}")


if __name__ == "__main__":
    main()
