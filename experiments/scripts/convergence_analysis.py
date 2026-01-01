#!/usr/bin/env python3
"""
Comprehensive Convergence/Funding Rate Analysis

This explores the "funding rate" style trade in prediction markets:
- The spread between expected outcome E[Y|q] and market price q
- How this spread evolves over time (time-to-expiry effects)
- Cross-market convergence (correlated markets should converge together)
- Optimal holding periods and position sizing
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# =============================================================================
# Convergence Analysis
# =============================================================================

def analyze_calibration_by_bin(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """
    Analyze calibration error by price bin.
    
    Returns DataFrame with:
    - bin_center: center of price bin
    - n_samples: count of markets in bin
    - expected_outcome: E[Y|q] (actual frequency of Y=1)
    - mean_price: average market price q
    - spread: E[Y|q] - q (the "funding rate")
    - spread_se: standard error of spread
    - t_stat: t-statistic for spread != 0
    """
    prices = df['first_price'].values
    outcomes = df['y'].values
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(prices, bin_edges) - 1, 0, n_bins - 1)
    
    results = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        
        if n < 10:
            continue
        
        p = prices[mask]
        y = outcomes[mask]
        
        expected_outcome = y.mean()
        mean_price = p.mean()
        spread = expected_outcome - mean_price
        
        # Standard error of spread
        residuals = y - p
        spread_se = residuals.std() / np.sqrt(n)
        t_stat = spread / spread_se if spread_se > 0 else 0
        
        results.append({
            'bin_center': (bin_edges[b] + bin_edges[b+1]) / 2,
            'n_samples': n,
            'expected_outcome': expected_outcome,
            'mean_price': mean_price,
            'spread': spread,
            'spread_se': spread_se,
            't_stat': t_stat,
        })
    
    return pd.DataFrame(results)


def analyze_spread_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze spread by market category."""
    results = []
    
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        n = len(cat_data)
        
        if n < 20:
            continue
        
        p = cat_data['first_price'].values
        y = cat_data['y'].values
        
        spread = (y - p).mean()
        spread_se = (y - p).std() / np.sqrt(n)
        t_stat = spread / spread_se if spread_se > 0 else 0
        
        results.append({
            'category': cat,
            'n_samples': n,
            'mean_price': p.mean(),
            'win_rate': y.mean(),
            'spread': spread,
            'spread_se': spread_se,
            't_stat': t_stat,
        })
    
    return pd.DataFrame(results).sort_values('t_stat', key=abs, ascending=False)


def analyze_time_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how spread changes with time-to-expiry.
    
    Hypothesis: Spread should be larger far from expiry, smaller near expiry
    (like funding rate that converges to zero at settlement)
    """
    if 'resolution_time' not in df.columns or 'first_trade_time' not in df.columns:
        return pd.DataFrame()
    
    # This would require trade timestamps which we don't have in aggregated data
    # For now, just analyze by resolution month
    if 'resolution_time' in df.columns:
        df = df.copy()
        df['resolution_month'] = pd.to_datetime(df['resolution_time']).dt.to_period('M')
        
        results = []
        for month, group in df.groupby('resolution_month'):
            if len(group) < 20:
                continue
            
            p = group['first_price'].values
            y = group['y'].values
            spread = (y - p).mean()
            
            results.append({
                'period': str(month),
                'n_samples': len(group),
                'spread': spread,
                'mean_price': p.mean(),
                'win_rate': y.mean(),
            })
        
        return pd.DataFrame(results)
    
    return pd.DataFrame()


# =============================================================================
# Enhanced Convergence Strategy
# =============================================================================

def convergence_strategy_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Spread parameters
    spread_threshold: float = 0.03,
    confidence_threshold: float = 2.0,       # t-stat threshold
    n_bins: int = 10,
    use_category_spreads: bool = True,
    category_weight: float = 0.5,            # Weight for category vs bin spread
    # Time decay
    decay_rate: float = 0.0,                 # Reduce position as markets age
    # Position sizing
    initial_bankroll: float = 10000.0,
    base_size_pct: float = 0.05,             # Base position size
    spread_scaling: float = 2.0,             # Scale position by spread magnitude
    max_position_pct: float = 0.25,
    # Risk management
    fee: float = 0.01,
    max_drawdown_stop: float = 0.50,
    max_daily_loss: float = 0.10,            # Stop if daily loss exceeds this
    # Diversification
    max_per_category: float = 0.30,          # Max exposure per category
) -> Dict[str, Any]:
    """
    Enhanced convergence strategy with multiple spread signals.
    """
    
    # Compute bin spreads from training
    bin_analysis = analyze_calibration_by_bin(train, n_bins=n_bins)
    bin_spreads = {
        row['bin_center']: {
            'spread': row['spread'],
            't_stat': row['t_stat'],
            'n': row['n_samples'],
        }
        for _, row in bin_analysis.iterrows()
    }
    
    # Compute category spreads
    category_analysis = analyze_spread_by_category(train)
    category_spreads = {
        row['category']: {
            'spread': row['spread'],
            't_stat': row['t_stat'],
            'n': row['n_samples'],
        }
        for _, row in category_analysis.iterrows()
    }
    
    # Trade on test set
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    daily_pnl = 0
    last_date = None
    pnls = []
    wins = 0
    trades_by_category = defaultdict(float)
    
    trade_log = []
    
    for _, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        cat = row.get('category', 'other')
        
        # Daily reset
        current_date = row.get('resolution_time')
        if current_date is not None and current_date != last_date:
            daily_pnl = 0
            last_date = current_date
        
        # Check stops
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        if daily_pnl < -max_daily_loss * bankroll:
            continue
        
        # Get bin spread signal
        bin_centers = list(bin_spreads.keys())
        if not bin_centers:
            continue
        
        closest_bin = min(bin_centers, key=lambda x: abs(x - price))
        bin_signal = bin_spreads.get(closest_bin, {'spread': 0, 't_stat': 0, 'n': 0})
        
        # Get category spread signal
        cat_signal = category_spreads.get(cat, {'spread': 0, 't_stat': 0, 'n': 0})
        
        # Combine signals
        if use_category_spreads and cat_signal['n'] >= 20:
            spread = (1 - category_weight) * bin_signal['spread'] + category_weight * cat_signal['spread']
            t_stat = (1 - category_weight) * bin_signal['t_stat'] + category_weight * cat_signal['t_stat']
        else:
            spread = bin_signal['spread']
            t_stat = bin_signal['t_stat']
        
        # Check if signal is strong enough
        if abs(spread) < spread_threshold or abs(t_stat) < confidence_threshold:
            continue
        
        # Check category exposure
        if trades_by_category[cat] >= max_per_category * bankroll:
            continue
        
        # Direction and sizing
        direction = 1 if spread > 0 else -1
        
        # Scale position by spread magnitude
        spread_scale = min(abs(spread) / spread_threshold, spread_scaling)
        position_frac = base_size_pct * spread_scale
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        position = bankroll * position_frac
        
        # Track exposure
        trades_by_category[cat] += position
        
        # Calculate PnL
        if direction > 0:
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price if price > 0.01 else 0
                wins += 1
            else:
                pnl = -position - position * fee / price if price > 0.01 else -position
        else:
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price) if price < 0.99 else 0
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price) if price < 0.99 else -position
        
        # Clamp extreme PnL
        pnl = np.clip(pnl, -position * 2, position * 10)
        
        pnls.append(pnl)
        bankroll += pnl
        daily_pnl += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        trade_log.append({
            'price': price,
            'direction': direction,
            'spread_signal': spread,
            't_stat': t_stat,
            'outcome': outcome,
            'pnl': pnl,
            'bankroll': bankroll,
        })
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {
            'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0,
            'max_dd': 0, 'final_bankroll': initial_bankroll, 'return_pct': 0,
        }
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
        'avg_trade_pnl': float(pnls.mean()),
        'trade_log': trade_log if len(trade_log) <= 100 else trade_log[:100],  # Sample
    }


# =============================================================================
# Cross-Market Convergence
# =============================================================================

def find_related_markets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find related markets that might move together.
    
    Examples:
    - "Trump wins" and "Republican wins"
    - "Bitcoin > 100k" and "Bitcoin > 90k"
    - Markets with same entities/themes
    """
    # Simple keyword-based grouping
    groups = defaultdict(list)
    
    for _, row in df.iterrows():
        q = str(row.get('question', '')).lower()
        market_id = str(row.get('goldsky_id', row.name))
        
        # Entity extraction
        if 'trump' in q:
            groups['trump'].append(market_id)
        if 'biden' in q:
            groups['biden'].append(market_id)
        if 'harris' in q:
            groups['harris'].append(market_id)
        if 'bitcoin' in q or 'btc' in q:
            groups['bitcoin'].append(market_id)
        if 'ethereum' in q or 'eth' in q:
            groups['ethereum'].append(market_id)
        if 'election' in q:
            groups['election'].append(market_id)
        if 'fed' in q or 'rate' in q:
            groups['fed_rates'].append(market_id)
    
    # Filter to groups with multiple markets
    return {k: v for k, v in groups.items() if len(v) >= 2}


# =============================================================================
# Optimization with Extended Parameters
# =============================================================================

def walk_forward_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    holdout_frac: float = 0.2,
) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], Optional[pd.DataFrame]]:
    """Generate walk-forward splits."""
    n = len(df)
    holdout_size = int(n * holdout_frac)
    cv_data = df.iloc[:-holdout_size].copy() if holdout_size > 0 else df.copy()
    holdout = df.iloc[-holdout_size:].copy() if holdout_size > 0 else None
    
    cv_n = len(cv_data)
    fold_size = cv_n // (n_folds + 1)
    
    splits = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        
        train = cv_data.iloc[:train_end]
        test = cv_data.iloc[train_end:test_end]
        
        if len(train) >= 500 and len(test) >= 100:
            splits.append((train, test))
    
    return splits, holdout


def optimize_convergence_v2(
    data: pd.DataFrame,
    n_trials: int = 1000,
    n_folds: int = 5,
    n_jobs: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Optimize convergence strategy with extended parameters."""
    
    splits, holdout = walk_forward_split(data, n_folds=n_folds)
    
    if not splits:
        return {'error': 'Not enough data for splits'}
    
    print(f"\n{'='*60}")
    print("CONVERGENCE STRATEGY V2 OPTIMIZATION")
    print(f"Dataset: {len(data):,} markets")
    print(f"CV folds: {len(splits)}")
    print(f"Trials: {n_trials}")
    print(f"{'='*60}\n")
    
    # First, print calibration analysis
    print("=== CALIBRATION ANALYSIS ===")
    bin_analysis = analyze_calibration_by_bin(data)
    print("\nBy Price Bin:")
    print(bin_analysis.to_string(index=False))
    
    cat_analysis = analyze_spread_by_category(data)
    print("\nBy Category (top 10 by |t-stat|):")
    print(cat_analysis.head(10).to_string(index=False))
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'spread_threshold': trial.suggest_float('spread_threshold', 0.01, 0.20),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 1.0, 4.0),
            'n_bins': trial.suggest_int('n_bins', 5, 25),
            'use_category_spreads': trial.suggest_categorical('use_category_spreads', [True, False]),
            'category_weight': trial.suggest_float('category_weight', 0.1, 0.9),
            'initial_bankroll': trial.suggest_float('initial_bankroll', 1000, 100000, log=True),
            'base_size_pct': trial.suggest_float('base_size_pct', 0.01, 0.15),
            'spread_scaling': trial.suggest_float('spread_scaling', 1.0, 5.0),
            'max_position_pct': trial.suggest_float('max_position_pct', 0.05, 0.40),
            'fee': trial.suggest_float('fee', 0.005, 0.02),
            'max_drawdown_stop': trial.suggest_float('max_drawdown_stop', 0.20, 0.70),
            'max_daily_loss': trial.suggest_float('max_daily_loss', 0.05, 0.30),
            'max_per_category': trial.suggest_float('max_per_category', 0.10, 0.50),
        }
        
        sharpes = []
        for train, test in splits:
            result = convergence_strategy_v2(train, test, **params)
            if result['n_trades'] >= 10:
                sharpes.append(result['sharpe'])
        
        return np.mean(sharpes) if sharpes else float('-inf')
    
    sampler = CmaEsSampler(n_startup_trials=50, warn_independent_sampling=False)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Best CV Sharpe: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Holdout evaluation
    if holdout is not None and len(holdout) > 0:
        train_all = data.iloc[:-len(holdout)]
        holdout_result = convergence_strategy_v2(train_all, holdout, **study.best_params)
        
        print(f"\n=== HOLDOUT RESULTS ===")
        print(f"  Sharpe: {holdout_result['sharpe']:.2f}")
        print(f"  PnL: ${holdout_result['pnl']:,.2f}")
        print(f"  Return: {holdout_result['return_pct']:.1f}%")
        print(f"  Trades: {holdout_result['n_trades']}")
        print(f"  Win Rate: {holdout_result['win_rate']:.1%}")
        print(f"  Max DD: {holdout_result['max_dd']:.1%}")
    else:
        holdout_result = None
    
    return {
        'cv_sharpe': study.best_value,
        'best_params': study.best_params,
        'holdout': holdout_result,
        'calibration_by_bin': bin_analysis.to_dict('records'),
        'calibration_by_category': cat_analysis.to_dict('records'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-path', type=str, default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, default='runs/convergence_v2')
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=32)
    parser.add_argument('--analyze-only', action='store_true', help='Only run analysis, no optimization')
    args = parser.parse_args()
    
    # Load data
    cache_path = Path(args.cache_path)
    print(f"Loading data from {cache_path}")
    data = pd.read_parquet(cache_path)
    print(f"Loaded {len(data):,} markets")
    
    if args.analyze_only:
        # Just run analysis
        print("\n" + "="*60)
        print("CALIBRATION ANALYSIS")
        print("="*60)
        
        print("\n--- By Price Bin ---")
        bin_analysis = analyze_calibration_by_bin(data)
        print(bin_analysis.to_string(index=False))
        
        print("\n--- By Category ---")
        cat_analysis = analyze_spread_by_category(data)
        print(cat_analysis.to_string(index=False))
        
        print("\n--- Related Market Groups ---")
        groups = find_related_markets(data)
        for group, markets in groups.items():
            print(f"  {group}: {len(markets)} markets")
        
        return
    
    # Run optimization
    results = optimize_convergence_v2(
        data=data,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        n_jobs=args.n_jobs,
        verbose=True,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj
        
        # Remove trade_log from holdout to keep file size small
        if results.get('holdout') and 'trade_log' in results['holdout']:
            del results['holdout']['trade_log']
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
