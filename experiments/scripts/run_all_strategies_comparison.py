#!/usr/bin/env python3
"""
Run ALL strategies on both Polymarket and Kalshi with realistic impact models.
Runs strategies in parallel for speed.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd
from dataclasses import dataclass

from stress_test_impact_models import (
    ImpactModelParams,
    create_almgren_chriss_base,
    compute_fill_probability,
    compute_price_impact,
    compute_adverse_selection_cost,
    estimate_liquidity,
)
from tail_risk_metrics import compute_tail_risk_metrics


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df = df.copy()
    if 'avg_price' in df.columns:
        df = df.rename(columns={'avg_price': 'price'})
    if 'y' in df.columns:
        df = df.rename(columns={'y': 'outcome'})
    df['price'] = df['price'].clip(0.01, 0.99)
    df['outcome'] = df['outcome'].astype(float)
    if 'category' not in df.columns:
        df['category'] = 'general'
    return df


def run_calibration_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    n_sims: int = 30,
    bankroll: float = 10000.0,
    spread_threshold: float = 0.05,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.10,
) -> Dict[str, Any]:
    """Calibration-based mean reversion strategy."""
    
    # Learn calibration
    train['price_bin'] = pd.cut(train['price'], bins=10, labels=False)
    cal = train.groupby('price_bin').agg({'price': 'mean', 'outcome': 'mean'})
    cal['spread'] = cal['outcome'] - cal['price']
    
    test = test.copy()
    test['price_bin'] = pd.cut(test['price'], bins=10, labels=False)
    test = test.merge(cal[['spread']], left_on='price_bin', right_index=True, how='left')
    test['spread'] = test['spread'].fillna(0)
    
    all_pnls = []
    n_trades = 0
    n_fills = 0
    
    for sim in range(n_sims):
        rng = np.random.default_rng(42 + sim)
        current_bankroll = bankroll
        
        for _, row in test.iterrows():
            price = row['price']
            spread = row.get('spread', 0)
            outcome = row.get('outcome', 0)
            category = str(row.get('category', ''))
            
            if abs(spread) < spread_threshold:
                continue
            
            edge = abs(spread)
            odds = (1 - price) / price if spread > 0 else price / (1 - price)
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * kelly_fraction
            
            size = current_bankroll * min(kelly, max_position_pct)
            size = min(size, current_bankroll * 0.5)
            
            if size < 1:
                continue
            
            n_trades += 1
            liquidity = estimate_liquidity(price, category, impact_params)
            fill_prob = compute_fill_probability(size, price, liquidity, impact_params, rng)
            
            if rng.random() >= fill_prob:
                continue
            
            n_fills += 1
            temp_imp, perm_imp = compute_price_impact(size, liquidity, impact_params)
            impact = (temp_imp + perm_imp) * size
            as_cost = compute_adverse_selection_cost(edge, 1.0, True, impact_params) * size
            
            raw_pnl = (outcome - price) * size if spread > 0 else ((1 - outcome) - (1 - price)) * size
            net_pnl = raw_pnl - impact - as_cost - (size * 0.01)
            all_pnls.append(net_pnl)
            current_bankroll += net_pnl
            
            if current_bankroll <= 0:
                break
    
    if len(all_pnls) < 10:
        return {'strategy': 'calibration_mean_reversion', 'error': 'too_few_trades'}
    
    pnls = np.array(all_pnls)
    metrics = compute_tail_risk_metrics(pnls)
    
    return {
        'strategy': 'calibration_mean_reversion',
        'sharpe': float(metrics.sharpe),
        'es_sharpe': float(metrics.es_sharpe_5pct),
        'sortino': float(metrics.sortino),
        'win_rate': float(metrics.win_rate),
        'n_trades': n_fills,
        'total_pnl': float(np.sum(pnls)),
        'fill_rate': n_fills / n_trades if n_trades > 0 else 0,
    }


def run_momentum_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    n_sims: int = 30,
    bankroll: float = 10000.0,
    momentum_threshold: float = 0.10,
    persistence_window: int = 20,
) -> Dict[str, Any]:
    """Momentum strategy based on persistent calibration errors."""
    
    # Learn calibration
    train['price_bin'] = pd.cut(train['price'], bins=10, labels=False)
    cal = train.groupby('price_bin').agg({'price': 'mean', 'outcome': 'mean'})
    cal['spread'] = cal['outcome'] - cal['price']
    
    test = test.copy()
    test['price_bin'] = pd.cut(test['price'], bins=10, labels=False)
    test = test.merge(cal[['spread']], left_on='price_bin', right_index=True, how='left')
    test['spread'] = test['spread'].fillna(0)
    
    # Rolling momentum signal
    test['rolling_spread'] = test['spread'].rolling(window=persistence_window, min_periods=5).mean()
    test['momentum_signal'] = test['rolling_spread'].abs() > momentum_threshold
    
    all_pnls = []
    n_trades = 0
    n_fills = 0
    
    for sim in range(n_sims):
        rng = np.random.default_rng(42 + sim)
        current_bankroll = bankroll
        
        for _, row in test.iterrows():
            if not row.get('momentum_signal', False):
                continue
            
            price = row['price']
            spread = row.get('rolling_spread', 0)
            outcome = row.get('outcome', 0)
            category = str(row.get('category', ''))
            
            edge = abs(spread)
            odds = (1 - price) / price if spread > 0 else price / (1 - price)
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * 0.25
            
            size = current_bankroll * min(kelly, 0.10)
            size = min(size, current_bankroll * 0.5)
            
            if size < 1:
                continue
            
            n_trades += 1
            liquidity = estimate_liquidity(price, category, impact_params)
            fill_prob = compute_fill_probability(size, price, liquidity, impact_params, rng)
            
            if rng.random() >= fill_prob:
                continue
            
            n_fills += 1
            temp_imp, perm_imp = compute_price_impact(size, liquidity, impact_params)
            impact = (temp_imp + perm_imp) * size
            as_cost = compute_adverse_selection_cost(edge, 1.0, True, impact_params) * size
            
            raw_pnl = (outcome - price) * size if spread > 0 else ((1 - outcome) - (1 - price)) * size
            net_pnl = raw_pnl - impact - as_cost - (size * 0.01)
            all_pnls.append(net_pnl)
            current_bankroll += net_pnl
            
            if current_bankroll <= 0:
                break
    
    if len(all_pnls) < 10:
        return {'strategy': 'momentum', 'error': 'too_few_trades'}
    
    pnls = np.array(all_pnls)
    metrics = compute_tail_risk_metrics(pnls)
    
    return {
        'strategy': 'momentum',
        'sharpe': float(metrics.sharpe),
        'es_sharpe': float(metrics.es_sharpe_5pct),
        'sortino': float(metrics.sortino),
        'win_rate': float(metrics.win_rate),
        'n_trades': n_fills,
        'total_pnl': float(np.sum(pnls)),
        'fill_rate': n_fills / n_trades if n_trades > 0 else 0,
    }


def run_stat_arb_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    n_sims: int = 30,
    bankroll: float = 10000.0,
) -> Dict[str, Any]:
    """Statistical arbitrage based on category-level calibration."""
    
    if 'category' not in train.columns:
        return {'strategy': 'stat_arb', 'error': 'no_category_data'}
    
    # Learn category-level calibration
    cat_cal = train.groupby('category').agg({
        'price': 'mean',
        'outcome': 'mean',
    })
    cat_cal['cat_spread'] = cat_cal['outcome'] - cat_cal['price']
    cat_cal = cat_cal[cat_cal.index.notna()]
    
    test = test.copy()
    test = test.merge(cat_cal[['cat_spread']], left_on='category', right_index=True, how='left')
    test['cat_spread'] = test['cat_spread'].fillna(0)
    
    all_pnls = []
    n_trades = 0
    n_fills = 0
    
    for sim in range(n_sims):
        rng = np.random.default_rng(42 + sim)
        current_bankroll = bankroll
        
        for _, row in test.iterrows():
            price = row['price']
            spread = row.get('cat_spread', 0)
            outcome = row.get('outcome', 0)
            category = str(row.get('category', ''))
            
            if abs(spread) < 0.05:
                continue
            
            edge = abs(spread)
            odds = (1 - price) / price if spread > 0 else price / (1 - price)
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * 0.25
            
            size = current_bankroll * min(kelly, 0.10)
            size = min(size, current_bankroll * 0.5)
            
            if size < 1:
                continue
            
            n_trades += 1
            liquidity = estimate_liquidity(price, category, impact_params)
            fill_prob = compute_fill_probability(size, price, liquidity, impact_params, rng)
            
            if rng.random() >= fill_prob:
                continue
            
            n_fills += 1
            temp_imp, perm_imp = compute_price_impact(size, liquidity, impact_params)
            impact = (temp_imp + perm_imp) * size
            as_cost = compute_adverse_selection_cost(edge, 1.0, True, impact_params) * size
            
            raw_pnl = (outcome - price) * size if spread > 0 else ((1 - outcome) - (1 - price)) * size
            net_pnl = raw_pnl - impact - as_cost - (size * 0.01)
            all_pnls.append(net_pnl)
            current_bankroll += net_pnl
            
            if current_bankroll <= 0:
                break
    
    if len(all_pnls) < 10:
        return {'strategy': 'stat_arb', 'error': 'too_few_trades'}
    
    pnls = np.array(all_pnls)
    metrics = compute_tail_risk_metrics(pnls)
    
    return {
        'strategy': 'stat_arb',
        'sharpe': float(metrics.sharpe),
        'es_sharpe': float(metrics.es_sharpe_5pct),
        'sortino': float(metrics.sortino),
        'win_rate': float(metrics.win_rate),
        'n_trades': n_fills,
        'total_pnl': float(np.sum(pnls)),
        'fill_rate': n_fills / n_trades if n_trades > 0 else 0,
    }


def run_longshot_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    n_sims: int = 30,
    bankroll: float = 10000.0,
    price_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Longshot strategy - bet YES on underpriced low-probability events."""
    
    # Learn calibration for low-price bin
    low_train = train[train['price'] < price_threshold]
    if len(low_train) < 10:
        return {'strategy': 'longshot', 'error': 'insufficient_low_price_data'}
    
    # Expected edge on longshots
    longshot_edge = low_train['outcome'].mean() - low_train['price'].mean()
    
    all_pnls = []
    n_trades = 0
    n_fills = 0
    
    for sim in range(n_sims):
        rng = np.random.default_rng(42 + sim)
        current_bankroll = bankroll
        
        for _, row in test.iterrows():
            price = row['price']
            outcome = row.get('outcome', 0)
            category = str(row.get('category', ''))
            
            # Only trade longshots
            if price >= price_threshold:
                continue
            
            # Bet YES on longshots (they're underpriced on Kalshi)
            edge = longshot_edge
            if edge <= 0:
                continue
            
            odds = (1 - price) / price
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * 0.15  # Conservative kelly
            
            size = current_bankroll * min(kelly, 0.05)  # Small positions
            size = min(size, current_bankroll * 0.3)
            
            if size < 1:
                continue
            
            n_trades += 1
            liquidity = estimate_liquidity(price, category, impact_params)
            fill_prob = compute_fill_probability(size, price, liquidity, impact_params, rng)
            
            if rng.random() >= fill_prob:
                continue
            
            n_fills += 1
            temp_imp, perm_imp = compute_price_impact(size, liquidity, impact_params)
            impact = (temp_imp + perm_imp) * size
            as_cost = compute_adverse_selection_cost(edge, 1.0, True, impact_params) * size
            
            # Bet YES
            raw_pnl = (outcome - price) * size
            net_pnl = raw_pnl - impact - as_cost - (size * 0.01)
            all_pnls.append(net_pnl)
            current_bankroll += net_pnl
            
            if current_bankroll <= 0:
                break
    
    if len(all_pnls) < 10:
        return {'strategy': 'longshot', 'error': 'too_few_trades'}
    
    pnls = np.array(all_pnls)
    metrics = compute_tail_risk_metrics(pnls)
    
    return {
        'strategy': 'longshot',
        'sharpe': float(metrics.sharpe),
        'es_sharpe': float(metrics.es_sharpe_5pct),
        'sortino': float(metrics.sortino),
        'win_rate': float(metrics.win_rate),
        'n_trades': n_fills,
        'total_pnl': float(np.sum(pnls)),
        'fill_rate': n_fills / n_trades if n_trades > 0 else 0,
        'longshot_edge': float(longshot_edge),
    }


def run_single_strategy(args_tuple: Tuple) -> Dict[str, Any]:
    """Worker function for parallel execution."""
    strat_name, strat_func_name, platform_name, train_path, test_path, n_sims = args_tuple
    
    # Load data in worker
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    # Map function name to actual function
    func_map = {
        'run_calibration_strategy': run_calibration_strategy,
        'run_momentum_strategy': run_momentum_strategy,
        'run_stat_arb_strategy': run_stat_arb_strategy,
        'run_longshot_strategy': run_longshot_strategy,
    }
    strat_func = func_map[strat_func_name]
    
    impact_params = create_almgren_chriss_base()
    result = strat_func(train, test, impact_params, n_sims=n_sims)
    result['platform'] = platform_name
    result['strategy'] = strat_name
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polymarket", default="data/polymarket/optimization_cache.parquet")
    parser.add_argument("--kalshi", default="data/kalshi/kalshi_backtest_clean.parquet")
    parser.add_argument("--output", default="runs/all_strategies_comparison")
    parser.add_argument("--n-sims", type=int, default=30)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("ALL STRATEGIES COMPARISON: POLYMARKET vs KALSHI (PARALLEL)")
    print("=" * 100)
    
    # Load data
    print("\nLoading data...")
    poly_raw = pd.read_parquet(args.polymarket)
    kalshi_raw = pd.read_parquet(args.kalshi)
    
    poly = prepare_data(poly_raw)
    kalshi = prepare_data(kalshi_raw)
    
    # Split and save temp files for workers
    poly_train = poly.iloc[:int(len(poly)*0.7)]
    poly_test = poly.iloc[int(len(poly)*0.7):]
    kalshi_train = kalshi.iloc[:int(len(kalshi)*0.7)]
    kalshi_test = kalshi.iloc[int(len(kalshi)*0.7):]
    
    # Save to temp files for parallel workers
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    poly_train.to_parquet(temp_dir / 'poly_train.parquet')
    poly_test.to_parquet(temp_dir / 'poly_test.parquet')
    kalshi_train.to_parquet(temp_dir / 'kalshi_train.parquet')
    kalshi_test.to_parquet(temp_dir / 'kalshi_test.parquet')
    
    print(f"Polymarket: {len(poly_train):,} train, {len(poly_test):,} test")
    print(f"Kalshi: {len(kalshi_train):,} train, {len(kalshi_test):,} test")
    print(f"Running with {args.n_workers} parallel workers...")
    
    strategies = [
        ('calibration_mean_reversion', 'run_calibration_strategy'),
        ('momentum', 'run_momentum_strategy'),
        ('stat_arb', 'run_stat_arb_strategy'),
        ('longshot', 'run_longshot_strategy'),
    ]
    
    # Build job list
    jobs = []
    for strat_name, strat_func_name in strategies:
        for platform_name in ['polymarket', 'kalshi']:
            prefix = 'poly' if platform_name == 'polymarket' else 'kalshi'
            train_path = str(temp_dir / f'{prefix}_train.parquet')
            test_path = str(temp_dir / f'{prefix}_test.parquet')
            jobs.append((strat_name, strat_func_name, platform_name, train_path, test_path, args.n_sims))
    
    print(f"\nRunning {len(jobs)} strategy/platform combinations in parallel...")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_single_strategy, job): job for job in jobs}
        
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                strat_name = result.get('strategy', job[0])
                platform_name = result.get('platform', job[2])
                
                if 'error' not in result:
                    print(f"✓ {strat_name:<25} {platform_name:<12} Sharpe={result['sharpe']:.2f} "
                          f"ES={result['es_sharpe']:.2f} Win={result['win_rate']:.1%} Trades={result['n_trades']}")
                else:
                    print(f"✗ {strat_name:<25} {platform_name:<12} ERROR: {result.get('error', 'unknown')}")
            except Exception as e:
                print(f"✗ {job[0]:<25} {job[2]:<12} EXCEPTION: {str(e)[:50]}")
    
    # Clean up temp files
    for f in temp_dir.glob('*.parquet'):
        f.unlink()
    temp_dir.rmdir()
    
    # Save results
    with open(output_dir / 'all_strategies_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Strategy':<25} {'Platform':<12} {'Sharpe':>8} {'ES_Sharpe':>10} {'Sortino':>9} {'Win%':>8} {'Trades':>8} {'TotalPnL':>12}")
    print("-" * 100)
    
    for result in sorted(results, key=lambda x: (x.get('strategy', ''), x.get('platform', ''))):
        if 'error' not in result:
            print(f"{result['strategy']:<25} {result['platform']:<12} {result['sharpe']:>8.2f} "
                  f"{result['es_sharpe']:>10.2f} {result['sortino']:>9.2f} "
                  f"{result['win_rate']:>7.1%} {result['n_trades']:>8} ${result['total_pnl']:>11,.0f}")
        else:
            print(f"{result.get('strategy', 'unknown'):<25} {result.get('platform', 'unknown'):<12} {'ERROR':>8} {result.get('error', ''):<60}")
    
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 100)
    
    # Best strategies per platform
    poly_results = [r for r in results if r.get('platform') == 'polymarket' and 'error' not in r]
    kalshi_results = [r for r in results if r.get('platform') == 'kalshi' and 'error' not in r]
    
    if poly_results:
        best_poly = max(poly_results, key=lambda x: x['sharpe'])
        print(f"\nBest Polymarket strategy: {best_poly['strategy']} (Sharpe={best_poly['sharpe']:.2f})")
    
    if kalshi_results:
        best_kalshi = max(kalshi_results, key=lambda x: x['sharpe'])
        print(f"Best Kalshi strategy: {best_kalshi['strategy']} (Sharpe={best_kalshi['sharpe']:.2f})")
    
    print(f"\nResults saved to {output_dir / 'all_strategies_results.json'}")


if __name__ == "__main__":
    main()
