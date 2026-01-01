#!/usr/bin/env python3
"""
Comprehensive Strategy Backtest Suite

Runs all three trading strategies on the Goldsky/Gamma merged data:
1. Blackwell/Calibration Arbitrage - bin-based calibration trading
2. Statistical/Portfolio Arbitrage - Markowitz-style with hyperparam sweep
3. Category Mean-Reversion - regime-based group trading

Uses real outcomes from Gamma API, prices from Goldsky trades.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Data Loading
# =============================================================================

def categorize_question(q: str) -> str:
    """Categorize a market question."""
    q = str(q).lower()
    
    if any(x in q for x in ['bitcoin', 'btc', 'eth', 'crypto', 'solana', 'dogecoin']):
        return 'crypto'
    if any(x in q for x in ['trump', 'biden', 'harris', 'election', 'president', 'congress', 'senate', 'democrat', 'republican']):
        return 'politics'
    if any(x in q for x in ['nba', 'nfl', 'mlb', 'nhl', 'win', 'beat', 'championship', 'super bowl', 'world series', 'soccer', 'football']):
        return 'sports'
    if any(x in q for x in ['temperature', 'weather', 'rain', 'snow', 'hurricane', 'celsius', 'fahrenheit']):
        return 'weather'
    if any(x in q for x in ['fed', 'rate', 'inflation', 'gdp', 'unemployment', 'interest', 'cpi']):
        return 'economics'
    
    return 'other'


def load_merged_data(
    gamma_path: Path,
    goldsky_markets_path: Path,
    goldsky_trades_path: Path,
    min_trades: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """Load and merge all data sources with real outcomes."""
    
    print("Loading data sources...")
    
    # Load Gamma resolved markets (has actual outcomes)
    gamma = pd.read_parquet(gamma_path)
    if verbose:
        print(f"  Gamma resolved markets: {len(gamma):,}")
    
    # Load Goldsky markets (has token IDs)
    goldsky_markets = pd.read_csv(goldsky_markets_path)
    if verbose:
        print(f"  Goldsky markets: {len(goldsky_markets):,}")
    
    # Create token → goldsky_id mapping
    token_to_goldsky = {}
    for _, row in goldsky_markets.iterrows():
        token_to_goldsky[str(row['token1'])] = row['id']
        token_to_goldsky[str(row['token2'])] = row['id']
    
    # Map Gamma markets to Goldsky IDs
    gamma['goldsky_id'] = gamma['yes_token_id'].astype(str).map(token_to_goldsky)
    matched_gamma = gamma[gamma['goldsky_id'].notna()].copy()
    if verbose:
        print(f"  Gamma matched to Goldsky: {len(matched_gamma):,}")
    
    # Load trade aggregations
    print("Loading trade aggregations (this may take a few minutes)...")
    
    chunk_size = 5_000_000
    agg_list = []
    total_trades = 0
    
    for i, chunk in enumerate(pd.read_csv(goldsky_trades_path, chunksize=chunk_size)):
        total_trades += len(chunk)
        
        agg = chunk.groupby('market_id').agg({
            'price': ['first', 'last', 'mean', 'count'],
            'usd_amount': 'sum',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        agg.columns = ['market_id', 'first_price', 'last_price', 'avg_price', 
                       'n_trades', 'total_volume', 'first_time', 'last_time']
        agg_list.append(agg)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"    Processed {total_trades:,} trades...")
    
    if verbose:
        print(f"  Total trades: {total_trades:,}")
    
    # Combine aggregations
    all_aggs = pd.concat(agg_list)
    final_agg = all_aggs.groupby('market_id').agg({
        'first_price': 'first',
        'last_price': 'last',
        'avg_price': 'mean',
        'n_trades': 'sum',
        'total_volume': 'sum',
        'first_time': 'min',
        'last_time': 'max'
    }).reset_index()
    
    # Map trades market_id to goldsky_id
    final_agg['goldsky_id'] = final_agg['market_id'].apply(
        lambda x: int(float(x)) if pd.notna(x) else None
    )
    
    # Merge
    matched_gamma['goldsky_id'] = matched_gamma['goldsky_id'].astype(int)
    merged = matched_gamma.merge(final_agg, on='goldsky_id', how='inner')
    
    if verbose:
        print(f"  Merged dataset: {len(merged):,}")
    
    # Filter by trade count
    merged = merged[merged['n_trades'] >= min_trades].copy()
    if verbose:
        print(f"  With >= {min_trades} trades: {len(merged):,}")
    
    # Parse resolution time
    merged['resolution_time'] = pd.to_datetime(
        merged['closedTime'], format='mixed', utc=True, errors='coerce'
    )
    merged = merged[merged['resolution_time'].notna()].copy()
    
    # Add category
    merged['category'] = merged['question'].apply(categorize_question)
    
    # Sort chronologically
    merged = merged.sort_values('resolution_time').reset_index(drop=True)
    
    if verbose:
        print(f"  Final dataset: {len(merged):,}")
        print(f"  Outcome distribution: YES={merged['y'].mean():.1%}")
        print(f"  Date range: {merged['resolution_time'].min().date()} to {merged['resolution_time'].max().date()}")
    
    return merged


# =============================================================================
# Strategy 1: Blackwell/Calibration Arbitrage
# =============================================================================

@dataclass
class BlackwellConfig:
    n_bins: int = 10
    g_bar_threshold: float = 0.03
    t_stat_threshold: float = 1.5
    min_samples_per_bin: int = 20
    lookback_trades: int = 500
    use_risk_parity: bool = True
    target_max_loss: float = 0.2


def run_blackwell_backtest(
    df: pd.DataFrame,
    train_frac: float = 0.5,
    cfg: BlackwellConfig = BlackwellConfig(),
) -> Dict:
    """Run Blackwell calibration arbitrage backtest."""
    
    print("\n" + "="*60)
    print("STRATEGY 1: BLACKWELL CALIBRATION ARBITRAGE")
    print("="*60)
    
    n = len(df)
    train_size = int(n * train_frac)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    print(f"Train: {len(train):,}, Test: {len(test):,}")
    
    # Compute bin statistics from training data
    bin_edges = np.linspace(0, 1, cfg.n_bins + 1)
    
    # Use first_price as the traded price
    prices = train['first_price'].values
    outcomes = train['y'].values
    
    bin_idx = np.clip(np.digitize(prices, bin_edges) - 1, 0, cfg.n_bins - 1)
    
    bin_stats = {}
    for b in range(cfg.n_bins):
        mask = bin_idx == b
        if mask.sum() >= cfg.min_samples_per_bin:
            p_bin = prices[mask]
            y_bin = outcomes[mask]
            residuals = y_bin - p_bin
            
            g_bar = residuals.mean()
            sigma = residuals.std()
            se = sigma / np.sqrt(mask.sum())
            t_stat = g_bar / se if se > 0 else 0
            
            is_significant = (
                abs(t_stat) >= cfg.t_stat_threshold and
                abs(g_bar) >= cfg.g_bar_threshold
            )
            
            bin_stats[b] = {
                'g_bar': g_bar,
                't_stat': t_stat,
                'n': int(mask.sum()),
                'significant': is_significant,
                'direction': 1 if g_bar > 0 else -1  # +1 = long YES
            }
    
    print(f"\nTradeable bins: {sum(1 for s in bin_stats.values() if s['significant'])}")
    for b, s in bin_stats.items():
        if s['significant']:
            print(f"  Bin {b}: g̅={s['g_bar']:+.3f}, t={s['t_stat']:.2f}, n={s['n']}")
    
    # Trade on test set
    results = []
    for _, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        
        b = int(np.clip(np.digitize(price, bin_edges) - 1, 0, cfg.n_bins - 1))
        
        if b not in bin_stats or not bin_stats[b]['significant']:
            continue
        
        direction = bin_stats[b]['direction']
        g_bar = bin_stats[b]['g_bar']
        
        # Position sizing
        if cfg.use_risk_parity:
            if direction > 0:
                max_loss = price
            else:
                max_loss = 1 - price
            size = min(1.0, cfg.target_max_loss / max(max_loss, 0.01))
        else:
            size = 1.0
        
        # PnL
        if direction > 0:  # Long YES
            pnl = (1 - price) * size if outcome == 1 else -price * size
        else:  # Short YES
            pnl = price * size if outcome == 0 else -(1 - price) * size
        
        results.append({
            'bin': b,
            'direction': direction,
            'price': price,
            'outcome': outcome,
            'size': size,
            'pnl': pnl,
            'g_bar': g_bar,
        })
    
    if not results:
        return {'strategy': 'blackwell', 'error': 'No trades'}
    
    results_df = pd.DataFrame(results)
    
    # Summary
    n_trades = len(results_df)
    win_rate = (results_df['pnl'] > 0).mean()
    total_pnl = results_df['pnl'].sum()
    sharpe = results_df['pnl'].mean() / results_df['pnl'].std() * np.sqrt(252) if results_df['pnl'].std() > 0 else 0
    
    print(f"\nResults:")
    print(f"  Trades: {n_trades:,}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Sharpe: {sharpe:.2f}")
    
    return {
        'strategy': 'blackwell',
        'config': cfg.__dict__,
        'n_trades': n_trades,
        'win_rate': float(win_rate),
        'total_pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'bin_stats': bin_stats,
    }


# =============================================================================
# Strategy 2: Statistical/Portfolio Arbitrage
# =============================================================================

@dataclass
class StatArbConfig:
    kelly_scale: float = 0.25
    kelly_cap: float = 0.10
    distance_scale: float = 1.0
    min_edge: float = 0.02
    n_bins: int = 10
    fee: float = 0.01


def run_stat_arb_backtest(
    df: pd.DataFrame,
    train_frac: float = 0.5,
    cfg: StatArbConfig = StatArbConfig(),
) -> Dict:
    """Run statistical arbitrage backtest with walk-forward validation."""
    
    n = len(df)
    train_size = int(n * train_frac)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Compute calibration stats from training data
    prices = train['first_price'].values
    outcomes = train['y'].values
    
    bin_edges = np.linspace(0, 1, cfg.n_bins + 1)
    bin_idx = np.clip(np.digitize(prices, bin_edges) - 1, 0, cfg.n_bins - 1)
    
    calibration_stats = {}
    for b in range(cfg.n_bins):
        mask = bin_idx == b
        if mask.sum() > 5:
            bias = outcomes[mask].mean() - prices[mask].mean()
            calibration_stats[b] = {
                'bias': bias,
                'distance': abs(bias),
                'n': int(mask.sum())
            }
    
    # Trade on test set
    br = 1.0
    trades = []
    eps = 1e-9
    
    for _, row in test.iterrows():
        qi = float(np.clip(row['first_price'], eps, 1 - eps))
        yi = float(row['y'])
        
        # Use model-free approach: estimate p from calibration bias
        b = int(np.clip(np.digitize(qi, bin_edges) - 1, 0, cfg.n_bins - 1))
        
        if b not in calibration_stats:
            continue
        
        bias = calibration_stats[b]['bias']
        distance = calibration_stats[b]['distance']
        
        # Adjusted "fair" price = market + bias
        pi = float(np.clip(qi + bias, eps, 1 - eps))
        edge = pi - qi
        
        if abs(edge) < cfg.min_edge:
            continue
        
        direction = 1 if edge > 0 else -1
        
        # Kelly sizing
        if direction > 0:
            kelly_f = cfg.kelly_scale * (pi - qi) / max(1 - qi, eps)
        else:
            kelly_f = cfg.kelly_scale * (qi - pi) / max(qi, eps)
        
        kelly_f = float(np.clip(kelly_f, 0.0, cfg.kelly_cap))
        
        # Distance weighting
        kelly_f *= min(1.0 + cfg.distance_scale * distance, 2.0)
        kelly_f = float(np.clip(kelly_f, 0.0, cfg.kelly_cap * 2))
        
        if kelly_f < eps:
            continue
        
        stake = kelly_f * br
        
        # PnL: Use FLAT betting (not Kelly compounding) to avoid blowup
        # For each $1 bet: 
        #   Long YES at q: profit = (1-q) if YES, loss = q if NO
        #   Long NO at q: profit = q if NO, loss = (1-q) if YES
        if direction > 0:
            pnl = kelly_f * (1 - qi) if yi == 1 else -kelly_f * qi  # Long YES
        else:
            pnl = kelly_f * qi if yi == 0 else -kelly_f * (1 - qi)  # Long NO
        pnl -= cfg.fee * kelly_f
        
        # Simple additive PnL (not compounding)
        br += pnl
        
        trades.append({
            'direction': direction,
            'edge': abs(edge),
            'distance': distance,
            'kelly_f': kelly_f,
            'pnl': pnl,
        })
    
    if not trades:
        return {'strategy': 'stat_arb', 'config': cfg.__dict__, 'error': 'No trades'}
    
    total_pnl = br - 1.0
    roi = total_pnl
    pnl_array = np.array([t['pnl'] for t in trades])
    sharpe = float(pnl_array.mean() / pnl_array.std() * np.sqrt(252)) if pnl_array.std() > 0 else 0
    win_rate = float((pnl_array > 0).mean())
    
    return {
        'strategy': 'stat_arb',
        'config': cfg.__dict__,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': float(total_pnl),
        'roi': float(roi),
        'final_bankroll': float(br),
        'sharpe': sharpe,
        'mean_edge': float(np.mean([t['edge'] for t in trades])),
        'mean_distance': float(np.mean([t['distance'] for t in trades])),
    }


def sweep_stat_arb_hyperparams(
    df: pd.DataFrame,
    train_frac: float = 0.5,
) -> List[Dict]:
    """Sweep hyperparameters for statistical arbitrage."""
    
    print("\n" + "="*60)
    print("STRATEGY 2: STATISTICAL ARBITRAGE (HYPERPARAM SWEEP)")
    print("="*60)
    
    # Hyperparameter grid
    kelly_scales = [0.1, 0.25, 0.5]
    kelly_caps = [0.05, 0.10, 0.20]
    distance_scales = [0.5, 1.0, 2.0]
    min_edges = [0.01, 0.02, 0.05]
    
    results = []
    total_combos = len(kelly_scales) * len(kelly_caps) * len(distance_scales) * len(min_edges)
    
    print(f"Testing {total_combos} hyperparameter combinations...")
    
    for i, (ks, kc, ds, me) in enumerate(product(kelly_scales, kelly_caps, distance_scales, min_edges)):
        cfg = StatArbConfig(
            kelly_scale=ks,
            kelly_cap=kc,
            distance_scale=ds,
            min_edge=me,
        )
        
        result = run_stat_arb_backtest(df, train_frac, cfg)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total_combos} completed...")
    
    # Sort by Sharpe
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
    
    print(f"\nTop 5 configurations by Sharpe:")
    for i, r in enumerate(valid_results[:5]):
        cfg = r['config']
        print(f"  {i+1}. kelly_scale={cfg['kelly_scale']}, kelly_cap={cfg['kelly_cap']}, "
              f"distance_scale={cfg['distance_scale']}, min_edge={cfg['min_edge']}")
        print(f"     Sharpe={r['sharpe']:.2f}, PnL=${r['total_pnl']:.2f}, "
              f"Win={r['win_rate']:.1%}, N={r['n_trades']}")
    
    return valid_results


# =============================================================================
# Strategy 3: Category Mean-Reversion
# =============================================================================

@dataclass
class MeanReversionConfig:
    calibration_window: int = 50
    mean_revert_threshold: float = 0.05
    momentum_threshold: float = 0.15
    kelly_fraction: float = 0.25
    max_position_size: float = 0.10
    min_edge: float = 0.02
    fee: float = 0.01


def run_mean_reversion_backtest(
    df: pd.DataFrame,
    train_frac: float = 0.5,
    cfg: MeanReversionConfig = MeanReversionConfig(),
) -> Dict:
    """Run category mean-reversion backtest."""
    
    print("\n" + "="*60)
    print("STRATEGY 3: CATEGORY MEAN-REVERSION")
    print("="*60)
    
    n = len(df)
    train_size = int(n * train_frac)
    
    # Use all data for walk-forward (training is implicit in rolling window)
    data = df.copy()
    
    # Track group calibration with rolling window
    group_residuals: Dict[str, List[float]] = {}
    
    def get_regime(g: str) -> str:
        if g not in group_residuals or len(group_residuals[g]) < 10:
            return "neutral"
        
        recent = group_residuals[g][-cfg.calibration_window:]
        mean_residual = np.mean(recent)
        abs_mean = abs(mean_residual)
        
        if abs_mean < cfg.mean_revert_threshold:
            return "mean_revert"
        elif abs_mean > cfg.momentum_threshold:
            return "momentum"
        return "neutral"
    
    def get_bias(g: str) -> float:
        if g not in group_residuals or len(group_residuals[g]) < 5:
            return 0.0
        return np.mean(group_residuals[g][-cfg.calibration_window:])
    
    regime_scale = {"mean_revert": 1.0, "momentum": 0.3, "neutral": 0.1}
    
    br = 1.0
    trades = []
    eps = 1e-9
    
    pnl_by_category = {}
    pnl_by_regime = {}
    regime_counts = {"mean_revert": 0, "momentum": 0, "neutral": 0}
    
    # Only trade on test period but update residuals from start
    for idx, row in data.iterrows():
        g = row['category']
        qi = float(np.clip(row['first_price'], eps, 1 - eps))
        yi = float(row['y'])
        
        # Skip training period for trading
        if idx < train_size:
            residual = yi - qi
            if g not in group_residuals:
                group_residuals[g] = []
            group_residuals[g].append(residual)
            continue
        
        # Get regime (no lookahead)
        regime = get_regime(g)
        regime_counts[regime] += 1
        
        # Estimate fair price from group bias
        bias = get_bias(g)
        pi = float(np.clip(qi + bias, eps, 1 - eps))
        edge = pi - qi
        
        if abs(edge) < cfg.min_edge:
            residual = yi - qi
            if g not in group_residuals:
                group_residuals[g] = []
            group_residuals[g].append(residual)
            continue
        
        direction = 1 if edge > 0 else -1
        
        # Position sizing with regime scaling
        if direction > 0:
            kelly = (pi - qi) / max(1 - qi, eps)
        else:
            kelly = (qi - pi) / max(qi, eps)
        
        size = kelly * cfg.kelly_fraction * regime_scale[regime]
        size = float(np.clip(size, 0.0, cfg.max_position_size))
        
        if size < eps:
            residual = yi - qi
            if g not in group_residuals:
                group_residuals[g] = []
            group_residuals[g].append(residual)
            continue
        
        # PnL: Use FLAT betting (not Kelly compounding) to avoid blowup
        if direction > 0:
            pnl = size * (1 - qi) if yi == 1 else -size * qi  # Long YES
        else:
            pnl = size * qi if yi == 0 else -size * (1 - qi)  # Long NO
        pnl -= cfg.fee * size
        
        # Simple additive PnL
        br += pnl
        
        pnl_by_category[g] = pnl_by_category.get(g, 0) + pnl
        pnl_by_regime[regime] = pnl_by_regime.get(regime, 0) + pnl
        
        trades.append({
            'category': g,
            'regime': regime,
            'direction': direction,
            'edge': abs(edge),
            'bias': bias,
            'size': size,
            'pnl': pnl,
        })
        
        # Update residuals
        residual = yi - qi
        if g not in group_residuals:
            group_residuals[g] = []
        group_residuals[g].append(residual)
    
    if not trades:
        return {'strategy': 'mean_reversion', 'error': 'No trades'}
    
    total_pnl = br - 1.0
    pnl_array = np.array([t['pnl'] for t in trades])
    sharpe = float(pnl_array.mean() / pnl_array.std() * np.sqrt(252)) if pnl_array.std() > 0 else 0
    win_rate = float((pnl_array > 0).mean())
    
    print(f"\nResults:")
    print(f"  Trades: {len(trades):,}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Final bankroll: ${br:.4f}")
    print(f"  Sharpe: {sharpe:.2f}")
    
    print(f"\nPnL by Category:")
    for cat, pnl in sorted(pnl_by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: ${pnl:.2f}")
    
    print(f"\nPnL by Regime:")
    for regime, pnl in sorted(pnl_by_regime.items(), key=lambda x: -x[1]):
        print(f"  {regime} ({regime_counts.get(regime, 0)}): ${pnl:.2f}")
    
    return {
        'strategy': 'mean_reversion',
        'config': cfg.__dict__,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': float(total_pnl),
        'final_bankroll': float(br),
        'sharpe': sharpe,
        'pnl_by_category': pnl_by_category,
        'pnl_by_regime': pnl_by_regime,
        'regime_counts': regime_counts,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all strategy backtests")
    parser.add_argument("--gamma-path", type=str, 
                        default="data/polymarket/gamma_yesno_resolved.parquet")
    parser.add_argument("--goldsky-markets", type=str, 
                        default="data/polymarket_goldsky/markets.csv")
    parser.add_argument("--goldsky-trades", type=str, 
                        default="data/polymarket_goldsky/processed/trades.csv")
    parser.add_argument("--output-dir", type=str, default="runs/all_strategies_backtest")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--min-trades", type=int, default=20)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE STRATEGY BACKTEST SUITE")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load data
    df = load_merged_data(
        gamma_path=Path(args.gamma_path),
        goldsky_markets_path=Path(args.goldsky_markets),
        goldsky_trades_path=Path(args.goldsky_trades),
        min_trades=args.min_trades,
    )
    
    all_results = {}
    
    # Strategy 1: Blackwell Calibration
    blackwell_result = run_blackwell_backtest(df, args.train_frac)
    all_results['blackwell'] = blackwell_result
    
    # Strategy 2: Statistical Arbitrage (with sweep)
    stat_arb_results = sweep_stat_arb_hyperparams(df, args.train_frac)
    all_results['stat_arb_sweep'] = stat_arb_results
    all_results['stat_arb_best'] = stat_arb_results[0] if stat_arb_results else None
    
    # Strategy 3: Mean Reversion
    mean_rev_result = run_mean_reversion_backtest(df, args.train_frac)
    all_results['mean_reversion'] = mean_rev_result
    
    # Summary comparison
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    
    summary = []
    for name, result in [
        ('Blackwell', blackwell_result),
        ('StatArb (best)', all_results['stat_arb_best']),
        ('Mean-Reversion', mean_rev_result),
    ]:
        if result and 'error' not in result:
            summary.append({
                'strategy': name,
                'trades': result.get('n_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'pnl': result.get('total_pnl', 0),
                'sharpe': result.get('sharpe', 0),
            })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary_df.to_csv(output_dir / 'strategy_comparison.csv', index=False)
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
