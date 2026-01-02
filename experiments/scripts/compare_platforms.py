#!/usr/bin/env python3
"""
Compare strategy performance across Polymarket and Kalshi.

Analyzes:
1. Calibration differences
2. Strategy performance under realistic impact models
3. Microstructure explanations for differences
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Import our modules
from stress_test_impact_models import (
    ImpactModelParams,
    create_almgren_chriss_base,
    create_almgren_chriss_conservative,
    create_adversarial_mev,
    compute_fill_probability,
    compute_price_impact,
    compute_adverse_selection_cost,
    estimate_liquidity,
)
from tail_risk_metrics import compute_tail_risk_metrics, TailRiskMetrics


@dataclass
class PlatformData:
    """Container for platform data."""
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    calibration: pd.DataFrame
    categories: List[str]


def load_polymarket_data(path: str) -> PlatformData:
    """Load and prepare Polymarket data."""
    df = pd.read_parquet(path)
    
    # Standardize column names
    rename = {}
    if 'avg_price' in df.columns:
        rename['avg_price'] = 'price'
    if 'y' in df.columns:
        rename['y'] = 'outcome'
    df = df.rename(columns=rename)
    
    # Ensure required columns
    df['price'] = df['price'].clip(0.01, 0.99)
    df['outcome'] = df['outcome'].astype(float)
    if 'category' not in df.columns:
        df['category'] = 'general'
    
    # Split
    n = len(df)
    train_idx = int(n * 0.7)
    train = df.iloc[:train_idx].copy()
    test = df.iloc[train_idx:].copy()
    
    # Compute calibration
    train['price_bin'] = pd.cut(train['price'], bins=10, labels=False)
    calibration = train.groupby('price_bin').agg({
        'price': 'mean',
        'outcome': 'mean',
    }).rename(columns={'price': 'bin_price', 'outcome': 'outcome_rate'})
    calibration['spread'] = calibration['outcome_rate'] - calibration['bin_price']
    
    categories = df['category'].dropna().unique().tolist()
    
    return PlatformData(
        name='polymarket',
        train=train,
        test=test,
        calibration=calibration,
        categories=categories[:10],  # Top 10
    )


def load_kalshi_data(path: str) -> PlatformData:
    """Load and prepare Kalshi data."""
    df = pd.read_parquet(path)
    
    # Standardize column names
    rename = {}
    if 'avg_price' in df.columns:
        rename['avg_price'] = 'price'
    if 'y' in df.columns:
        rename['y'] = 'outcome'
    df = df.rename(columns=rename)
    
    # Ensure required columns
    df['price'] = df['price'].clip(0.01, 0.99)
    df['outcome'] = df['outcome'].astype(float)
    if 'category' not in df.columns:
        df['category'] = 'general'
    
    # Split
    n = len(df)
    train_idx = int(n * 0.7)
    train = df.iloc[:train_idx].copy()
    test = df.iloc[train_idx:].copy()
    
    # Compute calibration
    train['price_bin'] = pd.cut(train['price'], bins=10, labels=False)
    calibration = train.groupby('price_bin').agg({
        'price': 'mean',
        'outcome': 'mean',
    }).rename(columns={'price': 'bin_price', 'outcome': 'outcome_rate'})
    calibration['spread'] = calibration['outcome_rate'] - calibration['bin_price']
    
    categories = df['category'].dropna().unique().tolist()
    
    return PlatformData(
        name='kalshi',
        train=train,
        test=test,
        calibration=calibration,
        categories=categories[:10],
    )


def run_calibration_strategy(
    platform: PlatformData,
    impact_params: ImpactModelParams,
    strategy_params: Dict[str, Any],
    n_sims: int = 50,
    bankroll: float = 10000.0,
) -> Dict[str, Any]:
    """Run calibration-based strategy on platform data."""
    
    spread_threshold = strategy_params.get('spread_threshold', 0.05)
    kelly_fraction = strategy_params.get('kelly_fraction', 0.25)
    max_position_pct = strategy_params.get('max_position_pct', 0.10)
    fee = strategy_params.get('fee', 0.01)
    n_bins = strategy_params.get('n_bins', 10)
    
    # Apply calibration to test set
    test = platform.test.copy()
    test['price_bin'] = pd.cut(test['price'], bins=n_bins, labels=False)
    test = test.merge(
        platform.calibration[['spread']],
        left_on='price_bin',
        right_index=True,
        how='left'
    )
    test['spread'] = test['spread'].fillna(0)
    
    all_pnls = []
    all_outcomes = []
    
    for sim in range(n_sims):
        rng = np.random.default_rng(42 + sim)
        current_bankroll = bankroll
        sim_pnls = []
        
        for _, row in test.iterrows():
            price = row['price']
            spread = row.get('spread', 0)
            outcome = row.get('outcome', 0)
            category = row.get('category', '')
            
            if abs(spread) < spread_threshold:
                continue
            
            # Kelly sizing
            edge = abs(spread)
            if spread > 0:
                odds = (1 - price) / price if price > 0.01 else 99
            else:
                odds = price / (1 - price) if price < 0.99 else 99
            
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * kelly_fraction
            
            intended_size = current_bankroll * min(kelly, max_position_pct)
            intended_size = min(intended_size, current_bankroll * 0.5)
            
            if intended_size < 1:
                continue
            
            # Fill probability
            liquidity = estimate_liquidity(price, str(category), impact_params)
            fill_prob = compute_fill_probability(intended_size, price, liquidity, impact_params, rng)
            
            if rng.random() >= fill_prob:
                continue
            
            # Filled - compute PnL
            temp_impact, perm_impact = compute_price_impact(intended_size, liquidity, impact_params)
            impact_cost = (temp_impact + perm_impact) * intended_size
            as_cost = compute_adverse_selection_cost(edge, 1.0, True, impact_params) * intended_size
            
            if spread > 0:
                raw_pnl = (outcome - price) * intended_size
            else:
                raw_pnl = ((1 - outcome) - (1 - price)) * intended_size
            
            net_pnl = raw_pnl - impact_cost - as_cost - (intended_size * fee)
            sim_pnls.append(net_pnl)
            
            all_outcomes.append({
                'pnl': net_pnl,
                'price': price,
                'spread': spread,
                'outcome': outcome,
                'size': intended_size,
                'sim': sim,
            })
            
            current_bankroll += net_pnl
            if current_bankroll <= 0:
                break
        
        if sim_pnls:
            all_pnls.extend(sim_pnls)
    
    # Compute metrics
    if len(all_pnls) < 10:
        return {'error': 'too_few_trades', 'n_trades': len(all_pnls)}
    
    pnls = np.array(all_pnls)
    metrics = compute_tail_risk_metrics(pnls)
    
    return {
        'platform': platform.name,
        'n_trades': len(pnls),
        'total_pnl': float(np.sum(pnls)),
        'mean_pnl': float(np.mean(pnls)),
        'sharpe': float(metrics.sharpe),
        'es_sharpe': float(metrics.es_sharpe_5pct),
        'sortino': float(metrics.sortino),
        'var_5pct': float(metrics.var_5pct),
        'es_5pct': float(metrics.es_5pct),
        'win_rate': float(metrics.win_rate),
        'skewness': float(metrics.skewness),
        'kurtosis': float(metrics.kurtosis),
        'max_loss': float(metrics.max_loss),
        'max_gain': float(metrics.max_gain),
    }


def analyze_microstructure(
    poly: PlatformData,
    kalshi: PlatformData,
) -> Dict[str, Any]:
    """Analyze microstructure differences between platforms."""
    
    analysis = {}
    
    # 1. Calibration comparison
    analysis['calibration'] = {
        'polymarket': {
            'mean_spread': float(poly.calibration['spread'].mean()),
            'std_spread': float(poly.calibration['spread'].std()),
            'max_spread': float(poly.calibration['spread'].abs().max()),
        },
        'kalshi': {
            'mean_spread': float(kalshi.calibration['spread'].mean()),
            'std_spread': float(kalshi.calibration['spread'].std()),
            'max_spread': float(kalshi.calibration['spread'].abs().max()),
        },
    }
    
    # 2. Price distribution
    analysis['price_distribution'] = {
        'polymarket': {
            'mean': float(poly.train['price'].mean()),
            'median': float(poly.train['price'].median()),
            'std': float(poly.train['price'].std()),
            'pct_extreme': float(((poly.train['price'] < 0.1) | (poly.train['price'] > 0.9)).mean()),
        },
        'kalshi': {
            'mean': float(kalshi.train['price'].mean()),
            'median': float(kalshi.train['price'].median()),
            'std': float(kalshi.train['price'].std()),
            'pct_extreme': float(((kalshi.train['price'] < 0.1) | (kalshi.train['price'] > 0.9)).mean()),
        },
    }
    
    # 3. Outcome rates
    analysis['outcome_rates'] = {
        'polymarket': {
            'yes_rate': float(poly.train['outcome'].mean()),
            'n_markets': len(poly.train),
        },
        'kalshi': {
            'yes_rate': float(kalshi.train['outcome'].mean()),
            'n_markets': len(kalshi.train),
        },
    }
    
    # 4. Category diversity
    analysis['categories'] = {
        'polymarket': poly.categories,
        'kalshi': kalshi.categories,
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Compare Polymarket vs Kalshi")
    parser.add_argument("--polymarket", type=str, default="data/polymarket/optimization_cache.parquet")
    parser.add_argument("--kalshi", type=str, default="data/kalshi/kalshi_backtest_clean.parquet")
    parser.add_argument("--output", type=str, default="runs/platform_comparison")
    parser.add_argument("--n-sims", type=int, default=50)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("POLYMARKET VS KALSHI COMPARISON")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    poly = load_polymarket_data(args.polymarket)
    kalshi = load_kalshi_data(args.kalshi)
    
    print(f"Polymarket: {len(poly.train)} train, {len(poly.test)} test")
    print(f"Kalshi: {len(kalshi.train)} train, {len(kalshi.test)} test")
    
    # Microstructure analysis
    print("\n" + "=" * 80)
    print("MICROSTRUCTURE ANALYSIS")
    print("=" * 80)
    
    micro = analyze_microstructure(poly, kalshi)
    
    print("\n1. CALIBRATION DIFFERENCES:")
    print(f"   Polymarket: mean spread = {micro['calibration']['polymarket']['mean_spread']:.3f}")
    print(f"   Kalshi:     mean spread = {micro['calibration']['kalshi']['mean_spread']:.3f}")
    
    print("\n2. PRICE DISTRIBUTION:")
    print(f"   Polymarket: mean={micro['price_distribution']['polymarket']['mean']:.3f}, "
          f"median={micro['price_distribution']['polymarket']['median']:.3f}, "
          f"extreme={micro['price_distribution']['polymarket']['pct_extreme']:.1%}")
    print(f"   Kalshi:     mean={micro['price_distribution']['kalshi']['mean']:.3f}, "
          f"median={micro['price_distribution']['kalshi']['median']:.3f}, "
          f"extreme={micro['price_distribution']['kalshi']['pct_extreme']:.1%}")
    
    print("\n3. OUTCOME RATES:")
    print(f"   Polymarket: {micro['outcome_rates']['polymarket']['yes_rate']:.1%} YES rate")
    print(f"   Kalshi:     {micro['outcome_rates']['kalshi']['yes_rate']:.1%} YES rate")
    
    print("\n4. CATEGORIES:")
    print(f"   Polymarket: {', '.join(micro['categories']['polymarket'][:5])}")
    print(f"   Kalshi:     {', '.join(micro['categories']['kalshi'][:5])}")
    
    # Strategy comparison
    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("=" * 80)
    
    impact_models = [
        ("almgren_base", create_almgren_chriss_base()),
        ("almgren_conservative", create_almgren_chriss_conservative()),
        ("adversarial_mev", create_adversarial_mev()),
    ]
    
    strategy_params = {
        'spread_threshold': 0.05,
        'n_bins': 10,
        'kelly_fraction': 0.25,
        'max_position_pct': 0.10,
        'fee': 0.01,
    }
    
    results = []
    
    print(f"\n{'Platform':<12} {'Model':<22} {'Sharpe':>8} {'ES_Sharpe':>10} {'Trades':>8} {'Win%':>8} {'TotalPnL':>12}")
    print("-" * 90)
    
    for model_name, impact_params in impact_models:
        for platform in [poly, kalshi]:
            result = run_calibration_strategy(
                platform, impact_params, strategy_params,
                n_sims=args.n_sims, bankroll=10000.0
            )
            result['model'] = model_name
            results.append(result)
            
            if 'error' not in result:
                print(f"{platform.name:<12} {model_name:<22} {result['sharpe']:>8.2f} "
                      f"{result['es_sharpe']:>10.2f} {result['n_trades']:>8} "
                      f"{result['win_rate']:>7.1%} ${result['total_pnl']:>11,.0f}")
            else:
                print(f"{platform.name:<12} {model_name:<22} {'N/A':>8} {'N/A':>10} "
                      f"{result.get('n_trades', 0):>8} {'N/A':>8} {'N/A':>12}")
    
    # Explanation
    print("\n" + "=" * 80)
    print("MICROSTRUCTURE EXPLANATIONS FOR DIFFERENCES")
    print("=" * 80)
    print("""
1. MARKET TYPE DIFFERENCES:
   - Polymarket: Political, event-driven markets (elections, policy)
   - Kalshi: Sports/esports heavy (NBA, EPL, NHL, esports)
   
   Impact: Sports markets resolve quickly with high volume, political markets
   have longer time horizons and more information asymmetry.

2. PRICE DISTRIBUTION:
   - Polymarket: More uniform price distribution
   - Kalshi: Heavily skewed toward low prices (many long-shot bets)
   
   Impact: Low-price markets have higher volatility and different
   calibration patterns. Kelly sizing works differently.

3. CALIBRATION PATTERNS:
   - Polymarket: Systematic overpricing (prices > outcome rates)
   - Kalshi: Mixed patterns with extreme underpricing at low end
   
   Impact: Different edge profiles - Polymarket favors NO bets,
   Kalshi has opportunities at extremes.

4. USER BASE DIFFERENCES:
   - Polymarket: Crypto-native, retail speculators, some institutional
   - Kalshi: Regulated US platform, more retail sports bettors
   
   Impact: Different behavioral biases:
   - Polymarket: Overconfidence in political predictions
   - Kalshi: Favorite-longshot bias from sports betting culture

5. LIQUIDITY STRUCTURE:
   - Polymarket: AMM + orderbook hybrid, higher fees
   - Kalshi: Pure orderbook, tighter spreads on popular markets
   
   Impact: Fill rates and slippage differ by market type.
""")
    
    # Save results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump({
            'microstructure': micro,
            'strategy_results': results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
