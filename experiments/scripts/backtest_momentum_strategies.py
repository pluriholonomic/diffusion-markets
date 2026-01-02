"""
Backtest Momentum Strategies with Realistic Market Impact Models

This script runs the new momentum/dispersion strategies through the same
market impact framework used in stress_test_impact_models.py to get
realistic PnL estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import argparse
from datetime import datetime

# Import impact model infrastructure
from stress_test_impact_models import (
    ImpactModelParams,
    create_almgren_chriss_base,
    create_almgren_chriss_conservative,
    compute_fill_probability,
    compute_price_impact,
    compute_adverse_selection_cost,
    estimate_liquidity,
)

# Import our momentum strategies
from momentum_strategies import (
    compute_rolling_calibration_error,
)
from regime_detector import RegimeDetector, RegimeType


@dataclass
class RealisticBacktestResult:
    strategy_name: str
    impact_model: str
    total_pnl: float
    final_bankroll: float
    sharpe: float
    max_drawdown: float
    trades_attempted: int
    trades_filled: int
    fill_rate: float
    total_impact_cost: float
    total_adverse_selection: float
    win_rate: float


def run_calibration_momentum_with_impact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    # Strategy params
    persistence_window: int = 30,
    momentum_threshold: float = 0.10,
    min_persistence_days: int = 20,
    spread_threshold: float = 0.03,
    n_bins: int = 10,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.10,
    fee: float = 0.01,
    # Simulation params
    n_simulations: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run calibration momentum strategy with realistic market impact.
    """
    rng = np.random.default_rng(seed)
    
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    category_col = 'category' if 'category' in train.columns else None
    
    # Learn calibration from training data
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    train_copy['_outcome'] = train_copy[outcome_col]
    train_copy['price_bin'] = pd.cut(train_copy['_price'], bins=n_bins, labels=False)
    
    # Compute calibration error per bin
    calibration = train_copy.groupby('price_bin').agg({
        '_price': 'mean',
        '_outcome': 'mean'
    }).rename(columns={'_price': 'bin_price', '_outcome': 'outcome_rate'})
    calibration['spread'] = calibration['outcome_rate'] - calibration['bin_price']
    
    # Compute rolling calibration to detect persistence
    train_with_cal = compute_rolling_calibration_error(
        train, price_col, outcome_col, persistence_window, n_bins
    )
    
    # Build momentum model
    bin_momentum = {}
    for bin_id in range(n_bins):
        bin_data = train_with_cal[train_with_cal['price_bin'] == bin_id]
        if len(bin_data) < persistence_window:
            bin_momentum[bin_id] = {'direction': 0, 'strength': 0}
            continue
        
        last_valid = bin_data[bin_data['rolling_cal_error'].notna()].tail(1)
        if len(last_valid) == 0:
            bin_momentum[bin_id] = {'direction': 0, 'strength': 0}
            continue
            
        cal_error = last_valid['rolling_cal_error'].values[0]
        persistence = last_valid['error_persistence'].values[0]
        
        if abs(cal_error) >= momentum_threshold and persistence >= min_persistence_days:
            direction = 1 if cal_error > 0 else -1
            strength = min(abs(cal_error) / 0.20, 1.0)
            bin_momentum[bin_id] = {
                'direction': direction,
                'strength': strength,
                'cal_error': cal_error
            }
        else:
            bin_momentum[bin_id] = {'direction': 0, 'strength': 0}
    
    # Prepare test data
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    if category_col and category_col in test_copy.columns:
        test_copy['_category'] = test_copy[category_col].fillna('')
    else:
        test_copy['_category'] = ''
    test_copy['price_bin'] = pd.cut(test_copy['_price'], bins=n_bins, labels=False)
    
    # Run Monte Carlo simulations
    simulation_results = []
    
    for sim in range(n_simulations):
        sim_rng = np.random.default_rng(seed + sim)
        
        bankroll = initial_bankroll
        peak_bankroll = initial_bankroll
        pnl_list = []
        trades_attempted = 0
        trades_filled = 0
        total_impact = 0
        total_as_cost = 0
        wins = 0
        
        for _, row in test_copy.iterrows():
            price = row['_price']
            outcome = row['_outcome']
            bin_id = row['price_bin']
            category = row.get('_category', '')
            
            if pd.isna(bin_id) or bin_id not in bin_momentum:
                pnl_list.append(0)
                continue
            
            momentum = bin_momentum[bin_id]
            if momentum['direction'] == 0 or momentum['strength'] < 0.1:
                pnl_list.append(0)
                continue
            
            # This is a trade signal
            trades_attempted += 1
            
            # Compute position size
            edge = abs(momentum.get('cal_error', momentum_threshold))
            position_frac = kelly_fraction * momentum['strength'] * min(edge / 0.10, 1.0)
            position_frac = min(position_frac, max_position_pct)
            intended_size = initial_bankroll * position_frac
            
            if intended_size < 1:
                pnl_list.append(0)
                continue
            
            # Estimate liquidity
            liquidity = estimate_liquidity(price, category, impact_params)
            
            # Compute fill probability
            fill_prob = compute_fill_probability(
                intended_size, price, liquidity, impact_params, sim_rng
            )
            
            # Simulate fill
            filled = sim_rng.random() < fill_prob
            
            if not filled:
                pnl_list.append(0)
                continue
            
            trades_filled += 1
            
            # Compute price impact
            perm_impact, temp_impact = compute_price_impact(
                intended_size, liquidity, impact_params
            )
            impact_cost = (perm_impact + temp_impact) * intended_size
            total_impact += impact_cost
            
            # Adverse selection cost
            edge = abs(momentum.get('cal_error', 0.05))
            as_cost = compute_adverse_selection_cost(
                edge, 1.0, filled, impact_params
            ) * intended_size
            total_as_cost += as_cost
            
            # Direction
            direction = momentum['direction']
            
            # Compute PnL
            if direction > 0:  # Long YES
                gross_pnl = intended_size * (outcome - price)
            else:  # Long NO
                gross_pnl = intended_size * (price - outcome)
            
            # Net PnL after costs
            net_pnl = gross_pnl - impact_cost - as_cost - fee * intended_size
            
            pnl_list.append(net_pnl)
            bankroll += net_pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            
            if net_pnl > 0:
                wins += 1
        
        pnl_array = np.array(pnl_list)
        sharpe = np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252)
        max_dd = (peak_bankroll - min(bankroll, peak_bankroll)) / peak_bankroll if peak_bankroll > 0 else 0
        
        simulation_results.append({
            'total_pnl': np.sum(pnl_array),
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades_attempted': trades_attempted,
            'trades_filled': trades_filled,
            'total_impact': total_impact,
            'total_as_cost': total_as_cost,
            'wins': wins,
        })
    
    # Aggregate results
    mean_pnl = np.mean([r['total_pnl'] for r in simulation_results])
    mean_sharpe = np.mean([r['sharpe'] for r in simulation_results])
    mean_dd = np.mean([r['max_dd'] for r in simulation_results])
    total_attempted = np.mean([r['trades_attempted'] for r in simulation_results])
    total_filled = np.mean([r['trades_filled'] for r in simulation_results])
    total_impact = np.mean([r['total_impact'] for r in simulation_results])
    total_as = np.mean([r['total_as_cost'] for r in simulation_results])
    total_wins = np.mean([r['wins'] for r in simulation_results])
    
    return {
        'strategy': 'calibration_momentum',
        'impact_model': impact_params.name,
        'regime': impact_params.regime,
        'mean_pnl': mean_pnl,
        'mean_sharpe': mean_sharpe,
        'mean_drawdown': mean_dd,
        'trades_attempted': total_attempted,
        'trades_filled': total_filled,
        'fill_rate': total_filled / total_attempted if total_attempted > 0 else 0,
        'total_impact_cost': total_impact,
        'total_as_cost': total_as,
        'win_rate': total_wins / total_filled if total_filled > 0 else 0,
        'n_simulations': n_simulations,
        'min_sharpe': min(r['sharpe'] for r in simulation_results),
        'max_sharpe': max(r['sharpe'] for r in simulation_results),
        'p5_sharpe': np.percentile([r['sharpe'] for r in simulation_results], 5),
    }


def run_mean_reversion_with_impact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    # Strategy params
    spread_threshold: float = 0.03,
    n_bins: int = 10,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.10,
    fee: float = 0.01,
    # Simulation params
    n_simulations: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run calibration mean reversion (existing strategy) with realistic market impact.
    """
    rng = np.random.default_rng(seed)
    
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    category_col = 'category' if 'category' in train.columns else None
    
    # Learn calibration
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    train_copy['_outcome'] = train_copy[outcome_col]
    train_copy['price_bin'] = pd.cut(train_copy['_price'], bins=n_bins, labels=False)
    
    calibration = train_copy.groupby('price_bin').agg({
        '_price': 'mean',
        '_outcome': 'mean'
    }).rename(columns={'_price': 'bin_price', '_outcome': 'outcome_rate'})
    calibration['spread'] = calibration['outcome_rate'] - calibration['bin_price']
    
    # Prepare test data
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    if category_col and category_col in test_copy.columns:
        test_copy['_category'] = test_copy[category_col].fillna('')
    else:
        test_copy['_category'] = ''
    test_copy['price_bin'] = pd.cut(test_copy['_price'], bins=n_bins, labels=False)
    test_copy = test_copy.merge(
        calibration[['spread']], 
        left_on='price_bin', 
        right_index=True, 
        how='left'
    )
    test_copy['spread'] = test_copy['spread'].fillna(0)
    
    # Run simulations
    simulation_results = []
    
    for sim in range(n_simulations):
        sim_rng = np.random.default_rng(seed + sim)
        
        bankroll = initial_bankroll
        peak_bankroll = initial_bankroll
        pnl_list = []
        trades_attempted = 0
        trades_filled = 0
        total_impact = 0
        total_as_cost = 0
        wins = 0
        
        for _, row in test_copy.iterrows():
            price = row['_price']
            outcome = row['_outcome']
            spread = row.get('spread', 0)
            category = row.get('_category', '')
            
            if abs(spread) < spread_threshold:
                pnl_list.append(0)
                continue
            
            trades_attempted += 1
            
            # Position sizing
            edge = abs(spread)
            position_frac = kelly_fraction * min(edge / 0.10, 1.0)
            position_frac = min(position_frac, max_position_pct)
            intended_size = initial_bankroll * position_frac
            
            if intended_size < 1:
                pnl_list.append(0)
                continue
            
            # Liquidity
            liquidity = estimate_liquidity(price, category, impact_params)
            
            # Fill probability
            fill_prob = compute_fill_probability(
                intended_size, price, liquidity, impact_params, sim_rng
            )
            
            filled = sim_rng.random() < fill_prob
            
            if not filled:
                pnl_list.append(0)
                continue
            
            trades_filled += 1
            
            # Impact
            perm_impact, temp_impact = compute_price_impact(
                intended_size, liquidity, impact_params
            )
            impact_cost = (perm_impact + temp_impact) * intended_size
            total_impact += impact_cost
            
            # Adverse selection
            as_cost = compute_adverse_selection_cost(
                edge, 1.0, filled, impact_params
            ) * intended_size
            total_as_cost += as_cost
            
            # Direction (mean reversion: bet WITH the spread direction)
            direction = 1 if spread > 0 else -1
            
            # PnL
            if direction > 0:
                gross_pnl = intended_size * (outcome - price)
            else:
                gross_pnl = intended_size * (price - outcome)
            
            net_pnl = gross_pnl - impact_cost - as_cost - fee * intended_size
            
            pnl_list.append(net_pnl)
            bankroll += net_pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            
            if net_pnl > 0:
                wins += 1
        
        pnl_array = np.array(pnl_list)
        sharpe = np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252)
        max_dd = (peak_bankroll - min(bankroll, peak_bankroll)) / peak_bankroll if peak_bankroll > 0 else 0
        
        simulation_results.append({
            'total_pnl': np.sum(pnl_array),
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades_attempted': trades_attempted,
            'trades_filled': trades_filled,
            'total_impact': total_impact,
            'total_as_cost': total_as_cost,
            'wins': wins,
        })
    
    mean_pnl = np.mean([r['total_pnl'] for r in simulation_results])
    mean_sharpe = np.mean([r['sharpe'] for r in simulation_results])
    mean_dd = np.mean([r['max_dd'] for r in simulation_results])
    total_attempted = np.mean([r['trades_attempted'] for r in simulation_results])
    total_filled = np.mean([r['trades_filled'] for r in simulation_results])
    total_impact = np.mean([r['total_impact'] for r in simulation_results])
    total_as = np.mean([r['total_as_cost'] for r in simulation_results])
    total_wins = np.mean([r['wins'] for r in simulation_results])
    
    return {
        'strategy': 'mean_reversion',
        'impact_model': impact_params.name,
        'regime': impact_params.regime,
        'mean_pnl': mean_pnl,
        'mean_sharpe': mean_sharpe,
        'mean_drawdown': mean_dd,
        'trades_attempted': total_attempted,
        'trades_filled': total_filled,
        'fill_rate': total_filled / total_attempted if total_attempted > 0 else 0,
        'total_impact_cost': total_impact,
        'total_as_cost': total_as,
        'win_rate': total_wins / total_filled if total_filled > 0 else 0,
        'n_simulations': n_simulations,
        'min_sharpe': min(r['sharpe'] for r in simulation_results),
        'max_sharpe': max(r['sharpe'] for r in simulation_results),
        'p5_sharpe': np.percentile([r['sharpe'] for r in simulation_results], 5),
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest momentum strategies with impact')
    parser.add_argument('--data', type=str, default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, default='runs/momentum_backtest')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--n-simulations', type=int, default=50)
    parser.add_argument('--bankroll', type=float, default=10000.0)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Split
    n = len(df)
    train_idx = int(n * args.train_ratio)
    train = df.iloc[:train_idx].copy()
    test = df.iloc[train_idx:].copy()
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Impact models to test
    impact_models = [
        create_almgren_chriss_base(),
        create_almgren_chriss_conservative(),
    ]
    
    all_results = []
    
    for impact_model in impact_models:
        print(f"\n{'='*60}")
        print(f"Testing with impact model: {impact_model.name} ({impact_model.regime})")
        print(f"{'='*60}")
        
        # Test momentum strategy
        print("\nRunning Calibration Momentum...")
        momentum_result = run_calibration_momentum_with_impact(
            train, test, impact_model,
            initial_bankroll=args.bankroll,
            n_simulations=args.n_simulations,
        )
        print(f"  Mean PnL: ${momentum_result['mean_pnl']:,.0f}")
        print(f"  Mean Sharpe: {momentum_result['mean_sharpe']:.2f}")
        print(f"  Fill Rate: {momentum_result['fill_rate']:.1%}")
        print(f"  Trades Filled: {momentum_result['trades_filled']:.0f}")
        print(f"  Win Rate: {momentum_result['win_rate']:.1%}")
        print(f"  Impact Cost: ${momentum_result['total_impact_cost']:.2f}")
        print(f"  Adverse Selection: ${momentum_result['total_as_cost']:.2f}")
        all_results.append(momentum_result)
        
        # Test mean reversion strategy
        print("\nRunning Mean Reversion...")
        mr_result = run_mean_reversion_with_impact(
            train, test, impact_model,
            initial_bankroll=args.bankroll,
            n_simulations=args.n_simulations,
        )
        print(f"  Mean PnL: ${mr_result['mean_pnl']:,.0f}")
        print(f"  Mean Sharpe: {mr_result['mean_sharpe']:.2f}")
        print(f"  Fill Rate: {mr_result['fill_rate']:.1%}")
        print(f"  Trades Filled: {mr_result['trades_filled']:.0f}")
        print(f"  Win Rate: {mr_result['win_rate']:.1%}")
        print(f"  Impact Cost: ${mr_result['total_impact_cost']:.2f}")
        print(f"  Adverse Selection: ${mr_result['total_as_cost']:.2f}")
        all_results.append(mr_result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - REALISTIC BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} {'Model':<25} {'PnL':>12} {'Sharpe':>8} {'Fill%':>8} {'Trades':>8}")
    print("-" * 90)
    
    for r in all_results:
        print(f"{r['strategy']:<25} {r['impact_model']:<25} "
              f"${r['mean_pnl']:>11,.0f} {r['mean_sharpe']:>7.2f} "
              f"{r['fill_rate']:>7.1%} {r['trades_filled']:>7.0f}")
    
    # Save results
    with open(output_dir / 'momentum_backtest_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
