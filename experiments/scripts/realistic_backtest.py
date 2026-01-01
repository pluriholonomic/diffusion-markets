#!/usr/bin/env python3
"""
Realistic Backtesting with Market Impact and Fill Models

Key components:
1. Fill Probability Model - probability of getting filled based on size/liquidity
2. Price Impact Model - slippage when entering/exiting
3. Adverse Selection - correlation between fills and unfavorable moves
4. Exit Cost Model - cost to unwind positions (matters less for hold-to-expiry)

References:
- Almgren-Chriss market impact model
- Kyle's lambda for price impact
- Gueant-Lehalle-Fernandez-Tapia for optimal execution
"""

import argparse
import json
from dataclasses import dataclass, field
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
# Market Impact Models
# =============================================================================

@dataclass
class MarketImpactParams:
    """Parameters for market impact model."""
    
    # Fill probability parameters
    base_fill_prob: float = 0.80           # Base probability of fill for small orders
    fill_size_decay: float = 0.10          # How fill prob decays with size
    fill_spread_sensitivity: float = 2.0   # How fill prob decreases near 0.5 (competitive)
    
    # Price impact parameters (Almgren-Chriss style)
    permanent_impact: float = 0.001        # Permanent price impact per $ traded
    temporary_impact: float = 0.005        # Temporary slippage per $ traded
    impact_power: float = 0.5              # Impact scales as size^power (sqrt typical)
    
    # Adverse selection
    adverse_selection: float = 0.10        # Extra loss when filled (winner's curse)
    adverse_selection_decay: float = 0.5   # How adverse selection decays with hold time
    
    # Liquidity estimation
    base_liquidity: float = 10000.0        # Base daily liquidity in $
    liquidity_price_factor: float = 2.0    # Liquidity higher near 0.5
    
    # Exit costs (for early exit, not hold-to-expiry)
    exit_spread: float = 0.02              # Bid-ask spread for exit
    exit_impact_mult: float = 1.5          # Exit impact multiplier (urgency)


def estimate_market_liquidity(
    price: float,
    category: str,
    params: MarketImpactParams,
) -> float:
    """
    Estimate market liquidity based on price and category.
    
    Liquidity is typically higher:
    - Near price = 0.5 (competitive/uncertain markets)
    - For popular categories (politics, crypto)
    - Lower for extreme prices (near 0 or 1)
    """
    # Price factor: highest at 0.5, lowest at extremes
    price_factor = 4 * price * (1 - price)  # Peaks at 1.0 when price=0.5
    price_factor = max(price_factor, 0.1)   # Floor at 10%
    
    # Category factor
    category_factors = {
        'politics': 2.0,
        'crypto': 1.5,
        'sports': 1.2,
        'weather': 0.5,
        'economics': 0.8,
        'other': 1.0,
    }
    cat_factor = category_factors.get(category, 1.0)
    
    return params.base_liquidity * price_factor * cat_factor


def compute_fill_probability(
    order_size: float,
    price: float,
    liquidity: float,
    params: MarketImpactParams,
) -> float:
    """
    Compute probability of getting filled.
    
    Fill probability decreases with:
    - Larger order sizes (relative to liquidity)
    - Prices near 0.5 (more competition)
    - Lower overall liquidity
    """
    # Size effect: larger orders less likely to fill
    size_ratio = order_size / liquidity if liquidity > 0 else 1.0
    size_factor = np.exp(-params.fill_size_decay * size_ratio)
    
    # Spread effect: harder to fill near 0.5 (more competition)
    spread_factor = 1 - params.fill_spread_sensitivity * price * (1 - price) * 0.25
    spread_factor = max(spread_factor, 0.3)
    
    fill_prob = params.base_fill_prob * size_factor * spread_factor
    return np.clip(fill_prob, 0.01, 0.99)


def compute_price_impact(
    order_size: float,
    liquidity: float,
    params: MarketImpactParams,
) -> Tuple[float, float]:
    """
    Compute price impact using Almgren-Chriss style model.
    
    Returns:
        (permanent_impact, temporary_impact) as fractions
    """
    if liquidity <= 0:
        return (0.1, 0.1)  # Max impact for illiquid markets
    
    # Normalized size
    sigma = order_size / liquidity
    
    # Impact scales as size^power (typically sqrt)
    impact_base = sigma ** params.impact_power
    
    permanent = params.permanent_impact * impact_base
    temporary = params.temporary_impact * impact_base
    
    return (
        np.clip(permanent, 0, 0.1),
        np.clip(temporary, 0, 0.2),
    )


def compute_adverse_selection_cost(
    price: float,
    direction: int,  # 1 for buy YES, -1 for sell YES
    hold_time_days: float,
    params: MarketImpactParams,
) -> float:
    """
    Compute adverse selection cost.
    
    When you get filled, it's often because the market moved against you.
    This is the "winner's curse" - you win the order but the price is worse.
    
    Effect decays with holding time (less relevant for stat arb / long holds).
    """
    # Base adverse selection
    adverse = params.adverse_selection
    
    # Decay with hold time
    decay = np.exp(-params.adverse_selection_decay * hold_time_days)
    
    return adverse * decay


def compute_exit_cost(
    position_size: float,
    entry_price: float,
    current_price: float,
    liquidity: float,
    hold_to_expiry: bool,
    params: MarketImpactParams,
) -> float:
    """
    Compute cost to exit position.
    
    For hold-to-expiry strategies, this is zero (position settles at 0 or 1).
    For early exit, includes spread and impact costs.
    """
    if hold_to_expiry:
        return 0.0
    
    # Spread cost
    spread_cost = position_size * params.exit_spread / 2
    
    # Impact cost (worse than entry due to urgency)
    _, temp_impact = compute_price_impact(
        position_size, liquidity, params
    )
    impact_cost = position_size * temp_impact * params.exit_impact_mult
    
    return spread_cost + impact_cost


# =============================================================================
# Realistic Strategy Evaluation
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade."""
    market_id: str
    entry_price: float
    direction: int
    intended_size: float
    actual_size: float
    fill_prob: float
    was_filled: bool
    entry_slippage: float
    adverse_selection_cost: float
    outcome: int
    gross_pnl: float
    net_pnl: float
    bankroll_after: float


def realistic_convergence_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Strategy params
    spread_threshold: float = 0.05,
    confidence_threshold: float = 2.0,
    n_bins: int = 10,
    # Sizing
    initial_bankroll: float = 10000.0,
    base_size_pct: float = 0.03,
    max_position_pct: float = 0.10,
    # Market impact params
    impact_params: Optional[MarketImpactParams] = None,
    # Risk management
    fee: float = 0.01,
    max_drawdown_stop: float = 0.30,
    hold_to_expiry: bool = True,          # Stat arb holds to resolution
    # Monte Carlo
    n_simulations: int = 100,             # Run multiple sims for fill randomness
) -> Dict[str, Any]:
    """
    Realistic convergence strategy with market impact.
    
    Key differences from naive backtest:
    1. Orders may not fill (fill probability)
    2. Entry prices are worse (slippage)
    3. Adverse selection when filled
    4. Exit costs if not holding to expiry
    """
    if impact_params is None:
        impact_params = MarketImpactParams()
    
    # Compute calibration spreads from training
    prices = train['first_price'].values
    outcomes = train['y'].values
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(prices, bin_edges) - 1, 0, n_bins - 1)
    
    bin_spreads = {}
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 20:
            spread = outcomes[mask].mean() - prices[mask].mean()
            se = (outcomes[mask] - prices[mask]).std() / np.sqrt(mask.sum())
            t_stat = spread / se if se > 0 else 0
            bin_spreads[b] = {'spread': spread, 't_stat': t_stat}
    
    # Run multiple simulations to account for fill randomness
    all_results = []
    
    for sim in range(n_simulations):
        np.random.seed(sim)  # Reproducible randomness
        
        bankroll = initial_bankroll
        peak_bankroll = bankroll
        trades = []
        
        for _, row in test.iterrows():
            price = row['first_price']
            outcome = row['y']
            category = row.get('category', 'other')
            market_id = str(row.get('goldsky_id', row.name))
            
            # Check drawdown stop
            dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            if dd >= max_drawdown_stop:
                break
            
            if bankroll <= 0:
                break
            
            # Get signal
            b = int(np.clip(np.digitize(price, bin_edges) - 1, 0, n_bins - 1))
            if b not in bin_spreads:
                continue
            
            signal = bin_spreads[b]
            if abs(signal['spread']) < spread_threshold:
                continue
            if abs(signal['t_stat']) < confidence_threshold:
                continue
            
            direction = 1 if signal['spread'] > 0 else -1
            
            # Compute intended position size
            intended_size = bankroll * base_size_pct
            intended_size = min(intended_size, bankroll * max_position_pct)
            
            # Estimate liquidity
            liquidity = estimate_market_liquidity(price, category, impact_params)
            
            # Compute fill probability
            fill_prob = compute_fill_probability(
                intended_size, price, liquidity, impact_params
            )
            
            # Simulate fill
            was_filled = np.random.random() < fill_prob
            
            if not was_filled:
                trades.append(TradeRecord(
                    market_id=market_id,
                    entry_price=price,
                    direction=direction,
                    intended_size=intended_size,
                    actual_size=0,
                    fill_prob=fill_prob,
                    was_filled=False,
                    entry_slippage=0,
                    adverse_selection_cost=0,
                    outcome=outcome,
                    gross_pnl=0,
                    net_pnl=0,
                    bankroll_after=bankroll,
                ))
                continue
            
            # Compute slippage
            perm_impact, temp_impact = compute_price_impact(
                intended_size, liquidity, impact_params
            )
            entry_slippage = temp_impact
            
            # Compute adverse selection
            hold_time = 30  # Assume ~30 days average hold
            adverse_cost = compute_adverse_selection_cost(
                price, direction, hold_time, impact_params
            )
            
            # Actual entry price after slippage (worse for us)
            if direction > 0:  # Buying YES
                actual_entry = price * (1 + entry_slippage)
            else:  # Selling YES (buying NO)
                actual_entry = price * (1 - entry_slippage)
            
            actual_entry = np.clip(actual_entry, 0.01, 0.99)
            
            # Compute PnL (holding to expiry)
            if direction > 0:  # Long YES
                if outcome == 1:
                    gross_pnl = intended_size * (1 - actual_entry) / actual_entry
                else:
                    gross_pnl = -intended_size
            else:  # Short YES (long NO)
                if outcome == 0:
                    gross_pnl = intended_size * actual_entry / (1 - actual_entry)
                else:
                    gross_pnl = -intended_size
            
            # Net PnL after costs
            fee_cost = intended_size * fee
            adverse_cost_dollars = intended_size * adverse_cost
            
            net_pnl = gross_pnl - fee_cost - adverse_cost_dollars
            
            # Clamp to reasonable bounds
            net_pnl = np.clip(net_pnl, -intended_size * 1.5, intended_size * 5)
            
            bankroll += net_pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            
            trades.append(TradeRecord(
                market_id=market_id,
                entry_price=price,
                direction=direction,
                intended_size=intended_size,
                actual_size=intended_size,
                fill_prob=fill_prob,
                was_filled=True,
                entry_slippage=entry_slippage,
                adverse_selection_cost=adverse_cost,
                outcome=outcome,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                bankroll_after=bankroll,
            ))
        
        # Compute metrics for this simulation
        filled_trades = [t for t in trades if t.was_filled]
        
        if filled_trades:
            pnls = np.array([t.net_pnl for t in filled_trades])
            total_pnl = bankroll - initial_bankroll
            win_rate = sum(1 for t in filled_trades if t.net_pnl > 0) / len(filled_trades)
            sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
            max_dd = max(
                (peak_bankroll - t.bankroll_after) / peak_bankroll 
                for t in filled_trades
            ) if peak_bankroll > 0 else 0
            avg_fill_rate = np.mean([t.fill_prob for t in trades])
            actual_fill_rate = len(filled_trades) / len(trades) if trades else 0
            avg_slippage = np.mean([t.entry_slippage for t in filled_trades])
        else:
            total_pnl = 0
            win_rate = 0
            sharpe = 0
            max_dd = 0
            avg_fill_rate = 0
            actual_fill_rate = 0
            avg_slippage = 0
        
        all_results.append({
            'total_pnl': total_pnl,
            'final_bankroll': bankroll,
            'n_trades_attempted': len(trades),
            'n_trades_filled': len(filled_trades),
            'fill_rate': actual_fill_rate,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'avg_slippage': avg_slippage,
        })
    
    # Aggregate across simulations
    df_results = pd.DataFrame(all_results)
    
    return {
        'mean_pnl': float(df_results['total_pnl'].mean()),
        'std_pnl': float(df_results['total_pnl'].std()),
        'pnl_5th': float(df_results['total_pnl'].quantile(0.05)),
        'pnl_95th': float(df_results['total_pnl'].quantile(0.95)),
        'mean_sharpe': float(df_results['sharpe'].mean()),
        'std_sharpe': float(df_results['sharpe'].std()),
        'mean_final_bankroll': float(df_results['final_bankroll'].mean()),
        'mean_return_pct': float((df_results['final_bankroll'].mean() - initial_bankroll) / initial_bankroll * 100),
        'mean_trades_attempted': float(df_results['n_trades_attempted'].mean()),
        'mean_trades_filled': float(df_results['n_trades_filled'].mean()),
        'mean_fill_rate': float(df_results['fill_rate'].mean()),
        'mean_win_rate': float(df_results['win_rate'].mean()),
        'mean_max_dd': float(df_results['max_dd'].mean()),
        'mean_slippage': float(df_results['avg_slippage'].mean()),
        'n_simulations': n_simulations,
    }


# =============================================================================
# Optimization with Realistic Model
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


def optimize_realistic_strategy(
    data: pd.DataFrame,
    n_trials: int = 500,
    n_folds: int = 5,
    n_simulations: int = 50,
    n_jobs: int = 16,
    initial_bankroll: float = 10000.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Optimize strategy with realistic market impact model."""
    
    splits, holdout = walk_forward_split(data, n_folds=n_folds)
    
    if not splits:
        return {'error': 'Not enough data'}
    
    print(f"\n{'='*70}")
    print("REALISTIC BACKTEST OPTIMIZATION")
    print(f"Dataset: {len(data):,} markets")
    print(f"CV folds: {len(splits)}")
    print(f"Trials: {n_trials}")
    print(f"Simulations per trial: {n_simulations}")
    print(f"Initial bankroll: ${initial_bankroll:,.0f}")
    print(f"{'='*70}\n")
    
    def objective(trial: optuna.Trial) -> float:
        # Strategy params
        spread_threshold = trial.suggest_float('spread_threshold', 0.02, 0.15)
        confidence_threshold = trial.suggest_float('confidence_threshold', 1.5, 4.0)
        n_bins = trial.suggest_int('n_bins', 5, 20)
        base_size_pct = trial.suggest_float('base_size_pct', 0.01, 0.08)
        max_position_pct = trial.suggest_float('max_position_pct', 0.03, 0.15)
        fee = trial.suggest_float('fee', 0.005, 0.02)
        max_drawdown_stop = trial.suggest_float('max_drawdown_stop', 0.15, 0.40)
        
        # Market impact params
        impact_params = MarketImpactParams(
            base_fill_prob=trial.suggest_float('base_fill_prob', 0.5, 0.95),
            fill_size_decay=trial.suggest_float('fill_size_decay', 0.05, 0.30),
            permanent_impact=trial.suggest_float('permanent_impact', 0.0005, 0.005),
            temporary_impact=trial.suggest_float('temporary_impact', 0.002, 0.02),
            adverse_selection=trial.suggest_float('adverse_selection', 0.02, 0.15),
            base_liquidity=trial.suggest_float('base_liquidity', 5000, 50000),
        )
        
        sharpes = []
        for train, test in splits:
            result = realistic_convergence_strategy(
                train, test,
                spread_threshold=spread_threshold,
                confidence_threshold=confidence_threshold,
                n_bins=n_bins,
                initial_bankroll=initial_bankroll,
                base_size_pct=base_size_pct,
                max_position_pct=max_position_pct,
                impact_params=impact_params,
                fee=fee,
                max_drawdown_stop=max_drawdown_stop,
                n_simulations=n_simulations,
            )
            if result['mean_trades_filled'] >= 10:
                sharpes.append(result['mean_sharpe'])
        
        return np.mean(sharpes) if sharpes else float('-inf')
    
    sampler = CmaEsSampler(n_startup_trials=30, warn_independent_sampling=False)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"Best CV Sharpe: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Holdout evaluation with more simulations
    if holdout is not None and len(holdout) > 0:
        print(f"\n{'='*70}")
        print("HOLDOUT EVALUATION (500 simulations)")
        print(f"{'='*70}")
        
        train_all = data.iloc[:-len(holdout)]
        
        # Reconstruct impact params
        bp = study.best_params
        impact_params = MarketImpactParams(
            base_fill_prob=bp['base_fill_prob'],
            fill_size_decay=bp['fill_size_decay'],
            permanent_impact=bp['permanent_impact'],
            temporary_impact=bp['temporary_impact'],
            adverse_selection=bp['adverse_selection'],
            base_liquidity=bp['base_liquidity'],
        )
        
        holdout_result = realistic_convergence_strategy(
            train_all, holdout,
            spread_threshold=bp['spread_threshold'],
            confidence_threshold=bp['confidence_threshold'],
            n_bins=bp['n_bins'],
            initial_bankroll=initial_bankroll,
            base_size_pct=bp['base_size_pct'],
            max_position_pct=bp['max_position_pct'],
            impact_params=impact_params,
            fee=bp['fee'],
            max_drawdown_stop=bp['max_drawdown_stop'],
            n_simulations=500,
        )
        
        print(f"\nHoldout Results (averaged over 500 simulations):")
        print(f"  Mean Sharpe:     {holdout_result['mean_sharpe']:.2f} ± {holdout_result['std_sharpe']:.2f}")
        print(f"  Mean PnL:        ${holdout_result['mean_pnl']:,.2f} ± ${holdout_result['std_pnl']:,.2f}")
        print(f"  5th/95th PnL:    ${holdout_result['pnl_5th']:,.2f} / ${holdout_result['pnl_95th']:,.2f}")
        print(f"  Mean Return:     {holdout_result['mean_return_pct']:.1f}%")
        print(f"  Trades Attempted: {holdout_result['mean_trades_attempted']:.0f}")
        print(f"  Trades Filled:   {holdout_result['mean_trades_filled']:.0f}")
        print(f"  Fill Rate:       {holdout_result['mean_fill_rate']:.1%}")
        print(f"  Win Rate:        {holdout_result['mean_win_rate']:.1%}")
        print(f"  Max Drawdown:    {holdout_result['mean_max_dd']:.1%}")
        print(f"  Avg Slippage:    {holdout_result['mean_slippage']:.2%}")
    else:
        holdout_result = None
    
    return {
        'cv_sharpe': study.best_value,
        'best_params': study.best_params,
        'holdout': holdout_result,
        'initial_bankroll': initial_bankroll,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-path', type=str, default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, default='runs/realistic_backtest')
    parser.add_argument('--n-trials', type=int, default=500)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-simulations', type=int, default=50)
    parser.add_argument('--n-jobs', type=int, default=16)
    parser.add_argument('--initial-bankroll', type=float, default=10000.0)
    args = parser.parse_args()
    
    # Load data
    cache_path = Path(args.cache_path)
    print(f"Loading data from {cache_path}")
    data = pd.read_parquet(cache_path)
    print(f"Loaded {len(data):,} markets")
    
    # Sort by resolution time for proper temporal ordering
    if 'resolution_time' in data.columns:
        data = data.sort_values('resolution_time').reset_index(drop=True)
    
    # Run optimization
    results = optimize_realistic_strategy(
        data=data,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        n_simulations=args.n_simulations,
        n_jobs=args.n_jobs,
        initial_bankroll=args.initial_bankroll,
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
            return obj
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
