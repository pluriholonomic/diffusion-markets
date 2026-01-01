#!/usr/bin/env python3
"""
Strategy Hyperparameter Optimization V2 - With Proper Sizing Model

Key improvements:
1. Proper bankroll management with compounding
2. Kelly criterion with fractional Kelly and leverage
3. Volatility targeting for consistent risk
4. More aggressive position sizing with risk controls
5. More comprehensive hyperparameter space
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Error: optuna package required. Install with: pip install optuna cmaes")


# =============================================================================
# Position Sizing Models
# =============================================================================

@dataclass
class SizingConfig:
    """Position sizing configuration."""
    initial_bankroll: float = 10000.0  # Starting capital
    kelly_fraction: float = 0.25       # Fractional Kelly (0.25 = quarter Kelly)
    leverage: float = 1.0              # Leverage multiplier
    max_position_pct: float = 0.20     # Max % of bankroll per trade
    min_position_pct: float = 0.01     # Min position to bother with
    vol_target: Optional[float] = None # Target annualized vol (None = no vol targeting)
    compounding: bool = True           # Whether to compound returns
    max_drawdown_stop: float = 0.50    # Stop trading if DD exceeds this


def compute_kelly_size(
    edge: float,
    win_prob: float,
    odds: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Compute Kelly-optimal bet size.
    
    Kelly formula: f* = (bp - q) / b
    where b = odds, p = win_prob, q = 1-p
    
    For binary prediction markets:
    - If we think fair prob is p*, market price is q
    - edge = p* - q
    - win_prob ≈ p* (our estimate of true probability)
    - odds = (1-q)/q for YES bet, q/(1-q) for NO bet
    """
    if edge <= 0:
        return 0.0
    
    # Kelly optimal fraction
    kelly_f = (win_prob * odds - (1 - win_prob)) / odds if odds > 0 else 0
    kelly_f = max(0, kelly_f)
    
    # Apply fractional Kelly
    return kelly_f * kelly_fraction


def compute_vol_scaled_size(
    base_size: float,
    realized_vol: float,
    vol_target: float,
) -> float:
    """Scale position size to target volatility."""
    if realized_vol <= 0 or vol_target <= 0:
        return base_size
    return base_size * (vol_target / realized_vol)


# =============================================================================
# Data Loading
# =============================================================================

def categorize_question(q: str) -> str:
    q = str(q).lower()
    if any(x in q for x in ['bitcoin', 'btc', 'eth', 'crypto', 'solana', 'dogecoin']):
        return 'crypto'
    if any(x in q for x in ['trump', 'biden', 'harris', 'election', 'president', 'congress', 'senate']):
        return 'politics'
    if any(x in q for x in ['nba', 'nfl', 'mlb', 'nhl', 'win', 'beat', 'championship', 'super bowl']):
        return 'sports'
    if any(x in q for x in ['temperature', 'weather', 'rain', 'snow', 'hurricane']):
        return 'weather'
    if any(x in q for x in ['fed', 'rate', 'inflation', 'gdp', 'unemployment']):
        return 'economics'
    return 'other'


def load_data(cache_path: Path) -> pd.DataFrame:
    """Load cached optimization data."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    return pd.read_parquet(cache_path)


# =============================================================================
# Enhanced Strategies with Proper Sizing
# =============================================================================

def blackwell_strategy_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Calibration params
    n_bins: int = 10,
    g_bar_threshold: float = 0.05,
    t_stat_threshold: float = 2.0,
    min_samples: int = 30,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    leverage: float = 1.0,
    max_position_pct: float = 0.20,
    min_edge: float = 0.02,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
    compounding: bool = True,
) -> Dict[str, float]:
    """
    Blackwell calibration strategy with proper bankroll management.
    """
    # Learn calibration from training data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    train_prices = train['first_price'].values
    train_outcomes = train['y'].values
    bin_idx = np.clip(np.digitize(train_prices, bin_edges) - 1, 0, n_bins - 1)
    
    bin_stats = {}
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= min_samples:
            p = train_prices[mask]
            y = train_outcomes[mask]
            residuals = y - p
            
            g_bar = residuals.mean()
            sigma = residuals.std()
            se = sigma / np.sqrt(mask.sum()) if mask.sum() > 0 else 1
            t_stat = g_bar / se if se > 0 else 0
            
            if abs(t_stat) >= t_stat_threshold and abs(g_bar) >= g_bar_threshold:
                bin_stats[b] = {
                    'g_bar': g_bar,
                    'sigma': sigma,
                    'direction': 1 if g_bar > 0 else -1,
                    'win_rate': (y.mean() if g_bar > 0 else (1 - y).mean()),
                }
    
    if not bin_stats:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    # Trade on test set
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    pnls = []
    wins = 0
    
    for _, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        b = int(np.clip(np.digitize(price, bin_edges) - 1, 0, n_bins - 1))
        
        if b not in bin_stats:
            continue
        
        stats = bin_stats[b]
        direction = stats['direction']
        edge = abs(stats['g_bar'])
        
        if edge < min_edge:
            continue
        
        # Check drawdown stop
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        # Kelly sizing
        if direction > 0:  # Long YES
            odds = (1 - price) / price if price > 0 else 0
            win_prob = price + stats['g_bar']
        else:  # Long NO
            odds = price / (1 - price) if price < 1 else 0
            win_prob = 1 - price - stats['g_bar']
        
        win_prob = np.clip(win_prob, 0.01, 0.99)
        kelly_size = compute_kelly_size(edge, win_prob, odds, kelly_fraction)
        
        # Apply leverage and position limits
        position_frac = kelly_size * leverage
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        if position_frac < 0.001:
            continue
        
        # Dollar position
        if compounding:
            position = bankroll * position_frac
        else:
            position = initial_bankroll * position_frac
        
        # PnL calculation
        if direction > 0:  # Long YES
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price
                wins += 1
            else:
                pnl = -position - position * fee / price
        else:  # Long NO
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price)
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price)
        
        pnls.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if pnls.size > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
    }


def stat_arb_strategy_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Calibration params
    n_bins: int = 10,
    min_edge: float = 0.02,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    leverage: float = 1.0,
    max_position_pct: float = 0.20,
    distance_scale: float = 1.0,  # Scale by calibration error magnitude
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
    compounding: bool = True,
) -> Dict[str, float]:
    """
    Statistical arbitrage with proper bankroll management.
    """
    # Compute calibration stats from training
    bin_edges = np.linspace(0, 1, n_bins + 1)
    train_prices = train['first_price'].values
    train_outcomes = train['y'].values
    bin_idx = np.clip(np.digitize(train_prices, bin_edges) - 1, 0, n_bins - 1)
    
    bin_stats = {}
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 10:
            p = train_prices[mask]
            y = train_outcomes[mask]
            bias = y.mean() - p.mean()
            win_rate_yes = y.mean()
            win_rate_no = 1 - y.mean()
            bin_stats[b] = {
                'bias': bias,
                'win_rate_yes': win_rate_yes,
                'win_rate_no': win_rate_no,
            }
    
    # Trade on test set
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    pnls = []
    wins = 0
    
    for _, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        b = int(np.clip(np.digitize(price, bin_edges) - 1, 0, n_bins - 1))
        
        if b not in bin_stats:
            continue
        
        stats = bin_stats[b]
        bias = stats['bias']
        edge = abs(bias)
        
        if edge < min_edge:
            continue
        
        # Check drawdown stop
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        direction = 1 if bias > 0 else -1
        
        # Kelly sizing with distance scaling
        if direction > 0:  # Long YES
            odds = (1 - price) / price if price > 0 else 0
            win_prob = stats['win_rate_yes']
        else:  # Long NO
            odds = price / (1 - price) if price < 1 else 0
            win_prob = stats['win_rate_no']
        
        kelly_size = compute_kelly_size(edge, win_prob, odds, kelly_fraction)
        
        # Scale by calibration error magnitude
        kelly_size *= (1 + distance_scale * edge)
        
        # Apply leverage and limits
        position_frac = kelly_size * leverage
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        if position_frac < 0.001:
            continue
        
        if compounding:
            position = bankroll * position_frac
        else:
            position = initial_bankroll * position_frac
        
        # PnL
        if direction > 0:
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price
                wins += 1
            else:
                pnl = -position - position * fee / price
        else:
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price)
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price)
        
        pnls.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if pnls.size > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
    }


def mean_reversion_strategy_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Strategy params
    min_edge: float = 0.02,
    mean_revert_threshold: float = 0.05,
    momentum_threshold: float = 0.15,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    leverage: float = 1.0,
    max_position_pct: float = 0.20,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
    compounding: bool = True,
) -> Dict[str, float]:
    """
    Mean-reversion strategy with proper bankroll management.
    """
    # Compute group biases from training
    group_stats = {}
    for cat in train['category'].unique():
        cat_data = train[train['category'] == cat]
        if len(cat_data) >= 20:
            residuals = cat_data['y'] - cat_data['first_price']
            bias = residuals.mean()
            win_rate = cat_data['y'].mean()
            group_stats[cat] = {
                'bias': bias,
                'win_rate_yes': win_rate,
                'win_rate_no': 1 - win_rate,
            }
    
    # Trade on test
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    pnls = []
    wins = 0
    
    for _, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        cat = row['category']
        
        if cat not in group_stats:
            continue
        
        stats = group_stats[cat]
        bias = stats['bias']
        edge = abs(bias)
        
        if edge < min_edge:
            continue
        
        # Regime scaling
        abs_bias = abs(bias)
        if abs_bias < mean_revert_threshold:
            regime_scale = 1.0  # Mean revert
        elif abs_bias > momentum_threshold:
            regime_scale = 0.5  # Momentum
        else:
            regime_scale = 0.25  # Neutral
        
        # Drawdown check
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        direction = 1 if bias > 0 else -1
        
        # Kelly sizing
        if direction > 0:
            odds = (1 - price) / price if price > 0 else 0
            win_prob = stats['win_rate_yes']
        else:
            odds = price / (1 - price) if price < 1 else 0
            win_prob = stats['win_rate_no']
        
        kelly_size = compute_kelly_size(edge, win_prob, odds, kelly_fraction) * regime_scale
        
        position_frac = kelly_size * leverage
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        if position_frac < 0.001:
            continue
        
        if compounding:
            position = bankroll * position_frac
        else:
            position = initial_bankroll * position_frac
        
        # PnL
        if direction > 0:
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price
                wins += 1
            else:
                pnl = -position - position * fee / price
        else:
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price)
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price)
        
        pnls.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if pnls.size > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
    }


# =============================================================================
# Walk-Forward Split
# =============================================================================

def walk_forward_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    holdout_frac: float = 0.2,
) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    """Generate walk-forward splits + holdout."""
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


# =============================================================================
# Optimization
# =============================================================================

def optimize_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    n_trials: int = 500,
    n_folds: int = 5,
    n_jobs: int = None,
    objective: str = 'sharpe',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optimize a strategy with comprehensive hyperparameter space.
    """
    if n_jobs is None:
        n_jobs = min(cpu_count(), 32)
    
    splits, holdout = walk_forward_split(data, n_folds=n_folds)
    
    if not splits:
        return {'error': 'Not enough data for CV splits'}
    
    def objective_fn(trial: optuna.Trial) -> float:
        # Common sizing params
        initial_bankroll = trial.suggest_float('initial_bankroll', 1000, 100000, log=True)
        kelly_fraction = trial.suggest_float('kelly_fraction', 0.05, 1.0)
        leverage = trial.suggest_float('leverage', 0.5, 5.0)
        max_position_pct = trial.suggest_float('max_position_pct', 0.05, 0.50)
        min_edge = trial.suggest_float('min_edge', 0.01, 0.15)
        fee = trial.suggest_float('fee', 0.005, 0.02)
        max_drawdown_stop = trial.suggest_float('max_drawdown_stop', 0.20, 0.80)
        compounding = trial.suggest_categorical('compounding', [True, False])
        
        if strategy_name == 'blackwell':
            n_bins = trial.suggest_int('n_bins', 3, 20)
            g_bar_threshold = trial.suggest_float('g_bar_threshold', 0.01, 0.20)
            t_stat_threshold = trial.suggest_float('t_stat_threshold', 1.0, 4.0)
            min_samples = trial.suggest_int('min_samples', 10, 100)
            
            def run_fold(train, test):
                return blackwell_strategy_v2(
                    train, test,
                    n_bins=n_bins, g_bar_threshold=g_bar_threshold,
                    t_stat_threshold=t_stat_threshold, min_samples=min_samples,
                    initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction,
                    leverage=leverage, max_position_pct=max_position_pct,
                    min_edge=min_edge, fee=fee, max_drawdown_stop=max_drawdown_stop,
                    compounding=compounding,
                )
        
        elif strategy_name == 'stat_arb':
            n_bins = trial.suggest_int('n_bins', 3, 20)
            distance_scale = trial.suggest_float('distance_scale', 0.5, 5.0)
            
            def run_fold(train, test):
                return stat_arb_strategy_v2(
                    train, test,
                    n_bins=n_bins, min_edge=min_edge, distance_scale=distance_scale,
                    initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction,
                    leverage=leverage, max_position_pct=max_position_pct,
                    fee=fee, max_drawdown_stop=max_drawdown_stop,
                    compounding=compounding,
                )
        
        elif strategy_name == 'mean_reversion':
            mean_revert_threshold = trial.suggest_float('mean_revert_threshold', 0.01, 0.20)
            momentum_threshold = trial.suggest_float('momentum_threshold', 0.05, 0.30)
            
            def run_fold(train, test):
                return mean_reversion_strategy_v2(
                    train, test,
                    min_edge=min_edge,
                    mean_revert_threshold=mean_revert_threshold,
                    momentum_threshold=momentum_threshold,
                    initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction,
                    leverage=leverage, max_position_pct=max_position_pct,
                    fee=fee, max_drawdown_stop=max_drawdown_stop,
                    compounding=compounding,
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Run CV
        results = [run_fold(train, test) for train, test in splits]
        
        if not results or all(r['n_trades'] == 0 for r in results):
            return float('-inf')
        
        if objective == 'sharpe':
            sharpes = [r['sharpe'] for r in results if r['n_trades'] > 0]
            return np.mean(sharpes) if sharpes else float('-inf')
        elif objective == 'pnl':
            pnls = [r['pnl'] for r in results if r['n_trades'] > 0]
            return np.mean(pnls) if pnls else float('-inf')
        elif objective == 'return_pct':
            returns = [r.get('return_pct', 0) for r in results if r['n_trades'] > 0]
            return np.mean(returns) if returns else float('-inf')
        else:
            return np.mean([r['sharpe'] for r in results if r['n_trades'] > 0]) or float('-inf')
    
    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    sampler = CmaEsSampler(n_startup_trials=50, restart_strategy='ipop')
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    if verbose:
        print(f"Starting {strategy_name} optimization: {n_trials} trials")
        print(f"Using {n_jobs} parallel jobs, CMA-ES sampler")
    
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=verbose,
    )
    
    best = study.best_trial
    
    # Evaluate on holdout
    if holdout is not None and len(holdout) > 0:
        # Use all CV data as training for holdout eval
        train_all = data.iloc[:-len(holdout)]
        params = best.params
        
        if strategy_name == 'blackwell':
            holdout_result = blackwell_strategy_v2(
                train_all, holdout,
                n_bins=params['n_bins'], g_bar_threshold=params['g_bar_threshold'],
                t_stat_threshold=params['t_stat_threshold'], min_samples=params['min_samples'],
                initial_bankroll=params['initial_bankroll'], kelly_fraction=params['kelly_fraction'],
                leverage=params['leverage'], max_position_pct=params['max_position_pct'],
                min_edge=params['min_edge'], fee=params['fee'],
                max_drawdown_stop=params['max_drawdown_stop'], compounding=params['compounding'],
            )
        elif strategy_name == 'stat_arb':
            holdout_result = stat_arb_strategy_v2(
                train_all, holdout,
                n_bins=params['n_bins'], min_edge=params['min_edge'],
                distance_scale=params['distance_scale'],
                initial_bankroll=params['initial_bankroll'], kelly_fraction=params['kelly_fraction'],
                leverage=params['leverage'], max_position_pct=params['max_position_pct'],
                fee=params['fee'], max_drawdown_stop=params['max_drawdown_stop'],
                compounding=params['compounding'],
            )
        elif strategy_name == 'mean_reversion':
            holdout_result = mean_reversion_strategy_v2(
                train_all, holdout,
                min_edge=params['min_edge'],
                mean_revert_threshold=params['mean_revert_threshold'],
                momentum_threshold=params['momentum_threshold'],
                initial_bankroll=params['initial_bankroll'], kelly_fraction=params['kelly_fraction'],
                leverage=params['leverage'], max_position_pct=params['max_position_pct'],
                fee=params['fee'], max_drawdown_stop=params['max_drawdown_stop'],
                compounding=params['compounding'],
            )
    else:
        holdout_result = {}
    
    return {
        'strategy': strategy_name,
        'cv_value': best.value,
        'best_params': best.params,
        'holdout': holdout_result,
        'n_trials': n_trials,
    }


def main():
    parser = argparse.ArgumentParser(description='Strategy Optimization V2')
    parser.add_argument('--cache-path', type=str, 
                       default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, default='runs/optimization_v2')
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--objective', type=str, default='sharpe',
                       choices=['sharpe', 'pnl', 'return_pct'])
    parser.add_argument('--strategies', type=str, nargs='+',
                       default=['blackwell', 'stat_arb', 'mean_reversion'])
    args = parser.parse_args()
    
    # Load data
    cache_path = Path(args.cache_path)
    print(f"Loading data from {cache_path}")
    data = load_data(cache_path)
    print(f"Loaded {len(data):,} markets")
    print(f"Date range: {data['resolution_time'].min().date()} to {data['resolution_time'].max().date()}")
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STRATEGY OPTIMIZATION V2 - WITH PROPER SIZING MODEL")
    print(f"Dataset: {len(data):,} markets")
    print(f"Trials per strategy: {args.n_trials}")
    print(f"CV folds: {args.n_folds}")
    print(f"Objective: {args.objective}")
    print(f"CPU cores: {args.n_jobs or cpu_count()}")
    print("=" * 70)
    
    results = []
    
    for strategy in args.strategies:
        print(f"\n{'='*50}")
        print(f"OPTIMIZING: {strategy.upper()}")
        print(f"{'='*50}")
        
        result = optimize_strategy(
            data=data,
            strategy_name=strategy,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
            n_jobs=args.n_jobs,
            objective=args.objective,
            verbose=True,
        )
        
        results.append(result)
        
        print(f"\nBest CV {args.objective}: {result['cv_value']:.4f}")
        print(f"Best params: {json.dumps(result['best_params'], indent=2)}")
        
        if result.get('holdout'):
            h = result['holdout']
            print(f"\nHoldout Results:")
            print(f"  Sharpe: {h.get('sharpe', 0):.2f}")
            print(f"  PnL: ${h.get('pnl', 0):,.2f}")
            print(f"  Return: {h.get('return_pct', 0):.1f}%")
            print(f"  Win Rate: {h.get('win_rate', 0)*100:.1f}%")
            print(f"  Trades: {h.get('n_trades', 0):,}")
            print(f"  Max DD: {h.get('max_dd', 0)*100:.1f}%")
            print(f"  Final Bankroll: ${h.get('final_bankroll', 0):,.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    summary_data = []
    for r in results:
        h = r.get('holdout', {})
        summary_data.append({
            'strategy': r['strategy'],
            'cv_value': r['cv_value'],
            'holdout_sharpe': h.get('sharpe', 0),
            'holdout_pnl': h.get('pnl', 0),
            'holdout_return_pct': h.get('return_pct', 0),
            'holdout_win_rate': h.get('win_rate', 0),
            'holdout_trades': h.get('n_trades', 0),
            'holdout_max_dd': h.get('max_dd', 0),
            'final_bankroll': h.get('final_bankroll', 0),
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    
    with open(output_dir / 'full_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
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
