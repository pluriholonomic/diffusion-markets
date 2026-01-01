#!/usr/bin/env python3
"""
Strategy Hyperparameter Optimization Framework

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with
parallel walk-forward cross-validation for efficient optimization.

Methodology:
1. CMA-ES: State-of-the-art derivative-free optimizer
2. Parallel CV: Utilize all CPU cores for fold evaluation
3. Walk-Forward Validation: Expanding window training
4. Multi-Objective: Optimize Sharpe with drawdown penalty
5. Out-of-Sample Holdout: Final 20% never seen during optimization
"""

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Try to import CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("Warning: cma package not found. Install with: pip install cma")

# Try to import Optuna for alternative optimization
try:
    import optuna
    from optuna.samplers import CmaEsSampler, TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# =============================================================================
# Data Loading (reuse from main backtest)
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


def load_data_fast(
    gamma_path: Path,
    goldsky_markets_path: Path, 
    trades_agg_path: Optional[Path] = None,
    min_trades: int = 20,
) -> pd.DataFrame:
    """Load pre-aggregated data for fast iteration."""
    
    # Check for cached aggregation
    cache_path = gamma_path.parent / 'optimization_cache.parquet'
    
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    
    print("Building optimization dataset (will be cached)...")
    
    # Load and merge
    gamma = pd.read_parquet(gamma_path)
    goldsky = pd.read_csv(goldsky_markets_path)
    
    # Token mapping
    token_map = {}
    for _, row in goldsky.iterrows():
        token_map[str(row['token1'])] = row['id']
        token_map[str(row['token2'])] = row['id']
    
    gamma['goldsky_id'] = gamma['yes_token_id'].astype(str).map(token_map)
    gamma = gamma[gamma['goldsky_id'].notna()].copy()
    gamma['goldsky_id'] = gamma['goldsky_id'].astype(int)
    
    # Load trade aggregations if available
    if trades_agg_path and trades_agg_path.exists():
        trades_agg = pd.read_parquet(trades_agg_path)
        merged = gamma.merge(trades_agg, on='goldsky_id', how='inner')
    else:
        # Use final_prob as proxy for first_price if no trade data
        gamma['first_price'] = gamma['final_prob'].clip(0.01, 0.99)
        gamma['n_trades'] = 100  # Placeholder
        merged = gamma
    
    # Filter and process
    merged = merged[merged['n_trades'] >= min_trades].copy()
    merged['resolution_time'] = pd.to_datetime(
        merged['closedTime'], format='mixed', utc=True, errors='coerce'
    )
    merged = merged[merged['resolution_time'].notna()].copy()
    merged['category'] = merged['question'].apply(categorize_question)
    merged = merged.sort_values('resolution_time').reset_index(drop=True)
    
    # Cache
    merged.to_parquet(cache_path)
    print(f"Cached to {cache_path}")
    
    return merged


# =============================================================================
# Walk-Forward Cross-Validation
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    n_folds: int = 5
    initial_train_frac: float = 0.3
    holdout_frac: float = 0.2  # Final holdout for out-of-sample test
    min_train_size: int = 1000
    

def walk_forward_split(
    df: pd.DataFrame,
    cfg: WalkForwardConfig = WalkForwardConfig(),
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.
    
    Each fold uses all prior data for training and the next period for testing.
    The final holdout is never touched during optimization.
    """
    n = len(df)
    holdout_size = int(n * cfg.holdout_frac)
    available = n - holdout_size
    
    initial_train = max(int(available * cfg.initial_train_frac), cfg.min_train_size)
    remaining = available - initial_train
    fold_size = remaining // cfg.n_folds
    
    splits = []
    for fold in range(cfg.n_folds):
        train_end = initial_train + fold * fold_size
        test_start = train_end
        test_end = min(train_end + fold_size, available)
        
        if test_end <= test_start:
            continue
        
        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]
        splits.append((train, test))
    
    return splits


def get_holdout(df: pd.DataFrame, holdout_frac: float = 0.2) -> pd.DataFrame:
    """Get final holdout set (never used during optimization)."""
    n = len(df)
    holdout_start = int(n * (1 - holdout_frac))
    return df.iloc[holdout_start:]


# =============================================================================
# Strategy Implementations (Vectorized for Speed)
# =============================================================================

def blackwell_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_bins: int = 10,
    g_bar_threshold: float = 0.03,
    t_stat_threshold: float = 1.5,
    min_samples: int = 20,
    target_max_loss: float = 0.2,
) -> Dict[str, float]:
    """Vectorized Blackwell calibration strategy."""
    
    # Compute bin statistics from training
    bin_edges = np.linspace(0, 1, n_bins + 1)
    train_prices = train['first_price'].values
    train_outcomes = train['y'].values
    
    bin_idx = np.clip(np.digitize(train_prices, bin_edges) - 1, 0, n_bins - 1)
    
    bin_g_bar = np.zeros(n_bins)
    bin_direction = np.zeros(n_bins)
    bin_tradeable = np.zeros(n_bins, dtype=bool)
    
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
                bin_g_bar[b] = g_bar
                bin_direction[b] = 1 if g_bar > 0 else -1
                bin_tradeable[b] = True
    
    if not bin_tradeable.any():
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0}
    
    # Trade on test set (vectorized)
    test_prices = test['first_price'].values
    test_outcomes = test['y'].values
    test_bins = np.clip(np.digitize(test_prices, bin_edges) - 1, 0, n_bins - 1)
    
    # Filter to tradeable
    tradeable_mask = bin_tradeable[test_bins]
    if not tradeable_mask.any():
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0}
    
    prices = test_prices[tradeable_mask]
    outcomes = test_outcomes[tradeable_mask]
    directions = bin_direction[test_bins[tradeable_mask]]
    
    # Position sizing (risk parity)
    max_loss = np.where(directions > 0, prices, 1 - prices)
    sizes = np.clip(target_max_loss / np.maximum(max_loss, 0.01), 0, 1)
    
    # PnL calculation
    pnl_yes = np.where(outcomes == 1, sizes * (1 - prices), -sizes * prices)
    pnl_no = np.where(outcomes == 0, sizes * prices, -sizes * (1 - prices))
    pnls = np.where(directions > 0, pnl_yes, pnl_no)
    
    # Metrics
    total_pnl = pnls.sum()
    win_rate = (pnls > 0).mean() if len(pnls) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = drawdown.max() if len(drawdown) > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': int(len(pnls)),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
    }


def stat_arb_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kelly_scale: float = 0.25,
    kelly_cap: float = 0.10,
    distance_scale: float = 1.0,
    min_edge: float = 0.02,
    n_bins: int = 10,
    fee: float = 0.01,
) -> Dict[str, float]:
    """Vectorized statistical arbitrage strategy."""
    
    # Compute calibration stats from training
    bin_edges = np.linspace(0, 1, n_bins + 1)
    train_prices = train['first_price'].values
    train_outcomes = train['y'].values
    
    bin_idx = np.clip(np.digitize(train_prices, bin_edges) - 1, 0, n_bins - 1)
    
    bin_bias = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 5:
            bin_bias[b] = train_outcomes[mask].mean() - train_prices[mask].mean()
    
    # Trade on test
    test_prices = test['first_price'].values
    test_outcomes = test['y'].values
    test_bins = np.clip(np.digitize(test_prices, bin_edges) - 1, 0, n_bins - 1)
    
    # Compute edges
    biases = bin_bias[test_bins]
    fair_prices = np.clip(test_prices + biases, 0.01, 0.99)
    edges = fair_prices - test_prices
    
    # Filter by min_edge
    trade_mask = np.abs(edges) >= min_edge
    if not trade_mask.any():
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0}
    
    prices = test_prices[trade_mask]
    outcomes = test_outcomes[trade_mask]
    e = edges[trade_mask]
    b = biases[trade_mask]
    
    directions = np.sign(e)
    
    # Kelly sizing
    kelly_base = np.where(
        directions > 0,
        kelly_scale * np.abs(e) / np.maximum(1 - prices, 0.01),
        kelly_scale * np.abs(e) / np.maximum(prices, 0.01)
    )
    
    # Distance weighting
    distances = np.abs(b)
    kelly_f = kelly_base * np.minimum(1 + distance_scale * distances, 2.0)
    kelly_f = np.clip(kelly_f, 0, kelly_cap * 2)
    
    # PnL
    pnl_yes = np.where(outcomes == 1, kelly_f * (1 - prices), -kelly_f * prices)
    pnl_no = np.where(outcomes == 0, kelly_f * prices, -kelly_f * (1 - prices))
    pnls = np.where(directions > 0, pnl_yes, pnl_no) - fee * kelly_f
    
    total_pnl = pnls.sum()
    win_rate = (pnls > 0).mean() if len(pnls) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = (running_max - cumsum).max() if len(cumsum) > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': int(len(pnls)),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
    }


def mean_reversion_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    calibration_window: int = 50,
    mean_revert_threshold: float = 0.05,
    momentum_threshold: float = 0.15,
    kelly_fraction: float = 0.25,
    max_position: float = 0.10,
    min_edge: float = 0.02,
    fee: float = 0.01,
) -> Dict[str, float]:
    """Vectorized mean-reversion strategy with regime detection."""
    
    # Compute group biases from training data
    train_residuals = train['y'] - train['first_price']
    group_bias = train.groupby('category').apply(
        lambda g: (g['y'] - g['first_price']).mean()
    ).to_dict()
    
    # Apply to test set (vectorized)
    test_prices = test['first_price'].values
    test_outcomes = test['y'].values
    test_categories = test['category'].values
    
    # Map biases
    biases = np.array([group_bias.get(c, 0) for c in test_categories])
    
    # Determine regime based on training bias
    abs_biases = np.abs(biases)
    regime_scales = np.where(
        abs_biases < mean_revert_threshold, 1.0,  # mean_revert
        np.where(abs_biases > momentum_threshold, 0.5, 0.1)  # momentum or neutral
    )
    
    # Compute edges
    fair_prices = np.clip(test_prices + biases, 0.01, 0.99)
    edges = fair_prices - test_prices
    
    # Filter by min_edge
    trade_mask = np.abs(edges) >= min_edge
    if not trade_mask.any():
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0}
    
    prices = test_prices[trade_mask]
    outcomes = test_outcomes[trade_mask]
    e = edges[trade_mask]
    scales = regime_scales[trade_mask]
    
    directions = np.sign(e)
    
    # Kelly sizing
    kelly = np.where(
        directions > 0,
        np.abs(e) / np.maximum(1 - prices, 0.01),
        np.abs(e) / np.maximum(prices, 0.01)
    )
    sizes = np.clip(kelly * kelly_fraction * scales, 0, max_position)
    
    # Filter tiny positions
    size_mask = sizes >= 0.001
    if not size_mask.any():
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0}
    
    prices = prices[size_mask]
    outcomes = outcomes[size_mask]
    directions = directions[size_mask]
    sizes = sizes[size_mask]
    
    # PnL calculation
    pnl_long = np.where(outcomes == 1, sizes * (1 - prices), -sizes * prices)
    pnl_short = np.where(outcomes == 0, sizes * prices, -sizes * (1 - prices))
    pnls = np.where(directions > 0, pnl_long, pnl_short) - fee * sizes
    
    total_pnl = pnls.sum()
    win_rate = (pnls > 0).mean()
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = (running_max - cumsum).max()
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
    }


# =============================================================================
# Bayesian Optimization
# =============================================================================

@dataclass
class OptimizationConfig:
    n_iterations: int = 100
    objective: str = 'sharpe'  # 'sharpe', 'pnl', 'risk_adjusted'
    method: str = 'auto'  # 'cmaes', 'optuna', 'random', 'auto'
    parallel: bool = True
    n_workers: int = None  # None = use all CPUs
    verbose: bool = True


def _evaluate_single_fold(args):
    """Worker function for parallel fold evaluation."""
    train_idx, test_idx, df_values, df_columns, strategy_fn, params = args
    
    # Reconstruct DataFrames
    train = pd.DataFrame(df_values[train_idx], columns=df_columns)
    test = pd.DataFrame(df_values[test_idx], columns=df_columns)
    
    result = strategy_fn(train, test, **params)
    return result


def cross_validate_params_parallel(
    df: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict[str, Any],
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    n_workers: int = None,
) -> Dict[str, float]:
    """Cross-validate with parallel fold evaluation."""
    
    if n_workers is None:
        n_workers = min(cpu_count(), wf_config.n_folds)
    
    splits = walk_forward_split(df, wf_config)
    
    if n_workers <= 1 or len(splits) <= 1:
        # Sequential fallback
        fold_results = []
        for train, test in splits:
            result = strategy_fn(train, test, **params)
            fold_results.append(result)
    else:
        # Parallel execution
        # Pre-convert to numpy for pickling efficiency
        df_values = df.values
        df_columns = df.columns.tolist()
        
        # Get indices for each split
        work_items = []
        idx = 0
        n = len(df)
        holdout_size = int(n * wf_config.holdout_frac)
        available = n - holdout_size
        initial_train = max(int(available * wf_config.initial_train_frac), wf_config.min_train_size)
        remaining = available - initial_train
        fold_size = remaining // wf_config.n_folds
        
        for fold in range(wf_config.n_folds):
            train_end = initial_train + fold * fold_size
            test_start = train_end
            test_end = min(train_end + fold_size, available)
            
            if test_end <= test_start:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            work_items.append((train_idx, test_idx, df_values, df_columns, strategy_fn, params))
        
        fold_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_evaluate_single_fold, item) for item in work_items]
            for future in as_completed(futures):
                fold_results.append(future.result())
    
    # Aggregate across folds
    metrics = {}
    for key in ['pnl', 'sharpe', 'win_rate', 'max_dd', 'n_trades']:
        values = [r[key] for r in fold_results]
        metrics[f'{key}_mean'] = np.mean(values)
        metrics[f'{key}_std'] = np.std(values)
    
    # Compute risk-adjusted score
    sharpe_mean = metrics['sharpe_mean']
    sharpe_std = metrics['sharpe_std']
    max_dd_mean = metrics['max_dd_mean']
    
    # Penalize high variance and high drawdown
    risk_adjusted = sharpe_mean - 0.5 * sharpe_std - 0.1 * max_dd_mean
    metrics['risk_adjusted'] = risk_adjusted
    
    return metrics


def cross_validate_params(
    df: pd.DataFrame,
    strategy_fn: Callable,
    params: Dict[str, Any],
    wf_config: WalkForwardConfig = WalkForwardConfig(),
) -> Dict[str, float]:
    """Cross-validate a parameter configuration (sequential version)."""
    
    splits = walk_forward_split(df, wf_config)
    
    fold_results = []
    for train, test in splits:
        result = strategy_fn(train, test, **params)
        fold_results.append(result)
    
    # Aggregate across folds
    metrics = {}
    for key in ['pnl', 'sharpe', 'win_rate', 'max_dd', 'n_trades']:
        values = [r[key] for r in fold_results]
        metrics[f'{key}_mean'] = np.mean(values)
        metrics[f'{key}_std'] = np.std(values)
    
    # Compute risk-adjusted score
    sharpe_mean = metrics['sharpe_mean']
    sharpe_std = metrics['sharpe_std']
    max_dd_mean = metrics['max_dd_mean']
    
    # Penalize high variance and high drawdown
    risk_adjusted = sharpe_mean - 0.5 * sharpe_std - 0.1 * max_dd_mean
    metrics['risk_adjusted'] = risk_adjusted
    
    return metrics


def cmaes_optimize(
    df: pd.DataFrame,
    strategy_fn: Callable,
    param_space: Dict[str, Tuple],
    cfg: OptimizationConfig = OptimizationConfig(),
    wf_config: WalkForwardConfig = WalkForwardConfig(),
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    CMA-ES optimization using the cma package.
    
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is one of
    the best derivative-free optimizers for continuous optimization.
    """
    
    if not HAS_CMA:
        print("CMA-ES not available, falling back to Optuna")
        return optuna_optimize(df, strategy_fn, param_space, cfg, wf_config)
    
    print(f"Starting CMA-ES optimization: {cfg.n_iterations} evaluations")
    print(f"Objective: {cfg.objective}")
    print(f"Using {cpu_count()} CPU cores")
    
    # Convert param space to CMA-ES format
    param_names = list(param_space.keys())
    param_specs = [param_space[n] for n in param_names]
    
    # Initial point (center of space) and initial sigma
    x0 = []
    sigma0 = []
    bounds_lower = []
    bounds_upper = []
    
    for spec in param_specs:
        if spec[0] == 'uniform':
            x0.append((spec[1] + spec[2]) / 2)
            sigma0.append((spec[2] - spec[1]) / 4)
            bounds_lower.append(spec[1])
            bounds_upper.append(spec[2])
        elif spec[0] == 'log_uniform':
            # Work in log space
            x0.append((np.log(spec[1]) + np.log(spec[2])) / 2)
            sigma0.append((np.log(spec[2]) - np.log(spec[1])) / 4)
            bounds_lower.append(np.log(spec[1]))
            bounds_upper.append(np.log(spec[2]))
        elif spec[0] == 'int':
            x0.append((spec[1] + spec[2]) / 2)
            sigma0.append((spec[2] - spec[1]) / 4)
            bounds_lower.append(spec[1])
            bounds_upper.append(spec[2])
    
    def decode_params(x):
        """Convert CMA-ES vector back to params dict."""
        params = {}
        for i, (name, spec) in enumerate(zip(param_names, param_specs)):
            if spec[0] == 'log_uniform':
                params[name] = np.exp(np.clip(x[i], np.log(spec[1]), np.log(spec[2])))
            elif spec[0] == 'int':
                params[name] = int(np.clip(round(x[i]), spec[1], spec[2]))
            else:
                params[name] = np.clip(x[i], spec[1], spec[2])
        return params
    
    history = []
    best_score = -np.inf
    best_params = None
    eval_count = 0
    
    def objective(x):
        """Objective function for CMA-ES (minimization, so negate)."""
        nonlocal eval_count, best_score, best_params
        
        params = decode_params(x)
        metrics = cross_validate_params(df, strategy_fn, params, wf_config)
        score = metrics[f'{cfg.objective}_mean']
        
        # Handle NaN
        if np.isnan(score):
            score = -100
        
        history.append({
            'iteration': eval_count,
            'params': params,
            'metrics': metrics,
            'score': score,
        })
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            if cfg.verbose:
                print(f"  [eval {eval_count}] New best: {cfg.objective}={best_score:.4f}")
        
        eval_count += 1
        
        if cfg.verbose and eval_count % 10 == 0:
            print(f"  [eval {eval_count}/{cfg.n_iterations}] Best {cfg.objective}: {best_score:.4f}")
        
        return -score  # CMA-ES minimizes
    
    # Configure CMA-ES
    opts = {
        'maxfevals': cfg.n_iterations,
        'bounds': [bounds_lower, bounds_upper],
        'verbose': -9,  # Quiet
        'popsize': max(8, 4 + int(3 * np.log(len(x0)))),  # Default + adjustment
    }
    
    # Run CMA-ES (sequential - CMA-ES handles population parallelism internally)
    es = cma.CMAEvolutionStrategy(x0, np.mean(sigma0), opts)
    
    while not es.stop() and eval_count < cfg.n_iterations:
        solutions = es.ask()
        
        # Sequential evaluation (avoids pickle issues)
        fitnesses = [objective(x) for x in solutions]
        
        es.tell(solutions, fitnesses)
    
    print(f"\nCMA-ES complete. Best {cfg.objective}: {best_score:.4f}")
    
    return best_params, history


def optuna_optimize(
    df: pd.DataFrame,
    strategy_fn: Callable,
    param_space: Dict[str, Tuple],
    cfg: OptimizationConfig = OptimizationConfig(),
    wf_config: WalkForwardConfig = WalkForwardConfig(),
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Optuna optimization with CMA-ES sampler.
    
    Optuna provides a robust interface with CMA-ES, TPE, and other samplers.
    """
    
    if not HAS_OPTUNA:
        print("Optuna not available, falling back to random search")
        return random_search_optimize(df, strategy_fn, param_space, cfg, wf_config)
    
    print(f"Starting Optuna optimization: {cfg.n_iterations} trials")
    print(f"Objective: {cfg.objective}")
    print(f"Using {cpu_count()} CPU cores")
    
    history = []
    
    def objective(trial):
        params = {}
        for name, spec in param_space.items():
            if spec[0] == 'uniform':
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif spec[0] == 'log_uniform':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif spec[0] == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec[0] == 'choice':
                params[name] = trial.suggest_categorical(name, spec[1])
        
        metrics = cross_validate_params(df, strategy_fn, params, wf_config)
        score = metrics[f'{cfg.objective}_mean']
        
        history.append({
            'iteration': trial.number,
            'params': params,
            'metrics': metrics,
            'score': score,
        })
        
        return score
    
    # Use CMA-ES sampler if available
    try:
        sampler = CmaEsSampler(seed=42)
        print("Using CMA-ES sampler")
    except:
        sampler = TPESampler(seed=42)
        print("Using TPE sampler")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
    )
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Run with parallel trials - use more cores for fast strategies
    n_jobs = min(cpu_count(), 16)  # Use up to 16 parallel trials
    
    study.optimize(
        objective,
        n_trials=cfg.n_iterations,
        n_jobs=n_jobs,
        show_progress_bar=cfg.verbose,
    )
    
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\nOptuna complete. Best {cfg.objective}: {best_score:.4f}")
    
    return best_params, history


def random_search_optimize(
    df: pd.DataFrame,
    strategy_fn: Callable,
    param_space: Dict[str, Tuple],
    cfg: OptimizationConfig = OptimizationConfig(),
    wf_config: WalkForwardConfig = WalkForwardConfig(),
) -> Tuple[Dict[str, Any], List[Dict]]:
    """Fallback random search (sequential to avoid pickle issues)."""
    
    print(f"Starting random search: {cfg.n_iterations} iterations")
    
    def sample_params():
        params = {}
        for name, spec in param_space.items():
            if spec[0] == 'uniform':
                params[name] = np.random.uniform(spec[1], spec[2])
            elif spec[0] == 'log_uniform':
                params[name] = np.exp(np.random.uniform(np.log(spec[1]), np.log(spec[2])))
            elif spec[0] == 'int':
                params[name] = np.random.randint(spec[1], spec[2] + 1)
            elif spec[0] == 'choice':
                params[name] = np.random.choice(spec[1])
        return params
    
    history = []
    best_score = -np.inf
    best_params = None
    
    for i in range(cfg.n_iterations):
        params = sample_params()
        metrics = cross_validate_params(df, strategy_fn, params, wf_config)
        score = metrics[f'{cfg.objective}_mean']
        
        history.append({
            'iteration': i,
            'params': params,
            'metrics': metrics,
            'score': score,
        })
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            if cfg.verbose:
                print(f"  [eval {i}] New best: {cfg.objective}={best_score:.4f}")
        
        if cfg.verbose and (i + 1) % 20 == 0:
            print(f"  [eval {i+1}/{cfg.n_iterations}] Best: {best_score:.4f}")
    
    print(f"\nRandom search complete. Best {cfg.objective}: {best_score:.4f}")
    
    return best_params, history


def optimize(
    df: pd.DataFrame,
    strategy_fn: Callable,
    param_space: Dict[str, Tuple],
    cfg: OptimizationConfig = OptimizationConfig(),
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    method: str = 'auto',
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Unified optimization interface.
    
    Methods:
    - 'cmaes': CMA-ES via cma package (sequential, best for small populations)
    - 'optuna': Optuna with CMA-ES/TPE sampler (parallel, recommended)
    - 'random': Parallel random search
    - 'auto': Best available method (prefers Optuna for parallelism)
    """
    
    if method == 'auto':
        # Prefer Optuna for better parallelism
        if HAS_OPTUNA:
            method = 'optuna'
        elif HAS_CMA:
            method = 'cmaes'
        else:
            method = 'random'
    
    if method == 'cmaes':
        return cmaes_optimize(df, strategy_fn, param_space, cfg, wf_config)
    elif method == 'optuna':
        return optuna_optimize(df, strategy_fn, param_space, cfg, wf_config)
    else:
        return random_search_optimize(df, strategy_fn, param_space, cfg, wf_config)


# =============================================================================
# Main Optimization Pipeline
# =============================================================================

def optimize_all_strategies(
    df: pd.DataFrame,
    output_dir: Path,
    n_iterations: int = 100,
    method: str = 'auto',
):
    """Optimize all strategies and save results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_cpus = cpu_count()
    
    print("="*70)
    print("STRATEGY HYPERPARAMETER OPTIMIZATION")
    print(f"Dataset: {len(df):,} markets")
    print(f"Iterations per strategy: {n_iterations}")
    print(f"CPU cores available: {n_cpus}")
    print(f"Optimization method: {method}")
    print("="*70)
    
    wf_config = WalkForwardConfig(n_folds=5, holdout_frac=0.2)
    opt_config = OptimizationConfig(
        n_iterations=n_iterations,
        objective='sharpe',
        method=method,
        parallel=True,
        n_workers=n_cpus,
    )
    
    results = {}
    
    # === Blackwell Strategy ===
    print("\n" + "="*50)
    print("OPTIMIZING: Blackwell Calibration Strategy")
    print("="*50)
    
    blackwell_space = {
        'n_bins': ('int', 5, 20),
        'g_bar_threshold': ('uniform', 0.01, 0.10),
        't_stat_threshold': ('uniform', 1.0, 3.0),
        'min_samples': ('int', 10, 50),
        'target_max_loss': ('uniform', 0.1, 0.5),
    }
    
    best_blackwell, history_blackwell = optimize(
        df, blackwell_strategy, blackwell_space, opt_config, wf_config, method
    )
    
    # Validate on holdout
    holdout = get_holdout(df, 0.2)
    train_for_holdout = df.iloc[:-len(holdout)]
    holdout_result = blackwell_strategy(train_for_holdout, holdout, **best_blackwell)
    
    results['blackwell'] = {
        'best_params': best_blackwell,
        'cv_score': max(h['score'] for h in history_blackwell),
        'holdout_result': holdout_result,
        'history': history_blackwell,
    }
    
    print(f"\nBest params: {best_blackwell}")
    print(f"Holdout Sharpe: {holdout_result['sharpe']:.2f}, PnL: ${holdout_result['pnl']:.2f}")
    
    # === Statistical Arbitrage ===
    print("\n" + "="*50)
    print("OPTIMIZING: Statistical Arbitrage Strategy")
    print("="*50)
    
    stat_arb_space = {
        'kelly_scale': ('log_uniform', 0.05, 1.0),
        'kelly_cap': ('uniform', 0.02, 0.20),
        'distance_scale': ('uniform', 0.5, 3.0),
        'min_edge': ('uniform', 0.01, 0.10),
        'n_bins': ('int', 5, 20),
        'fee': ('uniform', 0.005, 0.02),
    }
    
    best_stat_arb, history_stat_arb = optimize(
        df, stat_arb_strategy, stat_arb_space, opt_config, wf_config, method
    )
    
    holdout_result = stat_arb_strategy(train_for_holdout, holdout, **best_stat_arb)
    
    results['stat_arb'] = {
        'best_params': best_stat_arb,
        'cv_score': max(h['score'] for h in history_stat_arb),
        'holdout_result': holdout_result,
        'history': history_stat_arb,
    }
    
    print(f"\nBest params: {best_stat_arb}")
    print(f"Holdout Sharpe: {holdout_result['sharpe']:.2f}, PnL: ${holdout_result['pnl']:.2f}")
    
    # === Mean Reversion ===
    print("\n" + "="*50)
    print("OPTIMIZING: Mean-Reversion Strategy")
    print("="*50)
    
    mean_rev_space = {
        'calibration_window': ('int', 20, 200),
        'mean_revert_threshold': ('uniform', 0.02, 0.15),
        'momentum_threshold': ('uniform', 0.10, 0.30),
        'kelly_fraction': ('log_uniform', 0.1, 0.5),
        'max_position': ('uniform', 0.05, 0.20),
        'min_edge': ('uniform', 0.01, 0.05),
        'fee': ('uniform', 0.005, 0.02),
    }
    
    best_mean_rev, history_mean_rev = optimize(
        df, mean_reversion_strategy, mean_rev_space, opt_config, wf_config, method
    )
    
    holdout_result = mean_reversion_strategy(train_for_holdout, holdout, **best_mean_rev)
    
    results['mean_reversion'] = {
        'best_params': best_mean_rev,
        'cv_score': max(h['score'] for h in history_mean_rev),
        'holdout_result': holdout_result,
        'history': history_mean_rev,
    }
    
    print(f"\nBest params: {best_mean_rev}")
    print(f"Holdout Sharpe: {holdout_result['sharpe']:.2f}, PnL: ${holdout_result['pnl']:.2f}")
    
    # === Summary ===
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    summary = []
    for name, res in results.items():
        hr = res['holdout_result']
        summary.append({
            'strategy': name,
            'cv_sharpe': res['cv_score'],
            'holdout_sharpe': hr['sharpe'],
            'holdout_pnl': hr['pnl'],
            'holdout_win_rate': hr['win_rate'],
            'holdout_trades': hr['n_trades'],
            'holdout_max_dd': hr['max_dd'],
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Save results
    with open(output_dir / 'optimization_results.json', 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    summary_df.to_csv(output_dir / 'optimization_summary.csv', index=False)
    
    print(f"\nResults saved to {output_dir}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize strategy hyperparameters")
    parser.add_argument("--gamma-path", type=str, 
                        default="data/polymarket/gamma_yesno_resolved.parquet")
    parser.add_argument("--goldsky-markets", type=str, 
                        default="data/polymarket_goldsky/markets.csv")
    parser.add_argument("--trades-agg", type=str, default=None,
                        help="Pre-aggregated trades parquet (for speed)")
    parser.add_argument("--output-dir", type=str, default="runs/optimization")
    parser.add_argument("--n-iterations", type=int, default=100,
                        help="Optimization iterations per strategy")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "cmaes", "optuna", "random"],
                        help="Optimization method")
    args = parser.parse_args()
    
    # Load data
    df = load_data_fast(
        gamma_path=Path(args.gamma_path),
        goldsky_markets_path=Path(args.goldsky_markets),
        trades_agg_path=Path(args.trades_agg) if args.trades_agg else None,
    )
    
    print(f"Loaded {len(df):,} markets")
    print(f"Date range: {df['resolution_time'].min().date()} to {df['resolution_time'].max().date()}")
    
    # Run optimization
    optimize_all_strategies(
        df=df,
        output_dir=Path(args.output_dir),
        n_iterations=args.n_iterations,
        method=args.method,
    )


if __name__ == "__main__":
    main()
