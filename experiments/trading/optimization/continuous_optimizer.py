"""
Continuous Hyperparameter Optimizer

Runs continuously, cycling through different optimization tasks:
1. Individual strategy optimization (calibration, stat_arb, longshot)
2. Portfolio-level optimization (strategy weights, correlation)
3. Position management optimization (profit-take, stop-loss thresholds)
4. Online learning parameter optimization

Keeps all cores busy and saves best parameters periodically.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

# CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("Warning: cma not installed. Install with: pip install cma")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationTask:
    """Defines an optimization task."""
    name: str
    param_space: Dict[str, Tuple[float, float]]  # param -> (min, max)
    objective_weights: Dict[str, float]
    data_file: str
    iterations: int = 30
    priority: int = 1  # Higher = more important


@dataclass 
class OptimizationResult:
    """Result of an optimization run."""
    task_name: str
    best_params: Dict[str, float]
    best_metrics: Dict[str, float]
    objective_value: float
    evaluations: int
    elapsed_time: float
    timestamp: str


# Define optimization tasks
OPTIMIZATION_TASKS = [
    # Calibration strategy
    OptimizationTask(
        name="calibration_strategy",
        param_space={
            'kelly_fraction': (0.1, 0.5),
            'spread_threshold': (0.02, 0.15),
            'min_edge': (0.02, 0.15),
            'max_position_pct': (0.05, 0.25),
            'profit_take_pct': (10.0, 50.0),
            'stop_loss_pct': (5.0, 30.0),
        },
        objective_weights={
            'realized_pnl': 0.25,
            'sharpe': 0.30,
            'max_drawdown': 0.25,
            'es_sharpe': 0.20,
        },
        data_file="data/polymarket/optimization_cache.parquet",
        iterations=30,
        priority=3,
    ),
    
    # Statistical arbitrage
    OptimizationTask(
        name="stat_arb_strategy",
        param_space={
            'kelly_fraction': (0.1, 0.4),
            'spread_threshold': (0.03, 0.12),
            'min_edge': (0.03, 0.12),
            'max_position_pct': (0.05, 0.20),
            'category_weight_politics': (0.5, 2.0),
            'category_weight_crypto': (0.5, 2.0),
            'category_weight_sports': (0.5, 2.0),
        },
        objective_weights={
            'realized_pnl': 0.20,
            'sharpe': 0.35,
            'max_drawdown': 0.25,
            'es_sharpe': 0.20,
        },
        data_file="data/polymarket/optimization_cache.parquet",
        iterations=25,
        priority=2,
    ),
    
    # Longshot strategy (Kalshi)
    OptimizationTask(
        name="longshot_strategy",
        param_space={
            'kelly_fraction': (0.05, 0.25),
            'min_odds': (3.0, 20.0),
            'max_price': (0.05, 0.25),
            'min_expected_edge': (0.10, 0.40),
            'max_position_pct': (0.02, 0.10),
        },
        objective_weights={
            'realized_pnl': 0.30,
            'sharpe': 0.20,
            'max_drawdown': 0.30,
            'es_sharpe': 0.20,
        },
        data_file="data/kalshi/full_history.parquet",
        iterations=20,
        priority=1,
    ),
    
    # Portfolio optimization (strategy weights)
    OptimizationTask(
        name="portfolio_weights",
        param_space={
            'weight_calibration': (0.1, 0.5),
            'weight_stat_arb': (0.1, 0.4),
            'weight_longshot': (0.05, 0.3),
            'weight_momentum': (0.05, 0.3),
            'correlation_penalty': (0.0, 0.5),
            'diversification_bonus': (0.0, 0.3),
            'max_single_strategy': (0.3, 0.6),
        },
        objective_weights={
            'portfolio_sharpe': 0.35,
            'portfolio_pnl': 0.25,
            'max_drawdown': 0.25,
            'diversification_ratio': 0.15,
        },
        data_file="data/polymarket/optimization_cache.parquet",
        iterations=25,
        priority=3,
    ),
    
    # Position management
    OptimizationTask(
        name="position_management",
        param_space={
            'profit_take_pct': (5.0, 60.0),
            'stop_loss_pct': (3.0, 40.0),
            'trailing_stop_pct': (2.0, 20.0),
            'time_decay_hours': (1.0, 72.0),
            'ema_alpha_fast': (0.05, 0.30),
            'ema_alpha_slow': (0.01, 0.10),
            'learning_rate': (0.01, 0.20),
        },
        objective_weights={
            'realized_pnl': 0.25,
            'sharpe': 0.25,
            'max_drawdown': 0.30,
            'win_rate': 0.20,
        },
        data_file="data/polymarket/optimization_cache.parquet",
        iterations=30,
        priority=2,
    ),
    
    # Risk management
    OptimizationTask(
        name="risk_management",
        param_space={
            'max_daily_loss_pct': (5.0, 20.0),
            'max_position_size_pct': (5.0, 25.0),
            'max_correlation': (0.3, 0.8),
            'var_limit_pct': (2.0, 10.0),
            'drawdown_pause_pct': (10.0, 30.0),
        },
        objective_weights={
            'risk_adjusted_return': 0.30,
            'max_drawdown': 0.35,
            'tail_risk': 0.20,
            'recovery_time': 0.15,
        },
        data_file="data/polymarket/optimization_cache.parquet",
        iterations=20,
        priority=2,
    ),
]


def run_single_backtest(params: Dict[str, float], data: pd.DataFrame, 
                        task: OptimizationTask) -> Dict[str, float]:
    """Run a single backtest with given parameters."""
    # Extract common parameters
    kelly_fraction = params.get('kelly_fraction', 0.25)
    spread_threshold = params.get('spread_threshold', 0.08)
    min_edge = params.get('min_edge', 0.05)
    max_position_pct = params.get('max_position_pct', 0.10)
    profit_take_pct = params.get('profit_take_pct', 20.0)
    stop_loss_pct = params.get('stop_loss_pct', 15.0)
    
    initial_bankroll = 10000
    bankroll = initial_bankroll
    pnl_series = []
    wins = 0
    losses = 0
    peak = bankroll
    max_drawdown = 0
    
    # Impact cost
    impact_cost = 0.02
    
    for _, row in data.iterrows():
        price = row.get('price', row.get('avg_price', 0.5))
        outcome = row.get('outcome', row.get('y', None))
        
        if outcome is None or pd.isna(outcome):
            continue
        
        # Compute spread
        spread = outcome - price
        
        if abs(spread) < spread_threshold:
            continue
        
        edge = abs(spread)
        if edge < min_edge:
            continue
        
        side = 'yes' if spread > 0 else 'no'
        
        # Position sizing (fixed, not compounding)
        if side == 'yes':
            odds = (1 - price) / price if price > 0.01 else 99
        else:
            odds = price / (1 - price) if price < 0.99 else 99
        
        kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
        kelly = max(0, min(kelly, 1)) * kelly_fraction
        kelly = min(kelly, max_position_pct)
        
        if kelly < 0.01:
            continue
        
        size = initial_bankroll * kelly
        
        # Apply impact cost
        if side == 'yes':
            entry_price = min(0.99, price * (1 + impact_cost))
        else:
            entry_price = min(0.99, (1 - price) * (1 + impact_cost))
        
        # Compute PnL
        if side == 'yes':
            if outcome == 1:
                pnl = size * (1 - entry_price) / entry_price
                wins += 1
            else:
                pnl = -size
                losses += 1
        else:
            if outcome == 0:
                pnl = size * (1 - entry_price) / entry_price
                wins += 1
            else:
                pnl = -size
                losses += 1
        
        # Apply profit-take / stop-loss
        return_pct = (pnl / size) * 100 if size > 0 else 0
        if return_pct > profit_take_pct:
            pnl = size * (profit_take_pct / 100)
        elif return_pct < -stop_loss_pct:
            pnl = -size * (stop_loss_pct / 100)
        
        pnl_series.append(pnl)
        bankroll += pnl
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak * 100 if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Compute metrics
    if not pnl_series:
        return {
            'realized_pnl': 0, 'sharpe': 0, 'sortino': 0,
            'max_drawdown': 0, 'es_sharpe': 0, 'win_rate': 0,
            'total_trades': 0, 'portfolio_sharpe': 0, 'portfolio_pnl': 0,
            'diversification_ratio': 1.0, 'risk_adjusted_return': 0,
            'tail_risk': 0, 'recovery_time': 0,
        }
    
    pnl_array = np.array(pnl_series)
    total_trades = len(pnl_series)
    realized_pnl = float(np.sum(pnl_array))
    
    mean_pnl = np.mean(pnl_array)
    std_pnl = np.std(pnl_array)
    sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0
    
    negative_pnl = pnl_array[pnl_array < 0]
    downside_std = np.std(negative_pnl) if len(negative_pnl) > 0 else 1e-6
    sortino = (mean_pnl / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    # ES (Expected Shortfall)
    var_95 = np.percentile(pnl_array, 5)
    tail = pnl_array[pnl_array <= var_95]
    es = np.mean(tail) if len(tail) > 0 else var_95
    es_sharpe = -mean_pnl / es if es < 0 else 0
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'realized_pnl': realized_pnl,
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_drawdown),
        'es_sharpe': float(es_sharpe),
        'win_rate': float(win_rate),
        'total_trades': total_trades,
        'portfolio_sharpe': float(sharpe),
        'portfolio_pnl': realized_pnl,
        'diversification_ratio': 1.0,
        'risk_adjusted_return': realized_pnl / (max_drawdown + 1),
        'tail_risk': float(-es) if es < 0 else 0,
        'recovery_time': max_drawdown / (mean_pnl + 1e-6) if mean_pnl > 0 else 100,
    }


def compute_objective(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute weighted objective from metrics."""
    # Normalize weights
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    objective = 0
    
    # PnL component
    if 'realized_pnl' in norm_weights:
        objective += norm_weights['realized_pnl'] * metrics.get('realized_pnl', 0) / 1000
    if 'portfolio_pnl' in norm_weights:
        objective += norm_weights['portfolio_pnl'] * metrics.get('portfolio_pnl', 0) / 1000
    
    # Sharpe components
    if 'sharpe' in norm_weights:
        objective += norm_weights['sharpe'] * metrics.get('sharpe', 0) * 100
    if 'portfolio_sharpe' in norm_weights:
        objective += norm_weights['portfolio_sharpe'] * metrics.get('portfolio_sharpe', 0) * 100
    
    # Drawdown (minimize -> negate)
    if 'max_drawdown' in norm_weights:
        dd = metrics.get('max_drawdown', 100)
        objective -= norm_weights['max_drawdown'] * dd
    
    # ES Sharpe
    if 'es_sharpe' in norm_weights:
        objective += norm_weights['es_sharpe'] * metrics.get('es_sharpe', 0) * 10
    
    # Win rate
    if 'win_rate' in norm_weights:
        objective += norm_weights['win_rate'] * metrics.get('win_rate', 0)
    
    # Risk metrics
    if 'risk_adjusted_return' in norm_weights:
        objective += norm_weights['risk_adjusted_return'] * metrics.get('risk_adjusted_return', 0) / 100
    if 'tail_risk' in norm_weights:
        objective -= norm_weights['tail_risk'] * metrics.get('tail_risk', 0)
    if 'diversification_ratio' in norm_weights:
        objective += norm_weights['diversification_ratio'] * metrics.get('diversification_ratio', 1) * 100
    
    return objective


def evaluate_params(args: Tuple) -> Tuple[Dict, Dict, float]:
    """Evaluate parameters (for parallel execution)."""
    params, data, task = args
    metrics = run_single_backtest(params, data, task)
    objective = compute_objective(metrics, task.objective_weights)
    return params, metrics, objective


def run_cma_optimization(task: OptimizationTask, data: pd.DataFrame, 
                         n_workers: int) -> OptimizationResult:
    """Run CMA-ES optimization for a task."""
    if not HAS_CMA:
        raise ImportError("CMA-ES not installed")
    
    start_time = time.time()
    param_names = list(task.param_space.keys())
    bounds = [task.param_space[p] for p in param_names]
    
    # Initial point (middle of bounds)
    x0 = [(b[0] + b[1]) / 2 for b in bounds]
    sigma0 = 0.3
    
    # Scale to [0, 1]
    def to_normalized(x):
        return [(x[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) 
                for i in range(len(x))]
    
    def from_normalized(x):
        return [bounds[i][0] + x[i] * (bounds[i][1] - bounds[i][0]) 
                for i in range(len(x))]
    
    def to_params(x):
        real_x = from_normalized(x)
        return {param_names[i]: real_x[i] for i in range(len(param_names))}
    
    # CMA-ES options
    opts = {
        'bounds': [0, 1],
        'maxiter': task.iterations,
        'popsize': min(n_workers, 20),
        'verbose': -9,
    }
    
    es = cma.CMAEvolutionStrategy(to_normalized(x0), sigma0, opts)
    
    best_params = None
    best_metrics = None
    best_objective = float('-inf')
    total_evals = 0
    
    while not es.stop():
        solutions = es.ask()
        
        # Evaluate in parallel
        eval_args = [(to_params(s), data, task) for s in solutions]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(evaluate_params, eval_args))
        
        objectives = []
        for params, metrics, obj in results:
            objectives.append(-obj)  # CMA minimizes
            total_evals += 1
            
            if obj > best_objective:
                best_objective = obj
                best_params = params
                best_metrics = metrics
        
        es.tell(solutions, objectives)
    
    elapsed = time.time() - start_time
    
    return OptimizationResult(
        task_name=task.name,
        best_params=best_params or {},
        best_metrics=best_metrics or {},
        objective_value=best_objective,
        evaluations=total_evals,
        elapsed_time=elapsed,
        timestamp=datetime.utcnow().isoformat(),
    )


def save_results(results: List[OptimizationResult], output_dir: Path):
    """Save optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    for result in results:
        result_file = output_dir / f"{result.task_name}_best.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    # Save summary
    summary = {
        'last_updated': datetime.utcnow().isoformat(),
        'total_tasks': len(results),
        'results': [asdict(r) for r in results],
    }
    
    with open(output_dir / "optimization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved {len(results)} optimization results to {output_dir}")


def run_continuous_optimization(
    data_dir: str = "data",
    output_dir: str = "logs/continuous_optimization",
    n_workers: int = None,
    cycle_delay: int = 60,
):
    """Run continuous optimization loop."""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting continuous optimization with {n_workers} workers")
    logger.info(f"Output directory: {output_path}")
    
    # Load data for each task
    task_data = {}
    for task in OPTIMIZATION_TASKS:
        data_file = Path(data_dir).parent / task.data_file
        if data_file.exists():
            task_data[task.name] = pd.read_parquet(data_file)
            logger.info(f"Loaded {len(task_data[task.name])} samples for {task.name}")
        else:
            logger.warning(f"Data file not found for {task.name}: {data_file}")
    
    # Sort tasks by priority
    sorted_tasks = sorted(OPTIMIZATION_TASKS, key=lambda t: -t.priority)
    
    cycle = 0
    all_results = []
    
    while True:
        cycle += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZATION CYCLE {cycle}")
        logger.info(f"{'='*60}")
        
        for task in sorted_tasks:
            if task.name not in task_data:
                continue
            
            logger.info(f"\nOptimizing: {task.name} (priority={task.priority})")
            
            try:
                result = run_cma_optimization(
                    task, 
                    task_data[task.name], 
                    n_workers
                )
                
                logger.info(f"  Best objective: {result.objective_value:.2f}")
                logger.info(f"  Evaluations: {result.evaluations}")
                logger.info(f"  Time: {result.elapsed_time:.1f}s")
                
                # Update results
                all_results = [r for r in all_results if r.task_name != task.name]
                all_results.append(result)
                
                # Save after each task
                save_results(all_results, output_path)
                
            except Exception as e:
                logger.error(f"Error optimizing {task.name}: {e}")
                continue
        
        logger.info(f"\nCycle {cycle} complete. Waiting {cycle_delay}s before next cycle...")
        time.sleep(cycle_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Hyperparameter Optimizer")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="logs/continuous_optimization", 
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of workers (default: all CPUs)")
    parser.add_argument("--cycle-delay", type=int, default=60,
                        help="Seconds between optimization cycles")
    
    args = parser.parse_args()
    
    run_continuous_optimization(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_workers=args.workers,
        cycle_delay=args.cycle_delay,
    )
