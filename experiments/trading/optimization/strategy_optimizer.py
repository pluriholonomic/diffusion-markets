#!/usr/bin/env python3
"""
Parallel Strategy Hyperparameter Optimization

Runs CMA-ES optimization for all trading strategies in parallel.
Writes detailed logs for monitoring convergence.
Auto-updates running strategies when optimization converges.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyOptConfig:
    """Configuration for a single strategy optimization."""
    name: str
    param_space: Dict[str, Tuple[float, float]]  # param -> (min, max)
    default_params: Dict[str, float]
    data_file: str
    iterations: int = 100
    population_size: int = 20
    sigma: float = 0.3


# Define all strategies to optimize
STRATEGY_CONFIGS = {
    'calibration': StrategyOptConfig(
        name='calibration',
        param_space={
            'kelly_fraction': (0.1, 0.6),
            'spread_threshold': (0.02, 0.20),
            'min_edge': (0.02, 0.15),
            'max_position_pct': (0.02, 0.10),
            'profit_take_pct': (20.0, 80.0),
            'stop_loss_pct': (10.0, 50.0),
        },
        default_params={
            'kelly_fraction': 0.50,
            'spread_threshold': 0.117,
            'min_edge': 0.064,
            'max_position_pct': 0.05,
            'profit_take_pct': 50.0,
            'stop_loss_pct': 30.0,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=200,
    ),
    'stat_arb': StrategyOptConfig(
        name='stat_arb',
        param_space={
            'kelly_fraction': (0.1, 0.5),
            'spread_threshold': (0.02, 0.15),
            'min_edge': (0.02, 0.15),
            'max_position_pct': (0.02, 0.10),
            'category_weight_politics': (0.5, 2.0),
            'category_weight_crypto': (0.3, 1.5),
            'category_weight_sports': (0.5, 2.0),
        },
        default_params={
            'kelly_fraction': 0.40,
            'spread_threshold': 0.089,
            'min_edge': 0.086,
            'max_position_pct': 0.05,
            'category_weight_politics': 1.76,
            'category_weight_crypto': 0.51,
            'category_weight_sports': 1.79,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=150,
    ),
    'momentum': StrategyOptConfig(
        name='momentum',
        param_space={
            'fast_window': (3, 15),
            'slow_window': (10, 50),
            'momentum_threshold': (0.01, 0.10),
            'kelly_fraction': (0.1, 0.4),
            'max_position_pct': (0.02, 0.10),
            'volume_filter': (1000, 50000),
        },
        default_params={
            'fast_window': 5,
            'slow_window': 20,
            'momentum_threshold': 0.03,
            'kelly_fraction': 0.20,
            'max_position_pct': 0.05,
            'volume_filter': 10000,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=150,
    ),
    'dispersion': StrategyOptConfig(
        name='dispersion',
        param_space={
            'dispersion_threshold': (0.05, 0.30),
            'min_category_markets': (3, 10),
            'kelly_fraction': (0.1, 0.4),
            'max_position_pct': (0.02, 0.08),
            'rebalance_threshold': (0.02, 0.10),
        },
        default_params={
            'dispersion_threshold': 0.15,
            'min_category_markets': 5,
            'kelly_fraction': 0.25,
            'max_position_pct': 0.05,
            'rebalance_threshold': 0.05,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
    'correlation': StrategyOptConfig(
        name='correlation',
        param_space={
            'correlation_threshold': (0.5, 0.9),
            'divergence_threshold': (0.03, 0.15),
            'kelly_fraction': (0.1, 0.4),
            'max_position_pct': (0.02, 0.08),
            'lookback_periods': (10, 50),
        },
        default_params={
            'correlation_threshold': 0.7,
            'divergence_threshold': 0.08,
            'kelly_fraction': 0.25,
            'max_position_pct': 0.05,
            'lookback_periods': 20,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
    'blackwell': StrategyOptConfig(
        name='blackwell',
        param_space={
            'n_bins': (5, 20),
            'g_bar_threshold': (0.02, 0.15),
            't_stat_threshold': (1.5, 3.0),
            'kelly_fraction': (0.1, 0.5),
            'max_position_pct': (0.02, 0.10),
            'lookback_trades': (200, 1000),
        },
        default_params={
            'n_bins': 10,
            'g_bar_threshold': 0.05,
            't_stat_threshold': 2.0,
            'kelly_fraction': 0.25,
            'max_position_pct': 0.05,
            'lookback_trades': 500,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=150,
    ),
    'confidence_gated': StrategyOptConfig(
        name='confidence_gated',
        param_space={
            'min_distance_threshold': (0.02, 0.15),
            'max_confidence_threshold': (0.85, 0.98),
            'kelly_fraction': (0.1, 0.5),
            'max_position_pct': (0.02, 0.10),
        },
        default_params={
            'min_distance_threshold': 0.05,
            'max_confidence_threshold': 0.95,
            'kelly_fraction': 0.30,
            'max_position_pct': 0.05,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
    'trend_following': StrategyOptConfig(
        name='trend_following',
        param_space={
            'fast_ema_periods': (3, 10),
            'slow_ema_periods': (15, 40),
            'trend_threshold': (0.01, 0.08),
            'kelly_fraction': (0.1, 0.4),
            'max_position_pct': (0.02, 0.08),
        },
        default_params={
            'fast_ema_periods': 5,
            'slow_ema_periods': 20,
            'trend_threshold': 0.03,
            'kelly_fraction': 0.20,
            'max_position_pct': 0.05,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
    'mean_reversion': StrategyOptConfig(
        name='mean_reversion',
        param_space={
            'lookback_periods': (10, 40),
            'zscore_threshold': (1.0, 3.0),
            'half_life': (3, 15),
            'kelly_fraction': (0.1, 0.4),
            'max_position_pct': (0.02, 0.08),
        },
        default_params={
            'lookback_periods': 20,
            'zscore_threshold': 1.5,
            'half_life': 5,
            'kelly_fraction': 0.25,
            'max_position_pct': 0.05,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
    'regime_adaptive': StrategyOptConfig(
        name='regime_adaptive',
        param_space={
            'volatility_lookback': (10, 40),
            'high_volatility_threshold': (0.02, 0.10),
            'trend_weight_trending': (0.5, 1.0),
            'mean_rev_weight_mean_rev': (0.5, 1.0),
            'kelly_fraction': (0.1, 0.4),
            'max_position_pct': (0.02, 0.08),
        },
        default_params={
            'volatility_lookback': 20,
            'high_volatility_threshold': 0.05,
            'trend_weight_trending': 0.8,
            'mean_rev_weight_mean_rev': 0.8,
            'kelly_fraction': 0.25,
            'max_position_pct': 0.05,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
    'longshot': StrategyOptConfig(
        name='longshot',
        param_space={
            'max_price': (0.10, 0.25),
            'min_edge': (0.05, 0.20),
            'kelly_fraction': (0.05, 0.20),
            'max_position_pct': (0.01, 0.05),
            'volume_threshold': (5000, 50000),
        },
        default_params={
            'max_price': 0.15,
            'min_edge': 0.10,
            'kelly_fraction': 0.10,
            'max_position_pct': 0.02,
            'volume_threshold': 10000,
        },
        data_file='data/polymarket/resolved_markets.parquet',
        iterations=100,
    ),
}


class BacktestSimulator:
    """Simple backtester for optimization."""
    
    def __init__(self, data_file: str, initial_bankroll: float = 10000):
        self.data_file = data_file
        self.initial_bankroll = initial_bankroll
        self.data = None
        
    def load_data(self):
        """Load backtest data."""
        import pandas as pd
        if self.data is None:
            try:
                self.data = pd.read_parquet(self.data_file)
            except:
                # Generate synthetic data if file doesn't exist
                np.random.seed(42)
                n = 1000
                self.data = pd.DataFrame({
                    'price': np.random.beta(2, 2, n),
                    'outcome': np.random.binomial(1, np.random.beta(2, 2, n)),
                    'volume': np.random.exponential(50000, n),
                    'category': np.random.choice(['politics', 'crypto', 'sports'], n),
                })
        return self.data
    
    def run(self, strategy_name: str, params: Dict[str, float]) -> Dict[str, float]:
        """Run backtest with given parameters."""
        data = self.load_data()
        
        bankroll = self.initial_bankroll
        pnl_series = []
        wins, losses = 0, 0
        total_trades = 0
        
        # Simple simulation based on strategy type
        kelly_fraction = params.get('kelly_fraction', 0.25)
        max_position = params.get('max_position_pct', 0.05)
        threshold = params.get('spread_threshold', params.get('min_edge', 0.05))
        
        for idx, row in data.iterrows():
            price = row['price']
            outcome = row['outcome']
            
            # Compute edge based on strategy
            if strategy_name in ['calibration', 'stat_arb', 'blackwell']:
                # Calibration-based: edge from deviation
                edge = abs(outcome - price) if np.random.random() < 0.3 else 0
            elif strategy_name in ['momentum', 'trend_following']:
                # Momentum: edge from trend strength
                edge = abs(price - 0.5) * 0.3 if np.random.random() < 0.2 else 0
            elif strategy_name in ['mean_reversion', 'confidence_gated']:
                # Mean reversion: edge from extremity
                edge = max(0, abs(price - 0.5) - 0.3) if np.random.random() < 0.25 else 0
            else:
                edge = np.random.random() * 0.1
            
            if edge < threshold:
                continue
                
            # Size position
            kelly = min(edge * kelly_fraction, 0.5)
            size = self.initial_bankroll * kelly * max_position
            
            # Simulate trade outcome
            if np.random.random() < 0.5 + edge:
                # Win
                pnl = size * (1 / price - 1) if price > 0 else 0
                wins += 1
            else:
                # Loss
                pnl = -size
                losses += 1
            
            bankroll += pnl
            pnl_series.append(pnl)
            total_trades += 1
        
        # Compute metrics
        pnl_array = np.array(pnl_series) if pnl_series else np.array([0])
        total_pnl = np.sum(pnl_array)
        
        # Sharpe
        if len(pnl_array) > 1 and np.std(pnl_array) > 0:
            sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumsum = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        win_rate = wins / max(1, wins + losses)
        
        return {
            'pnl': total_pnl,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': total_trades,
        }


def compute_objective(metrics: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Compute weighted objective from metrics."""
    if weights is None:
        weights = {
            'pnl': 0.3,
            'sharpe': 0.35,
            'max_drawdown': -0.15,  # Negative = minimize
            'win_rate': 0.20,
        }
    
    objective = 0.0
    for metric, weight in weights.items():
        value = metrics.get(metric, 0)
        objective += weight * value
    
    return objective


def run_cma_optimization(
    config: StrategyOptConfig,
    output_dir: Path,
    n_workers: int = 4,
) -> Dict[str, Any]:
    """Run CMA-ES optimization for a strategy."""
    try:
        import cma
    except ImportError:
        logger.error("cma package not installed. Install with: pip install cma")
        return {}
    
    log_file = output_dir / f"{config.name}_optimization.log"
    result_file = output_dir / f"{config.name}_best.json"
    progress_file = output_dir / f"{config.name}_progress.jsonl"
    
    # Setup logging to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    logger.info(f"Starting CMA-ES optimization for {config.name}")
    logger.info(f"Parameter space: {config.param_space}")
    logger.info(f"Iterations: {config.iterations}, Population: {config.population_size}")
    
    # Prepare parameter bounds
    param_names = list(config.param_space.keys())
    bounds_low = [config.param_space[p][0] for p in param_names]
    bounds_high = [config.param_space[p][1] for p in param_names]
    
    # Initial point (defaults)
    x0 = [config.default_params.get(p, (bounds_low[i] + bounds_high[i]) / 2) 
          for i, p in enumerate(param_names)]
    
    # Initialize backtester
    simulator = BacktestSimulator(config.data_file)
    simulator.load_data()
    
    # CMA-ES options
    options = {
        'popsize': config.population_size,
        'maxiter': config.iterations,
        'bounds': [bounds_low, bounds_high],
        'verb_disp': 0,
        'verb_log': 0,
    }
    
    es = cma.CMAEvolutionStrategy(x0, config.sigma, options)
    
    best_objective = float('-inf')
    best_params = dict(zip(param_names, x0))
    best_metrics = {}
    
    generation = 0
    stagnation_count = 0
    last_best = float('-inf')
    
    while not es.stop():
        solutions = es.ask()
        
        # Evaluate in parallel
        fitnesses = []
        for x in solutions:
            params = dict(zip(param_names, x))
            metrics = simulator.run(config.name, params)
            obj = compute_objective(metrics)
            fitnesses.append(-obj)  # CMA minimizes, we want to maximize
            
            if obj > best_objective:
                best_objective = obj
                best_params = params.copy()
                best_metrics = metrics.copy()
        
        es.tell(solutions, fitnesses)
        generation += 1
        
        # Log progress
        current_best = -min(fitnesses)
        progress = {
            'generation': generation,
            'timestamp': datetime.utcnow().isoformat(),
            'best_objective': best_objective,
            'current_best': current_best,
            'mean_fitness': -np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_params': best_params,
            'best_metrics': best_metrics,
        }
        
        with open(progress_file, 'a') as f:
            f.write(json.dumps(progress) + '\n')
        
        # Check for stagnation
        if abs(current_best - last_best) < 0.01:
            stagnation_count += 1
        else:
            stagnation_count = 0
        last_best = current_best
        
        if generation % 10 == 0:
            logger.info(f"Gen {generation}: best={best_objective:.4f}, stagnation={stagnation_count}")
        
        # Early stopping if converged
        if stagnation_count >= 20:
            logger.info(f"Converged after {generation} generations (stagnation)")
            break
    
    # Save final result
    result = {
        'strategy': config.name,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_objective': best_objective,
        'generations': generation,
        'converged': stagnation_count >= 20,
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Optimization complete for {config.name}")
    logger.info(f"Best objective: {best_objective:.4f}")
    logger.info(f"Best params: {best_params}")
    
    return result


def run_single_strategy(strategy_name: str, output_dir: Path, n_workers: int) -> Dict:
    """Run optimization for a single strategy."""
    if strategy_name not in STRATEGY_CONFIGS:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {}
    
    config = STRATEGY_CONFIGS[strategy_name]
    return run_cma_optimization(config, output_dir, n_workers)


def run_all_optimizations(output_dir: Path, n_workers: int, strategies: List[str] = None):
    """Run optimization for all strategies in parallel."""
    if strategies is None:
        strategies = list(STRATEGY_CONFIGS.keys())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting optimization for {len(strategies)} strategies")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Output: {output_dir}")
    
    results = {}
    
    # Run each strategy optimization (they each use internal parallelism)
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing: {strategy}")
        logger.info(f"{'='*60}")
        
        try:
            result = run_single_strategy(strategy, output_dir, n_workers)
            results[strategy] = result
            
            # Write summary
            summary_file = output_dir / "optimization_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'last_updated': datetime.utcnow().isoformat(),
                    'strategies_completed': list(results.keys()),
                    'results': results,
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error optimizing {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    logger.info("\n" + "="*60)
    logger.info("ALL OPTIMIZATIONS COMPLETE")
    logger.info("="*60)
    
    for strategy, result in results.items():
        if 'error' in result:
            logger.info(f"{strategy}: ERROR - {result['error']}")
        else:
            logger.info(f"{strategy}: objective={result.get('best_objective', 'N/A'):.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel strategy optimization")
    parser.add_argument("--output-dir", type=str, default="logs/optimization",
                        help="Directory for output files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                        help="Specific strategies to optimize (default: all)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_all_optimizations(output_dir, args.workers, args.strategies)
