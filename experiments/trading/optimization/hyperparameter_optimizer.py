#!/usr/bin/env python3
"""
Hyperparameter Optimizer for Trading Strategies

Optimizes a weighted combination of:
- Unrealized PnL
- Realized PnL  
- Sharpe Ratio
- Max Drawdown (minimized)
- Expected Shortfall Sharpe (ES Sharpe)

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with parallel evaluation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import multiprocessing as mp

# Try to import cma, install if needed
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("CMA-ES not installed. Run: pip install cma")

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space."""
    
    # Position management
    profit_take_pct: Tuple[float, float] = (5.0, 50.0)      # Take profit threshold
    stop_loss_pct: Tuple[float, float] = (5.0, 30.0)        # Stop loss threshold
    
    # Kelly fraction and position sizing
    kelly_fraction: Tuple[float, float] = (0.05, 0.5)       # Kelly multiplier
    max_position_pct: Tuple[float, float] = (0.05, 0.20)    # Max position size
    
    # Signal thresholds
    min_edge: Tuple[float, float] = (0.02, 0.15)            # Minimum edge to trade
    min_confidence: Tuple[float, float] = (0.2, 0.8)        # Minimum signal confidence
    spread_threshold: Tuple[float, float] = (0.03, 0.20)    # Calibration spread threshold
    
    # Online learner
    learning_rate: Tuple[float, float] = (0.001, 0.1)       # Online learning rate
    l2_regularization: Tuple[float, float] = (0.0, 0.1)     # L2 regularization
    
    # EMA parameters
    ema_alpha_fast: Tuple[float, float] = (0.05, 0.3)       # Fast EMA alpha
    ema_alpha_slow: Tuple[float, float] = (0.01, 0.1)       # Slow EMA alpha
    
    # Risk management
    max_drawdown_pct: Tuple[float, float] = (0.15, 0.40)    # Max drawdown before halt
    max_daily_loss_pct: Tuple[float, float] = (0.10, 0.30)  # Max daily loss


@dataclass
class ObjectiveWeights:
    """Weights for combining multiple objectives."""
    realized_pnl: float = 0.25          # Weight for realized PnL
    unrealized_pnl: float = 0.10        # Weight for unrealized PnL
    sharpe: float = 0.25                # Weight for Sharpe ratio
    max_drawdown: float = 0.20          # Weight for max drawdown (penalty)
    es_sharpe: float = 0.20             # Weight for Expected Shortfall Sharpe
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.realized_pnl + self.unrealized_pnl + self.sharpe + self.max_drawdown + self.es_sharpe
        if total > 0:
            self.realized_pnl /= total
            self.unrealized_pnl /= total
            self.sharpe /= total
            self.max_drawdown /= total
            self.es_sharpe /= total


@dataclass
class OptimizationResult:
    """Result from a single hyperparameter evaluation."""
    params: Dict[str, float]
    metrics: Dict[str, float]
    objective_value: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def compute_es_sharpe(pnl_series: List[float], confidence: float = 0.95) -> float:
    """
    Compute Expected Shortfall Sharpe ratio.
    
    ES Sharpe = Mean Return / Expected Shortfall
    
    This is a tail-risk adjusted Sharpe that penalizes strategies with fat tails.
    """
    if not pnl_series or len(pnl_series) < 10:
        return 0.0
    
    pnl = np.array(pnl_series)
    mean_return = np.mean(pnl)
    
    # Compute Expected Shortfall (CVaR)
    var_threshold = np.percentile(pnl, (1 - confidence) * 100)
    tail_losses = pnl[pnl <= var_threshold]
    
    if len(tail_losses) == 0:
        # No tail losses - very good
        return mean_return / 0.01 if mean_return > 0 else 0.0
    
    expected_shortfall = -np.mean(tail_losses)
    
    if expected_shortfall <= 0:
        return mean_return / 0.01 if mean_return > 0 else 0.0
    
    return mean_return / expected_shortfall


def compute_objective(
    metrics: Dict[str, float],
    weights: ObjectiveWeights,
    initial_bankroll: float = 10000,
) -> float:
    """
    Compute the combined objective value from metrics.
    
    Higher is better. Max drawdown is inverted (lower drawdown = higher score).
    """
    weights.normalize()
    
    # Normalize metrics to comparable scales
    realized_pnl_norm = metrics.get('realized_pnl', 0) / initial_bankroll  # As fraction of bankroll
    unrealized_pnl_norm = metrics.get('unrealized_pnl', 0) / initial_bankroll
    sharpe = metrics.get('sharpe', 0) / 3.0  # Normalize assuming good Sharpe is ~3
    max_dd = metrics.get('max_drawdown', 0) / 100  # Already percentage
    es_sharpe = metrics.get('es_sharpe', 0) / 2.0  # Normalize
    
    # Combine (max_drawdown is penalized, so we invert it)
    objective = (
        weights.realized_pnl * realized_pnl_norm +
        weights.unrealized_pnl * unrealized_pnl_norm +
        weights.sharpe * sharpe +
        weights.max_drawdown * (1 - max_dd) +  # Invert: lower drawdown = higher score
        weights.es_sharpe * es_sharpe
    )
    
    return objective


class BacktestSimulator:
    """
    Simulates trading with given hyperparameters to compute metrics.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        initial_bankroll: float = 10000,
    ):
        self.data = historical_data
        self.initial_bankroll = initial_bankroll
    
    def run(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Run a backtest with the given hyperparameters.
        
        Returns metrics dictionary.
        """
        # Extract parameters
        profit_take_pct = params.get('profit_take_pct', 20.0)
        stop_loss_pct = params.get('stop_loss_pct', 15.0)
        kelly_fraction = params.get('kelly_fraction', 0.25)
        max_position_pct = params.get('max_position_pct', 0.10)
        min_edge = params.get('min_edge', 0.05)
        spread_threshold = params.get('spread_threshold', 0.10)
        
        # Simulate trades
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        pnl_series = []
        realized_pnl = 0
        unrealized_pnl = 0
        wins = 0
        losses = 0
        
        for _, row in self.data.iterrows():
            price = row.get('price', row.get('avg_price', 0.5))
            outcome = row.get('outcome', row.get('y', None))
            
            if outcome is None or pd.isna(outcome):
                continue
            
            # Compute calibration spread
            spread = outcome - price
            
            # Check if we would trade
            if abs(spread) < spread_threshold:
                continue
            
            edge = abs(spread)
            if edge < min_edge:
                continue
            
            # Determine side
            side = 'yes' if spread > 0 else 'no'
            
            # Compute position size
            if side == 'yes':
                odds = (1 - price) / price if price > 0.01 else 99
            else:
                odds = price / (1 - price) if price < 0.99 else 99
            
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * kelly_fraction
            kelly = min(kelly, max_position_pct)
            
            if kelly < 0.01:
                continue
            
            # Use INITIAL bankroll for sizing (not compounding) to prevent overflow
            # This makes results comparable and realistic
            size = self.initial_bankroll * kelly
            
            # Apply 2% impact cost (slippage) - we pay worse price than market
            impact_cost = 0.02
            if side == 'yes':
                entry_price = min(0.99, price * (1 + impact_cost))
            else:
                entry_price = min(0.99, (1 - price) * (1 + impact_cost))
            
            # Compute PnL including impact cost
            if side == 'yes':
                if outcome == 1:
                    # Win: receive $1, paid entry_price
                    pnl = size * (1 - entry_price) / entry_price
                    wins += 1
                else:
                    pnl = -size
                    losses += 1
            else:
                if outcome == 0:
                    # Win on NO: receive $1, paid entry_price (NO price)
                    pnl = size * (1 - entry_price) / entry_price
                    wins += 1
                else:
                    pnl = -size
                    losses += 1
            
            # Apply profit-take and stop-loss simulation
            # (In real trading, these would trigger during position lifetime)
            return_pct = (pnl / size) * 100 if size > 0 else 0
            
            if return_pct > profit_take_pct:
                pnl = size * (profit_take_pct / 100)
            elif return_pct < -stop_loss_pct:
                pnl = -size * (stop_loss_pct / 100)
            
            pnl_series.append(pnl)
            realized_pnl += pnl
            bankroll += pnl
            peak_bankroll = max(peak_bankroll, bankroll)
        
        # Compute metrics
        if not pnl_series:
            return {
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'sharpe': 0,
                'sortino': 0,
                'max_drawdown': 0,
                'es_sharpe': 0,
                'win_rate': 0,
                'total_trades': 0,
            }
        
        pnl_array = np.array(pnl_series)
        returns = pnl_array / self.initial_bankroll
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0001
        
        # Sharpe (annualized)
        sharpe = (mean_return / std_return) * np.sqrt(252 * 20) if std_return > 0 else 0
        
        # Sortino
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns, ddof=1)
            sortino = (mean_return / downside_std) * np.sqrt(252 * 20) if downside_std > 0 else sharpe
        else:
            sortino = sharpe * 2
        
        # Max drawdown
        cumulative = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative + self.initial_bankroll)
        drawdowns = (running_max - (cumulative + self.initial_bankroll)) / running_max
        max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
        
        # ES Sharpe
        es_sharpe = compute_es_sharpe(pnl_series)
        
        # Win rate
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        return {
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'es_sharpe': es_sharpe,
            'win_rate': win_rate,
            'total_trades': len(pnl_series),
            'final_bankroll': bankroll,
        }


def evaluate_params_worker(
    data_records: List[Dict],
    params: Dict[str, float],
    weights_dict: Dict[str, float],
    initial_bankroll: float,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Worker function for parallel evaluation.
    Must be a top-level function to be picklable.
    """
    # Reconstruct DataFrame
    df = pd.DataFrame(data_records)
    
    # Run simulation
    simulator = BacktestSimulator(df, initial_bankroll)
    metrics = simulator.run(params)
    
    # Compute objective
    weights = ObjectiveWeights(**weights_dict)
    objective = compute_objective(metrics, weights, initial_bankroll)
    
    return objective, metrics, params


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using CMA-ES, grid search, or random search.
    Supports parallel evaluation across multiple CPU cores.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        param_space: HyperparameterSpace = None,
        weights: ObjectiveWeights = None,
        initial_bankroll: float = 10000,
    ):
        self.data = historical_data
        self.param_space = param_space or HyperparameterSpace()
        self.weights = weights or ObjectiveWeights()
        self.initial_bankroll = initial_bankroll
        
        self.simulator = BacktestSimulator(historical_data, initial_bankroll)
        self.results: List[OptimizationResult] = []
    
    def sample_params(self, n_samples: int = 1, method: str = 'random') -> List[Dict[str, float]]:
        """Sample hyperparameters from the search space."""
        params_list = []
        
        for _ in range(n_samples):
            params = {}
            
            for field_name in self.param_space.__dataclass_fields__:
                bounds = getattr(self.param_space, field_name)
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    if method == 'random':
                        params[field_name] = np.random.uniform(bounds[0], bounds[1])
                    else:
                        # For grid, use midpoint
                        params[field_name] = (bounds[0] + bounds[1]) / 2
            
            params_list.append(params)
        
        return params_list
    
    def grid_search(self, n_points_per_dim: int = 3) -> List[OptimizationResult]:
        """
        Perform grid search over key hyperparameters.
        """
        # Define grid for most important parameters
        grids = {
            'profit_take_pct': np.linspace(10, 40, n_points_per_dim),
            'stop_loss_pct': np.linspace(5, 25, n_points_per_dim),
            'kelly_fraction': np.linspace(0.1, 0.4, n_points_per_dim),
            'spread_threshold': np.linspace(0.05, 0.15, n_points_per_dim),
            'min_edge': np.linspace(0.03, 0.10, n_points_per_dim),
        }
        
        # Generate all combinations
        keys = list(grids.keys())
        values = [grids[k] for k in keys]
        
        results = []
        total = np.prod([len(v) for v in values])
        
        logger.info(f"Grid search: {total} combinations")
        
        for i, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            
            # Add default values for other params
            params.setdefault('max_position_pct', 0.10)
            params.setdefault('learning_rate', 0.01)
            params.setdefault('l2_regularization', 0.001)
            params.setdefault('ema_alpha_fast', 0.1)
            params.setdefault('ema_alpha_slow', 0.02)
            
            result = self.evaluate(params)
            results.append(result)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{total}")
        
        self.results.extend(results)
        return results
    
    def random_search(self, n_iterations: int = 100) -> List[OptimizationResult]:
        """
        Perform random search over hyperparameters.
        """
        results = []
        
        for i in range(n_iterations):
            params = self.sample_params(1, method='random')[0]
            result = self.evaluate(params)
            results.append(result)
            
            if (i + 1) % 20 == 0:
                best = max(results, key=lambda r: r.objective_value)
                logger.info(f"Iteration {i+1}/{n_iterations}, best objective: {best.objective_value:.4f}")
        
        self.results.extend(results)
        return results
    
    def evaluate(self, params: Dict[str, float]) -> OptimizationResult:
        """Evaluate a single set of hyperparameters."""
        metrics = self.simulator.run(params)
        objective = compute_objective(metrics, self.weights, self.initial_bankroll)
        
        return OptimizationResult(
            params=params,
            metrics=metrics,
            objective_value=objective,
        )
    
    def cma_es_optimize(
        self,
        n_iterations: int = 50,
        population_size: int = None,
        n_workers: int = None,
        sigma0: float = 0.3,
    ) -> List[OptimizationResult]:
        """
        Optimize using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
        
        CMA-ES is a powerful derivative-free optimization algorithm that:
        - Adapts the search distribution based on successful samples
        - Handles non-convex, multi-modal objective functions
        - Can be parallelized across population evaluations
        
        Args:
            n_iterations: Maximum number of generations
            population_size: Population size (default: 4 + 3*ln(n_params))
            n_workers: Number of parallel workers (default: CPU count)
            sigma0: Initial step size (as fraction of parameter range)
        
        Returns:
            List of optimization results
        """
        if not HAS_CMA:
            logger.error("CMA-ES not available. Install with: pip install cma")
            return self.random_search(n_iterations * 10)
        
        # Define parameter names and bounds
        param_names = [
            'profit_take_pct', 'stop_loss_pct', 'kelly_fraction', 
            'max_position_pct', 'min_edge', 'spread_threshold',
            'learning_rate', 'ema_alpha_fast', 'ema_alpha_slow',
        ]
        
        bounds_lower = []
        bounds_upper = []
        x0 = []
        
        for name in param_names:
            bounds = getattr(self.param_space, name, (0, 1))
            bounds_lower.append(bounds[0])
            bounds_upper.append(bounds[1])
            x0.append((bounds[0] + bounds[1]) / 2)  # Start at midpoint
        
        bounds_lower = np.array(bounds_lower)
        bounds_upper = np.array(bounds_upper)
        x0 = np.array(x0)
        
        # Normalize to [0, 1] for CMA-ES
        def normalize(x):
            return (x - bounds_lower) / (bounds_upper - bounds_lower)
        
        def denormalize(x_norm):
            return x_norm * (bounds_upper - bounds_lower) + bounds_lower
        
        x0_norm = normalize(x0)
        
        # Setup workers
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        logger.info(f"CMA-ES optimization with {n_workers} parallel workers")
        
        # Create objective function (negative because CMA-ES minimizes)
        def objective_single(x_norm):
            x = denormalize(np.clip(x_norm, 0, 1))
            params = dict(zip(param_names, x))
            result = self.evaluate(params)
            self.results.append(result)
            return -result.objective_value  # Negative for minimization
        
        # Parallel evaluation function
        def objective_parallel(X_norm):
            """Evaluate population in parallel."""
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for x_norm in X_norm:
                    x = denormalize(np.clip(x_norm, 0, 1))
                    params = dict(zip(param_names, x))
                    futures.append(executor.submit(evaluate_params_worker, 
                                                   self.data.to_dict('records'),
                                                   params,
                                                   self.weights.__dict__,
                                                   self.initial_bankroll))
                
                results = []
                for future in futures:
                    try:
                        obj_value, metrics, params = future.result()
                        result = OptimizationResult(
                            params=params,
                            metrics=metrics,
                            objective_value=obj_value,
                        )
                        self.results.append(result)
                        results.append(-obj_value)  # Negative for minimization
                    except Exception as e:
                        logger.warning(f"Worker failed: {e}")
                        results.append(0)  # Penalize failed evaluations
                
                return results
        
        # CMA-ES options
        opts = {
            'maxiter': n_iterations,
            'bounds': [0, 1],  # Normalized bounds
            'verbose': 1,
            'verb_disp': 10,
        }
        
        if population_size:
            opts['popsize'] = population_size
        
        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(x0_norm.tolist(), sigma0, opts)
        
        generation = 0
        while not es.stop():
            # Get population
            X = es.ask()
            
            # Evaluate in parallel
            try:
                fitnesses = objective_parallel(X)
            except Exception as e:
                logger.error(f"Parallel evaluation failed: {e}")
                # Fallback to sequential
                fitnesses = [objective_single(x) for x in X]
            
            # Update CMA-ES
            es.tell(X, fitnesses)
            
            generation += 1
            if generation % 5 == 0:
                best_so_far = self.get_best(1)[0] if self.results else None
                if best_so_far:
                    logger.info(f"Generation {generation}: best objective = {best_so_far.objective_value:.4f}")
        
        # Get final result
        best_x_norm = es.result.xbest
        best_x = denormalize(np.clip(best_x_norm, 0, 1))
        best_params = dict(zip(param_names, best_x))
        
        logger.info(f"CMA-ES completed after {generation} generations")
        logger.info(f"Total evaluations: {len(self.results)}")
        
        return self.results
    
    def parallel_random_search(
        self,
        n_iterations: int = 200,
        n_workers: int = None,
    ) -> List[OptimizationResult]:
        """
        Parallel random search using multiple workers.
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        logger.info(f"Parallel random search with {n_workers} workers, {n_iterations} iterations")
        
        # Generate all parameter sets upfront
        all_params = self.sample_params(n_iterations, method='random')
        
        # Convert data to records for pickling
        data_records = self.data.to_dict('records')
        weights_dict = {
            'realized_pnl': self.weights.realized_pnl,
            'unrealized_pnl': self.weights.unrealized_pnl,
            'sharpe': self.weights.sharpe,
            'max_drawdown': self.weights.max_drawdown,
            'es_sharpe': self.weights.es_sharpe,
        }
        
        results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_params_worker,
                    data_records,
                    params,
                    weights_dict,
                    self.initial_bankroll
                ): params for params in all_params
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    obj_value, metrics, params = future.result()
                    result = OptimizationResult(
                        params=params,
                        metrics=metrics,
                        objective_value=obj_value,
                    )
                    results.append(result)
                    
                    if completed % 50 == 0:
                        best = max(results, key=lambda r: r.objective_value)
                        logger.info(f"Progress: {completed}/{n_iterations}, best: {best.objective_value:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
        
        self.results.extend(results)
        return results
    
    def get_best(self, n: int = 1) -> List[OptimizationResult]:
        """Get the top N results."""
        sorted_results = sorted(self.results, key=lambda r: r.objective_value, reverse=True)
        return sorted_results[:n]
    
    def get_pareto_front(self) -> List[OptimizationResult]:
        """
        Get Pareto-optimal results (non-dominated solutions).
        
        A result is Pareto-optimal if no other result is better in all objectives.
        """
        pareto = []
        
        for result in self.results:
            is_dominated = False
            
            for other in self.results:
                if other is result:
                    continue
                
                # Check if other dominates result
                other_better_in_all = (
                    other.metrics.get('realized_pnl', 0) >= result.metrics.get('realized_pnl', 0) and
                    other.metrics.get('sharpe', 0) >= result.metrics.get('sharpe', 0) and
                    other.metrics.get('max_drawdown', 100) <= result.metrics.get('max_drawdown', 100) and
                    other.metrics.get('es_sharpe', 0) >= result.metrics.get('es_sharpe', 0)
                )
                other_strictly_better = (
                    other.metrics.get('realized_pnl', 0) > result.metrics.get('realized_pnl', 0) or
                    other.metrics.get('sharpe', 0) > result.metrics.get('sharpe', 0) or
                    other.metrics.get('max_drawdown', 100) < result.metrics.get('max_drawdown', 100) or
                    other.metrics.get('es_sharpe', 0) > result.metrics.get('es_sharpe', 0)
                )
                
                if other_better_in_all and other_strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(result)
        
        return pareto
    
    def save_results(self, path: str):
        """Save results to JSON file."""
        data = [
            {
                'params': r.params,
                'metrics': r.metrics,
                'objective_value': r.objective_value,
                'timestamp': r.timestamp,
            }
            for r in self.results
        ]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.results:
            return {'status': 'no results'}
        
        best = self.get_best(1)[0]
        pareto = self.get_pareto_front()
        
        return {
            'total_evaluations': len(self.results),
            'best_objective': best.objective_value,
            'best_params': best.params,
            'best_metrics': best.metrics,
            'pareto_front_size': len(pareto),
            'objective_weights': {
                'realized_pnl': self.weights.realized_pnl,
                'unrealized_pnl': self.weights.unrealized_pnl,
                'sharpe': self.weights.sharpe,
                'max_drawdown': self.weights.max_drawdown,
                'es_sharpe': self.weights.es_sharpe,
            },
        }


def run_optimization(
    data_path: str,
    output_path: str = "optimization_results.json",
    method: str = "cma",
    n_iterations: int = 50,
    n_workers: int = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization.
    
    Args:
        data_path: Path to historical data parquet file
        output_path: Path to save results
        method: 'cma' (CMA-ES), 'parallel' (parallel random), 'random', or 'grid'
        n_iterations: Number of iterations/generations
        n_workers: Number of parallel workers (default: CPU count)
        weights: Optional custom weights for objectives
    
    Returns:
        Optimization summary
    """
    import time
    start_time = time.time()
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Standardize columns
    if 'avg_price' in df.columns:
        df = df.rename(columns={'avg_price': 'price'})
    if 'y' in df.columns:
        df = df.rename(columns={'y': 'outcome'})
    
    df = df.dropna(subset=['price', 'outcome'])
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    logger.info(f"Loaded {len(df)} samples for optimization")
    logger.info(f"Method: {method}, Workers: {n_workers}, Iterations: {n_iterations}")
    
    # Setup weights
    if weights:
        obj_weights = ObjectiveWeights(**weights)
    else:
        obj_weights = ObjectiveWeights()
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        historical_data=df,
        weights=obj_weights,
    )
    
    # Run optimization
    if method == 'cma':
        optimizer.cma_es_optimize(n_iterations=n_iterations, n_workers=n_workers)
    elif method == 'parallel':
        optimizer.parallel_random_search(n_iterations=n_iterations * 10, n_workers=n_workers)
    elif method == 'grid':
        optimizer.grid_search(n_points_per_dim=4)
    else:
        optimizer.random_search(n_iterations=n_iterations)
    
    elapsed = time.time() - start_time
    
    # Save results
    optimizer.save_results(output_path)
    
    # Get summary
    summary = optimizer.summary()
    summary['elapsed_seconds'] = elapsed
    summary['n_workers'] = n_workers
    summary['method'] = method
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Method: {method.upper()}")
    print(f"Workers: {n_workers}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Best objective: {summary['best_objective']:.4f}")
    print(f"Pareto front size: {summary['pareto_front_size']}")
    print("\nBest parameters:")
    for k, v in summary['best_params'].items():
        print(f"  {k}: {v:.4f}")
    print("\nBest metrics:")
    for k, v in summary['best_metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with CMA-ES")
    parser.add_argument("--data", required=True, help="Path to historical data parquet")
    parser.add_argument("--output", default="optimization_results.json", help="Output path")
    parser.add_argument("--method", default="cma", 
                        choices=["cma", "parallel", "random", "grid"],
                        help="Optimization method: cma (CMA-ES), parallel (parallel random), random, grid")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of generations (CMA-ES) or iterations")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--weight-pnl", type=float, default=0.25,
                        help="Weight for realized PnL objective")
    parser.add_argument("--weight-sharpe", type=float, default=0.25,
                        help="Weight for Sharpe ratio objective")
    parser.add_argument("--weight-drawdown", type=float, default=0.20,
                        help="Weight for max drawdown objective (penalty)")
    parser.add_argument("--weight-es-sharpe", type=float, default=0.20,
                        help="Weight for ES Sharpe objective")
    
    args = parser.parse_args()
    
    weights = {
        'realized_pnl': args.weight_pnl,
        'unrealized_pnl': 0.10,
        'sharpe': args.weight_sharpe,
        'max_drawdown': args.weight_drawdown,
        'es_sharpe': args.weight_es_sharpe,
    }
    
    run_optimization(
        data_path=args.data,
        output_path=args.output,
        method=args.method,
        n_iterations=args.iterations,
        n_workers=args.workers,
        weights=weights,
    )
