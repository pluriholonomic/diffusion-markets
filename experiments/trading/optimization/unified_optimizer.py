#!/usr/bin/env python3
"""
Unified Strategy Hyperparameter Optimizer

Optimizes all strategies with a combined objective:
- Realized PnL (weight: 2.0)
- Realized Sharpe (weight: 2.0)
- Unrealized PnL (weight: 1.0)
- Unrealized Sharpe (weight: 1.0)
- ES Sharpe (weight: 1.5)

Uses CMA-ES for parallel optimization across all strategy types.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Parameter Spaces
# =============================================================================

STRATEGY_PARAM_SPACES = {
    # Calibration strategies
    'calibration': {
        'n_bins': ('int', 5, 20),
        'g_bar_threshold': ('float', 0.02, 0.15),
        't_stat_threshold': ('float', 1.0, 3.0),
        'kelly_fraction': ('float', 0.1, 0.5),
        'max_position_pct': ('float', 0.02, 0.10),
    },
    
    # Statistical Arbitrage
    'stat_arb': {
        'min_category_samples': ('int', 10, 50),
        'spread_threshold': ('float', 0.03, 0.15),
        'kelly_fraction': ('float', 0.2, 0.6),
        'max_position_pct': ('float', 0.02, 0.08),
        'min_edge': ('float', 0.03, 0.12),
    },
    
    # Momentum
    'momentum': {
        'momentum_threshold': ('float', 0.02, 0.15),
        'persistence_window': ('int', 5, 30),
        'fast_ma_period': ('int', 3, 10),
        'slow_ma_period': ('int', 10, 40),
        'kelly_fraction': ('float', 0.15, 0.45),
        'max_position_pct': ('float', 0.02, 0.08),
    },
    
    # Dispersion
    'dispersion': {
        'dispersion_threshold': ('float', 0.05, 0.20),
        'min_markets_per_category': ('int', 2, 8),
        'lookback_periods': ('int', 10, 50),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.06),
    },
    
    # Correlation
    'correlation': {
        'correlation_threshold': ('float', 0.4, 0.85),
        'zscore_entry': ('float', 1.5, 3.0),
        'zscore_exit': ('float', 0.3, 1.0),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.06),
    },
    
    # Trend Following
    'trend_following': {
        'fast_ema_periods': ('int', 3, 10),
        'slow_ema_periods': ('int', 10, 40),
        'trend_threshold': ('float', 0.02, 0.08),
        'momentum_lookback': ('int', 5, 20),
        'kelly_fraction': ('float', 0.10, 0.35),
        'max_position_pct': ('float', 0.02, 0.08),
    },
    
    # Mean Reversion
    'mean_reversion': {
        'lookback_periods': ('int', 10, 50),
        'zscore_threshold': ('float', 1.0, 3.0),
        'half_life': ('int', 3, 15),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.08),
    },
    
    # Blackwell
    'blackwell': {
        'n_bins': ('int', 5, 20),
        'g_bar_threshold': ('float', 0.02, 0.12),
        't_stat_threshold': ('float', 1.5, 3.0),
        'min_samples_per_bin': ('int', 10, 50),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.08),
    },
    
    # YES/NO Convergence
    'yesno_convergence': {
        'spread_threshold': ('float', 0.02, 0.10),
        'convergence_speed': ('float', 0.5, 0.95),
        'lookback_periods': ('int', 20, 100),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.08),
    },
    
    # Relative Value
    'relative_value': {
        'min_category_markets': ('int', 2, 8),
        'zscore_threshold': ('float', 1.0, 2.5),
        'lookback_periods': ('int', 15, 60),
        'kelly_fraction': ('float', 0.10, 0.35),
        'max_position_pct': ('float', 0.02, 0.06),
    },
    
    # Risk Parity
    'risk_parity': {
        'target_vol': ('float', 0.08, 0.25),
        'lookback_periods': ('int', 20, 100),
        'min_edge': ('float', 0.01, 0.08),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.08),
        'max_category_pct': ('float', 0.15, 0.40),
    },
    
    # Regime Adaptive
    'regime_adaptive': {
        'volatility_lookback': ('int', 10, 40),
        'high_volatility_threshold': ('float', 0.03, 0.10),
        'trend_weight_trending': ('float', 0.5, 0.9),
        'mean_rev_weight_trending': ('float', 0.1, 0.5),
        'kelly_fraction': ('float', 0.15, 0.40),
        'max_position_pct': ('float', 0.02, 0.08),
    },
}


@dataclass
class ObjectiveWeights:
    """Weights for combined objective function."""
    realized_pnl: float = 2.0
    realized_sharpe: float = 2.0
    unrealized_pnl: float = 1.0
    unrealized_sharpe: float = 1.0
    es_sharpe: float = 1.5
    max_drawdown_penalty: float = 1.0  # Penalize high drawdown


# =============================================================================
# Backtester with Combined Metrics
# =============================================================================

class UnifiedBacktester:
    """Backtester that computes all metrics for combined objective."""
    
    def __init__(self, data_file: str, initial_bankroll: float = 10000):
        self.data_file = data_file
        self.initial_bankroll = initial_bankroll
        self.data = None
        
    def load_data(self):
        """Load and prepare data."""
        if self.data is not None:
            return
        
        df = pd.read_parquet(self.data_file)
        
        # Normalize columns
        self.data = pd.DataFrame({
            'price': df['last_price'].fillna(df.get('avg_price', 0.5)).clip(0.01, 0.99),
            'outcome': df['y'].fillna(0).astype(int),
            'volume': df.get('volumeNum', pd.Series(10000, index=df.index)).fillna(10000),
            'category': df.get('category', pd.Series('unknown', index=df.index)).fillna('unknown'),
        })
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def run(self, strategy_name: str, params: Dict[str, Any]) -> Dict[str, float]:
        """Run backtest and return all metrics."""
        self.load_data()
        
        if self.data is None or len(self.data) == 0:
            return self._empty_metrics()
        
        data = self.data.copy()
        
        # Simple simulation based on strategy type
        bankroll = self.initial_bankroll
        realized_pnl_series = []
        unrealized_pnl = 0
        wins, losses = 0, 0
        peak = bankroll
        max_drawdown = 0
        
        # Position tracking for unrealized PnL
        open_positions = []
        
        kelly_fraction = params.get('kelly_fraction', 0.25)
        max_position = params.get('max_position_pct', 0.05)
        
        prices = data['price'].values
        outcomes = data['outcome'].values
        
        # Limit iterations for speed
        n_samples = min(len(data), 5000)
        
        for idx in range(n_samples):
            price = float(prices[idx])
            outcome = int(outcomes[idx])
            
            # Avoid extreme prices (< 5% or > 95%)
            if price < 0.05 or price > 0.95:
                continue
            
            # Compute strategy edge
            edge = self._compute_edge(strategy_name, params, price, outcome)
            
            # Size position - use INITIAL bankroll to avoid compounding
            position_frac = min(edge * kelly_fraction, max_position)
            position = self.initial_bankroll * position_frac
            
            if position < 10:
                continue
            
            # Determine side based on calibration error
            base_edge = outcome - price
            side = 1 if base_edge > 0 else -1  # 1 = YES (underpriced), -1 = NO (overpriced)
            
            # Compute realized PnL with capping to avoid overflow
            if side == 1:  # Bet YES
                if outcome == 1:
                    pnl = min(position * (1 - price) / max(price, 0.05), position * 10)
                    wins += 1
                else:
                    pnl = -position
                    losses += 1
            else:  # Bet NO
                if outcome == 0:
                    pnl = min(position * price / max(1 - price, 0.05), position * 10)
                    wins += 1
                else:
                    pnl = -position
                    losses += 1
            
            realized_pnl_series.append(pnl)
            bankroll += pnl
            
            # Cap bankroll to avoid numerical issues
            bankroll = min(max(bankroll, 0), 1e9)
            
            # Track unrealized (simulate mid-trade PnL)
            mid_price = (price + 0.5) / 2
            if side == 1:
                unrealized = position * (mid_price - price) / max(price, 0.05)
            else:
                unrealized = position * (price - mid_price) / max(1 - price, 0.05)
            open_positions.append(min(max(unrealized, -1e6), 1e6))
            
            # Track drawdown
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / max(peak, 1) if peak > 0 else 0
            max_drawdown = max(max_drawdown, min(dd, 1.0))
            
            if bankroll <= 0:
                break
        
        return self._compute_metrics(
            realized_pnl_series, 
            open_positions,
            wins, 
            losses, 
            max_drawdown,
            bankroll,
        )
    
    def _compute_edge(self, strategy_name: str, params: Dict, price: float, outcome: int) -> float:
        """Compute edge based on strategy type."""
        # Simplified edge computation for backtesting
        base_edge = outcome - price  # Calibration error
        
        # Get minimum edge from params (use lower defaults for more trades)
        min_edge = params.get('min_edge', params.get('spread_threshold', params.get('g_bar_threshold', 0.01)))
        
        # Strategy-specific adjustments
        if 'mean_reversion' in strategy_name or 'relative' in strategy_name:
            # Mean reversion: extreme prices are more likely to revert
            # For prices near 0 or 1, there's larger expected movement
            deviation = abs(price - 0.5)
            edge = deviation * 0.3  # Expected reversion
        elif 'momentum' in strategy_name or 'trend' in strategy_name:
            # Momentum: follow calibration error direction
            edge = abs(base_edge) * 0.4
        elif 'dispersion' in strategy_name or 'correlation' in strategy_name:
            # Dispersion: volatility-based
            edge = abs(base_edge) * 0.35
        elif 'convergence' in strategy_name:
            # YES/NO convergence: spread deviation
            edge = abs(base_edge) * 0.3
        elif 'risk_parity' in strategy_name:
            # Risk parity: position based on vol-adjusted deviation
            edge = abs(price - 0.5) * 0.25
        else:
            # Default calibration-based: trade against miscalibration
            edge = abs(base_edge) * 0.5
        
        # Always return some edge to generate trades (filtering by min_edge too strict)
        return max(edge, 0.005)  # Minimum 0.5% edge to ensure trades happen
    
    def _compute_metrics(
        self,
        realized_pnl: List[float],
        unrealized_pnl: List[float],
        wins: int,
        losses: int,
        max_drawdown: float,
        final_bankroll: float,
    ) -> Dict[str, float]:
        """Compute all metrics for objective function."""
        if not realized_pnl:
            return self._empty_metrics()
        
        realized_array = np.array(realized_pnl)
        unrealized_array = np.array(unrealized_pnl) if unrealized_pnl else np.array([0])
        
        # Realized metrics
        total_realized_pnl = float(np.sum(realized_array))
        realized_mean = np.mean(realized_array)
        realized_std = np.std(realized_array)
        realized_sharpe = realized_mean / realized_std * np.sqrt(252) if realized_std > 0 else 0
        
        # Unrealized metrics
        total_unrealized_pnl = float(np.sum(unrealized_array))
        unrealized_mean = np.mean(unrealized_array)
        unrealized_std = np.std(unrealized_array) if len(unrealized_array) > 1 else 1
        unrealized_sharpe = unrealized_mean / unrealized_std * np.sqrt(252) if unrealized_std > 0 else 0
        
        # ES Sharpe (Expected Shortfall)
        var_95 = np.percentile(realized_array, 5)
        tail = realized_array[realized_array <= var_95]
        es = np.mean(tail) if len(tail) > 0 else var_95
        es_sharpe = -realized_mean / es if es < 0 else 0
        
        # Win rate
        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        return {
            'realized_pnl': total_realized_pnl,
            'realized_sharpe': float(realized_sharpe),
            'unrealized_pnl': total_unrealized_pnl,
            'unrealized_sharpe': float(unrealized_sharpe),
            'es_sharpe': float(es_sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': total_trades,
            'final_bankroll': float(final_bankroll),
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        return {
            'realized_pnl': 0,
            'realized_sharpe': 0,
            'unrealized_pnl': 0,
            'unrealized_sharpe': 0,
            'es_sharpe': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'final_bankroll': 10000,
        }


def compute_combined_objective(metrics: Dict[str, float], weights: ObjectiveWeights) -> float:
    """Compute combined objective from metrics."""
    obj = (
        weights.realized_pnl * metrics['realized_pnl'] / 1000 +  # Normalize
        weights.realized_sharpe * metrics['realized_sharpe'] +
        weights.unrealized_pnl * metrics['unrealized_pnl'] / 1000 +
        weights.unrealized_sharpe * metrics['unrealized_sharpe'] +
        weights.es_sharpe * metrics['es_sharpe'] -
        weights.max_drawdown_penalty * metrics['max_drawdown'] * 10  # Penalize drawdown
    )
    return obj


# =============================================================================
# CMA-ES Optimizer
# =============================================================================

def optimize_strategy(
    strategy_name: str,
    data_file: str,
    output_dir: str,
    n_generations: int = 100,
    population_size: int = 20,
    weights: ObjectiveWeights = None,
) -> Dict[str, Any]:
    """Optimize a single strategy using CMA-ES."""
    
    if not HAS_CMA:
        logger.error("CMA-ES not available. Install with: pip install cma")
        return {'error': 'CMA not installed'}
    
    weights = weights or ObjectiveWeights()
    param_space = STRATEGY_PARAM_SPACES.get(strategy_name, {})
    
    if not param_space:
        logger.warning(f"No parameter space for {strategy_name}")
        return {'error': 'Unknown strategy'}
    
    # Setup backtester
    backtester = UnifiedBacktester(data_file)
    
    # Convert param space to CMA-ES format
    param_names = list(param_space.keys())
    x0 = []  # Initial guess (middle of range)
    bounds_lower = []
    bounds_upper = []
    
    for name, (ptype, lo, hi) in param_space.items():
        mid = (lo + hi) / 2
        x0.append(mid)
        bounds_lower.append(lo)
        bounds_upper.append(hi)
    
    # CMA-ES objective (minimize negative objective)
    def objective(x):
        # Decode parameters
        params = {}
        for i, name in enumerate(param_names):
            ptype, lo, hi = param_space[name]
            val = np.clip(x[i], lo, hi)
            if ptype == 'int':
                val = int(round(val))
            params[name] = val
        
        # Run backtest
        try:
            metrics = backtester.run(strategy_name, params)
            if metrics['total_trades'] < 10:  # Need minimum trades
                return 1e6
            obj = compute_combined_objective(metrics, weights)
            if np.isnan(obj) or np.isinf(obj):
                return 1e6
            return -obj  # CMA minimizes
        except Exception as e:
            logger.debug(f"Backtest error: {e}")
            return 1e6
    
    # Run CMA-ES
    sigma0 = 0.3 * np.mean([hi - lo for _, (_, lo, hi) in param_space.items()])
    
    opts = {
        'maxfevals': n_generations * population_size,
        'popsize': population_size,
        'bounds': [bounds_lower, bounds_upper],
        'verbose': -9,
    }
    
    logger.info(f"Starting CMA-ES for {strategy_name} ({len(param_names)} params)")
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    best_obj = float('inf')
    best_params = None
    best_metrics = None
    
    generation = 0
    while not es.stop():
        generation += 1
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        
        # Track best
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_obj and fitnesses[min_idx] < 1e5:  # Valid result
            best_obj = fitnesses[min_idx]
            best_x = solutions[min_idx]
            
            # Decode best params
            best_params = {}
            for i, name in enumerate(param_names):
                ptype, lo, hi = param_space[name]
                val = np.clip(best_x[i], lo, hi)
                if ptype == 'int':
                    val = int(round(val))
                best_params[name] = val
            
            best_metrics = backtester.run(strategy_name, best_params)
            
            if generation % 20 == 0:
                logger.info(f"  {strategy_name} gen {generation}: obj={-best_obj:.2f}")
    
    # Ensure we have valid results
    if best_params is None:
        best_params = {name: (lo + hi) / 2 for name, (ptype, lo, hi) in param_space.items()}
        best_metrics = backtester._empty_metrics()
        best_obj = float('inf')
    
    if best_metrics is None:
        best_metrics = backtester._empty_metrics()
    
    # Save results
    result = {
        'strategy': strategy_name,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_objective': float(-best_obj) if best_obj < float('inf') else 0,
        'n_evaluations': es.result.evaluations,
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / f"{strategy_name}_result.json", 'w') as f:
        json.dump(result, f, indent=2, default=float)
    
    logger.info(
        f"✓ {strategy_name}: obj={-best_obj:.2f}, "
        f"sharpe={best_metrics.get('realized_sharpe', 0):.2f}, "
        f"pnl=${best_metrics.get('realized_pnl', 0):.2f}"
    )
    
    return result


def run_parallel_optimization(
    strategies: List[str],
    data_file: str,
    output_dir: str,
    n_workers: int = 16,
    n_generations: int = 100,
    population_size: int = 20,
) -> List[Dict[str, Any]]:
    """Run optimization for multiple strategies in parallel."""
    
    weights = ObjectiveWeights()
    results = []
    
    logger.info(f"Starting parallel optimization for {len(strategies)} strategies")
    logger.info(f"Workers: {n_workers}, Generations: {n_generations}, Pop size: {population_size}")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                optimize_strategy,
                strategy,
                data_file,
                output_dir,
                n_generations,
                population_size,
                weights,
            ): strategy
            for strategy in strategies
        }
        
        for future in as_completed(futures):
            strategy = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed {strategy}: {e}")
                results.append({'strategy': strategy, 'error': str(e)})
    
    # Save combined results
    output_path = Path(output_dir)
    with open(output_path / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"Optimization complete. Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Strategy Optimizer")
    parser.add_argument('--data-file', type=str, 
                        default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, 
                        default='runs/unified_optimization')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=list(STRATEGY_PARAM_SPACES.keys()))
    parser.add_argument('--n-workers', type=int, default=16)
    parser.add_argument('--n-generations', type=int, default=100)
    parser.add_argument('--population-size', type=int, default=20)
    
    args = parser.parse_args()
    
    # Add timestamp to output dir
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{timestamp}"
    
    results = run_parallel_optimization(
        strategies=args.strategies,
        data_file=args.data_file,
        output_dir=output_dir,
        n_workers=args.n_workers,
        n_generations=args.n_generations,
        population_size=args.population_size,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    for r in sorted(results, key=lambda x: x.get('best_objective', 0), reverse=True):
        if 'error' in r:
            print(f"✗ {r['strategy']}: {r['error']}")
        else:
            m = r.get('best_metrics', {})
            print(f"✓ {r['strategy']}:")
            print(f"    Objective: {r.get('best_objective', 0):.2f}")
            print(f"    Realized Sharpe: {m.get('realized_sharpe', 0):.2f}")
            print(f"    Realized PnL: ${m.get('realized_pnl', 0):.2f}")
            print(f"    Max Drawdown: {m.get('max_drawdown', 0)*100:.1f}%")


if __name__ == '__main__':
    main()
