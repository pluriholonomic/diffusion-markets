#!/usr/bin/env python3
"""
Kalshi-Specific Hyperparameter Optimizer

Parallel optimization for strategies on Kalshi markets with:
- Kalshi-specific market characteristics (different fee structure, regulations)
- Different market categories (politics, economics, weather, entertainment)
- CMA-ES optimization with multiple objective functions
"""

import os
import sys
import json
import time
import argparse
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Kalshi-Specific Configuration
# ============================================================================

# Kalshi has different characteristics than Polymarket
KALSHI_CONFIG = {
    'fee_rate': 0.01,  # 1% fee on profits
    'min_trade_size': 1.0,  # Minimum $1 trade
    'max_position_per_market': 25000,  # $25K max per market
    'settlement_delay_hours': 24,  # Settlement typically within 24 hours
}

# Kalshi categories with different volatility/liquidity profiles
KALSHI_CATEGORIES = {
    'politics': {'volatility': 0.8, 'liquidity': 1.0, 'resolution_speed': 0.3},
    'economics': {'volatility': 0.5, 'liquidity': 0.8, 'resolution_speed': 0.7},
    'weather': {'volatility': 0.9, 'liquidity': 0.6, 'resolution_speed': 0.9},
    'entertainment': {'volatility': 0.7, 'liquidity': 0.5, 'resolution_speed': 0.5},
    'science': {'volatility': 0.4, 'liquidity': 0.4, 'resolution_speed': 0.2},
    'sports': {'volatility': 0.95, 'liquidity': 0.7, 'resolution_speed': 0.95},
}

# Strategy configurations for Kalshi
KALSHI_STRATEGIES = {
    'calibration': {
        'kelly_fraction': (0.1, 0.5),
        'spread_threshold': (0.02, 0.15),
        'min_edge': (0.02, 0.12),
        'max_position_pct': (0.02, 0.08),
        'profit_take_pct': (25.0, 70.0),
        'stop_loss_pct': (15.0, 45.0),
    },
    'stat_arb': {
        'kelly_fraction': (0.1, 0.4),
        'spread_threshold': (0.02, 0.12),
        'min_edge': (0.02, 0.12),
        'max_position_pct': (0.02, 0.08),
        'correlation_threshold': (0.5, 0.9),
    },
    'momentum': {
        'fast_window': (3, 12),
        'slow_window': (12, 40),
        'momentum_threshold': (0.015, 0.08),
        'kelly_fraction': (0.1, 0.35),
        'max_position_pct': (0.02, 0.08),
    },
    'mean_reversion': {
        'lookback_periods': (8, 35),
        'zscore_threshold': (1.2, 2.8),
        'kelly_fraction': (0.1, 0.35),
        'max_position_pct': (0.02, 0.07),
    },
    'trend_following': {
        'fast_ema_periods': (3, 8),
        'slow_ema_periods': (12, 35),
        'trend_threshold': (0.015, 0.06),
        'kelly_fraction': (0.1, 0.35),
        'max_position_pct': (0.02, 0.07),
    },
    'confidence_gated': {
        'min_distance_threshold': (0.02, 0.12),
        'max_confidence_threshold': (0.80, 0.95),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.08),
    },
    'longshot': {
        'max_price': (0.08, 0.20),
        'min_edge': (0.08, 0.25),
        'kelly_fraction': (0.03, 0.15),
        'max_position_pct': (0.01, 0.04),
    },
}

# Objective function configurations
OBJECTIVE_CONFIGS = {
    'balanced': {
        'pnl_weight': 0.30,
        'sharpe_weight': 0.35,
        'drawdown_weight': -0.15,
        'es_sharpe_weight': 0.10,
        'win_rate_weight': 0.10,
    },
    'pnl_focused': {
        'pnl_weight': 0.55,
        'sharpe_weight': 0.25,
        'drawdown_weight': -0.10,
        'es_sharpe_weight': 0.05,
        'win_rate_weight': 0.05,
    },
    'risk_adjusted': {
        'pnl_weight': 0.20,
        'sharpe_weight': 0.40,
        'drawdown_weight': -0.20,
        'es_sharpe_weight': 0.15,
        'win_rate_weight': 0.05,
    },
    'conservative': {
        'pnl_weight': 0.15,
        'sharpe_weight': 0.30,
        'drawdown_weight': -0.30,
        'es_sharpe_weight': 0.20,
        'win_rate_weight': 0.05,
    },
}


# ============================================================================
# Kalshi Backtester
# ============================================================================

class KalshiBacktester:
    """Backtester specifically tuned for Kalshi market characteristics."""
    
    def __init__(
        self,
        data_file: str,
        initial_bankroll: float = 10000,
        impact_pct: float = 1.5,  # Lower than Polymarket due to regulation
    ):
        self.data_file = data_file
        self.initial_bankroll = initial_bankroll
        self.impact_pct = impact_pct
        self.data = None
        
    def load_data(self):
        """Load Kalshi market data."""
        import pandas as pd
        
        if self.data is not None:
            return self.data
        
        try:
            df = pd.read_parquet(self.data_file)
            
            # Normalize column names for Kalshi format
            if 'last_price' in df.columns:
                price_col = 'last_price'
            elif 'close' in df.columns:
                price_col = 'close'
            elif 'yes_price' in df.columns:
                price_col = 'yes_price'
            else:
                price_col = df.columns[0]
            
            # Kalshi prices are typically in cents (0-100), normalize to 0-1
            prices = df[price_col].fillna(50)
            if prices.max() > 1:
                prices = prices / 100
            
            # Determine outcome column
            if 'expiration_value' in df.columns:
                outcomes = df['expiration_value'].apply(
                    lambda x: 1 if str(x).lower() == 'yes' else 0
                )
            elif 'outcome' in df.columns:
                outcomes = df['outcome'].fillna(0)
            elif 'y' in df.columns:
                outcomes = df['y'].fillna(0)
            else:
                outcomes = (prices > 0.5).astype(int)
            
            # Get volume if available
            if 'volume' in df.columns:
                volumes = df['volume'].fillna(5000)
            elif 'volumeNum' in df.columns:
                volumes = df['volumeNum'].fillna(5000)
            else:
                volumes = np.full(len(df), 5000.0)
            
            # Get category if available
            if 'category' in df.columns:
                categories = df['category'].fillna('general')
            elif 'event_ticker' in df.columns:
                # Extract category from event ticker (e.g., "ECON-CPI-..." -> "economics")
                categories = df['event_ticker'].apply(self._extract_category)
            else:
                categories = pd.Series(['general'] * len(df))
            
            self.data = pd.DataFrame({
                'price': prices.clip(0.01, 0.99),
                'outcome': outcomes.astype(int),
                'volume': volumes,
                'category': categories,
                'spread': 0.015,  # Kalshi typically has tighter spreads
            })
            
        except Exception as e:
            logger.warning(f"Failed to load Kalshi data: {e}, using synthetic data")
            np.random.seed(42)
            n = 3000
            self.data = pd.DataFrame({
                'price': np.random.beta(2, 2, n),
                'outcome': np.random.binomial(1, np.random.beta(2, 2, n)),
                'volume': np.random.exponential(8000, n),
                'category': np.random.choice(list(KALSHI_CATEGORIES.keys()), n),
                'spread': 0.015,
            })
        
        return self.data
    
    def _extract_category(self, ticker: str) -> str:
        """Extract category from Kalshi event ticker."""
        if not ticker:
            return 'general'
        
        ticker = str(ticker).upper()
        
        if any(x in ticker for x in ['PRES', 'ELECT', 'VOTE', 'GOV', 'CONG']):
            return 'politics'
        elif any(x in ticker for x in ['CPI', 'GDP', 'FED', 'RATE', 'ECON', 'INFL']):
            return 'economics'
        elif any(x in ticker for x in ['TEMP', 'RAIN', 'SNOW', 'HURR', 'WEATHER']):
            return 'weather'
        elif any(x in ticker for x in ['NFL', 'NBA', 'MLB', 'NCAA', 'SPORT']):
            return 'sports'
        elif any(x in ticker for x in ['OSCAR', 'EMMY', 'MOVIE', 'TV']):
            return 'entertainment'
        else:
            return 'general'
    
    def run(self, strategy_name: str, params: Dict[str, float]) -> Dict[str, float]:
        """Run backtest with given parameters."""
        data = self.load_data()
        
        bankroll = self.initial_bankroll
        pnl_series = []
        wins, losses = 0, 0
        total_trades = 0
        peak_bankroll = bankroll
        max_drawdown = 0
        
        kelly_fraction = params.get('kelly_fraction', 0.20)
        max_position = params.get('max_position_pct', 0.05)
        threshold = params.get('spread_threshold', params.get('min_edge', 0.04))
        
        # Category weights for stat arb
        category_weights = {
            'politics': params.get('category_weight_politics', 1.0),
            'economics': params.get('category_weight_economics', 1.0),
            'sports': params.get('category_weight_sports', 1.0),
            'weather': params.get('category_weight_weather', 1.0),
        }
        
        prices = data['price'].values
        outcomes = data['outcome'].values
        volumes = data['volume'].values
        categories = data['category'].values
        
        for idx in range(len(data)):
            price = float(prices[idx])
            outcome = int(outcomes[idx])
            volume = float(volumes[idx])
            category = str(categories[idx])
            
            # Apply category-specific adjustments
            cat_config = KALSHI_CATEGORIES.get(category, {'volatility': 0.5, 'liquidity': 0.5})
            
            # Compute edge based on strategy
            edge = self._compute_edge(strategy_name, params, price, outcome, category)
            
            # Apply category weight
            edge *= category_weights.get(category, 1.0)
            
            if edge < threshold:
                continue
            
            # Position sizing with Kalshi limits
            kelly = min(edge * kelly_fraction, 0.4)
            size = min(
                self.initial_bankroll * kelly * max_position,
                KALSHI_CONFIG['max_position_per_market']
            )
            
            # Minimum trade size
            if size < KALSHI_CONFIG['min_trade_size']:
                continue
            
            # Compute market impact (lower for Kalshi due to regulation)
            impact = (self.impact_pct / 100) * (1 + size / max(volume, 1000) * 0.5)
            
            # Effective entry price
            effective_price = price * (1 + impact)
            
            # Simulate trade outcome with category volatility
            win_prob = 0.5 + edge - impact - (cat_config['volatility'] - 0.5) * 0.1
            
            if np.random.random() < win_prob:
                # Win - apply Kalshi fee on profit
                gross_pnl = size * (1 / effective_price - 1) if effective_price > 0 else 0
                fee = max(0, gross_pnl * KALSHI_CONFIG['fee_rate'])
                pnl = gross_pnl - fee
                wins += 1
            else:
                # Loss
                pnl = -size
                losses += 1
            
            bankroll += pnl
            pnl_series.append(pnl)
            total_trades += 1
            
            # Track drawdown
            peak_bankroll = max(peak_bankroll, bankroll)
            drawdown = (peak_bankroll - bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, drawdown)
        
        return self._compute_metrics(pnl_series, wins, losses, total_trades, max_drawdown)
    
    def _compute_edge(
        self, 
        strategy_name: str, 
        params: Dict, 
        price: float, 
        outcome: int,
        category: str
    ) -> float:
        """Compute expected edge for a trade."""
        cat_config = KALSHI_CATEGORIES.get(category, {'volatility': 0.5})
        
        if strategy_name in ['calibration', 'stat_arb', 'confidence_gated']:
            base_edge = abs(float(outcome) - price)
            edge = base_edge * (1 + cat_config['volatility'] * 0.2) if np.random.random() < 0.25 else 0
        elif strategy_name in ['momentum', 'trend_following']:
            edge = abs(price - 0.5) * 0.25 * cat_config['volatility'] if np.random.random() < 0.18 else 0
        elif strategy_name == 'mean_reversion':
            edge = max(0, abs(price - 0.5) - 0.25) * (1 - cat_config['volatility'] * 0.3) if np.random.random() < 0.22 else 0
        elif strategy_name == 'longshot':
            edge = 0.18 if price < params.get('max_price', 0.12) and np.random.random() < 0.08 else 0
        else:
            edge = np.random.random() * 0.08
        
        return float(edge)
    
    def _compute_metrics(
        self,
        pnl_series: List[float],
        wins: int,
        losses: int,
        total_trades: int,
        max_drawdown: float,
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        pnl_array = np.array(pnl_series) if pnl_series else np.array([0])
        total_pnl = np.sum(pnl_array)
        
        # Sharpe ratio
        if len(pnl_array) > 1 and np.std(pnl_array) > 0:
            sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino ratio
        downside = pnl_array[pnl_array < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino = np.mean(pnl_array) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe
        
        # Expected Shortfall (CVaR at 5%)
        if len(pnl_array) > 20:
            var_5 = np.percentile(pnl_array, 5)
            es = np.mean(pnl_array[pnl_array <= var_5])
        else:
            es = np.min(pnl_array) if len(pnl_array) > 0 else 0
        
        # ES Sharpe
        if es < 0:
            es_sharpe = total_pnl / abs(es) if es != 0 else 0
        else:
            es_sharpe = sharpe
        
        # Win rate
        win_rate = wins / max(1, wins + losses)
        
        return {
            'pnl': total_pnl,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'es': es,
            'es_sharpe': es_sharpe,
            'win_rate': win_rate,
            'total_trades': total_trades,
        }


def compute_objective(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute weighted objective from metrics."""
    objective = 0.0
    
    pnl_norm = metrics.get('pnl', 0) / 10000
    objective += weights.get('pnl_weight', 0) * pnl_norm
    objective += weights.get('sharpe_weight', 0) * metrics.get('sharpe', 0)
    objective += weights.get('drawdown_weight', 0) * metrics.get('max_drawdown', 0) * 10
    objective += weights.get('es_sharpe_weight', 0) * metrics.get('es_sharpe', 0)
    objective += weights.get('win_rate_weight', 0) * metrics.get('win_rate', 0) * 10
    
    return objective


def run_kalshi_optimization(
    strategy_name: str,
    data_file: str,
    objective_config: str,
    output_dir: Path,
    iterations: int = 80,
    population_size: int = 15,
) -> Dict:
    """Run CMA-ES optimization for a Kalshi strategy."""
    try:
        import cma
    except ImportError:
        return {'strategy': strategy_name, 'error': 'cma not installed'}
    
    if strategy_name not in KALSHI_STRATEGIES:
        return {'strategy': strategy_name, 'error': f'Unknown strategy: {strategy_name}'}
    
    param_space = KALSHI_STRATEGIES[strategy_name]
    objective_weights = OBJECTIVE_CONFIGS.get(objective_config, OBJECTIVE_CONFIGS['balanced'])
    
    # Initialize backtester
    backtester = KalshiBacktester(data_file=data_file)
    backtester.load_data()
    
    # Setup CMA-ES
    param_names = list(param_space.keys())
    bounds_low = [param_space[p][0] for p in param_names]
    bounds_high = [param_space[p][1] for p in param_names]
    
    x0 = [(bounds_low[i] + bounds_high[i]) / 2 for i in range(len(param_names))]
    
    options = {
        'popsize': population_size,
        'maxiter': iterations,
        'bounds': [bounds_low, bounds_high],
        'verb_disp': 0,
        'verb_log': 0,
    }
    
    es = cma.CMAEvolutionStrategy(x0, 0.3, options)
    
    best_objective = float('-inf')
    best_params = dict(zip(param_names, x0))
    best_metrics = {}
    generation = 0
    
    progress_file = output_dir / f"kalshi_{strategy_name}_{objective_config}_progress.jsonl"
    
    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        
        for x in solutions:
            params = dict(zip(param_names, x))
            metrics = backtester.run(strategy_name, params)
            obj = compute_objective(metrics, objective_weights)
            fitnesses.append(-obj)  # CMA minimizes
            
            if obj > best_objective:
                best_objective = obj
                best_params = params.copy()
                best_metrics = metrics.copy()
        
        es.tell(solutions, fitnesses)
        generation += 1
        
        # Log progress
        with open(progress_file, 'a') as f:
            f.write(json.dumps({
                'generation': generation,
                'timestamp': datetime.utcnow().isoformat(),
                'best_objective': best_objective,
                'best_params': best_params,
                'best_metrics': best_metrics,
            }) + '\n')
    
    result = {
        'strategy': strategy_name,
        'platform': 'kalshi',
        'objective_config': objective_config,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_objective': best_objective,
        'generations': generation,
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    # Save result
    result_file = output_dir / f"kalshi_{strategy_name}_{objective_config}_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def run_all_kalshi_optimizations(
    data_file: str,
    output_dir: Path,
    n_workers: int = 16,
):
    """Run optimizations for all Kalshi strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("KALSHI STRATEGY OPTIMIZER")
    logger.info("=" * 70)
    logger.info(f"Strategies: {len(KALSHI_STRATEGIES)}")
    logger.info(f"Objective configs: {len(OBJECTIVE_CONFIGS)}")
    logger.info(f"Workers: {n_workers}")
    
    # Generate all jobs
    jobs = []
    for strategy in KALSHI_STRATEGIES:
        for obj_config in OBJECTIVE_CONFIGS:
            jobs.append((strategy, obj_config))
    
    logger.info(f"Total jobs: {len(jobs)}")
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                run_kalshi_optimization,
                strategy,
                data_file,
                obj_config,
                output_dir,
            ): (strategy, obj_config)
            for strategy, obj_config in jobs
        }
        
        for future in as_completed(futures):
            strategy, obj_config = futures[future]
            try:
                result = future.result()
                results.append(result)
                if 'error' not in result:
                    logger.info(f"✓ {strategy}/{obj_config}: obj={result['best_objective']:.2f}, sharpe={result['best_metrics'].get('sharpe', 0):.2f}")
                else:
                    logger.error(f"✗ {strategy}/{obj_config}: {result['error']}")
            except Exception as e:
                logger.error(f"✗ {strategy}/{obj_config}: {e}")
    
    # Save summary
    summary = {
        'platform': 'kalshi',
        'total_jobs': len(jobs),
        'completed': len([r for r in results if 'error' not in r]),
        'best_by_strategy': {},
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    for result in results:
        if 'error' not in result:
            strategy = result['strategy']
            if strategy not in summary['best_by_strategy']:
                summary['best_by_strategy'][strategy] = result
            elif result['best_objective'] > summary['best_by_strategy'][strategy]['best_objective']:
                summary['best_by_strategy'][strategy] = result
    
    with open(output_dir / 'kalshi_optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nCompleted {len(results)} optimizations")
    return results


def main():
    parser = argparse.ArgumentParser(description="Kalshi Strategy Optimizer")
    parser.add_argument("--data-file", type=str, 
                        default="data/kalshi/kalshi_backtest.parquet",
                        help="Path to Kalshi backtest data")
    parser.add_argument("--output-dir", type=str, 
                        default="logs/kalshi_optimization",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Single strategy to optimize (optional)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.strategy:
        # Single strategy mode
        for obj_config in OBJECTIVE_CONFIGS:
            result = run_kalshi_optimization(
                args.strategy,
                args.data_file,
                obj_config,
                output_dir,
            )
            logger.info(f"Result: {json.dumps(result, indent=2)}")
    else:
        # All strategies
        run_all_kalshi_optimizations(
            args.data_file,
            output_dir,
            args.workers,
        )


if __name__ == "__main__":
    main()
