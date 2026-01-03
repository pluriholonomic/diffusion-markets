#!/usr/bin/env python3
"""
Multi-Objective Hyperparameter Explorer

Continuously explores strategy hyperparameters across:
1. Different objective function weightings (PnL, Sharpe, Drawdown, ES)
2. Different impact/market models (linear, sqrt, quadratic)
3. Adversarial vs Average case scenarios

Keeps all CPUs active with a job queue system.
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
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import queue
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Objective Function Configurations
# ============================================================================

OBJECTIVE_CONFIGS = {
    'balanced': {
        'pnl_weight': 0.30,
        'sharpe_weight': 0.35,
        'drawdown_weight': -0.15,
        'es_sharpe_weight': 0.10,
        'win_rate_weight': 0.10,
    },
    'pnl_focused': {
        'pnl_weight': 0.60,
        'sharpe_weight': 0.20,
        'drawdown_weight': -0.10,
        'es_sharpe_weight': 0.05,
        'win_rate_weight': 0.05,
    },
    'sharpe_focused': {
        'pnl_weight': 0.15,
        'sharpe_weight': 0.55,
        'drawdown_weight': -0.15,
        'es_sharpe_weight': 0.10,
        'win_rate_weight': 0.05,
    },
    'risk_averse': {
        'pnl_weight': 0.15,
        'sharpe_weight': 0.25,
        'drawdown_weight': -0.30,
        'es_sharpe_weight': 0.20,
        'win_rate_weight': 0.10,
    },
    'tail_risk': {
        'pnl_weight': 0.20,
        'sharpe_weight': 0.20,
        'drawdown_weight': -0.20,
        'es_sharpe_weight': 0.30,
        'win_rate_weight': 0.10,
    },
    'high_frequency': {
        'pnl_weight': 0.25,
        'sharpe_weight': 0.40,
        'drawdown_weight': -0.10,
        'es_sharpe_weight': 0.05,
        'win_rate_weight': 0.20,
    },
    'max_sharpe': {
        'pnl_weight': 0.05,
        'sharpe_weight': 0.80,
        'drawdown_weight': -0.10,
        'es_sharpe_weight': 0.05,
        'win_rate_weight': 0.00,
    },
    'max_pnl': {
        'pnl_weight': 0.85,
        'sharpe_weight': 0.05,
        'drawdown_weight': -0.05,
        'es_sharpe_weight': 0.05,
        'win_rate_weight': 0.00,
    },
}


# ============================================================================
# Impact Model Configurations
# ============================================================================

@dataclass
class ImpactModel:
    """Market impact model configuration."""
    name: str
    base_impact_pct: float  # Base impact as percentage
    impact_type: str  # 'linear', 'sqrt', 'quadratic', 'log'
    volume_sensitivity: float  # How much volume affects impact
    spread_multiplier: float  # Multiplier for bid-ask spread
    
    def compute_impact(self, order_size: float, volume: float, spread: float) -> float:
        """Compute market impact for an order."""
        # Base impact
        impact = self.base_impact_pct / 100
        
        # Size-dependent component
        if volume > 0:
            size_ratio = order_size / volume
        else:
            size_ratio = 0.1
        
        if self.impact_type == 'linear':
            size_impact = size_ratio * self.volume_sensitivity
        elif self.impact_type == 'sqrt':
            size_impact = np.sqrt(size_ratio) * self.volume_sensitivity
        elif self.impact_type == 'quadratic':
            size_impact = size_ratio ** 2 * self.volume_sensitivity
        elif self.impact_type == 'log':
            size_impact = np.log1p(size_ratio) * self.volume_sensitivity
        else:
            size_impact = size_ratio * self.volume_sensitivity
        
        # Spread component
        spread_impact = spread * self.spread_multiplier
        
        return impact + size_impact + spread_impact


IMPACT_MODELS = {
    'optimistic': ImpactModel(
        name='optimistic',
        base_impact_pct=0.5,
        impact_type='sqrt',
        volume_sensitivity=0.01,
        spread_multiplier=0.5,
    ),
    'realistic': ImpactModel(
        name='realistic',
        base_impact_pct=2.0,
        impact_type='sqrt',
        volume_sensitivity=0.05,
        spread_multiplier=1.0,
    ),
    'conservative': ImpactModel(
        name='conservative',
        base_impact_pct=3.0,
        impact_type='linear',
        volume_sensitivity=0.10,
        spread_multiplier=1.5,
    ),
    'pessimistic': ImpactModel(
        name='pessimistic',
        base_impact_pct=5.0,
        impact_type='quadratic',
        volume_sensitivity=0.15,
        spread_multiplier=2.0,
    ),
    'adversarial': ImpactModel(
        name='adversarial',
        base_impact_pct=7.0,
        impact_type='quadratic',
        volume_sensitivity=0.25,
        spread_multiplier=3.0,
    ),
}


# ============================================================================
# Scenario Configurations
# ============================================================================

@dataclass
class ScenarioConfig:
    """Scenario for backtesting."""
    name: str
    description: str
    # Market conditions
    volatility_multiplier: float = 1.0
    trend_strength: float = 0.0  # -1 to 1
    correlation_boost: float = 0.0  # Additional correlation
    # Adversarial
    is_adversarial: bool = False
    adversary_strength: float = 0.0  # 0 to 1
    # Sampling
    bootstrap_samples: int = 1  # Number of bootstrap samples


SCENARIOS = {
    'average_case': ScenarioConfig(
        name='average_case',
        description='Normal market conditions',
        volatility_multiplier=1.0,
        trend_strength=0.0,
        is_adversarial=False,
    ),
    'high_volatility': ScenarioConfig(
        name='high_volatility',
        description='2x normal volatility',
        volatility_multiplier=2.0,
        trend_strength=0.0,
        is_adversarial=False,
    ),
    'trending_up': ScenarioConfig(
        name='trending_up',
        description='Bull market conditions',
        volatility_multiplier=1.2,
        trend_strength=0.3,
        is_adversarial=False,
    ),
    'trending_down': ScenarioConfig(
        name='trending_down',
        description='Bear market conditions',
        volatility_multiplier=1.5,
        trend_strength=-0.3,
        is_adversarial=False,
    ),
    'high_correlation': ScenarioConfig(
        name='high_correlation',
        description='Markets move together',
        volatility_multiplier=1.0,
        correlation_boost=0.3,
        is_adversarial=False,
    ),
    'adversarial_weak': ScenarioConfig(
        name='adversarial_weak',
        description='Mild adversarial conditions',
        is_adversarial=True,
        adversary_strength=0.3,
    ),
    'adversarial_medium': ScenarioConfig(
        name='adversarial_medium',
        description='Medium adversarial conditions',
        is_adversarial=True,
        adversary_strength=0.5,
    ),
    'adversarial_strong': ScenarioConfig(
        name='adversarial_strong',
        description='Strong adversarial conditions',
        is_adversarial=True,
        adversary_strength=0.7,
    ),
    'worst_case': ScenarioConfig(
        name='worst_case',
        description='Combination of adverse conditions',
        volatility_multiplier=2.0,
        trend_strength=-0.2,
        correlation_boost=0.2,
        is_adversarial=True,
        adversary_strength=0.5,
    ),
}


# ============================================================================
# Strategy Configurations
# ============================================================================

STRATEGIES = [
    'calibration',
    'stat_arb',
    'momentum',
    'dispersion',
    'correlation',
    'blackwell',
    'confidence_gated',
    'trend_following',
    'mean_reversion',
    'regime_adaptive',
    'longshot',
]

STRATEGY_PARAM_SPACES = {
    'calibration': {
        'kelly_fraction': (0.1, 0.6),
        'spread_threshold': (0.02, 0.20),
        'min_edge': (0.02, 0.15),
        'max_position_pct': (0.02, 0.10),
        'profit_take_pct': (20.0, 80.0),
        'stop_loss_pct': (10.0, 50.0),
    },
    'stat_arb': {
        'kelly_fraction': (0.1, 0.5),
        'spread_threshold': (0.02, 0.15),
        'min_edge': (0.02, 0.15),
        'max_position_pct': (0.02, 0.10),
    },
    'momentum': {
        'fast_window': (3, 15),
        'slow_window': (10, 50),
        'momentum_threshold': (0.01, 0.10),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.10),
    },
    'dispersion': {
        'dispersion_threshold': (0.05, 0.30),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.08),
    },
    'correlation': {
        'correlation_threshold': (0.5, 0.9),
        'divergence_threshold': (0.03, 0.15),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.08),
    },
    'blackwell': {
        'g_bar_threshold': (0.02, 0.15),
        't_stat_threshold': (1.5, 3.0),
        'kelly_fraction': (0.1, 0.5),
        'max_position_pct': (0.02, 0.10),
    },
    'confidence_gated': {
        'min_distance_threshold': (0.02, 0.15),
        'max_confidence_threshold': (0.85, 0.98),
        'kelly_fraction': (0.1, 0.5),
        'max_position_pct': (0.02, 0.10),
    },
    'trend_following': {
        'fast_ema_periods': (3, 10),
        'slow_ema_periods': (15, 40),
        'trend_threshold': (0.01, 0.08),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.08),
    },
    'mean_reversion': {
        'lookback_periods': (10, 40),
        'zscore_threshold': (1.0, 3.0),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.08),
    },
    'regime_adaptive': {
        'volatility_lookback': (10, 40),
        'high_volatility_threshold': (0.02, 0.10),
        'trend_weight_trending': (0.5, 1.0),
        'mean_rev_weight_mean_rev': (0.5, 1.0),
        'kelly_fraction': (0.1, 0.4),
        'max_position_pct': (0.02, 0.08),
    },
    'longshot': {
        'max_price': (0.10, 0.25),
        'min_edge': (0.05, 0.20),
        'kelly_fraction': (0.05, 0.20),
        'max_position_pct': (0.01, 0.05),
    },
}


# ============================================================================
# Job Definition
# ============================================================================

@dataclass
class OptimizationJob:
    """A single optimization job configuration."""
    job_id: str
    strategy: str
    objective_config_name: str
    objective_weights: Dict[str, float]
    impact_model_name: str
    impact_model: ImpactModel
    scenario_name: str
    scenario: ScenarioConfig
    param_space: Dict[str, Tuple[float, float]]
    iterations: int = 100
    population_size: int = 20
    
    def to_dict(self) -> Dict:
        return {
            'job_id': self.job_id,
            'strategy': self.strategy,
            'objective_config_name': self.objective_config_name,
            'objective_weights': self.objective_weights,
            'impact_model_name': self.impact_model_name,
            'scenario_name': self.scenario_name,
            'param_space': self.param_space,
            'iterations': self.iterations,
        }


# ============================================================================
# Backtester with Impact and Scenarios
# ============================================================================

class AdvancedBacktester:
    """Backtester with configurable impact and scenario models."""
    
    def __init__(
        self,
        data_file: str,
        impact_model: ImpactModel,
        scenario: ScenarioConfig,
        initial_bankroll: float = 10000,
    ):
        self.data_file = data_file
        self.impact_model = impact_model
        self.scenario = scenario
        self.initial_bankroll = initial_bankroll
        self.data = None
        
    def load_data(self):
        """Load and transform data based on scenario."""
        import pandas as pd
        if self.data is None:
            try:
                df = pd.read_parquet(self.data_file)
                # Normalize column names to expected format
                self.data = pd.DataFrame({
                    'price': df['last_price'].fillna(df['avg_price']).clip(0.01, 0.99),
                    'outcome': df['y'].fillna(0).astype(int),
                    'volume': df['volumeNum'].fillna(10000),
                    'spread': 0.02,  # Default spread
                    'category': df['category'].fillna('unknown'),
                })
            except Exception as e:
                # Generate synthetic data as fallback
                logger.warning(f"Failed to load data: {e}, using synthetic data")
                np.random.seed(42)
                n = 2000
                self.data = pd.DataFrame({
                    'price': np.random.beta(2, 2, n),
                    'outcome': np.random.binomial(1, np.random.beta(2, 2, n)),
                    'volume': np.random.exponential(50000, n),
                    'spread': np.random.uniform(0.01, 0.05, n),
                    'category': np.random.choice(['politics', 'crypto', 'sports'], n),
                })
            
            # Apply scenario transformations
            if self.scenario.volatility_multiplier != 1.0:
                # Add noise to prices
                noise = np.random.normal(0, 0.05 * (self.scenario.volatility_multiplier - 1), len(self.data))
                self.data['price'] = np.clip(self.data['price'] + noise, 0.01, 0.99)
            
            if self.scenario.trend_strength != 0:
                # Add trend bias
                trend = np.linspace(0, self.scenario.trend_strength * 0.1, len(self.data))
                self.data['price'] = np.clip(self.data['price'] + trend, 0.01, 0.99)
            
            if self.scenario.is_adversarial:
                # Adversary picks worst outcomes for our positions
                self._apply_adversarial()
        
        return self.data
    
    def _apply_adversarial(self):
        """Apply adversarial transformations."""
        strength = self.scenario.adversary_strength
        
        # Adversary makes outcomes worse for extreme prices
        prices = self.data['price'].values
        outcomes = self.data['outcome'].values.copy()
        
        for idx in range(len(self.data)):
            price = prices[idx]
            
            # If we'd bet YES (low price), adversary reduces outcome probability
            if price < 0.3:
                if np.random.random() < strength:
                    outcomes[idx] = 0
            # If we'd bet NO (high price), adversary increases outcome probability
            elif price > 0.7:
                if np.random.random() < strength:
                    outcomes[idx] = 1
        
        self.data['outcome'] = outcomes
    
    def run(self, strategy_name: str, params: Dict[str, float]) -> Dict[str, float]:
        """Run backtest with given parameters."""
        data = self.load_data()
        
        bankroll = self.initial_bankroll
        pnl_series = []
        wins, losses = 0, 0
        total_trades = 0
        peak_bankroll = bankroll
        max_drawdown = 0
        
        kelly_fraction = params.get('kelly_fraction', 0.25)
        max_position = params.get('max_position_pct', 0.05)
        threshold = params.get('spread_threshold', params.get('min_edge', 0.05))
        
        # Convert to numpy arrays for faster iteration
        prices = data['price'].values
        outcomes = data['outcome'].values
        volumes = data['volume'].values if 'volume' in data.columns else np.full(len(data), 50000.0)
        spreads = data['spread'].values if 'spread' in data.columns else np.full(len(data), 0.02)
        
        for idx in range(len(data)):
            price = float(prices[idx])
            outcome = int(outcomes[idx])
            volume = float(volumes[idx])
            spread = float(spreads[idx])
            
            # Compute edge based on strategy
            edge = self._compute_edge(strategy_name, params, price, outcome)
            
            if edge < threshold:
                continue
            
            # Compute position size
            kelly = min(edge * kelly_fraction, 0.5)
            size = self.initial_bankroll * kelly * max_position
            
            # Compute market impact
            impact = self.impact_model.compute_impact(size, volume, spread)
            
            # Adjust entry price for impact
            effective_price = price * (1 + impact)
            
            # Simulate trade outcome
            if np.random.random() < 0.5 + edge - impact:
                # Win
                pnl = size * (1 / effective_price - 1) if effective_price > 0 else 0
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
    
    def _compute_edge(self, strategy_name: str, params: Dict, price: float, outcome: int) -> float:
        """Compute expected edge for a trade."""
        if strategy_name in ['calibration', 'stat_arb', 'blackwell']:
            edge = abs(float(outcome) - price) if np.random.random() < 0.3 else 0
        elif strategy_name in ['momentum', 'trend_following']:
            edge = abs(price - 0.5) * 0.3 if np.random.random() < 0.2 else 0
        elif strategy_name in ['mean_reversion', 'confidence_gated']:
            edge = max(0, abs(price - 0.5) - 0.3) if np.random.random() < 0.25 else 0
        elif strategy_name == 'longshot':
            edge = 0.2 if price < params.get('max_price', 0.15) and np.random.random() < 0.1 else 0
        else:
            edge = np.random.random() * 0.1
        
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
        
        # Sortino ratio (downside deviation)
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
    
    # PnL component (normalized)
    pnl_norm = metrics.get('pnl', 0) / 10000  # Normalize to ~1
    objective += weights.get('pnl_weight', 0) * pnl_norm
    
    # Sharpe component
    objective += weights.get('sharpe_weight', 0) * metrics.get('sharpe', 0)
    
    # Drawdown component (negative weight = penalize)
    objective += weights.get('drawdown_weight', 0) * metrics.get('max_drawdown', 0) * 10
    
    # ES Sharpe component
    objective += weights.get('es_sharpe_weight', 0) * metrics.get('es_sharpe', 0)
    
    # Win rate component
    objective += weights.get('win_rate_weight', 0) * metrics.get('win_rate', 0) * 10
    
    return objective


# ============================================================================
# Job Executor
# ============================================================================

def run_single_job(job: OptimizationJob, data_file: str, output_dir: Path) -> Dict:
    """Run a single optimization job."""
    try:
        import cma
    except ImportError:
        return {'job_id': job.job_id, 'error': 'cma not installed'}
    
    job_output_dir = output_dir / job.job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = job_output_dir / "progress.jsonl"
    result_file = job_output_dir / "result.json"
    
    # Initialize backtester
    backtester = AdvancedBacktester(
        data_file=data_file,
        impact_model=job.impact_model,
        scenario=job.scenario,
    )
    backtester.load_data()
    
    # Setup CMA-ES
    param_names = list(job.param_space.keys())
    bounds_low = [job.param_space[p][0] for p in param_names]
    bounds_high = [job.param_space[p][1] for p in param_names]
    
    x0 = [(bounds_low[i] + bounds_high[i]) / 2 for i in range(len(param_names))]
    
    options = {
        'popsize': job.population_size,
        'maxiter': job.iterations,
        'bounds': [bounds_low, bounds_high],
        'verb_disp': 0,
        'verb_log': 0,
    }
    
    es = cma.CMAEvolutionStrategy(x0, 0.3, options)
    
    best_objective = float('-inf')
    best_params = dict(zip(param_names, x0))
    best_metrics = {}
    generation = 0
    
    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        
        for x in solutions:
            params = dict(zip(param_names, x))
            metrics = backtester.run(job.strategy, params)
            obj = compute_objective(metrics, job.objective_weights)
            fitnesses.append(-obj)  # CMA minimizes
            
            if obj > best_objective:
                best_objective = obj
                best_params = params.copy()
                best_metrics = metrics.copy()
        
        es.tell(solutions, fitnesses)
        generation += 1
        
        # Log progress
        progress = {
            'generation': generation,
            'timestamp': datetime.utcnow().isoformat(),
            'best_objective': best_objective,
            'best_params': best_params,
            'best_metrics': best_metrics,
        }
        
        with open(progress_file, 'a') as f:
            f.write(json.dumps(progress) + '\n')
    
    # Save result
    result = {
        'job_id': job.job_id,
        'strategy': job.strategy,
        'objective_config': job.objective_config_name,
        'impact_model': job.impact_model_name,
        'scenario': job.scenario_name,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_objective': best_objective,
        'generations': generation,
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


# ============================================================================
# Job Generator
# ============================================================================

class JobGenerator:
    """Generates optimization jobs for exploration."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.completed_jobs = set()
        self._load_completed()
    
    def _load_completed(self):
        """Load list of completed job IDs."""
        completed_file = self.output_dir / "completed_jobs.txt"
        if completed_file.exists():
            with open(completed_file) as f:
                self.completed_jobs = set(line.strip() for line in f)
    
    def _save_completed(self, job_id: str):
        """Save completed job ID."""
        completed_file = self.output_dir / "completed_jobs.txt"
        with open(completed_file, 'a') as f:
            f.write(job_id + '\n')
        self.completed_jobs.add(job_id)
    
    def _make_job_id(self, strategy: str, obj_name: str, impact_name: str, scenario_name: str) -> str:
        """Create unique job ID."""
        key = f"{strategy}_{obj_name}_{impact_name}_{scenario_name}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def generate_jobs(self, batch_size: int = 20) -> List[OptimizationJob]:
        """Generate a batch of jobs to run."""
        jobs = []
        
        # Priority order for exploration
        priority_objectives = ['balanced', 'sharpe_focused', 'pnl_focused', 'risk_averse', 'tail_risk']
        priority_impacts = ['realistic', 'conservative', 'adversarial']
        priority_scenarios = ['average_case', 'adversarial_medium', 'worst_case', 'high_volatility']
        
        for strategy in STRATEGIES:
            for obj_name in priority_objectives:
                for impact_name in priority_impacts:
                    for scenario_name in priority_scenarios:
                        job_id = self._make_job_id(strategy, obj_name, impact_name, scenario_name)
                        
                        if job_id in self.completed_jobs:
                            continue
                        
                        job = OptimizationJob(
                            job_id=job_id,
                            strategy=strategy,
                            objective_config_name=obj_name,
                            objective_weights=OBJECTIVE_CONFIGS[obj_name],
                            impact_model_name=impact_name,
                            impact_model=IMPACT_MODELS[impact_name],
                            scenario_name=scenario_name,
                            scenario=SCENARIOS[scenario_name],
                            param_space=STRATEGY_PARAM_SPACES.get(strategy, {}),
                            iterations=80,
                            population_size=15,
                        )
                        jobs.append(job)
                        
                        if len(jobs) >= batch_size:
                            return jobs
        
        # If we've covered priorities, expand to all combinations
        for strategy in STRATEGIES:
            for obj_name in OBJECTIVE_CONFIGS:
                for impact_name in IMPACT_MODELS:
                    for scenario_name in SCENARIOS:
                        job_id = self._make_job_id(strategy, obj_name, impact_name, scenario_name)
                        
                        if job_id in self.completed_jobs:
                            continue
                        
                        job = OptimizationJob(
                            job_id=job_id,
                            strategy=strategy,
                            objective_config_name=obj_name,
                            objective_weights=OBJECTIVE_CONFIGS[obj_name],
                            impact_model_name=impact_name,
                            impact_model=IMPACT_MODELS[impact_name],
                            scenario_name=scenario_name,
                            scenario=SCENARIOS[scenario_name],
                            param_space=STRATEGY_PARAM_SPACES.get(strategy, {}),
                            iterations=60,
                            population_size=12,
                        )
                        jobs.append(job)
                        
                        if len(jobs) >= batch_size:
                            return jobs
        
        return jobs


# ============================================================================
# Main Runner
# ============================================================================

class MultiObjectiveExplorer:
    """Main exploration runner."""
    
    def __init__(
        self,
        data_file: str,
        output_dir: Path,
        n_workers: int = 16,
        jobs_per_batch: int = 20,
    ):
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        self.jobs_per_batch = jobs_per_batch
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.job_generator = JobGenerator(self.output_dir)
        
    def run(self):
        """Run continuous exploration."""
        logger.info("=" * 70)
        logger.info("MULTI-OBJECTIVE HYPERPARAMETER EXPLORER")
        logger.info("=" * 70)
        logger.info(f"Workers: {self.n_workers}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Strategies: {len(STRATEGIES)}")
        logger.info(f"Objective configs: {len(OBJECTIVE_CONFIGS)}")
        logger.info(f"Impact models: {len(IMPACT_MODELS)}")
        logger.info(f"Scenarios: {len(SCENARIOS)}")
        total_combinations = len(STRATEGIES) * len(OBJECTIVE_CONFIGS) * len(IMPACT_MODELS) * len(SCENARIOS)
        logger.info(f"Total combinations to explore: {total_combinations}")
        
        batch_num = 0
        total_jobs = 0
        
        while True:
            # Generate next batch of jobs
            jobs = self.job_generator.generate_jobs(self.jobs_per_batch)
            
            if not jobs:
                logger.info("All combinations explored! Restarting with variations...")
                self.job_generator.completed_jobs.clear()
                time.sleep(60)
                continue
            
            batch_num += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"BATCH {batch_num}: {len(jobs)} jobs")
            logger.info(f"{'='*50}")
            
            # Log job details
            for job in jobs[:5]:
                logger.info(f"  {job.strategy} | {job.objective_config_name} | {job.impact_model_name} | {job.scenario_name}")
            if len(jobs) > 5:
                logger.info(f"  ... and {len(jobs) - 5} more")
            
            # Run jobs in parallel
            results = []
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(run_single_job, job, self.data_file, self.output_dir): job
                    for job in jobs
                }
                
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.job_generator._save_completed(job.job_id)
                        
                        if 'error' not in result:
                            logger.info(f"✓ {job.strategy}/{job.objective_config_name}/{job.impact_model_name}: obj={result['best_objective']:.2f}, sharpe={result['best_metrics'].get('sharpe', 0):.2f}")
                    except Exception as e:
                        import traceback
                        logger.error(f"✗ {job.job_id} ({job.strategy}): {type(e).__name__}: {e}")
                        logger.debug(traceback.format_exc())
            
            total_jobs += len(jobs)
            
            # Update summary
            self._update_summary(results)
            
            logger.info(f"Batch {batch_num} complete. Total jobs: {total_jobs}")
    
    def _update_summary(self, results: List[Dict]):
        """Update exploration summary."""
        summary_file = self.output_dir / "exploration_summary.json"
        
        # Load existing summary
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
        else:
            summary = {
                'best_by_strategy': {},
                'best_by_objective': {},
                'best_overall': None,
                'total_jobs': 0,
            }
        
        # Update with new results
        for result in results:
            if 'error' in result:
                continue
            
            strategy = result['strategy']
            obj_config = result['objective_config']
            objective = result['best_objective']
            
            # Track best by strategy
            if strategy not in summary['best_by_strategy']:
                summary['best_by_strategy'][strategy] = result
            elif objective > summary['best_by_strategy'][strategy].get('best_objective', 0):
                summary['best_by_strategy'][strategy] = result
            
            # Track best by objective config
            if obj_config not in summary['best_by_objective']:
                summary['best_by_objective'][obj_config] = result
            elif objective > summary['best_by_objective'][obj_config].get('best_objective', 0):
                summary['best_by_objective'][obj_config] = result
            
            # Track best overall
            if summary['best_overall'] is None or objective > summary['best_overall'].get('best_objective', 0):
                summary['best_overall'] = result
        
        summary['total_jobs'] += len(results)
        summary['last_updated'] = datetime.utcnow().isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Multi-objective hyperparameter exploration")
    parser.add_argument("--data-file", type=str, 
                        default="data/polymarket/resolved_markets.parquet",
                        help="Path to backtest data")
    parser.add_argument("--output-dir", type=str, 
                        default="logs/multi_objective_exploration",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Jobs per batch")
    
    args = parser.parse_args()
    
    explorer = MultiObjectiveExplorer(
        data_file=args.data_file,
        output_dir=Path(args.output_dir),
        n_workers=args.workers,
        jobs_per_batch=args.batch_size,
    )
    
    explorer.run()


if __name__ == "__main__":
    main()
