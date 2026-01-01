#!/usr/bin/env python3
"""
Comprehensive Stress Testing Framework for Market Impact Models

Two regimes:
1. ADVERSARIAL / Worst-Case: Models that try to destroy PnL/Sharpe
2. AVERAGE-CASE: Mean-field models (Almgren-Chriss style)

Includes:
- Fill probability models
- Price impact (temporary & permanent)
- MEV / Adverse selection from blockchain reordering
- Slippage models
- Liquidity estimation

All models are fit on historical data where possible.
"""

import argparse
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import warnings
from datetime import datetime
import hashlib
import time

warnings.filterwarnings('ignore')

# Try to import optuna
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Please install optuna: pip install optuna")
    exit(1)


# =============================================================================
# IMPACT MODEL DEFINITIONS
# =============================================================================

@dataclass
class ImpactModelParams:
    """Base parameters for all impact models"""
    name: str = "base"
    regime: str = "average"  # "average" or "adversarial"
    
    # Fill probability
    base_fill_prob: float = 0.80
    fill_size_decay: float = 0.10
    fill_spread_sensitivity: float = 2.0
    
    # Price impact (Almgren-Chriss style)
    permanent_impact: float = 0.001
    temporary_impact: float = 0.005
    impact_power: float = 0.5
    
    # Adverse selection
    adverse_selection: float = 0.10
    adverse_selection_decay: float = 0.5
    
    # MEV / Blockchain specific
    mev_extraction_rate: float = 0.0  # Fraction of edge extracted by MEV
    reorder_probability: float = 0.0  # Probability of unfavorable reordering
    block_delay_cost: float = 0.0  # Cost per block of delay
    
    # Liquidity
    base_liquidity: float = 10000.0
    liquidity_price_factor: float = 2.0
    
    # Exit costs
    exit_spread: float = 0.02
    exit_impact_mult: float = 1.5
    
    # Stress multipliers (for adversarial regime)
    stress_multiplier: float = 1.0


# -----------------------------------------------------------------------------
# AVERAGE-CASE MODELS (Mean-Field / Almgren-Chriss Style)
# -----------------------------------------------------------------------------

def create_almgren_chriss_base() -> ImpactModelParams:
    """Standard Almgren-Chriss model - baseline"""
    return ImpactModelParams(
        name="almgren_chriss_base",
        regime="average",
        base_fill_prob=0.85,
        fill_size_decay=0.08,
        permanent_impact=0.0008,
        temporary_impact=0.004,
        impact_power=0.5,
        adverse_selection=0.05,
        mev_extraction_rate=0.0,
    )


def create_almgren_chriss_conservative() -> ImpactModelParams:
    """Conservative Almgren-Chriss with higher costs"""
    return ImpactModelParams(
        name="almgren_chriss_conservative",
        regime="average",
        base_fill_prob=0.75,
        fill_size_decay=0.12,
        permanent_impact=0.0015,
        temporary_impact=0.008,
        impact_power=0.6,
        adverse_selection=0.08,
        exit_spread=0.03,
    )


def create_low_liquidity_average() -> ImpactModelParams:
    """Average case in low liquidity environment"""
    return ImpactModelParams(
        name="low_liquidity_average",
        regime="average",
        base_fill_prob=0.65,
        fill_size_decay=0.15,
        permanent_impact=0.002,
        temporary_impact=0.012,
        impact_power=0.55,
        base_liquidity=3000.0,
        adverse_selection=0.10,
    )


def create_crypto_average() -> ImpactModelParams:
    """Average case for crypto prediction markets (with mild MEV)"""
    return ImpactModelParams(
        name="crypto_average",
        regime="average",
        base_fill_prob=0.80,
        fill_size_decay=0.10,
        permanent_impact=0.001,
        temporary_impact=0.006,
        mev_extraction_rate=0.02,  # 2% MEV on average
        reorder_probability=0.05,
        block_delay_cost=0.001,
        adverse_selection=0.08,
    )


def create_high_frequency_average() -> ImpactModelParams:
    """Average case for frequent trading"""
    return ImpactModelParams(
        name="high_frequency_average",
        regime="average",
        base_fill_prob=0.90,
        fill_size_decay=0.05,
        permanent_impact=0.0005,
        temporary_impact=0.002,
        impact_power=0.4,
        adverse_selection=0.03,
        exit_impact_mult=1.2,
    )


# -----------------------------------------------------------------------------
# ADVERSARIAL / WORST-CASE MODELS
# -----------------------------------------------------------------------------

def create_adversarial_fills() -> ImpactModelParams:
    """Adversary controls fill probability - you only get filled when it's bad"""
    return ImpactModelParams(
        name="adversarial_fills",
        regime="adversarial",
        base_fill_prob=0.40,  # Very low fills
        fill_size_decay=0.25,  # Larger orders almost never fill
        fill_spread_sensitivity=5.0,  # Spread kills fills
        permanent_impact=0.002,
        temporary_impact=0.015,
        adverse_selection=0.25,  # High adverse selection - you get filled when wrong
        stress_multiplier=2.0,
    )


def create_adversarial_impact() -> ImpactModelParams:
    """Adversary maximizes price impact"""
    return ImpactModelParams(
        name="adversarial_impact",
        regime="adversarial",
        base_fill_prob=0.70,
        permanent_impact=0.005,  # 5x normal
        temporary_impact=0.025,  # 5x normal
        impact_power=0.8,  # More concave = worse for large orders
        adverse_selection=0.15,
        exit_impact_mult=2.5,  # Exits are very expensive
        stress_multiplier=1.5,
    )


def create_adversarial_mev() -> ImpactModelParams:
    """Adversary extracts maximum MEV - blockchain adversary"""
    return ImpactModelParams(
        name="adversarial_mev",
        regime="adversarial",
        base_fill_prob=0.60,
        mev_extraction_rate=0.15,  # 15% of edge extracted
        reorder_probability=0.30,  # 30% chance of bad reorder
        block_delay_cost=0.005,  # 0.5% per block delay
        permanent_impact=0.002,
        temporary_impact=0.010,
        adverse_selection=0.20,
        stress_multiplier=1.8,
    )


def create_adversarial_liquidity() -> ImpactModelParams:
    """Adversary creates liquidity crises"""
    return ImpactModelParams(
        name="adversarial_liquidity",
        regime="adversarial",
        base_fill_prob=0.50,
        fill_size_decay=0.30,
        base_liquidity=1000.0,  # Very low liquidity
        liquidity_price_factor=0.5,  # Liquidity doesn't improve near 0.5
        permanent_impact=0.004,
        temporary_impact=0.020,
        exit_spread=0.05,
        exit_impact_mult=3.0,
        stress_multiplier=2.5,
    )


def create_adversarial_selection() -> ImpactModelParams:
    """Maximum adverse selection - you only trade when wrong"""
    return ImpactModelParams(
        name="adversarial_selection",
        regime="adversarial",
        base_fill_prob=0.75,
        adverse_selection=0.35,  # 35% adverse selection
        adverse_selection_decay=0.2,  # Slow decay
        permanent_impact=0.002,
        temporary_impact=0.010,
        stress_multiplier=2.0,
    )


def create_adversarial_exit() -> ImpactModelParams:
    """Adversary makes exits maximally expensive"""
    return ImpactModelParams(
        name="adversarial_exit",
        regime="adversarial",
        base_fill_prob=0.80,
        permanent_impact=0.001,
        temporary_impact=0.005,
        exit_spread=0.08,  # 8% exit spread
        exit_impact_mult=4.0,  # 4x impact on exit
        adverse_selection=0.15,
        stress_multiplier=1.5,
    )


def create_adversarial_combined() -> ImpactModelParams:
    """Combined worst-case: everything is bad"""
    return ImpactModelParams(
        name="adversarial_combined",
        regime="adversarial",
        base_fill_prob=0.35,
        fill_size_decay=0.25,
        permanent_impact=0.004,
        temporary_impact=0.020,
        impact_power=0.75,
        mev_extraction_rate=0.10,
        reorder_probability=0.25,
        block_delay_cost=0.003,
        adverse_selection=0.30,
        base_liquidity=2000.0,
        exit_spread=0.06,
        exit_impact_mult=3.0,
        stress_multiplier=3.0,
    )


# -----------------------------------------------------------------------------
# FITTED MODELS (from historical data)
# -----------------------------------------------------------------------------

def fit_impact_model_from_data(df: pd.DataFrame) -> ImpactModelParams:
    """
    Fit impact model parameters from historical trade data.
    Uses empirical observations to estimate realistic parameters.
    """
    # Estimate base liquidity from trade volumes
    vol_col = 'volumeNum' if 'volumeNum' in df.columns else 'volume'
    if vol_col in df.columns:
        base_liquidity = df[vol_col].median()
    else:
        base_liquidity = 10000.0
    
    # Estimate impact from price movements around trades
    price_col = 'avg_price' if 'avg_price' in df.columns else 'price'
    final_col = 'final_prob' if 'final_prob' in df.columns else 'final_price'
    
    if price_col in df.columns and final_col in df.columns:
        price_changes = np.abs(df[final_col] - df[price_col])
        permanent_impact = price_changes.mean() * 0.5
        temporary_impact = price_changes.mean() * 1.5
    else:
        permanent_impact = 0.001
        temporary_impact = 0.005
    
    # Estimate adverse selection from outcome correlation
    outcome_col = 'y' if 'y' in df.columns else 'outcome'
    if price_col in df.columns and outcome_col in df.columns:
        # Markets where we bought high and lost
        df_copy = df.copy()
        df_copy['predicted_up'] = df_copy[price_col] > 0.5
        df_copy['was_right'] = df_copy['predicted_up'] == df_copy[outcome_col]
        adverse_selection = 1.0 - df_copy['was_right'].mean()
    else:
        adverse_selection = 0.10
    
    return ImpactModelParams(
        name="fitted_from_data",
        regime="average",
        base_fill_prob=0.80,
        permanent_impact=min(permanent_impact, 0.01),
        temporary_impact=min(temporary_impact, 0.05),
        adverse_selection=min(adverse_selection, 0.30),
        base_liquidity=base_liquidity,
    )


# =============================================================================
# IMPACT COST COMPUTATION
# =============================================================================

def compute_fill_probability(
    size: float,
    price: float,
    liquidity: float,
    params: ImpactModelParams,
    rng: np.random.Generator,
) -> float:
    """Compute probability of fill given order parameters"""
    # Base probability
    p = params.base_fill_prob
    
    # Size effect (larger orders less likely to fill)
    size_ratio = size / max(liquidity, 1.0)
    p *= np.exp(-params.fill_size_decay * size_ratio * 10)
    
    # Spread effect (orders near 0/1 less likely to fill due to thin liquidity)
    spread_factor = 4 * price * (1 - price)  # Peaks at 0.5
    p *= (1 - params.fill_spread_sensitivity * 0.1 * (1 - spread_factor))
    
    # Apply stress multiplier for adversarial
    if params.regime == "adversarial":
        p *= (1 / params.stress_multiplier)
    
    return np.clip(p, 0.01, 0.99)


def compute_price_impact(
    size: float,
    liquidity: float,
    params: ImpactModelParams,
) -> tuple[float, float]:
    """
    Compute temporary and permanent price impact.
    Returns (temporary_impact, permanent_impact)
    """
    # Normalized size
    sigma = size / max(liquidity, 1.0)
    
    # Almgren-Chriss style impact
    temp = params.temporary_impact * np.power(sigma, params.impact_power)
    perm = params.permanent_impact * sigma
    
    # Apply stress multiplier
    if params.regime == "adversarial":
        temp *= params.stress_multiplier
        perm *= params.stress_multiplier
    
    return temp, perm


def compute_mev_cost(
    edge: float,
    params: ImpactModelParams,
    rng: np.random.Generator,
) -> float:
    """Compute MEV extraction cost (blockchain-specific)"""
    mev_cost = 0.0
    
    # Direct MEV extraction (searchers front-running)
    mev_cost += edge * params.mev_extraction_rate
    
    # Reordering cost (unfavorable block position)
    if rng.random() < params.reorder_probability:
        mev_cost += edge * 0.5  # Lose half edge to reordering
    
    # Block delay cost
    blocks_delayed = rng.poisson(1)  # Average 1 block delay
    mev_cost += params.block_delay_cost * blocks_delayed
    
    return mev_cost


def compute_adverse_selection_cost(
    edge: float,
    hold_time: float,
    filled: bool,
    params: ImpactModelParams,
) -> float:
    """
    Compute adverse selection cost.
    The idea: if you got filled, it might be because informed traders
    knew the price was moving against you.
    """
    if not filled:
        return 0.0
    
    # Base adverse selection
    as_cost = edge * params.adverse_selection
    
    # Decay with hold time (longer holds dilute adverse selection)
    decay = np.exp(-params.adverse_selection_decay * hold_time)
    as_cost *= decay
    
    # Adversarial regime amplifies
    if params.regime == "adversarial":
        as_cost *= params.stress_multiplier
    
    return as_cost


def compute_exit_cost(
    size: float,
    price: float,
    liquidity: float,
    params: ImpactModelParams,
) -> float:
    """Compute cost of exiting position"""
    # Spread cost
    exit_cost = params.exit_spread
    
    # Impact cost on exit
    _, perm = compute_price_impact(size, liquidity, params)
    exit_cost += perm * params.exit_impact_mult
    
    return exit_cost


def estimate_liquidity(
    price: float,
    category: str,
    params: ImpactModelParams,
) -> float:
    """Estimate market liquidity"""
    # Base liquidity
    liq = params.base_liquidity
    
    # Price factor (more liquidity near 0.5)
    price_mult = 1.0 + params.liquidity_price_factor * (4 * price * (1 - price) - 1)
    liq *= max(price_mult, 0.1)
    
    # Category adjustments
    category_mults = {
        'politics': 2.0,
        'crypto': 1.5,
        'sports': 1.2,
        'entertainment': 0.8,
        'science': 0.6,
    }
    liq *= category_mults.get(category.lower() if category else '', 1.0)
    
    return liq


# =============================================================================
# STRATEGY EVALUATION WITH IMPACT MODEL
# =============================================================================

def evaluate_strategy_with_impact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    impact_params: ImpactModelParams,
    strategy_params: Dict[str, Any],
    n_simulations: int = 20,
    initial_bankroll: float = 10000.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a calibration-based strategy under a specific impact model.
    
    Returns performance metrics across multiple Monte Carlo simulations.
    """
    rng = np.random.default_rng(seed)
    
    # Detect column names (handle different formats)
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    category_col = 'category' if 'category' in train.columns else None
    volume_col = 'volumeNum' if 'volumeNum' in train.columns else 'volume'
    
    # Extract strategy parameters
    spread_threshold = strategy_params.get('spread_threshold', 0.03)
    n_bins = strategy_params.get('n_bins', 10)
    kelly_fraction = strategy_params.get('kelly_fraction', 0.25)
    max_position_pct = strategy_params.get('max_position_pct', 0.10)
    fee = strategy_params.get('fee', 0.01)
    
    # Learn calibration from training data
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    train_copy['_outcome'] = train_copy[outcome_col]
    train_copy['price_bin'] = pd.cut(train_copy['_price'], bins=n_bins, labels=False)
    
    calibration = train_copy.groupby('price_bin').agg({
        '_price': 'mean',
        '_outcome': 'mean'
    }).rename(columns={'_price': 'bin_avg_price', '_outcome': 'outcome_rate'})
    calibration['spread'] = calibration['outcome_rate'] - calibration['bin_avg_price']
    
    # Prepare test data
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    if category_col:
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
    
    # Run Monte Carlo simulations
    simulation_results = []
    
    for sim in range(n_simulations):
        sim_rng = np.random.default_rng(seed + sim)
        
        bankroll = initial_bankroll
        peak_bankroll = bankroll
        total_pnl = 0.0
        trades = 0
        fills = 0
        total_impact_cost = 0.0
        total_mev_cost = 0.0
        total_adverse_selection_cost = 0.0
        
        pnl_series = []
        
        for _, row in test_copy.iterrows():
            price = row['_price']
            spread = row.get('spread', 0)
            outcome = row.get('_outcome', 0)
            category = row.get('_category', '')
            
            # Skip if spread below threshold
            if abs(spread) < spread_threshold:
                continue
            
            trades += 1
            
            # Estimate liquidity
            liquidity = estimate_liquidity(price, category, impact_params)
            
            # Position sizing (Kelly-based)
            edge = abs(spread)
            if spread > 0:  # Buy YES
                odds = (1 - price) / price if price > 0.01 else 99
            else:  # Buy NO
                odds = price / (1 - price) if price < 0.99 else 99
            
            kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, 1)) * kelly_fraction
            
            intended_size = bankroll * min(kelly, max_position_pct)
            intended_size = min(intended_size, bankroll * 0.5)  # Never risk more than 50%
            
            if intended_size < 1:
                continue
            
            # Fill probability
            fill_prob = compute_fill_probability(
                intended_size, price, liquidity, impact_params, sim_rng
            )
            
            filled = sim_rng.random() < fill_prob
            
            if not filled:
                continue
            
            fills += 1
            
            # Compute costs
            temp_impact, perm_impact = compute_price_impact(
                intended_size, liquidity, impact_params
            )
            
            mev_cost = compute_mev_cost(edge, impact_params, sim_rng)
            
            # Simulated hold time (in "periods")
            hold_time = sim_rng.exponential(5)
            
            as_cost = compute_adverse_selection_cost(
                edge, hold_time, True, impact_params
            )
            
            exit_cost = compute_exit_cost(intended_size, price, liquidity, impact_params)
            
            # Total friction
            total_friction = temp_impact + perm_impact + mev_cost + as_cost + exit_cost + fee
            
            total_impact_cost += temp_impact + perm_impact
            total_mev_cost += mev_cost
            total_adverse_selection_cost += as_cost
            
            # Compute PnL
            if spread > 0:  # Bet on YES
                if outcome == 1:
                    gross_return = (1 - price) / price
                else:
                    gross_return = -1
            else:  # Bet on NO
                if outcome == 0:
                    gross_return = price / (1 - price) if price < 0.99 else 0
                else:
                    gross_return = -1
            
            # Apply friction
            net_return = gross_return - total_friction
            
            pnl = intended_size * net_return
            total_pnl += pnl
            bankroll += pnl
            bankroll = max(bankroll, 0)
            peak_bankroll = max(peak_bankroll, bankroll)
            
            pnl_series.append(pnl)
        
        # Compute metrics for this simulation
        pnl_array = np.array(pnl_series) if pnl_series else np.array([0])
        
        result = {
            'final_bankroll': bankroll,
            'total_pnl': total_pnl,
            'return_pct': (bankroll - initial_bankroll) / initial_bankroll * 100,
            'trades': trades,
            'fills': fills,
            'fill_rate': fills / trades if trades > 0 else 0,
            'sharpe': np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252),
            'max_drawdown': (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0,
            'total_impact_cost': total_impact_cost,
            'total_mev_cost': total_mev_cost,
            'total_adverse_selection_cost': total_adverse_selection_cost,
        }
        
        simulation_results.append(result)
    
    # Aggregate across simulations
    df_results = pd.DataFrame(simulation_results)
    
    return {
        'mean_sharpe': df_results['sharpe'].mean(),
        'std_sharpe': df_results['sharpe'].std(),
        'min_sharpe': df_results['sharpe'].min(),
        'p5_sharpe': df_results['sharpe'].quantile(0.05),
        'p25_sharpe': df_results['sharpe'].quantile(0.25),
        'median_sharpe': df_results['sharpe'].median(),
        'mean_return': df_results['return_pct'].mean(),
        'mean_fill_rate': df_results['fill_rate'].mean(),
        'mean_drawdown': df_results['max_drawdown'].mean(),
        'mean_impact_cost': df_results['total_impact_cost'].mean(),
        'mean_mev_cost': df_results['total_mev_cost'].mean(),
        'mean_as_cost': df_results['total_adverse_selection_cost'].mean(),
        'n_simulations': n_simulations,
        'impact_model': impact_params.name,
        'regime': impact_params.regime,
    }


# =============================================================================
# OPTUNA OPTIMIZATION
# =============================================================================

def create_objective(
    df: pd.DataFrame,
    impact_model: ImpactModelParams,
    n_folds: int = 3,
    n_simulations: int = 10,
) -> Callable:
    """Create an Optuna objective for strategy optimization under a given impact model"""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample strategy parameters
        strategy_params = {
            'spread_threshold': trial.suggest_float('spread_threshold', 0.01, 0.15),
            'n_bins': trial.suggest_int('n_bins', 5, 25),
            'kelly_fraction': trial.suggest_float('kelly_fraction', 0.05, 0.50),
            'max_position_pct': trial.suggest_float('max_position_pct', 0.02, 0.25),
            'fee': trial.suggest_float('fee', 0.005, 0.03),
        }
        
        # Cross-validation
        fold_size = len(df) // (n_folds + 1)
        sharpes = []
        
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = test_start + fold_size
            
            train = df.iloc[:train_end].copy()
            test = df.iloc[test_start:test_end].copy()
            
            if len(train) < 100 or len(test) < 50:
                continue
            
            result = evaluate_strategy_with_impact(
                train, test, impact_model, strategy_params,
                n_simulations=n_simulations,
                seed=fold * 1000 + trial.number,
            )
            
            sharpes.append(result['mean_sharpe'])
        
        if not sharpes:
            return -10.0
        
        return np.mean(sharpes)
    
    return objective


def run_stress_test_optimization(
    df: pd.DataFrame,
    impact_models: List[ImpactModelParams],
    output_dir: Path,
    n_trials: int = 100,
    n_folds: int = 3,
    n_simulations: int = 15,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run optimization across multiple impact models.
    Returns best parameters and performance for each model.
    """
    results = {}
    
    for model in impact_models:
        print(f"\n{'='*60}")
        print(f"Testing Impact Model: {model.name} ({model.regime})")
        print(f"{'='*60}")
        
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
        )
        
        objective = create_objective(df, model, n_folds, n_simulations)
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )
        except Exception as e:
            print(f"Error optimizing {model.name}: {e}")
            continue
        
        # Get best result
        best_params = study.best_params
        best_value = study.best_value
        
        # Evaluate best params more thoroughly
        final_result = evaluate_strategy_with_impact(
            df.iloc[:int(len(df)*0.7)],
            df.iloc[int(len(df)*0.7):],
            model,
            best_params,
            n_simulations=50,  # More simulations for final eval
        )
        
        results[model.name] = {
            'impact_model': asdict(model),
            'best_params': best_params,
            'optimization_sharpe': best_value,
            'final_evaluation': final_result,
            'all_trials': [
                {'number': t.number, 'value': t.value, 'params': t.params}
                for t in study.trials if t.value is not None
            ],
        }
        
        print(f"\nBest Sharpe: {best_value:.3f}")
        print(f"Final Mean Sharpe: {final_result['mean_sharpe']:.3f}")
        print(f"Final Min Sharpe: {final_result['min_sharpe']:.3f}")
        print(f"Best params: {best_params}")
        
        # Save intermediate results
        model_output = output_dir / f"{model.name}_results.json"
        with open(model_output, 'w') as f:
            json.dump(results[model.name], f, indent=2, default=str)
    
    return results


# =============================================================================
# CONTINUOUS STRESS TESTING
# =============================================================================

def get_all_impact_models() -> List[ImpactModelParams]:
    """Get all defined impact models"""
    return [
        # Average case models
        create_almgren_chriss_base(),
        create_almgren_chriss_conservative(),
        create_low_liquidity_average(),
        create_crypto_average(),
        create_high_frequency_average(),
        
        # Adversarial models
        create_adversarial_fills(),
        create_adversarial_impact(),
        create_adversarial_mev(),
        create_adversarial_liquidity(),
        create_adversarial_selection(),
        create_adversarial_exit(),
        create_adversarial_combined(),
    ]


def generate_random_impact_model(
    regime: str,
    rng: np.random.Generator,
    model_id: int,
) -> ImpactModelParams:
    """Generate a random impact model for exploration"""
    
    if regime == "adversarial":
        return ImpactModelParams(
            name=f"random_adversarial_{model_id}",
            regime="adversarial",
            base_fill_prob=rng.uniform(0.20, 0.60),
            fill_size_decay=rng.uniform(0.15, 0.40),
            fill_spread_sensitivity=rng.uniform(3.0, 8.0),
            permanent_impact=rng.uniform(0.002, 0.010),
            temporary_impact=rng.uniform(0.010, 0.050),
            impact_power=rng.uniform(0.6, 0.9),
            mev_extraction_rate=rng.uniform(0.05, 0.25),
            reorder_probability=rng.uniform(0.10, 0.40),
            block_delay_cost=rng.uniform(0.002, 0.010),
            adverse_selection=rng.uniform(0.15, 0.40),
            adverse_selection_decay=rng.uniform(0.1, 0.4),
            base_liquidity=rng.uniform(500, 5000),
            exit_spread=rng.uniform(0.03, 0.10),
            exit_impact_mult=rng.uniform(2.0, 5.0),
            stress_multiplier=rng.uniform(1.5, 4.0),
        )
    else:
        return ImpactModelParams(
            name=f"random_average_{model_id}",
            regime="average",
            base_fill_prob=rng.uniform(0.60, 0.95),
            fill_size_decay=rng.uniform(0.03, 0.15),
            fill_spread_sensitivity=rng.uniform(1.0, 4.0),
            permanent_impact=rng.uniform(0.0005, 0.003),
            temporary_impact=rng.uniform(0.002, 0.015),
            impact_power=rng.uniform(0.4, 0.7),
            mev_extraction_rate=rng.uniform(0.0, 0.05),
            reorder_probability=rng.uniform(0.0, 0.15),
            block_delay_cost=rng.uniform(0.0, 0.003),
            adverse_selection=rng.uniform(0.02, 0.15),
            adverse_selection_decay=rng.uniform(0.3, 0.8),
            base_liquidity=rng.uniform(3000, 20000),
            exit_spread=rng.uniform(0.01, 0.04),
            exit_impact_mult=rng.uniform(1.0, 2.5),
            stress_multiplier=1.0,
        )


def continuous_stress_test(
    df: pd.DataFrame,
    output_dir: Path,
    n_trials_per_model: int = 50,
    n_random_models_per_round: int = 5,
    max_rounds: int = 1000,
    n_folds: int = 3,
    n_simulations: int = 10,
) -> None:
    """
    Run continuous stress testing with many impact models.
    Keeps running until max_rounds or interrupted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with predefined models
    all_models = get_all_impact_models()
    
    rng = np.random.default_rng(42)
    round_num = 0
    
    # Master results log
    master_log = output_dir / "master_results.jsonl"
    
    print(f"\n{'#'*70}")
    print("CONTINUOUS STRESS TESTING FRAMEWORK")
    print(f"Output: {output_dir}")
    print(f"{'#'*70}\n")
    
    while round_num < max_rounds:
        round_num += 1
        round_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} / {max_rounds}")
        print(f"Time: {datetime.now().isoformat()}")
        print(f"{'='*60}")
        
        # Select models for this round
        if round_num <= len(all_models):
            # First, run through predefined models
            models_this_round = [all_models[round_num - 1]]
        else:
            # Then generate random models
            models_this_round = []
            for i in range(n_random_models_per_round):
                regime = rng.choice(["average", "adversarial"])
                model = generate_random_impact_model(
                    regime, rng, 
                    model_id=round_num * 100 + i
                )
                models_this_round.append(model)
        
        # Run optimization for each model
        for model in models_this_round:
            try:
                results = run_stress_test_optimization(
                    df=df,
                    impact_models=[model],
                    output_dir=output_dir,
                    n_trials=n_trials_per_model,
                    n_folds=n_folds,
                    n_simulations=n_simulations,
                )
                
                # Log to master file
                for model_name, result in results.items():
                    log_entry = {
                        'round': round_num,
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'regime': result['impact_model']['regime'],
                        'optimization_sharpe': result['optimization_sharpe'],
                        'final_mean_sharpe': result['final_evaluation']['mean_sharpe'],
                        'final_min_sharpe': result['final_evaluation']['min_sharpe'],
                        'final_p5_sharpe': result['final_evaluation']['p5_sharpe'],
                        'mean_fill_rate': result['final_evaluation']['mean_fill_rate'],
                        'mean_drawdown': result['final_evaluation']['mean_drawdown'],
                        'best_params': result['best_params'],
                    }
                    
                    with open(master_log, 'a') as f:
                        f.write(json.dumps(log_entry, default=str) + '\n')
                    
            except Exception as e:
                print(f"Error in round {round_num}: {e}")
                continue
        
        round_time = time.time() - round_start
        print(f"\nRound {round_num} completed in {round_time:.1f}s")
        
        # Print summary every 10 rounds
        if round_num % 10 == 0:
            try:
                df_log = pd.read_json(master_log, lines=True)
                print(f"\n{'='*60}")
                print("CUMULATIVE SUMMARY")
                print(f"{'='*60}")
                print(f"Total models tested: {len(df_log)}")
                print(f"\nBy Regime:")
                print(df_log.groupby('regime')['final_mean_sharpe'].agg(['mean', 'min', 'max']))
                print(f"\nTop 5 Models (by mean Sharpe):")
                print(df_log.nlargest(5, 'final_mean_sharpe')[
                    ['model_name', 'regime', 'final_mean_sharpe', 'final_min_sharpe']
                ].to_string())
                print(f"\nWorst 5 Models (by min Sharpe):")
                print(df_log.nsmallest(5, 'final_min_sharpe')[
                    ['model_name', 'regime', 'final_mean_sharpe', 'final_min_sharpe']
                ].to_string())
            except Exception:
                pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stress Test Impact Models')
    parser.add_argument('--cache-path', type=str, 
                       default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, 
                       default='runs/stress_test_impact')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Trials per impact model')
    parser.add_argument('--n-folds', type=int, default=3)
    parser.add_argument('--n-simulations', type=int, default=15)
    parser.add_argument('--max-rounds', type=int, default=1000,
                       help='Maximum rounds of testing')
    parser.add_argument('--n-random-models', type=int, default=5,
                       help='Random models per round after predefined')
    parser.add_argument('--worker-id', type=int, default=0,
                       help='Worker ID for parallel execution')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.cache_path}")
    df = pd.read_parquet(args.cache_path)
    print(f"Loaded {len(df):,} markets")
    
    # Setup output
    output_dir = Path(args.output_dir) / f"worker_{args.worker_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print("STRESS TEST CONFIGURATION")
    print(f"{'#'*70}")
    print(f"Worker ID: {args.worker_id}")
    print(f"Output: {output_dir}")
    print(f"Trials per model: {args.n_trials}")
    print(f"CV folds: {args.n_folds}")
    print(f"Simulations: {args.n_simulations}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Random models/round: {args.n_random_models}")
    print(f"{'#'*70}\n")
    
    # Run continuous stress testing
    continuous_stress_test(
        df=df,
        output_dir=output_dir,
        n_trials_per_model=args.n_trials,
        n_random_models_per_round=args.n_random_models,
        max_rounds=args.max_rounds,
        n_folds=args.n_folds,
        n_simulations=args.n_simulations,
    )


if __name__ == '__main__':
    main()
