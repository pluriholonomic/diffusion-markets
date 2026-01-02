#!/usr/bin/env python3
"""
Category-Aware Hyperparameter Optimization for Clustering Algorithms.

Different market categories (sports, politics, crypto, tech) have different
dynamics. This script optimizes hyperparameters separately for each category
and recommends the best algorithm per category.

Usage:
    python optimize_by_category.py --synthetic
    python optimize_by_category.py --data data/polymarket.parquet
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecastbench.clustering import (
    SWOCC, OLRCM, BirthDeathHawkes, StreamingDPClustering,
    DynamicGraphAttention,
)
from forecastbench.clustering.survival_weighted import SWOCCConfig
from forecastbench.clustering.online_factor import OLRCMConfig
from forecastbench.clustering.birth_death_hawkes import BDHPConfig
from forecastbench.clustering.streaming_dp import SDPMConfig
from forecastbench.clustering.dynamic_graph import DGATConfig


# =============================================================================
# Category Definitions - Each has distinct market dynamics
# =============================================================================

@dataclass
class CategoryConfig:
    """Configuration for a market category's dynamics."""
    name: str
    
    # Lifetime characteristics
    mean_lifetime_days: float
    lifetime_std_days: float
    
    # Correlation structure
    n_clusters: int
    within_cluster_corr: float
    between_cluster_corr: float
    
    # Volatility
    base_volatility: float
    volatility_clustering: float  # GARCH-like persistence
    
    # Event sensitivity
    event_sensitivity: float  # How much news affects prices
    
    # Typical market count
    typical_n_markets: int


CATEGORY_CONFIGS = {
    'sports': CategoryConfig(
        name='sports',
        mean_lifetime_days=3,      # Very short - games happen within days
        lifetime_std_days=2,
        n_clusters=6,              # Different sports/leagues
        within_cluster_corr=0.6,   # High within same sport
        between_cluster_corr=0.1,  # Low across sports
        base_volatility=0.15,
        volatility_clustering=0.3,
        event_sensitivity=0.8,     # Very event-driven
        typical_n_markets=50,
    ),
    'politics': CategoryConfig(
        name='politics',
        mean_lifetime_days=90,     # Elections are months out
        lifetime_std_days=60,
        n_clusters=4,              # Ideology/region clusters
        within_cluster_corr=0.5,
        between_cluster_corr=0.25, # Moderate cross-cluster (national mood)
        base_volatility=0.08,
        volatility_clustering=0.6, # News cycles cause clustering
        event_sensitivity=0.9,     # Very news-sensitive
        typical_n_markets=30,
    ),
    'crypto': CategoryConfig(
        name='crypto',
        mean_lifetime_days=30,
        lifetime_std_days=20,
        n_clusters=3,              # BTC, ETH, altcoins
        within_cluster_corr=0.75,  # Very high correlation
        between_cluster_corr=0.5,  # Also correlated across
        base_volatility=0.25,      # High volatility
        volatility_clustering=0.8, # Strong GARCH effects
        event_sensitivity=0.7,
        typical_n_markets=40,
    ),
    'tech': CategoryConfig(
        name='tech',
        mean_lifetime_days=45,
        lifetime_std_days=30,
        n_clusters=5,              # Different sectors (AI, hardware, etc.)
        within_cluster_corr=0.45,
        between_cluster_corr=0.2,
        base_volatility=0.12,
        volatility_clustering=0.5,
        event_sensitivity=0.6,     # Product launches, earnings
        typical_n_markets=35,
    ),
}


# =============================================================================
# Category-Specific Data Generation
# =============================================================================

def generate_category_data(
    category: CategoryConfig,
    n_days: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int]]]:
    """
    Generate synthetic data matching a category's characteristics.
    
    Returns:
        prices: (n_days, n_markets) price matrix
        outcomes: (n_markets,) binary outcomes
        market_ids: list of market identifiers
        death_events: list of (day, market_idx) death events
    """
    np.random.seed(seed)
    
    n_markets = category.typical_n_markets
    
    # Generate cluster assignments
    cluster_sizes = np.random.multinomial(
        n_markets, 
        [1/category.n_clusters] * category.n_clusters
    )
    cluster_assignments = np.repeat(
        np.arange(category.n_clusters),
        cluster_sizes
    )
    np.random.shuffle(cluster_assignments)
    
    # Build correlation matrix
    corr_matrix = np.eye(n_markets) * (1 - category.within_cluster_corr)
    for i in range(n_markets):
        for j in range(n_markets):
            if i != j:
                if cluster_assignments[i] == cluster_assignments[j]:
                    corr_matrix[i, j] = category.within_cluster_corr
                else:
                    corr_matrix[i, j] = category.between_cluster_corr
    
    corr_matrix = np.diag(np.ones(n_markets)) * (1 - category.within_cluster_corr) + corr_matrix
    
    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Generate market lifetimes
    lifetimes = np.maximum(
        1,
        np.random.normal(
            category.mean_lifetime_days,
            category.lifetime_std_days,
            n_markets
        ).astype(int)
    )
    lifetimes = np.minimum(lifetimes, n_days - 1)
    
    # Generate start times (staggered)
    start_times = np.random.randint(0, max(1, n_days - int(category.mean_lifetime_days)), n_markets)
    end_times = np.minimum(start_times + lifetimes, n_days - 1)
    
    # Generate correlated returns with volatility clustering
    chol = np.linalg.cholesky(corr_matrix)
    
    prices = np.full((n_days, n_markets), np.nan)
    volatility = np.ones(n_markets) * category.base_volatility
    
    for t in range(n_days):
        # Update volatility (GARCH-like)
        innovations = np.random.randn(n_markets)
        corr_innovations = chol @ innovations
        
        # Volatility clustering
        volatility = (
            category.base_volatility * (1 - category.volatility_clustering) +
            category.volatility_clustering * volatility +
            0.1 * np.abs(corr_innovations)
        )
        
        for m in range(n_markets):
            if start_times[m] <= t <= end_times[m]:
                if t == start_times[m]:
                    prices[t, m] = 0.5  # Start at 50%
                else:
                    prev_price = prices[t-1, m]
                    if not np.isnan(prev_price):
                        # Price change with mean reversion to outcome
                        outcome_drift = 0.001 * (1.0 if np.random.random() > 0.5 else 0.0 - prev_price)
                        change = outcome_drift + volatility[m] * corr_innovations[m]
                        prices[t, m] = np.clip(prev_price + change, 0.01, 0.99)
    
    # Record death events
    death_events = [(int(end_times[m]), m) for m in range(n_markets) if end_times[m] < n_days - 1]
    death_events.sort(key=lambda x: x[0])
    
    # Generate outcomes
    outcomes = np.random.binomial(1, 0.5, n_markets).astype(float)
    
    # Market IDs
    market_ids = [f"{category.name}_{i}" for i in range(n_markets)]
    
    return prices, outcomes, market_ids, death_events


# =============================================================================
# Category-Specific Hyperparameter Grids
# =============================================================================

def get_category_hyperparams(category: CategoryConfig) -> Dict[str, List[Dict]]:
    """
    Get hyperparameter grids tuned for a specific category.
    """
    grids = {}
    
    # SWOCC - Good for short-lived, event-driven markets
    if category.mean_lifetime_days < 30:  # Short-lived (sports, crypto)
        grids['SWOCC'] = [
            {'ema_alpha': 0.3, 'use_survival_weights': True, 'shrinkage': 0.1, 'recluster_every': 5},
            {'ema_alpha': 0.4, 'use_survival_weights': True, 'shrinkage': 0.05, 'recluster_every': 3},
            {'ema_alpha': 0.2, 'use_survival_weights': True, 'shrinkage': 0.15, 'recluster_every': 5},
        ]
    else:  # Longer-lived (politics, tech)
        grids['SWOCC'] = [
            {'ema_alpha': 0.1, 'use_survival_weights': True, 'shrinkage': 0.2, 'recluster_every': 20},
            {'ema_alpha': 0.15, 'use_survival_weights': True, 'shrinkage': 0.15, 'recluster_every': 15},
            {'ema_alpha': 0.05, 'use_survival_weights': False, 'shrinkage': 0.1, 'recluster_every': 10},
        ]
    
    # OLRCM - Good for factor-driven markets (crypto, tech)
    if category.within_cluster_corr > 0.5:  # Strong factor structure
        grids['OLRCM'] = [
            {'n_factors': 10, 'learning_rate': 0.02, 'l2_reg': 0.001, 'recluster_every': 10},
            {'n_factors': 15, 'learning_rate': 0.02, 'l2_reg': 0.01, 'recluster_every': 10},
            {'n_factors': 8, 'learning_rate': 0.01, 'l2_reg': 0.001, 'recluster_every': 5},
        ]
    else:  # Weaker factor structure
        grids['OLRCM'] = [
            {'n_factors': 5, 'learning_rate': 0.01, 'l2_reg': 0.001, 'recluster_every': 10},
            {'n_factors': 8, 'learning_rate': 0.015, 'l2_reg': 0.01, 'recluster_every': 15},
            {'n_factors': 3, 'learning_rate': 0.01, 'l2_reg': 0.0001, 'recluster_every': 20},
        ]
    
    # SDPM - Good for evolving cluster structure (politics, tech)
    if category.event_sensitivity > 0.7:  # High event sensitivity
        grids['SDPM'] = [
            {'concentration': 2.0, 'prior_precision': 0.2, 'temperature': 1.0, 'reassign_every': 5},
            {'concentration': 3.0, 'prior_precision': 0.15, 'temperature': 0.8, 'reassign_every': 10},
            {'concentration': 1.5, 'prior_precision': 0.25, 'temperature': 1.2, 'reassign_every': 5},
        ]
    else:
        grids['SDPM'] = [
            {'concentration': 5.0, 'prior_precision': 0.2, 'temperature': 1.0, 'reassign_every': 20},
            {'concentration': 4.0, 'prior_precision': 0.1, 'temperature': 1.0, 'reassign_every': 15},
            {'concentration': 3.0, 'prior_precision': 0.2, 'temperature': 0.5, 'reassign_every': 10},
        ]
    
    # BDHP - Good for event-driven correlations
    if category.event_sensitivity > 0.7:
        grids['BDHP'] = [
            {'decay_rate': 0.3, 'base_intensity': 0.15, 'self_excitation': 0.4, 'price_change_threshold': 0.01},
            {'decay_rate': 0.2, 'base_intensity': 0.2, 'self_excitation': 0.5, 'price_change_threshold': 0.015},
            {'decay_rate': 0.4, 'base_intensity': 0.1, 'self_excitation': 0.3, 'price_change_threshold': 0.02},
        ]
    else:
        grids['BDHP'] = [
            {'decay_rate': 0.1, 'base_intensity': 0.1, 'self_excitation': 0.2, 'price_change_threshold': 0.02},
            {'decay_rate': 0.15, 'base_intensity': 0.15, 'self_excitation': 0.3, 'price_change_threshold': 0.025},
            {'decay_rate': 0.2, 'base_intensity': 0.1, 'self_excitation': 0.25, 'price_change_threshold': 0.03},
        ]
    
    # DGAT - Good for complex, non-linear relationships
    grids['DGAT'] = [
        {'hidden_dim': 16, 'n_attention_heads': 2, 'learning_rate': 0.01, 'k_neighbors': 5},
        {'hidden_dim': 32, 'n_attention_heads': 4, 'learning_rate': 0.02, 'k_neighbors': 10},
        {'hidden_dim': 24, 'n_attention_heads': 2, 'learning_rate': 0.015, 'k_neighbors': 8},
    ]
    
    return grids


# =============================================================================
# Trading Backtest (simplified from main script)
# =============================================================================

def compute_expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    if len(returns) == 0:
        return 0.0
    var = np.percentile(returns, alpha * 100)
    return float(np.mean(returns[returns <= var])) if np.any(returns <= var) else float(var)


def create_algorithm(name: str, config: Dict):
    """Factory to create clustering algorithm instances."""
    if name == 'SWOCC':
        return SWOCC(SWOCCConfig(
            ema_alpha=config.get('ema_alpha', 0.1),
            use_survival_weights=config.get('use_survival_weights', True),
            shrinkage=config.get('shrinkage', 0.1),
        ))
    elif name == 'OLRCM':
        return OLRCM(OLRCMConfig(
            n_factors=config.get('n_factors', 5),
            learning_rate=config.get('learning_rate', 0.01),
            l2_reg=config.get('l2_reg', 0.001),
        ))
    elif name == 'SDPM':
        return StreamingDPClustering(SDPMConfig(
            concentration=config.get('concentration', 1.0),
            prior_precision=config.get('prior_precision', 0.1),
            temperature=config.get('temperature', 1.0),
        ))
    elif name == 'BDHP':
        return BirthDeathHawkes(BDHPConfig(
            decay_rate=config.get('decay_rate', 0.1),
            base_intensity=config.get('base_intensity', 0.1),
            self_excitation=config.get('self_excitation', 0.2),
        ))
    elif name == 'DGAT':
        return DynamicGraphAttention(DGATConfig(
            hidden_dim=config.get('hidden_dim', 16),
            n_attention_heads=config.get('n_attention_heads', 2),
            learning_rate=config.get('learning_rate', 0.01),
        ))
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def run_backtest(
    algo_name: str,
    config: Dict,
    prices: np.ndarray,
    death_events: List[Tuple[int, int]],
    recluster_every: int = 10,
) -> Dict[str, float]:
    """Run a simple mean-reversion backtest with clustering."""
    n_days, n_markets = prices.shape
    market_ids = [str(m) for m in range(n_markets)]
    
    try:
        algo = create_algorithm(algo_name, config)
    except Exception as e:
        return {'sharpe': float('-inf'), 'pnl': 0, 'es': 0, 'error': str(e)}
    
    # Track which markets have been added
    added_markets = set()
    
    # Track deaths
    death_set = set()
    death_dict = {}
    for day, mkt in death_events:
        death_dict.setdefault(day, []).append(mkt)
    
    daily_pnl = []
    positions = np.zeros(n_markets)
    n_clusters_list = []
    
    for t in range(1, n_days):
        # Process deaths
        if t in death_dict:
            for mkt in death_dict[t]:
                death_set.add(mkt)
                try:
                    algo.remove_market(str(mkt), timestamp=float(t))
                except:
                    pass
        
        # Get current prices
        curr_prices = prices[t]
        prev_prices = prices[t-1]
        
        # Calculate returns
        valid = ~np.isnan(curr_prices) & ~np.isnan(prev_prices)
        returns = np.zeros(n_markets)
        returns[valid] = (curr_prices[valid] - prev_prices[valid]) / np.maximum(prev_prices[valid], 0.01)
        
        # Add new markets that just became active
        for m in range(n_markets):
            if m not in added_markets and m not in death_set and not np.isnan(curr_prices[m]):
                try:
                    algo.add_market(str(m), timestamp=float(t), initial_price=float(curr_prices[m]))
                    added_markets.add(m)
                except:
                    pass
        
        # Update clustering with price dict
        if t % recluster_every == 0:
            try:
                price_dict = {
                    str(m): float(curr_prices[m])
                    for m in range(n_markets)
                    if m not in death_set and not np.isnan(curr_prices[m]) and m in added_markets
                }
                if price_dict:
                    algo.update(float(t), price_dict)
            except Exception as e:
                pass
        
        # Get cluster assignments
        try:
            clusters = algo.get_clusters()
            n_clusters = len(clusters)
            n_clusters_list.append(n_clusters)
        except:
            n_clusters_list.append(1)
        
        # Cluster-aware mean-reversion strategy
        active = [m for m in range(n_markets) if m not in death_set and not np.isnan(curr_prices[m])]
        
        # Get cluster-based positions
        try:
            clusters = algo.get_clusters()
            if len(clusters) > 0:
                # Mean-revert within each cluster
                for cluster_id, members in clusters.items():
                    member_indices = [int(m) for m in members if int(m) in active]
                    if len(member_indices) >= 2:
                        cluster_returns = returns[member_indices]
                        cluster_mean = np.mean(cluster_returns)
                        for m in member_indices:
                            deviation = returns[m] - cluster_mean
                            positions[m] = -np.clip(deviation * 15, -1, 1)
            else:
                # Fallback: global mean-reversion
                if len(active) > 1:
                    active_returns = returns[active]
                    mean_return = np.mean(active_returns)
                    for m in active:
                        deviation = returns[m] - mean_return
                        positions[m] = -np.clip(deviation * 10, -1, 1)
        except:
            # Fallback: global mean-reversion
            if len(active) > 1:
                active_returns = returns[active]
                mean_return = np.mean(active_returns)
                for m in active:
                    deviation = returns[m] - mean_return
                    positions[m] = -np.clip(deviation * 10, -1, 1)
        
        # Calculate PnL
        pnl = np.nansum(positions * returns)
        daily_pnl.append(pnl)
        
        # Decay positions
        positions *= 0.95
    
    daily_pnl = np.array(daily_pnl)
    
    if len(daily_pnl) == 0 or np.std(daily_pnl) == 0:
        return {'sharpe': 0, 'pnl': 0, 'es': 0, 'avg_clusters': 1}
    
    sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    total_pnl = np.sum(daily_pnl)
    es = compute_expected_shortfall(daily_pnl)
    avg_clusters = np.mean(n_clusters_list) if n_clusters_list else 1
    
    return {
        'sharpe': float(sharpe),
        'pnl': float(total_pnl),
        'es': float(es),
        'avg_clusters': float(avg_clusters),
    }


# =============================================================================
# Main Optimization Loop
# =============================================================================

def optimize_category(
    category: CategoryConfig,
    n_days: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimize all algorithms for a specific category."""
    
    print(f"\n{'='*60}")
    print(f"Optimizing for category: {category.name.upper()}")
    print(f"  Avg lifetime: {category.mean_lifetime_days:.0f} days")
    print(f"  Clusters: {category.n_clusters}")
    print(f"  Within-cluster corr: {category.within_cluster_corr:.2f}")
    print(f"{'='*60}")
    
    # Generate data
    prices, outcomes, market_ids, death_events = generate_category_data(
        category, n_days=n_days, seed=seed
    )
    
    print(f"Generated {prices.shape[1]} markets, {prices.shape[0]} days, {len(death_events)} deaths")
    
    # Get category-specific hyperparameters
    hyperparams = get_category_hyperparams(category)
    
    results = {}
    
    for algo_name, configs in hyperparams.items():
        best_result = None
        best_sharpe = float('-inf')
        
        for config in configs:
            recluster = config.pop('recluster_every', 10)
            result = run_backtest(algo_name, config, prices, death_events, recluster)
            config['recluster_every'] = recluster  # Restore
            
            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = {
                    'config': config.copy(),
                    **result
                }
        
        if best_result:
            results[algo_name] = best_result
            print(f"  {algo_name:20s}: Sharpe={best_result['sharpe']:+.3f}, "
                  f"PnL=${best_result['pnl']:,.0f}, Clusters={best_result['avg_clusters']:.1f}")
    
    # Find best algorithm for this category
    best_algo = max(results.keys(), key=lambda a: results[a]['sharpe'])
    
    return {
        'category': category.name,
        'best_algorithm': best_algo,
        'best_config': results[best_algo]['config'],
        'best_sharpe': results[best_algo]['sharpe'],
        'all_results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Category-aware clustering optimization')
    parser.add_argument('--categories', nargs='+', default=list(CATEGORY_CONFIGS.keys()),
                        help='Categories to optimize')
    parser.add_argument('--n-days', type=int, default=300,
                        help='Number of days for synthetic data')
    parser.add_argument('--output-dir', type=str, default='runs/category_optimization',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for cat_name in args.categories:
        if cat_name not in CATEGORY_CONFIGS:
            print(f"Unknown category: {cat_name}")
            continue
        
        category = CATEGORY_CONFIGS[cat_name]
        result = optimize_category(category, n_days=args.n_days, seed=args.seed)
        all_results[cat_name] = result
    
    # Print summary
    print("\n" + "="*80)
    print("RECOMMENDED ALGORITHMS BY CATEGORY")
    print("="*80)
    print(f"{'Category':<15} {'Best Algorithm':<25} {'Sharpe':>10} {'Config'}")
    print("-"*80)
    
    for cat_name, result in all_results.items():
        config_str = json.dumps(result['best_config'], separators=(',', ':'))
        if len(config_str) > 40:
            config_str = config_str[:37] + "..."
        print(f"{cat_name:<15} {result['best_algorithm']:<25} {result['best_sharpe']:>+10.3f} {config_str}")
    
    # Save results
    results_file = output_dir / 'category_optimization_results.json'
    
    def convert_types(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save recommended configs for production use
    recommended = {
        cat: {
            'algorithm': r['best_algorithm'],
            'config': r['best_config'],
        }
        for cat, r in all_results.items()
    }
    
    with open(output_dir / 'recommended_configs.json', 'w') as f:
        json.dump(convert_types(recommended), f, indent=2)
    
    print(f"Recommended configs saved to {output_dir / 'recommended_configs.json'}")


if __name__ == '__main__':
    main()
