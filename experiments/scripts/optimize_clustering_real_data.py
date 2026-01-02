#!/usr/bin/env python3
"""
Clustering Optimization on Real Polymarket Data.

Loads historical Polymarket CLOB price data and runs clustering algorithm
optimization by category.

Usage:
    python optimize_clustering_real_data.py
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from collections import defaultdict
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
# Data Loading
# =============================================================================

def load_market_metadata(cache_path: Path, clob_dir: Path) -> pd.DataFrame:
    """Load market metadata from optimization cache, filtering to markets with CLOB data."""
    df = pd.read_parquet(cache_path)
    
    # Get available CLOB files
    clob_files = set(f.replace('.parquet', '') for f in os.listdir(clob_dir))
    
    # Filter to markets with CLOB data
    df['yes_token_id'] = df['yes_token_id'].astype(str)
    df['has_clob'] = df['yes_token_id'].isin(clob_files)
    df = df[df['has_clob']].copy()
    
    # Parse dates
    df['resolution_time'] = pd.to_datetime(df['resolution_time'], errors='coerce')
    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
    
    # Filter to resolved markets with outcomes
    df = df[df['y'].notna()].copy()
    df['y'] = df['y'].astype(int)
    
    # Categorize
    df['category'] = df['category'].fillna('other')
    df['category'] = df['category'].str.lower()
    
    # Map to our standard categories
    def map_category(cat):
        cat = str(cat).lower()
        if 'crypto' in cat or 'bitcoin' in cat:
            return 'crypto'
        elif 'politic' in cat or 'election' in cat:
            return 'politics'
        elif 'sport' in cat or 'nba' in cat or 'nfl' in cat:
            return 'sports'
        elif 'tech' in cat or 'ai' in cat:
            return 'tech'
        else:
            return 'other'
    
    df['category_mapped'] = df['category'].apply(map_category)
    
    return df


def load_clob_prices(clob_dir: Path, metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Load CLOB price history for specified markets.
    
    Args:
        clob_dir: Directory containing CLOB parquet files
        metadata: DataFrame with 'id' and 'yes_token_id' columns
    
    Returns dict mapping market_id -> DataFrame with ['t', 'p'] columns.
    """
    prices = {}
    
    for _, row in metadata.iterrows():
        market_id = str(row['id'])
        token_id = str(row['yes_token_id'])
        
        file_path = clob_dir / f"{token_id}.parquet"
        if not file_path.exists():
            continue
        
        try:
            df = pd.read_parquet(file_path)
            # Handle timestamp - could be Unix epoch or already datetime
            if df['t'].dtype in ['int64', 'float64']:
                # Unix timestamp in seconds
                df['t'] = pd.to_datetime(df['t'], unit='s', utc=True)
            else:
                df['t'] = pd.to_datetime(df['t'])
            df = df.sort_values('t')
            # Drop duplicates keeping last observation per timestamp
            df = df.drop_duplicates(subset=['t'], keep='last')
            prices[market_id] = df[['t', 'p']].copy()
        except Exception as e:
            continue
    
    return prices


def build_price_matrix(
    prices: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    freq: str = 'D',
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], pd.DatetimeIndex]:
    """
    Build a (T, N) price matrix from individual price series.
    
    Returns:
        price_matrix: (n_days, n_markets) array
        market_ids: list of market IDs
        death_events: list of (day_idx, market_idx) for market resolutions
        date_index: pandas DatetimeIndex
    """
    if not prices:
        raise ValueError("No price data available")
    
    market_ids = list(prices.keys())
    n_markets = len(market_ids)
    
    # First, resample each market to daily and collect
    daily_prices = {}
    all_dates = set()
    
    for mid in market_ids:
        df = prices[mid].copy()
        # Normalize to date only (remove time component)
        df['date'] = df['t'].dt.normalize()
        # Keep last price per day
        daily = df.groupby('date')['p'].last()
        daily_prices[mid] = daily
        all_dates.update(daily.index)
    
    # Create date range
    all_dates = sorted(all_dates)
    date_range = pd.DatetimeIndex(all_dates)
    n_days = len(date_range)
    
    # Build matrix
    price_matrix = np.full((n_days, n_markets), np.nan)
    
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    
    for i, mid in enumerate(market_ids):
        daily = daily_prices[mid]
        for date, p in daily.items():
            if date in date_to_idx:
                price_matrix[date_to_idx[date], i] = p
    
    # Forward fill prices
    for i in range(n_markets):
        last_valid = np.nan
        for t in range(n_days):
            if np.isnan(price_matrix[t, i]) and not np.isnan(last_valid):
                price_matrix[t, i] = last_valid
            elif not np.isnan(price_matrix[t, i]):
                last_valid = price_matrix[t, i]
    
    # Determine death events from metadata
    death_events = []
    resolution_times = metadata.set_index('id')['resolution_time'].to_dict()
    
    for i, mid in enumerate(market_ids):
        res_time = resolution_times.get(mid)
        if pd.notna(res_time):
            try:
                idx = date_range.get_indexer([res_time], method='ffill')[0]
                if 0 <= idx < n_days:
                    death_events.append((idx, i))
            except:
                continue
    
    death_events.sort(key=lambda x: x[0])
    
    return price_matrix, market_ids, death_events, date_range


# =============================================================================
# Algorithm Creation
# =============================================================================

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


# =============================================================================
# Hyperparameter Configs by Category
# =============================================================================

HYPERPARAMS = {
    'sports': {
        'SWOCC': [
            {'ema_alpha': 0.3, 'use_survival_weights': True, 'shrinkage': 0.1, 'recluster_every': 5},
            {'ema_alpha': 0.4, 'use_survival_weights': True, 'shrinkage': 0.05, 'recluster_every': 3},
        ],
        'OLRCM': [
            {'n_factors': 5, 'learning_rate': 0.02, 'l2_reg': 0.001, 'recluster_every': 5},
            {'n_factors': 8, 'learning_rate': 0.01, 'l2_reg': 0.01, 'recluster_every': 5},
        ],
        'SDPM': [
            {'concentration': 2.0, 'prior_precision': 0.2, 'temperature': 1.0, 'reassign_every': 5},
            {'concentration': 3.0, 'prior_precision': 0.15, 'temperature': 0.8, 'reassign_every': 3},
        ],
        'BDHP': [
            {'decay_rate': 0.3, 'base_intensity': 0.15, 'self_excitation': 0.4, 'price_change_threshold': 0.01},
        ],
        'DGAT': [
            {'hidden_dim': 16, 'n_attention_heads': 2, 'learning_rate': 0.02, 'k_neighbors': 5},
        ],
    },
    'politics': {
        'SWOCC': [
            {'ema_alpha': 0.1, 'use_survival_weights': True, 'shrinkage': 0.2, 'recluster_every': 20},
            {'ema_alpha': 0.05, 'use_survival_weights': False, 'shrinkage': 0.15, 'recluster_every': 15},
        ],
        'OLRCM': [
            {'n_factors': 10, 'learning_rate': 0.01, 'l2_reg': 0.001, 'recluster_every': 15},
            {'n_factors': 15, 'learning_rate': 0.02, 'l2_reg': 0.01, 'recluster_every': 10},
        ],
        'SDPM': [
            {'concentration': 3.0, 'prior_precision': 0.15, 'temperature': 0.8, 'reassign_every': 10},
            {'concentration': 5.0, 'prior_precision': 0.2, 'temperature': 1.0, 'reassign_every': 20},
        ],
        'BDHP': [
            {'decay_rate': 0.1, 'base_intensity': 0.1, 'self_excitation': 0.2, 'price_change_threshold': 0.02},
        ],
        'DGAT': [
            {'hidden_dim': 32, 'n_attention_heads': 4, 'learning_rate': 0.01, 'k_neighbors': 10},
        ],
    },
    'crypto': {
        'SWOCC': [
            {'ema_alpha': 0.2, 'use_survival_weights': True, 'shrinkage': 0.1, 'recluster_every': 10},
        ],
        'OLRCM': [
            {'n_factors': 10, 'learning_rate': 0.02, 'l2_reg': 0.001, 'recluster_every': 10},
            {'n_factors': 15, 'learning_rate': 0.02, 'l2_reg': 0.01, 'recluster_every': 10},
        ],
        'SDPM': [
            {'concentration': 5.0, 'prior_precision': 0.2, 'temperature': 1.0, 'reassign_every': 10},
        ],
        'BDHP': [
            {'decay_rate': 0.2, 'base_intensity': 0.1, 'self_excitation': 0.3, 'price_change_threshold': 0.02},
        ],
        'DGAT': [
            {'hidden_dim': 32, 'n_attention_heads': 4, 'learning_rate': 0.02, 'k_neighbors': 10},
        ],
    },
    'tech': {
        'SWOCC': [
            {'ema_alpha': 0.1, 'use_survival_weights': False, 'shrinkage': 0.15, 'recluster_every': 15},
        ],
        'OLRCM': [
            {'n_factors': 8, 'learning_rate': 0.015, 'l2_reg': 0.01, 'recluster_every': 15},
        ],
        'SDPM': [
            {'concentration': 4.0, 'prior_precision': 0.15, 'temperature': 1.0, 'reassign_every': 15},
        ],
        'BDHP': [
            {'decay_rate': 0.15, 'base_intensity': 0.1, 'self_excitation': 0.25, 'price_change_threshold': 0.02},
        ],
        'DGAT': [
            {'hidden_dim': 24, 'n_attention_heads': 2, 'learning_rate': 0.015, 'k_neighbors': 8},
        ],
    },
}


# =============================================================================
# Backtest
# =============================================================================

def compute_expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    if len(returns) == 0:
        return 0.0
    var = np.percentile(returns, alpha * 100)
    return float(np.mean(returns[returns <= var])) if np.any(returns <= var) else float(var)


def run_backtest(
    algo_name: str,
    config: Dict,
    prices: np.ndarray,
    death_events: List[Tuple[int, int]],
    recluster_every: int = 10,
) -> Dict[str, float]:
    """Run cluster-aware mean-reversion backtest."""
    n_days, n_markets = prices.shape
    
    try:
        algo = create_algorithm(algo_name, config)
    except Exception as e:
        return {'sharpe': float('-inf'), 'pnl': 0, 'es': 0, 'error': str(e)}
    
    added_markets = set()
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
        
        curr_prices = prices[t]
        prev_prices = prices[t-1]
        
        # Calculate returns
        valid = ~np.isnan(curr_prices) & ~np.isnan(prev_prices)
        returns = np.zeros(n_markets)
        returns[valid] = (curr_prices[valid] - prev_prices[valid]) / np.maximum(prev_prices[valid], 0.01)
        
        # Add new markets
        for m in range(n_markets):
            if m not in added_markets and m not in death_set and not np.isnan(curr_prices[m]):
                try:
                    algo.add_market(str(m), timestamp=float(t), initial_price=float(curr_prices[m]))
                    added_markets.add(m)
                except:
                    pass
        
        # Update clustering
        if t % recluster_every == 0:
            try:
                price_dict = {
                    str(m): float(curr_prices[m])
                    for m in range(n_markets)
                    if m not in death_set and not np.isnan(curr_prices[m]) and m in added_markets
                }
                if price_dict:
                    algo.update(float(t), price_dict)
            except:
                pass
        
        # Get clusters
        try:
            clusters = algo.get_clusters()
            n_clusters = len(clusters)
            n_clusters_list.append(n_clusters)
        except:
            clusters = {}
            n_clusters_list.append(1)
        
        # Cluster-aware mean-reversion
        active = [m for m in range(n_markets) if m not in death_set and not np.isnan(curr_prices[m])]
        
        if len(clusters) > 0:
            for cluster_id, members in clusters.items():
                member_indices = [int(m) for m in members if int(m) in active]
                if len(member_indices) >= 2:
                    cluster_returns = returns[member_indices]
                    cluster_mean = np.mean(cluster_returns)
                    for m in member_indices:
                        deviation = returns[m] - cluster_mean
                        positions[m] = -np.clip(deviation * 15, -1, 1)
        else:
            if len(active) > 1:
                active_returns = returns[active]
                mean_return = np.mean(active_returns)
                for m in active:
                    deviation = returns[m] - mean_return
                    positions[m] = -np.clip(deviation * 10, -1, 1)
        
        pnl = np.nansum(positions * returns)
        daily_pnl.append(pnl)
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Clustering optimization on real Polymarket data')
    parser.add_argument('--cache-path', type=str, 
                        default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--clob-dir', type=str,
                        default='data/polymarket/clob_history_yes_f1')
    parser.add_argument('--categories', nargs='+', 
                        default=['sports', 'politics', 'crypto', 'tech'])
    parser.add_argument('--max-markets', type=int, default=100,
                        help='Max markets per category')
    parser.add_argument('--output-dir', type=str, 
                        default='runs/clustering_real_data')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = Path(args.cache_path)
    clob_dir = Path(args.clob_dir)
    
    if not cache_path.exists():
        print(f"ERROR: Cache file not found at {cache_path}")
        return
    
    if not clob_dir.exists():
        print(f"ERROR: CLOB directory not found at {clob_dir}")
        return
    
    # Load metadata (filtering to markets with CLOB data)
    print("Loading market metadata (with CLOB data filter)...")
    metadata = load_market_metadata(cache_path, clob_dir)
    print(f"Loaded {len(metadata)} resolved markets with CLOB data")
    print(f"Categories: {metadata['category_mapped'].value_counts().to_dict()}")
    
    all_results = {}
    
    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Processing category: {category.upper()}")
        print(f"{'='*60}")
        
        # Filter markets by category
        cat_markets = metadata[metadata['category_mapped'] == category].copy()
        if len(cat_markets) == 0:
            print(f"No markets found for category {category}")
            continue
        
        # Sort by volume and take top markets
        cat_markets = cat_markets.sort_values('volumeNum', ascending=False).head(args.max_markets)
        print(f"Selected {len(cat_markets)} markets by volume")
        
        # Load prices
        print("Loading price data...")
        prices_dict = load_clob_prices(clob_dir, cat_markets)
        print(f"Loaded prices for {len(prices_dict)} markets")
        
        if len(prices_dict) < 5:
            print(f"Not enough price data for {category}")
            continue
        
        # Build price matrix
        try:
            price_matrix, market_ids, death_events, date_range = build_price_matrix(
                prices_dict, cat_markets
            )
            print(f"Price matrix: {price_matrix.shape[0]} days x {price_matrix.shape[1]} markets")
            print(f"Death events: {len(death_events)}")
        except Exception as e:
            print(f"Error building price matrix: {e}")
            continue
        
        # Run optimization
        hyperparams = HYPERPARAMS.get(category, HYPERPARAMS['tech'])
        results = {}
        
        for algo_name, configs in hyperparams.items():
            best_result = None
            best_sharpe = float('-inf')
            
            for config in configs:
                recluster = config.pop('recluster_every', config.pop('reassign_every', 10))
                result = run_backtest(algo_name, config, price_matrix, death_events, recluster)
                config['recluster_every'] = recluster
                
                if result['sharpe'] > best_sharpe:
                    best_sharpe = result['sharpe']
                    best_result = {'config': config.copy(), **result}
            
            if best_result:
                results[algo_name] = best_result
                print(f"  {algo_name:20s}: Sharpe={best_result['sharpe']:+.3f}, "
                      f"PnL=${best_result['pnl']:,.0f}, Clusters={best_result['avg_clusters']:.1f}")
        
        if results:
            best_algo = max(results.keys(), key=lambda a: results[a]['sharpe'])
            all_results[category] = {
                'best_algorithm': best_algo,
                'best_config': results[best_algo]['config'],
                'best_sharpe': results[best_algo]['sharpe'],
                'n_markets': len(market_ids),
                'n_days': price_matrix.shape[0],
                'all_results': results,
            }
    
    # Print summary
    print("\n" + "="*80)
    print("RECOMMENDED ALGORITHMS BY CATEGORY (REAL DATA)")
    print("="*80)
    print(f"{'Category':<15} {'Best Algorithm':<20} {'Sharpe':>10} {'Markets':>10}")
    print("-"*80)
    
    for cat, result in all_results.items():
        print(f"{cat:<15} {result['best_algorithm']:<20} {result['best_sharpe']:>+10.3f} {result['n_markets']:>10}")
    
    # Save results
    results_file = output_dir / 'real_data_optimization_results.json'
    
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


if __name__ == '__main__':
    main()
