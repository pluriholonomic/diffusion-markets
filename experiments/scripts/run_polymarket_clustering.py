#!/usr/bin/env python3
"""
Polymarket Cluster Discovery Experiment.

Runs clustering algorithms on historical Polymarket data to:
1. Discover natural clusters
2. Validate cluster quality metrics
3. Test if intra-cluster correlation predicts future co-movements

Usage:
    python run_polymarket_clustering.py --data data/polymarket/resolved_markets.parquet
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecastbench.clustering import (
    SWOCC,
    OLRCM,
    BirthDeathHawkes,
    StreamingDPClustering,
    ClusteringEvaluator,
)
from forecastbench.clustering.survival_weighted import SWOCCConfig
from forecastbench.clustering.online_factor import OLRCMConfig
from forecastbench.clustering.birth_death_hawkes import BDHPConfig
from forecastbench.clustering.streaming_dp import SDPMConfig


def load_polymarket_data(path: str) -> pd.DataFrame:
    """Load Polymarket data from parquet file."""
    df = pd.read_parquet(path)
    
    # Ensure required columns
    required = ['market_id', 'timestamp', 'price']
    for col in required:
        if col not in df.columns:
            # Try alternative names
            if col == 'market_id' and 'id' in df.columns:
                df['market_id'] = df['id']
            elif col == 'price' and 'avg_price' in df.columns:
                df['price'] = df['avg_price']
            elif col == 'timestamp' and 'createdAt' in df.columns:
                df['timestamp'] = pd.to_datetime(df['createdAt'])
    
    return df


def prepare_price_matrix(
    df: pd.DataFrame,
    time_resolution: str = 'D',
) -> tuple:
    """
    Convert long-form data to price matrix.
    
    Returns:
        prices: (T, n_markets) array
        market_ids: list of market IDs
        death_events: list of (timestep, market_id)
    """
    # Pivot to wide format
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.floor(time_resolution)
    
    pivot = df.pivot_table(
        index='date',
        columns='market_id',
        values='price',
        aggfunc='last'
    )
    
    market_ids = list(pivot.columns)
    prices = pivot.values
    
    # Forward fill then replace remaining NaN with 0.5
    prices = pd.DataFrame(prices).ffill().fillna(0.5).values
    
    # Detect death events (last non-NaN value)
    death_events = []
    for i, mid in enumerate(market_ids):
        last_valid = pivot[mid].last_valid_index()
        if last_valid is not None:
            t = list(pivot.index).index(last_valid)
            if t < len(pivot) - 1:  # Only if not at end
                death_events.append((t, mid))
    
    return prices, market_ids, death_events


def run_clustering_experiment(
    prices: np.ndarray,
    market_ids: List[str],
    death_events: List[tuple],
    output_dir: Path,
    category_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run all clustering algorithms and evaluate.
    """
    # Initialize algorithms
    algorithms = {
        "SWOCC": SWOCC(SWOCCConfig(
            ema_alpha=0.1,
            use_survival_weights=True,
            recluster_every=5,
        )),
        "SWOCC_NoSurvival": SWOCC(SWOCCConfig(
            ema_alpha=0.1,
            use_survival_weights=False,
            recluster_every=5,
        )),
        "OLRCM": OLRCM(OLRCMConfig(
            n_factors=10,
            recluster_every=5,
        )),
        "BDHP": BirthDeathHawkes(BDHPConfig(
            decay_rate=0.1,
            recluster_every=5,
        )),
        "SDPM": StreamingDPClustering(SDPMConfig(
            concentration=1.0,
            reassign_every=5,
        )),
    }
    
    evaluator = ClusteringEvaluator()
    results = {}
    
    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        
        try:
            metrics = evaluator.evaluate_clustering(
                algorithm=algo,
                price_data=prices,
                death_events=death_events,
                market_ids=market_ids,
                true_labels=category_labels,
            )
            
            results[name] = metrics.to_dict()
            
            print(f"  Clusters: {metrics.n_clusters}")
            print(f"  Intra-cluster corr: {metrics.intra_cluster_corr:.3f}")
            print(f"  Inter-cluster corr: {metrics.inter_cluster_corr:.3f}")
            print(f"  Stability: {metrics.cluster_stability:.3f}")
            print(f"  Update time: {metrics.update_time_ms:.2f} ms")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Polymarket Clustering Experiment')
    parser.add_argument('--data', type=str, default='data/polymarket/resolved_markets.parquet',
                        help='Path to Polymarket data')
    parser.add_argument('--output-dir', type=str, default='runs/clustering',
                        help='Output directory')
    parser.add_argument('--time-resolution', type=str, default='H',
                        help='Time resolution (D=daily, H=hourly)')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{timestamp}_polymarket"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.data}...")
    
    try:
        df = load_polymarket_data(args.data)
        print(f"Loaded {len(df)} rows, {df['market_id'].nunique()} markets")
        
        prices, market_ids, death_events = prepare_price_matrix(df, args.time_resolution)
        print(f"Price matrix: {prices.shape}")
        print(f"Death events: {len(death_events)}")
        
        # Extract category labels if available
        category_labels = None
        if 'category' in df.columns:
            categories = df.groupby('market_id')['category'].first()
            category_map = {cat: i for i, cat in enumerate(categories.unique())}
            category_labels = np.array([category_map[categories.get(mid, 'unknown')] for mid in market_ids])
        
        # Run experiment
        results = run_clustering_experiment(
            prices, market_ids, death_events, output_dir, category_labels
        )
        
    except FileNotFoundError:
        print(f"Data file not found: {args.data}")
        print("Running on synthetic data instead...")
        
        # Generate synthetic data for testing
        from forecastbench.clustering.generators import BlockCorrelationGenerator, BlockCorrelationConfig
        
        config = BlockCorrelationConfig(
            n_clusters=5,
            markets_per_cluster=20,
            n_timesteps=500,
            intra_cluster_corr=0.7,
            inter_cluster_corr=0.1,
            death_rate=0.01,
        )
        gen = BlockCorrelationGenerator(config)
        prices, death_events, labels, true_corr = gen.generate()
        market_ids = gen.get_market_ids()
        
        print(f"Generated synthetic data: {prices.shape}")
        
        results = run_clustering_experiment(
            prices, market_ids, death_events, output_dir, labels
        )
        
        # Add synthetic data info
        results["_synthetic"] = True
        results["_true_labels_available"] = True
    
    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Clusters':>10} {'Intra':>10} {'Inter':>10} {'Ratio':>10}")
    print("-" * 60)
    
    for name, metrics in results.items():
        if name.startswith('_') or 'error' in metrics:
            continue
        
        intra = metrics.get('intra_cluster_corr', 0)
        inter = metrics.get('inter_cluster_corr', 0)
        ratio = intra / inter if inter > 0.01 else float('inf')
        
        print(f"{name:<20} {metrics.get('n_clusters', 0):>10} {intra:>10.3f} {inter:>10.3f} {ratio:>10.2f}")


if __name__ == '__main__':
    main()
