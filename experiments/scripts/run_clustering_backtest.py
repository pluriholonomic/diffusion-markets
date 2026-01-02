#!/usr/bin/env python3
"""
Comprehensive Clustering Algorithm Backtest.

Runs all clustering algorithms on synthetic and real data,
computes metrics, and generates comparison reports.

Usage:
    python run_clustering_backtest.py --synthetic --n-trials 10
    python run_clustering_backtest.py --data path/to/data.parquet
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore')

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
from forecastbench.clustering.dynamic_graph import DynamicGraphAttention, DGATConfig

from forecastbench.clustering.generators import (
    BlockCorrelationGenerator,
    BlockCorrelationConfig,
    FactorModelGenerator,
    FactorModelConfig,
    RegimeSwitchingGenerator,
    RegimeSwitchingConfig,
)


def create_algorithms() -> Dict[str, Any]:
    """Create instances of all clustering algorithms."""
    return {
        "SWOCC": SWOCC(SWOCCConfig(ema_alpha=0.1, use_survival_weights=True)),
        "SWOCC_NoSurvival": SWOCC(SWOCCConfig(ema_alpha=0.1, use_survival_weights=False)),
        "OLRCM_5": OLRCM(OLRCMConfig(n_factors=5)),
        "OLRCM_10": OLRCM(OLRCMConfig(n_factors=10)),
        "BDHP": BirthDeathHawkes(BDHPConfig(decay_rate=0.1)),
        "SDPM": StreamingDPClustering(SDPMConfig(concentration=1.0)),
        "DGAT": DynamicGraphAttention(DGATConfig(hidden_dim=16)),
    }


def run_synthetic_experiments(
    n_trials: int = 10,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run experiments on synthetic data with known ground truth.
    """
    print("=" * 60)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("=" * 60)
    
    evaluator = ClusteringEvaluator()
    all_results = {}
    
    # Experiment 1: Block Correlation
    print("\n1. Block Correlation Structure")
    print("-" * 40)
    
    block_results = {name: [] for name in create_algorithms().keys()}
    
    for trial in range(n_trials):
        config = BlockCorrelationConfig(
            n_clusters=5,
            markets_per_cluster=15,
            intra_cluster_corr=0.7,
            inter_cluster_corr=0.1,
            n_timesteps=300,
            death_rate=0.01,
            seed=42 + trial,
        )
        gen = BlockCorrelationGenerator(config)
        prices, deaths, labels, true_corr = gen.generate()
        market_ids = gen.get_market_ids()
        
        algorithms = create_algorithms()
        
        for name, algo in algorithms.items():
            metrics = evaluator.evaluate_clustering(
                algorithm=algo,
                price_data=prices,
                death_events=deaths,
                market_ids=market_ids,
                true_labels=labels,
                true_correlation=true_corr,
            )
            block_results[name].append(metrics.to_dict())
    
    # Aggregate block results
    all_results["block_correlation"] = {}
    for name, metrics_list in block_results.items():
        all_results["block_correlation"][name] = {
            key: {
                "mean": np.mean([m[key] for m in metrics_list]),
                "std": np.std([m[key] for m in metrics_list]),
            }
            for key in metrics_list[0].keys()
        }
        print(f"  {name}: ARI={all_results['block_correlation'][name]['ari']['mean']:.3f} ± "
              f"{all_results['block_correlation'][name]['ari']['std']:.3f}")
    
    # Experiment 2: Factor Model
    print("\n2. Factor Model Structure")
    print("-" * 40)
    
    factor_results = {name: [] for name in create_algorithms().keys()}
    
    for trial in range(n_trials):
        config = FactorModelConfig(
            n_factors=5,
            markets_per_factor=15,
            n_timesteps=300,
            seed=42 + trial,
        )
        gen = FactorModelGenerator(config)
        prices, deaths, labels, loadings = gen.generate()
        market_ids = [f"market_{i}" for i in range(prices.shape[1])]
        true_corr = gen.get_correlation_matrix(loadings)
        
        algorithms = create_algorithms()
        
        for name, algo in algorithms.items():
            metrics = evaluator.evaluate_clustering(
                algorithm=algo,
                price_data=prices,
                death_events=deaths,
                market_ids=market_ids,
                true_labels=labels,
                true_correlation=true_corr,
            )
            factor_results[name].append(metrics.to_dict())
    
    all_results["factor_model"] = {}
    for name, metrics_list in factor_results.items():
        if not metrics_list:
            continue
        all_results["factor_model"][name] = {
            key: {
                "mean": np.mean([m[key] for m in metrics_list]),
                "std": np.std([m[key] for m in metrics_list]),
            }
            for key in metrics_list[0].keys()
        }
        print(f"  {name}: ARI={all_results['factor_model'][name]['ari']['mean']:.3f}")
    
    # Experiment 3: Regime Switching
    print("\n3. Regime Switching Structure")
    print("-" * 40)
    
    regime_results = {name: [] for name in create_algorithms().keys()}
    
    for trial in range(n_trials):
        config = RegimeSwitchingConfig(
            n_markets=50,
            n_regimes=2,
            regime_persistence=0.95,
            n_timesteps=300,
            seed=42 + trial,
        )
        gen = RegimeSwitchingGenerator(config)
        prices, deaths, regimes, corr_matrices = gen.generate()
        market_ids = [f"market_{i}" for i in range(prices.shape[1])]
        
        # Use final regime's correlation as ground truth
        final_regime = regimes[-1]
        true_corr = corr_matrices[final_regime]
        labels = gen.get_cluster_labels_for_regime(final_regime, corr_matrices)
        
        algorithms = create_algorithms()
        
        for name, algo in algorithms.items():
            metrics = evaluator.evaluate_clustering(
                algorithm=algo,
                price_data=prices,
                death_events=deaths,
                market_ids=market_ids,
                true_labels=labels,
                true_correlation=true_corr,
            )
            regime_results[name].append(metrics.to_dict())
    
    all_results["regime_switching"] = {}
    for name, metrics_list in regime_results.items():
        if not metrics_list:
            continue
        all_results["regime_switching"][name] = {
            key: {
                "mean": np.mean([m[key] for m in metrics_list]),
                "std": np.std([m[key] for m in metrics_list]),
            }
            for key in metrics_list[0].keys()
        }
        print(f"  {name}: ARI={all_results['regime_switching'][name]['ari']['mean']:.3f}")
    
    return all_results


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Mean ARI Across Experiments")
    print("=" * 80)
    
    algo_names = list(create_algorithms().keys())
    exp_names = list(results.keys())
    
    # Header
    header = f"{'Algorithm':<20}"
    for exp in exp_names:
        header += f"{exp[:15]:>15}"
    print(header)
    print("-" * 80)
    
    # Rows
    for algo in algo_names:
        row = f"{algo:<20}"
        for exp in exp_names:
            if algo in results[exp]:
                ari = results[exp][algo].get('ari', {}).get('mean', 0)
                row += f"{ari:>15.3f}"
            else:
                row += f"{'N/A':>15}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Clustering Backtest')
    parser.add_argument('--synthetic', action='store_true',
                        help='Run synthetic experiments')
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Number of trials per experiment')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to real data (optional)')
    parser.add_argument('--output-dir', type=str, default='runs/clustering_backtest',
                        help='Output directory')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if args.synthetic or args.data is None:
        synthetic_results = run_synthetic_experiments(
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
        results.update(synthetic_results)
    
    if args.data:
        print("\n" + "=" * 60)
        print("REAL DATA EXPERIMENTS")
        print("=" * 60)
        print("(To be implemented with real data loading)")
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    results_file = output_dir / 'results.json'
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
