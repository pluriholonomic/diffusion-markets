#!/usr/bin/env python3
"""
Hyperparameter Optimization and Ranking of Clustering Algorithms.

Trains clustering algorithms by backtesting on historical data and ranks them
by trading performance metrics: PnL, Sharpe Ratio, and Expected Shortfall.

Usage:
    python optimize_and_rank_clustering.py --data data/polymarket.parquet
    python optimize_and_rank_clustering.py --synthetic --n-markets 100 --n-days 500
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Iterator
from itertools import product
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecastbench.clustering import (
    SWOCC, OLRCM, BirthDeathHawkes, StreamingDPClustering,
    DynamicGraphAttention, EnsembleClustering,
)
from forecastbench.clustering.survival_weighted import SWOCCConfig
from forecastbench.clustering.online_factor import OLRCMConfig
from forecastbench.clustering.birth_death_hawkes import BDHPConfig
from forecastbench.clustering.streaming_dp import SDPMConfig
from forecastbench.clustering.dynamic_graph import DGATConfig

from forecastbench.clustering.generators import (
    BlockCorrelationGenerator, BlockCorrelationConfig,
    FactorModelGenerator, FactorModelConfig,
)


@dataclass
class TradingResult:
    """Results from a trading backtest."""
    algorithm: str
    config: Dict[str, Any]
    
    # PnL metrics
    total_pnl: float = 0.0
    mean_daily_pnl: float = 0.0
    std_daily_pnl: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Tail risk
    expected_shortfall_5: float = 0.0  # ES at 5% (CVaR)
    expected_shortfall_1: float = 0.0  # ES at 1%
    var_5: float = 0.0  # VaR at 5%
    var_1: float = 0.0  # VaR at 1%
    
    # Trade stats
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Cluster quality
    avg_n_clusters: float = 0.0
    cluster_stability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def composite_score(self) -> float:
        """
        Composite score balancing Sharpe, ES, and PnL.
        Higher is better.
        """
        # Normalize components
        sharpe_contrib = self.sharpe_ratio * 0.4
        es_contrib = -self.expected_shortfall_5 * 0.3  # Negative ES is better
        pnl_contrib = np.sign(self.total_pnl) * np.log1p(abs(self.total_pnl)) * 0.3
        
        return sharpe_contrib + es_contrib + pnl_contrib


def compute_expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute Expected Shortfall (CVaR) at level alpha.
    
    ES_alpha = E[X | X <= VaR_alpha]
    """
    if len(returns) == 0:
        return 0.0
    
    var = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]
    
    if len(tail) == 0:
        return var
    
    return np.mean(tail)


def compute_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute Value at Risk at level alpha."""
    if len(returns) == 0:
        return 0.0
    return np.percentile(returns, alpha * 100)


# ============================================================================
# Hyperparameter Grids
# ============================================================================

SWOCC_GRID = {
    "ema_alpha": [0.05, 0.1, 0.2, 0.3],
    "use_survival_weights": [True, False],
    "shrinkage": [0.0, 0.1, 0.2],
    "recluster_every": [5, 10, 20],
}

OLRCM_GRID = {
    "n_factors": [3, 5, 10, 15],
    "learning_rate": [0.005, 0.01, 0.02],
    "l2_reg": [0.0001, 0.001, 0.01],
    "recluster_every": [5, 10, 20],
}

BDHP_GRID = {
    "decay_rate": [0.05, 0.1, 0.2, 0.5],
    "base_intensity": [0.05, 0.1, 0.2],
    "self_excitation": [0.2, 0.3, 0.5],
    "price_change_threshold": [0.01, 0.02, 0.05],
}

SDPM_GRID = {
    "concentration": [0.5, 1.0, 2.0, 5.0],
    "prior_precision": [0.05, 0.1, 0.2],
    "temperature": [0.5, 1.0, 2.0],
    "reassign_every": [5, 10, 20],
}

DGAT_GRID = {
    "hidden_dim": [8, 16, 32],
    "n_attention_heads": [2, 4],
    "learning_rate": [0.005, 0.01, 0.02],
    "k_neighbors": [5, 10, 15],
}


def generate_configs(grid: Dict[str, List]) -> Iterator[Dict[str, Any]]:
    """Generate all combinations of hyperparameters."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    
    for combo in product(*values):
        yield dict(zip(keys, combo))


def sample_configs(grid: Dict[str, List], n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Randomly sample n configurations from grid."""
    rng = np.random.default_rng(seed)
    all_configs = list(generate_configs(grid))
    
    if len(all_configs) <= n_samples:
        return all_configs
    
    indices = rng.choice(len(all_configs), size=n_samples, replace=False)
    return [all_configs[i] for i in indices]


def create_algorithm(algo_name: str, config: Dict[str, Any]):
    """Create algorithm instance from name and config."""
    if algo_name == "SWOCC":
        return SWOCC(SWOCCConfig(**config))
    elif algo_name == "OLRCM":
        return OLRCM(OLRCMConfig(**config))
    elif algo_name == "BDHP":
        return BirthDeathHawkes(BDHPConfig(**config))
    elif algo_name == "SDPM":
        return StreamingDPClustering(SDPMConfig(**config))
    elif algo_name == "DGAT":
        return DynamicGraphAttention(DGATConfig(**config))
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ============================================================================
# Trading Strategy
# ============================================================================

def run_cluster_trading_backtest(
    algorithm,
    prices: np.ndarray,
    outcomes: np.ndarray,
    market_ids: List[str],
    death_events: List[Tuple[int, str]],
    # Trading params
    spread_threshold: float = 0.03,
    position_size: float = 100.0,
    fee_rate: float = 0.01,
    # Train/test split
    train_ratio: float = 0.6,
) -> TradingResult:
    """
    Run trading backtest with clustering-based strategy.
    
    Strategy:
    1. Mean reversion within clusters
    2. Trade when market deviates from cluster average
    """
    T, n_markets = prices.shape
    train_end = int(T * train_ratio)
    
    # Reset algorithm
    algorithm.reset()
    
    # Create death lookup
    death_lookup = {}
    death_times = {mid: T for mid in market_ids}
    for t, mid in death_events:
        death_lookup.setdefault(t, []).append(mid)
        death_times[mid] = t
    
    # Initialize markets
    for i, mid in enumerate(market_ids):
        initial_price = prices[0, i]
        if not np.isnan(initial_price):
            algorithm.add_market(mid, timestamp=0.0, initial_price=initial_price)
    
    # Training phase: fit clustering
    for t in range(1, train_end):
        active_prices = {}
        for i, mid in enumerate(market_ids):
            if mid in algorithm._markets and algorithm._markets[mid].is_active:
                p = prices[t, i]
                if not np.isnan(p):
                    active_prices[mid] = p
        
        if active_prices:
            algorithm.update(timestamp=float(t), prices=active_prices)
        
        for mid in death_lookup.get(t, []):
            algorithm.remove_market(mid, timestamp=float(t))
    
    # Testing phase: trade and evaluate
    daily_pnls = []
    trade_pnls = []
    cluster_counts = []
    
    id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
    
    for t in range(train_end, T):
        day_pnl = 0.0
        
        # Update clustering
        active_prices = {}
        for i, mid in enumerate(market_ids):
            if mid in algorithm._markets and algorithm._markets[mid].is_active:
                p = prices[t, i]
                if not np.isnan(p):
                    active_prices[mid] = p
        
        if active_prices:
            algorithm.update(timestamp=float(t), prices=active_prices)
        
        # Get clusters
        clusters = algorithm.get_clusters()
        cluster_counts.append(len(clusters))
        
        # Trade within each cluster
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            
            # Get cluster prices
            member_prices = []
            for mid in members:
                if mid in id_to_idx:
                    idx = id_to_idx[mid]
                    p = prices[t, idx]
                    if not np.isnan(p):
                        member_prices.append((mid, idx, p))
            
            if len(member_prices) < 2:
                continue
            
            # Cluster average
            avg_price = np.mean([p for _, _, p in member_prices])
            
            # Trade deviations
            for mid, idx, price in member_prices:
                spread = price - avg_price
                
                if abs(spread) < spread_threshold:
                    continue
                
                # Direction: bet against deviation (mean reversion)
                direction = -1 if spread > 0 else 1
                
                # Get outcome (if market resolved)
                outcome = outcomes[idx] if idx < len(outcomes) else price
                
                # PnL calculation
                if direction > 0:  # Long YES
                    gross_pnl = position_size * (outcome - price)
                else:  # Long NO
                    gross_pnl = position_size * (price - outcome)
                
                net_pnl = gross_pnl - fee_rate * position_size
                
                trade_pnls.append(net_pnl)
                day_pnl += net_pnl
        
        daily_pnls.append(day_pnl)
        
        # Handle deaths
        for mid in death_lookup.get(t, []):
            algorithm.remove_market(mid, timestamp=float(t))
    
    # Compute metrics
    daily_pnls = np.array(daily_pnls)
    trade_pnls = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    
    total_pnl = np.sum(daily_pnls)
    mean_daily = np.mean(daily_pnls) if len(daily_pnls) > 0 else 0
    std_daily = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-6
    
    # Sharpe (annualized)
    sharpe = mean_daily / (std_daily + 1e-8) * np.sqrt(252)
    
    # Sortino (downside deviation)
    downside = daily_pnls[daily_pnls < 0]
    downside_std = np.std(downside) if len(downside) > 0 else std_daily
    sortino = mean_daily / (downside_std + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    cumsum = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cumsum) if len(cumsum) > 0 else np.array([0])
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Expected Shortfall and VaR
    es_5 = compute_expected_shortfall(daily_pnls, 0.05)
    es_1 = compute_expected_shortfall(daily_pnls, 0.01)
    var_5 = compute_var(daily_pnls, 0.05)
    var_1 = compute_var(daily_pnls, 0.01)
    
    # Trade statistics
    n_trades = len(trade_pnls)
    wins = np.sum(trade_pnls > 0)
    win_rate = wins / n_trades if n_trades > 0 else 0
    
    gross_profit = np.sum(trade_pnls[trade_pnls > 0])
    gross_loss = abs(np.sum(trade_pnls[trade_pnls < 0]))
    profit_factor = gross_profit / (gross_loss + 1e-8)
    
    # Cluster stats
    avg_n_clusters = np.mean(cluster_counts) if cluster_counts else 0
    
    return TradingResult(
        algorithm=algorithm.__class__.__name__,
        config={},  # Will be set by caller
        total_pnl=float(total_pnl),
        mean_daily_pnl=float(mean_daily),
        std_daily_pnl=float(std_daily),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        expected_shortfall_5=float(es_5),
        expected_shortfall_1=float(es_1),
        var_5=float(var_5),
        var_1=float(var_1),
        n_trades=n_trades,
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        avg_n_clusters=float(avg_n_clusters),
    )


# ============================================================================
# Optimization and Ranking
# ============================================================================

def optimize_algorithm(
    algo_name: str,
    grid: Dict[str, List],
    prices: np.ndarray,
    outcomes: np.ndarray,
    market_ids: List[str],
    death_events: List[Tuple[int, str]],
    n_configs: int = 20,
    seed: int = 42,
) -> Tuple[Dict[str, Any], TradingResult, List[TradingResult]]:
    """
    Optimize hyperparameters for an algorithm.
    
    Returns:
        best_config: Best hyperparameters
        best_result: Best trading result
        all_results: All results for analysis
    """
    configs = sample_configs(grid, n_configs, seed)
    
    results = []
    for config in configs:
        try:
            algo = create_algorithm(algo_name, config)
            result = run_cluster_trading_backtest(
                algorithm=algo,
                prices=prices,
                outcomes=outcomes,
                market_ids=market_ids,
                death_events=death_events,
            )
            result.config = config
            results.append(result)
        except Exception as e:
            print(f"  Config failed: {e}")
            continue
    
    if not results:
        return {}, TradingResult(algorithm=algo_name, config={}), []
    
    # Rank by composite score
    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    return results[0].config, results[0], results


def run_full_optimization(
    prices: np.ndarray,
    outcomes: np.ndarray,
    market_ids: List[str],
    death_events: List[Tuple[int, str]],
    n_configs_per_algo: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full optimization across all algorithms.
    """
    algorithm_grids = {
        "SWOCC": SWOCC_GRID,
        "OLRCM": OLRCM_GRID,
        "BDHP": BDHP_GRID,
        "SDPM": SDPM_GRID,
        "DGAT": DGAT_GRID,
    }
    
    all_best_results = []
    detailed_results = {}
    
    for algo_name, grid in algorithm_grids.items():
        print(f"\nOptimizing {algo_name}...")
        
        best_config, best_result, all_results = optimize_algorithm(
            algo_name=algo_name,
            grid=grid,
            prices=prices,
            outcomes=outcomes,
            market_ids=market_ids,
            death_events=death_events,
            n_configs=n_configs_per_algo,
            seed=seed,
        )
        
        all_best_results.append(best_result)
        detailed_results[algo_name] = {
            "best_config": best_config,
            "best_result": best_result.to_dict(),
            "all_results": [r.to_dict() for r in all_results],
        }
        
        print(f"  Best config: {best_config}")
        print(f"  Sharpe: {best_result.sharpe_ratio:.3f}")
        print(f"  Total PnL: ${best_result.total_pnl:,.0f}")
        print(f"  ES(5%): ${best_result.expected_shortfall_5:,.2f}")
    
    # Global ranking
    all_best_results.sort(key=lambda r: r.composite_score, reverse=True)
    
    return {
        "ranking": [r.to_dict() for r in all_best_results],
        "detailed": detailed_results,
    }


def generate_synthetic_data(
    n_markets: int = 100,
    n_days: int = 500,
    n_clusters: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, str]]]:
    """Generate synthetic data for testing."""
    config = BlockCorrelationConfig(
        n_clusters=n_clusters,
        markets_per_cluster=n_markets // n_clusters,
        n_timesteps=n_days,
        intra_cluster_corr=0.7,
        inter_cluster_corr=0.1,
        death_rate=0.005,
        death_correlation=True,
        seed=seed,
    )
    gen = BlockCorrelationGenerator(config)
    prices, death_events, labels, corr = gen.generate()
    market_ids = gen.get_market_ids()
    
    # Generate outcomes (final prices for resolved markets, simulated for others)
    rng = np.random.default_rng(seed + 1)
    outcomes = np.zeros(len(market_ids))
    
    death_market_ids = set(mid for _, mid in death_events)
    for i, mid in enumerate(market_ids):
        if mid in death_market_ids:
            # Use final valid price as indication
            final_price = prices[~np.isnan(prices[:, i]), i][-1] if any(~np.isnan(prices[:, i])) else 0.5
            # Binary outcome based on price
            outcomes[i] = rng.binomial(1, final_price)
        else:
            # Simulated outcome for non-resolved
            outcomes[i] = rng.binomial(1, prices[-1, i])
    
    return prices, outcomes, market_ids, death_events


def print_ranking_table(results: Dict[str, Any]) -> None:
    """Print formatted ranking table."""
    ranking = results["ranking"]
    
    print("\n" + "=" * 100)
    print("ALGORITHM RANKING BY COMPOSITE SCORE")
    print("=" * 100)
    print(f"{'Rank':<6} {'Algorithm':<12} {'Sharpe':>10} {'Total PnL':>14} {'ES(5%)':>12} "
          f"{'Win Rate':>10} {'Score':>10}")
    print("-" * 100)
    
    for i, r in enumerate(ranking, 1):
        score = (r['sharpe_ratio'] * 0.4 - r['expected_shortfall_5'] * 0.3 + 
                 np.sign(r['total_pnl']) * np.log1p(abs(r['total_pnl'])) * 0.3)
        
        print(f"{i:<6} {r['algorithm']:<12} {r['sharpe_ratio']:>10.3f} "
              f"${r['total_pnl']:>13,.0f} ${r['expected_shortfall_5']:>11,.2f} "
              f"{r['win_rate']:>10.1%} {score:>10.3f}")
    
    print("\n" + "=" * 100)
    print("DETAILED METRICS")
    print("=" * 100)
    
    for r in ranking:
        print(f"\n{r['algorithm']}:")
        print(f"  Best Config: {r.get('config', 'N/A')}")
        print(f"  PnL:           ${r['total_pnl']:>12,.2f}")
        print(f"  Sharpe:        {r['sharpe_ratio']:>12.3f}")
        print(f"  Sortino:       {r['sortino_ratio']:>12.3f}")
        print(f"  Max Drawdown:  ${r['max_drawdown']:>12,.2f}")
        print(f"  VaR(5%):       ${r['var_5']:>12,.2f}")
        print(f"  ES(5%):        ${r['expected_shortfall_5']:>12,.2f}")
        print(f"  VaR(1%):       ${r['var_1']:>12,.2f}")
        print(f"  ES(1%):        ${r['expected_shortfall_1']:>12,.2f}")
        print(f"  Trades:        {r['n_trades']:>12}")
        print(f"  Win Rate:      {r['win_rate']:>12.1%}")
        print(f"  Profit Factor: {r['profit_factor']:>12.2f}")
        print(f"  Avg Clusters:  {r['avg_n_clusters']:>12.1f}")


def main():
    parser = argparse.ArgumentParser(description='Optimize and Rank Clustering Algorithms')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to Polymarket data (parquet)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--n-markets', type=int, default=100,
                        help='Number of markets for synthetic data')
    parser.add_argument('--n-days', type=int, default=500,
                        help='Number of days for synthetic data')
    parser.add_argument('--n-configs', type=int, default=15,
                        help='Number of configs to try per algorithm')
    parser.add_argument('--output-dir', type=str, default='runs/clustering_optimization',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    if args.data and Path(args.data).exists():
        print(f"Loading data from {args.data}...")
        df = pd.read_parquet(args.data)
        
        # Prepare data
        price_col = 'avg_price' if 'avg_price' in df.columns else 'price'
        outcome_col = 'y' if 'y' in df.columns else 'outcome'
        
        # TODO: Convert to time series format
        # This would require proper data formatting
        raise NotImplementedError("Real data loading needs proper time series format")
    else:
        print("Generating synthetic data...")
        prices, outcomes, market_ids, death_events = generate_synthetic_data(
            n_markets=args.n_markets,
            n_days=args.n_days,
            seed=args.seed,
        )
        print(f"Generated: {prices.shape[1]} markets, {prices.shape[0]} days, {len(death_events)} deaths")
    
    # Run optimization
    print("\nStarting optimization...")
    results = run_full_optimization(
        prices=prices,
        outcomes=outcomes,
        market_ids=market_ids,
        death_events=death_events,
        n_configs_per_algo=args.n_configs,
        seed=args.seed,
    )
    
    # Print ranking
    print_ranking_table(results)
    
    # Save results
    results_file = output_dir / 'optimization_results.json'
    
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
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save best configs
    algo_name_map = {
        'SWOCC': 'SWOCC',
        'OLRCM': 'OLRCM', 
        'BirthDeathHawkes': 'BDHP',
        'StreamingDPClustering': 'SDPM',
        'DynamicGraphAttention': 'DGAT',
    }
    best_configs = {
        r['algorithm']: results['detailed'][algo_name_map.get(r['algorithm'], r['algorithm'])]['best_config']
        for r in results['ranking']
        if algo_name_map.get(r['algorithm'], r['algorithm']) in results['detailed']
    }
    
    with open(output_dir / 'best_configs.json', 'w') as f:
        json.dump(convert_types(best_configs), f, indent=2)
    
    print(f"Best configs saved to {output_dir / 'best_configs.json'}")


if __name__ == '__main__':
    main()
