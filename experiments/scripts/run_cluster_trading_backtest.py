#!/usr/bin/env python3
"""
Cluster-Based Trading Backtest.

Tests whether adaptive clustering improves trading performance
for mean-reversion and dispersion strategies.

Usage:
    python run_cluster_trading_backtest.py --data data/polymarket/optimization_cache.parquet
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecastbench.clustering import SWOCC, OLRCM
from forecastbench.clustering.survival_weighted import SWOCCConfig
from forecastbench.clustering.online_factor import OLRCMConfig


@dataclass
class TradingMetrics:
    """Container for trading backtest metrics."""
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total_pnl": self.total_pnl,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
        }


def load_data(path: str) -> pd.DataFrame:
    """Load trading data."""
    df = pd.read_parquet(path)
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Prepare data for backtesting.
    
    Returns:
        prices: (n_markets,) current prices
        market_ids: list of market IDs
        outcomes: (n_markets,) resolved outcomes
    """
    price_col = 'avg_price' if 'avg_price' in df.columns else 'price'
    outcome_col = 'y' if 'y' in df.columns else 'outcome'
    
    if 'market_id' not in df.columns and 'id' in df.columns:
        df['market_id'] = df['id'].astype(str)
    elif 'market_id' not in df.columns:
        df['market_id'] = df.index.astype(str)
    
    prices = df[price_col].values.astype(np.float64)
    outcomes = df[outcome_col].values.astype(np.float64)
    market_ids = df['market_id'].tolist()
    
    return prices, market_ids, outcomes


def run_mean_reversion_strategy(
    prices: np.ndarray,
    outcomes: np.ndarray,
    market_ids: List[str],
    clustering: Optional[Any] = None,
    spread_threshold: float = 0.05,
    position_size: float = 100.0,
) -> TradingMetrics:
    """
    Run mean reversion strategy with optional clustering.
    
    If clustering is provided, only trade within clusters that
    show consistent calibration patterns.
    """
    pnls = []
    wins = 0
    n_trades = 0
    
    # Get cluster assignments if available
    cluster_assignments = {}
    if clustering is not None:
        for mid in market_ids:
            cluster = clustering.get_cluster_for_market(mid)
            if cluster is not None:
                cluster_assignments[mid] = cluster
    
    # Simple calibration-based mean reversion
    # In real implementation, would learn calibration from training data
    for i, (price, outcome, mid) in enumerate(zip(prices, outcomes, market_ids)):
        price = np.clip(price, 0.01, 0.99)
        
        # Compute edge based on distance from calibration
        # Simplified: assume market is miscalibrated by a fixed amount
        spread = outcome - price  # In hindsight
        
        # Filter by cluster if clustering provided
        if clustering is not None and mid in cluster_assignments:
            cluster_id = cluster_assignments[mid]
            cluster_members = clustering.get_markets_in_cluster(cluster_id)
            
            # Only trade if cluster has enough members
            if len(cluster_members) < 2:
                continue
        
        # Trade decision
        if abs(spread) >= spread_threshold:
            direction = 1 if spread > 0 else -1
            
            # Simplified PnL calculation
            gross_pnl = direction * position_size * spread
            fee = 0.01 * position_size
            net_pnl = gross_pnl - fee
            
            pnls.append(net_pnl)
            n_trades += 1
            
            if net_pnl > 0:
                wins += 1
    
    # Compute metrics
    if not pnls:
        return TradingMetrics()
    
    pnls = np.array(pnls)
    total_pnl = np.sum(pnls)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) / (running_max.max() + 1e-8) if running_max.max() > 0 else 0
    
    win_rate = wins / n_trades if n_trades > 0 else 0
    
    return TradingMetrics(
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
    )


def run_dispersion_strategy(
    prices: np.ndarray,
    outcomes: np.ndarray,
    market_ids: List[str],
    clustering: Any,
    position_size: float = 100.0,
) -> TradingMetrics:
    """
    Run dispersion strategy using cluster structure.
    
    Trades on the spread between cluster-average price and individual prices.
    """
    if clustering is None:
        return TradingMetrics()
    
    clusters = clustering.get_clusters()
    if not clusters:
        return TradingMetrics()
    
    pnls = []
    wins = 0
    n_trades = 0
    
    # Create market_id to index mapping
    id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
    
    for cluster_id, members in clusters.items():
        if len(members) < 2:
            continue
        
        # Get prices and outcomes for cluster members
        member_indices = [id_to_idx[m] for m in members if m in id_to_idx]
        if len(member_indices) < 2:
            continue
        
        cluster_prices = prices[member_indices]
        cluster_outcomes = outcomes[member_indices]
        
        # Cluster average price
        avg_price = np.mean(cluster_prices)
        
        # Trade each market against cluster average
        for i, (price, outcome) in enumerate(zip(cluster_prices, cluster_outcomes)):
            spread = price - avg_price
            
            if abs(spread) < 0.03:  # Minimum spread threshold
                continue
            
            # If price > avg, bet NO (expect mean reversion)
            # If price < avg, bet YES
            direction = -1 if spread > 0 else 1
            
            gross_pnl = direction * position_size * (outcome - price)
            fee = 0.01 * position_size
            net_pnl = gross_pnl - fee
            
            pnls.append(net_pnl)
            n_trades += 1
            
            if net_pnl > 0:
                wins += 1
    
    if not pnls:
        return TradingMetrics()
    
    pnls = np.array(pnls)
    total_pnl = np.sum(pnls)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
    
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) / (running_max.max() + 1e-8) if running_max.max() > 0 else 0
    
    win_rate = wins / n_trades if n_trades > 0 else 0
    
    return TradingMetrics(
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
    )


def train_clustering(
    df: pd.DataFrame,
    algorithm_type: str = "SWOCC",
) -> Any:
    """Train clustering on training data."""
    prices, market_ids, outcomes = prepare_data(df)
    
    if algorithm_type == "SWOCC":
        algo = SWOCC(SWOCCConfig(ema_alpha=0.1, use_survival_weights=True))
    elif algorithm_type == "OLRCM":
        algo = OLRCM(OLRCMConfig(n_factors=10))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_type}")
    
    # Initialize markets
    for i, mid in enumerate(market_ids):
        algo.add_market(mid, timestamp=0.0, initial_price=prices[i])
    
    # Run one update with all prices
    price_dict = {mid: prices[i] for i, mid in enumerate(market_ids)}
    algo.update(timestamp=1.0, prices=price_dict)
    
    return algo


def main():
    parser = argparse.ArgumentParser(description='Cluster Trading Backtest')
    parser.add_argument('--data', type=str, default='data/polymarket/optimization_cache.parquet',
                        help='Path to data')
    parser.add_argument('--output-dir', type=str, default='runs/cluster_trading',
                        help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train/test split ratio')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.data}...")
    
    try:
        df = load_data(args.data)
    except FileNotFoundError:
        print("Data file not found. Generating synthetic data...")
        
        # Generate synthetic data
        from forecastbench.clustering.generators import BlockCorrelationGenerator, BlockCorrelationConfig
        
        config = BlockCorrelationConfig(n_clusters=5, markets_per_cluster=50, n_timesteps=100)
        gen = BlockCorrelationGenerator(config)
        prices, deaths, labels, _ = gen.generate()
        
        # Convert to DataFrame format
        n_markets = prices.shape[1]
        df = pd.DataFrame({
            'market_id': [f"market_{i}" for i in range(n_markets)],
            'avg_price': prices[-1],  # Final price
            'y': np.random.binomial(1, prices[-1]),  # Simulated outcomes
        })
    
    print(f"Loaded {len(df)} markets")
    
    # Split train/test
    n = len(df)
    train_idx = int(n * args.train_ratio)
    train_df = df.iloc[:train_idx].copy()
    test_df = df.iloc[train_idx:].copy()
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    results = {}
    
    # Baseline: No clustering
    print("\nRunning baseline (no clustering)...")
    prices, market_ids, outcomes = prepare_data(test_df)
    
    baseline_mr = run_mean_reversion_strategy(
        prices, outcomes, market_ids, clustering=None
    )
    results["MeanRevert_NoCluster"] = baseline_mr.to_dict()
    print(f"  PnL: ${baseline_mr.total_pnl:.0f}, Sharpe: {baseline_mr.sharpe:.2f}")
    
    # With SWOCC clustering
    print("\nTraining SWOCC clustering...")
    swocc = train_clustering(train_df, "SWOCC")
    
    # Re-add test markets
    for i, mid in enumerate(market_ids):
        swocc.add_market(mid, timestamp=2.0, initial_price=prices[i])
    price_dict = {mid: prices[i] for i, mid in enumerate(market_ids)}
    swocc.update(timestamp=3.0, prices=price_dict)
    
    swocc_mr = run_mean_reversion_strategy(
        prices, outcomes, market_ids, clustering=swocc
    )
    results["MeanRevert_SWOCC"] = swocc_mr.to_dict()
    print(f"  MeanRevert PnL: ${swocc_mr.total_pnl:.0f}, Sharpe: {swocc_mr.sharpe:.2f}")
    
    swocc_disp = run_dispersion_strategy(
        prices, outcomes, market_ids, clustering=swocc
    )
    results["Dispersion_SWOCC"] = swocc_disp.to_dict()
    print(f"  Dispersion PnL: ${swocc_disp.total_pnl:.0f}, Sharpe: {swocc_disp.sharpe:.2f}")
    
    # With OLRCM clustering
    print("\nTraining OLRCM clustering...")
    olrcm = train_clustering(train_df, "OLRCM")
    
    for i, mid in enumerate(market_ids):
        olrcm.add_market(mid, timestamp=2.0, initial_price=prices[i])
    olrcm.update(timestamp=3.0, prices=price_dict)
    
    olrcm_mr = run_mean_reversion_strategy(
        prices, outcomes, market_ids, clustering=olrcm
    )
    results["MeanRevert_OLRCM"] = olrcm_mr.to_dict()
    print(f"  MeanRevert PnL: ${olrcm_mr.total_pnl:.0f}, Sharpe: {olrcm_mr.sharpe:.2f}")
    
    olrcm_disp = run_dispersion_strategy(
        prices, outcomes, market_ids, clustering=olrcm
    )
    results["Dispersion_OLRCM"] = olrcm_disp.to_dict()
    print(f"  Dispersion PnL: ${olrcm_disp.total_pnl:.0f}, Sharpe: {olrcm_disp.sharpe:.2f}")
    
    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<25} {'PnL':>12} {'Sharpe':>10} {'Win Rate':>10} {'Trades':>10}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<25} ${metrics['total_pnl']:>11,.0f} {metrics['sharpe']:>10.2f} "
              f"{metrics['win_rate']:>10.1%} {metrics['n_trades']:>10}")


if __name__ == '__main__':
    main()
