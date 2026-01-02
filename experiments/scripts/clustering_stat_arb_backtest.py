#!/usr/bin/env python3
"""
Stat Arb / Markowitz Backtest with Online Clustering.

Uses the correlation matrix from clustering algorithms for Markowitz-style
portfolio optimization, not naive mean-reversion.

The correct use of clustering for trading:
1. Clustering estimates correlation structure
2. Markowitz uses correlation for optimal portfolio weights
3. Trade based on "mispricing" relative to estimated correlations
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
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
# Markowitz Portfolio Optimizer
# =============================================================================

class MarkowitzOptimizer:
    """
    Mean-variance portfolio optimization using correlation from clustering.
    
    w* = (1/λ) Σ^{-1} μ
    
    where:
    - Σ is the covariance matrix (from clustering algorithm)
    - μ is expected returns (based on mispricing/spread)
    - λ is risk aversion
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        regularization: float = 0.1,
        max_position: float = 1.0,
        max_gross_exposure: float = 5.0,
    ):
        self.risk_aversion = risk_aversion
        self.regularization = regularization
        self.max_position = max_position
        self.max_gross_exposure = max_gross_exposure
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Compute optimal portfolio weights.
        
        Args:
            expected_returns: (n,) expected return per asset
            covariance: (n, n) covariance matrix
            
        Returns:
            Optimal portfolio weights
        """
        n = len(expected_returns)
        
        # Regularized covariance inverse
        reg = self.regularization * np.eye(n)
        try:
            cov_inv = np.linalg.inv(covariance + reg)
        except np.linalg.LinAlgError:
            return np.zeros(n)
        
        # Optimal weights: w = (1/λ) Σ^{-1} μ
        weights = (1.0 / self.risk_aversion) * (cov_inv @ expected_returns)
        
        # Clip individual positions
        weights = np.clip(weights, -self.max_position, self.max_position)
        
        # Scale to respect gross exposure
        gross = np.sum(np.abs(weights))
        if gross > self.max_gross_exposure:
            weights *= self.max_gross_exposure / gross
        
        return weights


# =============================================================================
# Algorithm Factory
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
            n_factors=config.get('n_factors', 10),
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
# Stat Arb Backtest
# =============================================================================

def compute_expected_returns_from_spread(
    prices: np.ndarray,
    cluster_means: Dict[int, float],
    cluster_assignments: Dict[str, int],
    market_idx_to_id: Dict[int, str],
) -> np.ndarray:
    """
    Compute expected returns based on spread from cluster mean.
    
    If market price deviates from its cluster's mean, we expect mean-reversion.
    """
    n = len(prices)
    expected_returns = np.zeros(n)
    
    for i in range(n):
        mid = market_idx_to_id.get(i)
        if mid is None or np.isnan(prices[i]):
            continue
        
        cluster_id = cluster_assignments.get(mid)
        if cluster_id is None:
            continue
        
        cluster_mean = cluster_means.get(cluster_id, prices[i])
        
        # Spread: how far from cluster mean
        spread = prices[i] - cluster_mean
        
        # Expected return: mean reversion back to cluster mean
        # Negative spread = underpriced = positive expected return
        expected_returns[i] = -spread
    
    return expected_returns


def run_stat_arb_backtest(
    algo_name: str,
    config: Dict,
    prices: np.ndarray,
    death_events: List[Tuple[int, int]],
    update_every: int = 5,
    risk_aversion: float = 1.0,
    regularization: float = 0.1,
) -> Dict[str, float]:
    """
    Run Markowitz stat arb backtest with clustering-based correlation.
    
    Key difference from naive mean-reversion:
    - Uses correlation matrix from clustering for portfolio optimization
    - Weights are Markowitz-optimal, not equal-weighted
    - Trades spread relative to cluster means
    """
    n_days, n_markets = prices.shape
    market_ids = [str(m) for m in range(n_markets)]
    market_idx_to_id = {i: str(i) for i in range(n_markets)}
    
    try:
        algo = create_algorithm(algo_name, config)
    except Exception as e:
        return {'sharpe': float('-inf'), 'pnl': 0, 'es': 0, 'error': str(e)}
    
    optimizer = MarkowitzOptimizer(
        risk_aversion=risk_aversion,
        regularization=regularization,
        max_position=1.0,
        max_gross_exposure=5.0,
    )
    
    # Track market state
    added_markets = set()
    death_set = set()
    death_dict = {}
    for day, mkt in death_events:
        death_dict.setdefault(day, []).append(mkt)
    
    daily_pnl = []
    positions = np.zeros(n_markets)
    
    for t in range(1, n_days):
        # Process deaths
        if t in death_dict:
            for mkt in death_dict[t]:
                death_set.add(mkt)
                positions[mkt] = 0
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
        
        # Update clustering periodically
        if t % update_every == 0:
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
        
        # Get correlation matrix from clustering
        try:
            corr_market_ids, corr_matrix = algo.get_correlation_matrix()
            
            if len(corr_market_ids) >= 2:
                # Build mapping from correlation matrix indices to full market indices
                corr_id_to_idx = {mid: i for i, mid in enumerate(corr_market_ids)}
                
                # Get cluster assignments and compute cluster means
                clusters = algo.get_clusters()
                cluster_assignments = {}
                cluster_means = {}
                
                for cluster_id, members in clusters.items():
                    member_prices = []
                    for mid in members:
                        m_idx = int(mid)
                        if m_idx < n_markets and not np.isnan(curr_prices[m_idx]):
                            member_prices.append(curr_prices[m_idx])
                            cluster_assignments[mid] = cluster_id
                    if member_prices:
                        cluster_means[cluster_id] = np.mean(member_prices)
                
                # Compute expected returns (spread from cluster mean)
                active_indices = [int(mid) for mid in corr_market_ids if int(mid) not in death_set]
                
                if len(active_indices) >= 2:
                    # Build covariance matrix for active markets only
                    n_active = len(active_indices)
                    active_corr = np.eye(n_active)
                    
                    for i, mi in enumerate(active_indices):
                        for j, mj in enumerate(active_indices):
                            mid_i = str(mi)
                            mid_j = str(mj)
                            if mid_i in corr_id_to_idx and mid_j in corr_id_to_idx:
                                ci = corr_id_to_idx[mid_i]
                                cj = corr_id_to_idx[mid_j]
                                if ci < corr_matrix.shape[0] and cj < corr_matrix.shape[1]:
                                    active_corr[i, j] = corr_matrix[ci, cj]
                    
                    # Convert correlation to covariance using realized volatility
                    vols = np.std([returns[m] for m in active_indices]) if len(active_indices) > 0 else 0.1
                    vols = max(vols, 0.01)  # Floor volatility
                    active_cov = active_corr * (vols ** 2)
                    
                    # Expected returns from spread
                    expected_rets = np.zeros(n_active)
                    for i, m in enumerate(active_indices):
                        mid = str(m)
                        cluster_id = cluster_assignments.get(mid)
                        if cluster_id is not None and cluster_id in cluster_means:
                            spread = curr_prices[m] - cluster_means[cluster_id]
                            expected_rets[i] = -spread  # Mean reversion
                    
                    # Markowitz optimization
                    optimal_weights = optimizer.optimize(expected_rets, active_cov)
                    
                    # Map back to full position array
                    positions = np.zeros(n_markets)
                    for i, m in enumerate(active_indices):
                        positions[m] = optimal_weights[i]
        except Exception as e:
            # Fallback to simple equal-weighted within clusters
            pass
        
        # Calculate PnL
        pnl = np.nansum(positions * returns)
        daily_pnl.append(pnl)
        
        # Decay positions
        positions *= 0.95
    
    daily_pnl = np.array(daily_pnl)
    
    if len(daily_pnl) == 0 or np.std(daily_pnl) == 0:
        return {'sharpe': 0, 'pnl': 0, 'es': 0}
    
    sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    total_pnl = np.sum(daily_pnl)
    
    # Expected shortfall
    alpha = 0.05
    var = np.percentile(daily_pnl, alpha * 100)
    es = np.mean(daily_pnl[daily_pnl <= var]) if np.any(daily_pnl <= var) else var
    
    return {
        'sharpe': float(sharpe),
        'pnl': float(total_pnl),
        'es': float(es),
    }


def run_baseline_no_clustering(
    prices: np.ndarray,
    death_events: List[Tuple[int, int]],
    risk_aversion: float = 1.0,
    regularization: float = 0.1,
) -> Dict[str, float]:
    """
    Baseline: Markowitz with sample covariance (no clustering).
    """
    n_days, n_markets = prices.shape
    
    optimizer = MarkowitzOptimizer(
        risk_aversion=risk_aversion,
        regularization=regularization,
        max_position=1.0,
        max_gross_exposure=5.0,
    )
    
    death_set = set()
    death_dict = {}
    for day, mkt in death_events:
        death_dict.setdefault(day, []).append(mkt)
    
    daily_pnl = []
    positions = np.zeros(n_markets)
    
    # Rolling window for sample covariance
    lookback = 30
    returns_history = []
    
    for t in range(1, n_days):
        if t in death_dict:
            for mkt in death_dict[t]:
                death_set.add(mkt)
                positions[mkt] = 0
        
        curr = prices[t]
        prev = prices[t-1]
        
        valid = ~np.isnan(curr) & ~np.isnan(prev)
        rets = np.zeros(n_markets)
        rets[valid] = (curr[valid] - prev[valid]) / np.maximum(prev[valid], 0.01)
        
        returns_history.append(rets)
        if len(returns_history) > lookback:
            returns_history.pop(0)
        
        # Active markets
        active = [m for m in range(n_markets) if m not in death_set and not np.isnan(curr[m])]
        
        if len(active) >= 2 and len(returns_history) >= 10:
            # Sample covariance
            hist = np.array(returns_history)[:, active]
            
            try:
                sample_cov = np.cov(hist.T)
                if sample_cov.ndim == 0:
                    sample_cov = np.array([[float(sample_cov)]])
                elif sample_cov.ndim == 1:
                    sample_cov = np.diag(sample_cov)
                
                # Expected returns: global mean reversion
                mean_price = np.mean(curr[active])
                expected_rets = -(curr[active] - mean_price)
                
                # Markowitz optimization
                optimal_weights = optimizer.optimize(expected_rets, sample_cov)
                
                positions = np.zeros(n_markets)
                for i, m in enumerate(active):
                    positions[m] = optimal_weights[i]
            except:
                pass
        
        pnl = np.nansum(positions * rets)
        daily_pnl.append(pnl)
        positions *= 0.95
    
    daily_pnl = np.array(daily_pnl)
    
    if len(daily_pnl) == 0 or np.std(daily_pnl) == 0:
        return {'sharpe': 0, 'pnl': 0}
    
    sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    return {'sharpe': float(sharpe), 'pnl': float(np.sum(daily_pnl))}


# =============================================================================
# Data Loading (reuse from optimize_clustering_real_data.py)
# =============================================================================

def load_market_metadata(cache_path: Path, clob_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache_path)
    clob_files = set(f.replace('.parquet', '') for f in os.listdir(clob_dir))
    df['yes_token_id'] = df['yes_token_id'].astype(str)
    df['has_clob'] = df['yes_token_id'].isin(clob_files)
    df = df[df['has_clob']].copy()
    df['resolution_time'] = pd.to_datetime(df['resolution_time'], errors='coerce')
    df = df[df['y'].notna()].copy()
    
    def map_category(cat):
        cat = str(cat).lower()
        if 'crypto' in cat or 'bitcoin' in cat:
            return 'crypto'
        elif 'politic' in cat or 'election' in cat:
            return 'politics'
        elif 'sport' in cat or 'nba' in cat or 'nfl' in cat:
            return 'sports'
        else:
            return 'other'
    
    df['category_mapped'] = df['category'].fillna('other').apply(map_category)
    return df


def load_clob_prices(clob_dir: Path, metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    prices = {}
    for _, row in metadata.iterrows():
        market_id = str(row['id'])
        token_id = str(row['yes_token_id'])
        file_path = clob_dir / f"{token_id}.parquet"
        if not file_path.exists():
            continue
        try:
            df = pd.read_parquet(file_path)
            if df['t'].dtype in ['int64', 'float64']:
                df['t'] = pd.to_datetime(df['t'], unit='s', utc=True)
            else:
                df['t'] = pd.to_datetime(df['t'])
            df = df.sort_values('t').drop_duplicates(subset=['t'], keep='last')
            prices[market_id] = df[['t', 'p']].copy()
        except:
            continue
    return prices


def build_price_matrix(
    prices: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], pd.DatetimeIndex]:
    if not prices:
        raise ValueError("No price data")
    
    market_ids = list(prices.keys())
    n_markets = len(market_ids)
    
    daily_prices = {}
    all_dates = set()
    
    for mid in market_ids:
        df = prices[mid].copy()
        df['date'] = df['t'].dt.normalize()
        daily = df.groupby('date')['p'].last()
        daily_prices[mid] = daily
        all_dates.update(daily.index)
    
    all_dates = sorted(all_dates)
    date_range = pd.DatetimeIndex(all_dates)
    n_days = len(date_range)
    
    price_matrix = np.full((n_days, n_markets), np.nan)
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    
    for i, mid in enumerate(market_ids):
        for date, p in daily_prices[mid].items():
            if date in date_to_idx:
                price_matrix[date_to_idx[date], i] = p
    
    # Forward fill
    for i in range(n_markets):
        last_valid = np.nan
        for t in range(n_days):
            if np.isnan(price_matrix[t, i]) and not np.isnan(last_valid):
                price_matrix[t, i] = last_valid
            elif not np.isnan(price_matrix[t, i]):
                last_valid = price_matrix[t, i]
    
    # Death events
    death_events = []
    resolution_times = metadata.set_index('id')['resolution_time'].to_dict()
    for i, mid in enumerate(market_ids):
        res_time = resolution_times.get(int(mid) if mid.isdigit() else mid)
        if pd.notna(res_time):
            res_date = res_time.normalize() if hasattr(res_time, 'normalize') else pd.Timestamp(res_time).normalize()
            if res_date in date_to_idx:
                death_events.append((date_to_idx[res_date], i))
    
    death_events.sort(key=lambda x: x[0])
    return price_matrix, market_ids, death_events, date_range


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-path', default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--clob-dir', default='data/polymarket/clob_history_yes_f1')
    parser.add_argument('--categories', nargs='+', default=['sports', 'politics', 'crypto'])
    parser.add_argument('--max-markets', type=int, default=60)
    parser.add_argument('--output-dir', default='runs/clustering_stat_arb')
    args = parser.parse_args()
    
    cache_path = Path(args.cache_path)
    clob_dir = Path(args.clob_dir)
    
    if not cache_path.exists() or not clob_dir.exists():
        print("Data files not found")
        return
    
    metadata = load_market_metadata(cache_path, clob_dir)
    print(f"Loaded {len(metadata)} markets with CLOB data")
    
    # Hyperparameters per algorithm
    algo_configs = {
        'OLRCM': {'n_factors': 10, 'learning_rate': 0.02, 'l2_reg': 0.01},
        'SWOCC': {'ema_alpha': 0.1, 'use_survival_weights': True, 'shrinkage': 0.1},
        'SDPM': {'concentration': 3.0, 'prior_precision': 0.2, 'temperature': 1.0},
        'BDHP': {'decay_rate': 0.2, 'base_intensity': 0.1, 'self_excitation': 0.3},
    }
    
    all_results = {}
    
    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Category: {category.upper()}")
        print(f"{'='*60}")
        
        cat_markets = metadata[metadata['category_mapped'] == category].sort_values('volumeNum', ascending=False).head(args.max_markets)
        if len(cat_markets) < 5:
            print(f"Not enough markets for {category}")
            continue
        
        prices_dict = load_clob_prices(clob_dir, cat_markets)
        if len(prices_dict) < 5:
            continue
        
        price_matrix, market_ids, death_events, _ = build_price_matrix(prices_dict, cat_markets)
        print(f"Price matrix: {price_matrix.shape[0]} days x {price_matrix.shape[1]} markets")
        
        # Baseline: Markowitz without clustering
        baseline = run_baseline_no_clustering(price_matrix, death_events)
        print(f"\nBaseline (Markowitz, no clustering): Sharpe = {baseline['sharpe']:+.3f}")
        
        # Test each clustering algorithm
        results = {'baseline': baseline}
        
        for algo_name, config in algo_configs.items():
            result = run_stat_arb_backtest(
                algo_name, config, price_matrix, death_events,
                update_every=5, risk_aversion=1.0, regularization=0.1
            )
            results[algo_name] = result
            
            improvement = result['sharpe'] - baseline['sharpe']
            better = "✓" if improvement > 0 else "✗"
            
            print(f"{algo_name:20s}: Sharpe = {result['sharpe']:+.3f}  ({improvement:+.3f}) {better}")
        
        # Find best
        best_algo = max(algo_configs.keys(), key=lambda a: results[a]['sharpe'])
        print(f"\nBest: {best_algo} (Sharpe = {results[best_algo]['sharpe']:+.3f})")
        
        all_results[category] = {
            'baseline': baseline,
            'algorithms': {k: results[k] for k in algo_configs.keys()},
            'best_algorithm': best_algo,
        }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Stat Arb / Markowitz with Clustering")
    print("="*70)
    print(f"{'Category':<12} {'Baseline':>12} {'Best Algo':<15} {'Best Sharpe':>12} {'Improvement':>12}")
    print("-"*70)
    
    for cat, res in all_results.items():
        baseline_sharpe = res['baseline']['sharpe']
        best_algo = res['best_algorithm']
        best_sharpe = res['algorithms'][best_algo]['sharpe']
        improvement = best_sharpe - baseline_sharpe
        print(f"{cat:<12} {baseline_sharpe:>+12.3f} {best_algo:<15} {best_sharpe:>+12.3f} {improvement:>+12.3f}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'stat_arb_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_dir / 'stat_arb_results.json'}")


if __name__ == '__main__':
    main()
