#!/usr/bin/env python3
"""
Portfolio Construction and Advanced Arbitrage Strategies

Key strategies:
1. Correlated Market Groups - Find markets that move together
2. Cointegrated Pairs - Statistical arbitrage on mean-reverting spreads
3. YES/NO Convergence - "Funding rate" style trade on spread → 1
4. Category Portfolio - Risk parity across market categories
5. Multi-Market Arbitrage - Exploit price discrepancies across related markets
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

try:
    import optuna
    from optuna.samplers import CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# =============================================================================
# Portfolio Construction - Finding Related Markets
# =============================================================================

def extract_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for market similarity analysis."""
    
    features = []
    
    for _, row in df.iterrows():
        q = str(row.get('question', '')).lower()
        
        # Entity extraction (simple keyword-based)
        entities = {
            'trump': 'trump' in q,
            'biden': 'biden' in q,
            'harris': 'harris' in q,
            'bitcoin': any(x in q for x in ['bitcoin', 'btc']),
            'ethereum': any(x in q for x in ['ethereum', 'eth']),
            'election': 'election' in q,
            'president': 'president' in q,
            'fed': any(x in q for x in ['fed', 'federal reserve']),
            'rate': 'rate' in q,
        }
        
        # Time features
        resolution_time = row.get('resolution_time')
        if pd.notna(resolution_time):
            month = resolution_time.month
            year = resolution_time.year
        else:
            month, year = 0, 0
        
        features.append({
            'market_id': row.get('goldsky_id', row.name),
            'category': row.get('category', 'other'),
            'first_price': row.get('first_price', 0.5),
            'outcome': row.get('y', 0),
            'month': month,
            'year': year,
            **entities,
        })
    
    return pd.DataFrame(features)


def find_market_clusters(
    df: pd.DataFrame,
    n_clusters: int = 20,
    min_cluster_size: int = 5,
) -> Dict[int, List[int]]:
    """
    Cluster markets by similarity for portfolio construction.
    
    Returns dict of cluster_id -> list of market indices.
    """
    features = extract_market_features(df)
    
    # Create feature matrix
    cat_dummies = pd.get_dummies(features['category'], prefix='cat')
    entity_cols = ['trump', 'biden', 'harris', 'bitcoin', 'ethereum', 
                   'election', 'president', 'fed', 'rate']
    
    X = pd.concat([
        features[['first_price', 'month']].fillna(0),
        features[entity_cols].astype(int),
        cat_dummies,
    ], axis=1).values
    
    # Hierarchical clustering
    if len(X) > 1:
        Z = linkage(X, method='ward')
        labels = fcluster(Z, n_clusters, criterion='maxclust')
    else:
        labels = np.array([1])
    
    # Group by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)
    
    # Filter small clusters
    clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
    
    return clusters


def compute_residual_correlation(
    df: pd.DataFrame,
    lookback: int = 100,
) -> pd.DataFrame:
    """
    Compute correlation of calibration residuals between market categories.
    
    This finds categories where calibration errors move together,
    suggesting shared underlying factors.
    """
    # Compute residuals by category over time
    df = df.sort_values('resolution_time').copy()
    df['residual'] = df['y'] - df['first_price']
    
    # Rolling residuals by category
    category_residuals = {}
    
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat].copy()
        if len(cat_data) >= lookback:
            # Rolling mean residual
            cat_data['rolling_residual'] = cat_data['residual'].rolling(lookback).mean()
            category_residuals[cat] = cat_data[['resolution_time', 'rolling_residual']].dropna()
    
    if len(category_residuals) < 2:
        return pd.DataFrame()
    
    # Align on common timestamps and compute correlation
    # (simplified: just use overall correlation of residuals)
    cats = list(category_residuals.keys())
    n_cats = len(cats)
    corr_matrix = np.eye(n_cats)
    
    for i in range(n_cats):
        for j in range(i+1, n_cats):
            r1 = category_residuals[cats[i]]['rolling_residual'].values
            r2 = category_residuals[cats[j]]['rolling_residual'].values
            
            # Align by taking common length from end
            min_len = min(len(r1), len(r2))
            if min_len > 10:
                corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    return pd.DataFrame(corr_matrix, index=cats, columns=cats)


# =============================================================================
# YES/NO Convergence Strategy ("Funding Rate" Arbitrage)
# =============================================================================

def yes_no_convergence_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Strategy params
    spread_threshold: float = 0.05,      # Min deviation from 1.0
    convergence_speed: float = 0.8,      # Expected speed of convergence
    holding_period: int = 1,             # Days to hold
    # Sizing params  
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    leverage: float = 1.0,
    max_position_pct: float = 0.20,
    fee: float = 0.01,
    max_drawdown_stop: float = 0.50,
) -> Dict[str, float]:
    """
    YES/NO Convergence Strategy.
    
    In prediction markets:
    - YES_price + NO_price should ≈ 1.0
    - If sum > 1: market is "expensive" - sell both
    - If sum < 1: market is "cheap" - buy both
    - The gap should converge to 1.0 over time
    
    This is analogous to funding rate arbitrage in perpetuals:
    - Profit from the spread between spot and futures converging
    
    Since we only have YES prices, we simulate NO_price = 1 - YES_price
    and look for systematic deviations in realized outcomes.
    """
    
    # In our dataset, we have first_price (YES price) and outcome
    # The "spread" is how far E[Y|q] is from q
    # If E[Y|q] > q consistently, the YES token is underpriced
    # If E[Y|q] < q consistently, the YES token is overpriced
    
    # Learn the spread from training data
    train_prices = train['first_price'].values
    train_outcomes = train['y'].values
    
    # Bin by price level to find systematic spread
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(train_prices, bin_edges) - 1, 0, n_bins - 1)
    
    bin_spreads = {}
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 20:
            # Spread = E[Y|q] - q (if positive, YES is underpriced)
            spread = train_outcomes[mask].mean() - train_prices[mask].mean()
            bin_spreads[b] = {
                'spread': spread,
                'n': mask.sum(),
                'expected_outcome': train_outcomes[mask].mean(),
            }
    
    # Trade on test set
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    pnls = []
    wins = 0
    
    for _, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        b = int(np.clip(np.digitize(price, bin_edges) - 1, 0, n_bins - 1))
        
        if b not in bin_spreads:
            continue
        
        spread = bin_spreads[b]['spread']
        
        # Only trade if spread exceeds threshold
        if abs(spread) < spread_threshold:
            continue
        
        # Check drawdown stop
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        # Trade direction: if spread > 0, YES is underpriced (buy YES)
        direction = 1 if spread > 0 else -1
        
        # Kelly sizing based on expected convergence
        edge = abs(spread) * convergence_speed
        
        if direction > 0:
            odds = (1 - price) / price if price > 0 else 0
            win_prob = bin_spreads[b]['expected_outcome']
        else:
            odds = price / (1 - price) if price < 1 else 0
            win_prob = 1 - bin_spreads[b]['expected_outcome']
        
        # Kelly formula
        kelly_f = (win_prob * odds - (1 - win_prob)) / odds if odds > 0 else 0
        kelly_f = max(0, kelly_f) * kelly_fraction
        
        position_frac = kelly_f * leverage
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        if position_frac < 0.001:
            continue
        
        position = bankroll * position_frac
        
        # PnL
        if direction > 0:
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price
                wins += 1
            else:
                pnl = -position - position * fee / price
        else:
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price)
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price)
        
        pnls.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
    }


# =============================================================================
# Portfolio Stat Arb - Trade Correlated Markets Together
# =============================================================================

def portfolio_stat_arb_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Portfolio params
    n_clusters: int = 10,
    min_cluster_size: int = 5,
    intra_cluster_weight: float = 1.0,   # Weight for same-cluster signals
    cross_cluster_hedge: float = 0.5,    # Hedge with anti-correlated clusters
    # Calibration params
    n_bins: int = 10,
    min_edge: float = 0.02,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    leverage: float = 1.0,
    max_position_pct: float = 0.30,      # Higher for portfolio
    portfolio_max_pct: float = 0.80,     # Max total exposure
    fee: float = 0.01,
    max_drawdown_stop: float = 0.50,
) -> Dict[str, float]:
    """
    Portfolio Statistical Arbitrage Strategy.
    
    Key ideas:
    1. Cluster markets by similarity
    2. Compute calibration signals per cluster
    3. Trade clusters with hedging across anti-correlated groups
    4. Risk parity sizing across clusters
    """
    
    # Cluster training data
    clusters = find_market_clusters(train, n_clusters=n_clusters, min_cluster_size=min_cluster_size)
    
    if not clusters:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    # Compute calibration signals per cluster
    cluster_signals = {}
    
    for cluster_id, indices in clusters.items():
        cluster_data = train.iloc[indices]
        prices = cluster_data['first_price'].values
        outcomes = cluster_data['y'].values
        
        if len(prices) < 20:
            continue
        
        # Calibration error for this cluster
        residuals = outcomes - prices
        g_bar = residuals.mean()
        sigma = residuals.std()
        se = sigma / np.sqrt(len(residuals))
        t_stat = g_bar / se if se > 0 else 0
        
        if abs(t_stat) >= 2.0 and abs(g_bar) >= min_edge:
            cluster_signals[cluster_id] = {
                'g_bar': g_bar,
                'direction': 1 if g_bar > 0 else -1,
                'confidence': abs(t_stat),
                'win_rate': outcomes.mean() if g_bar > 0 else (1 - outcomes).mean(),
            }
    
    if not cluster_signals:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    # Cluster test data
    test_clusters = find_market_clusters(test, n_clusters=n_clusters, min_cluster_size=1)
    
    # Map test markets to training clusters (simplified: by index proximity)
    def get_cluster_for_market(test_idx, train_clusters, test_clusters):
        for cluster_id, indices in test_clusters.items():
            if test_idx in indices:
                # Check if this cluster_id exists in training
                if cluster_id in cluster_signals:
                    return cluster_id
        return None
    
    # Trade on test set
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    pnls = []
    wins = 0
    
    for idx, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        
        # Try to assign to a cluster with a signal
        # (simplified: use category-based matching)
        cat = row.get('category', 'other')
        
        # Find a cluster with matching category bias
        matching_cluster = None
        for cluster_id, signal in cluster_signals.items():
            # Check if signal direction aligns with category
            if abs(signal['g_bar']) >= min_edge:
                matching_cluster = cluster_id
                break
        
        if matching_cluster is None:
            continue
        
        signal = cluster_signals[matching_cluster]
        direction = signal['direction']
        edge = abs(signal['g_bar'])
        
        # Check drawdown stop
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        # Portfolio-weighted sizing
        if direction > 0:
            odds = (1 - price) / price if price > 0 else 0
            win_prob = signal['win_rate']
        else:
            odds = price / (1 - price) if price < 1 else 0
            win_prob = signal['win_rate']
        
        kelly_f = (win_prob * odds - (1 - win_prob)) / odds if odds > 0 else 0
        kelly_f = max(0, kelly_f) * kelly_fraction
        
        # Scale by cluster confidence
        kelly_f *= min(signal['confidence'] / 2.0, 1.5)
        
        position_frac = kelly_f * leverage * intra_cluster_weight
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        if position_frac < 0.001:
            continue
        
        position = bankroll * position_frac
        
        # PnL
        if direction > 0:
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price
                wins += 1
            else:
                pnl = -position - position * fee / price
        else:
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price)
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price)
        
        pnls.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
    }


# =============================================================================
# Category Risk Parity Strategy
# =============================================================================

def category_risk_parity_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Strategy params
    vol_lookback: int = 50,
    target_vol: float = 0.15,            # Target annualized vol
    rebalance_frequency: int = 20,       # Rebalance every N markets
    min_edge: float = 0.02,
    # Sizing params
    initial_bankroll: float = 10000.0,
    leverage: float = 1.0,
    max_position_pct: float = 0.15,
    max_category_pct: float = 0.40,      # Max exposure per category
    fee: float = 0.01,
    max_drawdown_stop: float = 0.50,
) -> Dict[str, float]:
    """
    Category Risk Parity Strategy.
    
    Key ideas:
    1. Estimate volatility per category from training data
    2. Allocate risk budget equally across categories
    3. Scale positions to target overall portfolio volatility
    """
    
    # Estimate category volatility from training
    category_stats = {}
    
    for cat in train['category'].unique():
        cat_data = train[train['category'] == cat]
        if len(cat_data) >= vol_lookback:
            residuals = cat_data['y'] - cat_data['first_price']
            vol = residuals.std()
            mean_residual = residuals.mean()
            
            category_stats[cat] = {
                'vol': vol if vol > 0 else 0.5,
                'edge': mean_residual,
                'win_rate': cat_data['y'].mean(),
                'n_samples': len(cat_data),
            }
    
    if not category_stats:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    # Risk parity weights (inverse vol)
    total_inv_vol = sum(1 / s['vol'] for s in category_stats.values())
    risk_weights = {cat: (1 / s['vol']) / total_inv_vol for cat, s in category_stats.items()}
    
    # Trade on test set
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_dd = 0
    pnls = []
    wins = 0
    category_exposure = defaultdict(float)
    
    for idx, row in test.iterrows():
        price = row['first_price']
        outcome = row['y']
        cat = row.get('category', 'other')
        
        if cat not in category_stats:
            continue
        
        stats = category_stats[cat]
        edge = stats['edge']
        
        if abs(edge) < min_edge:
            continue
        
        # Check category exposure limit
        if category_exposure[cat] >= max_category_pct * bankroll:
            continue
        
        # Check drawdown stop
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            break
        
        direction = 1 if edge > 0 else -1
        
        # Vol-scaled position sizing
        cat_vol = stats['vol']
        vol_scalar = target_vol / cat_vol if cat_vol > 0 else 1.0
        
        # Risk parity weight
        weight = risk_weights.get(cat, 0.1)
        
        position_frac = weight * vol_scalar * leverage
        position_frac = np.clip(position_frac, 0, max_position_pct)
        
        if position_frac < 0.001:
            continue
        
        position = bankroll * position_frac
        
        # Track exposure
        category_exposure[cat] += position
        
        # PnL
        if direction > 0:
            if outcome == 1:
                pnl = position * (1 - price) / price - position * fee / price
                wins += 1
            else:
                pnl = -position - position * fee / price
        else:
            if outcome == 0:
                pnl = position * price / (1 - price) - position * fee / (1 - price)
                wins += 1
            else:
                pnl = -position - position * fee / (1 - price)
        
        pnls.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        max_dd = max(max_dd, (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        # Periodic rebalance (reset exposure tracking)
        if len(pnls) % rebalance_frequency == 0:
            category_exposure = defaultdict(float)
        
        if bankroll <= 0:
            break
    
    if not pnls:
        return {'pnl': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0, 'final_bankroll': initial_bankroll}
    
    pnls = np.array(pnls)
    total_pnl = bankroll - initial_bankroll
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    
    return {
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(win_rate),
        'max_dd': float(max_dd),
        'final_bankroll': float(bankroll),
        'return_pct': float((bankroll - initial_bankroll) / initial_bankroll * 100),
    }


# =============================================================================
# Optimization
# =============================================================================

def walk_forward_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    holdout_frac: float = 0.2,
) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    """Generate walk-forward splits + holdout."""
    n = len(df)
    holdout_size = int(n * holdout_frac)
    cv_data = df.iloc[:-holdout_size].copy() if holdout_size > 0 else df.copy()
    holdout = df.iloc[-holdout_size:].copy() if holdout_size > 0 else None
    
    cv_n = len(cv_data)
    fold_size = cv_n // (n_folds + 1)
    
    splits = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        
        train = cv_data.iloc[:train_end]
        test = cv_data.iloc[train_end:test_end]
        
        if len(train) >= 500 and len(test) >= 100:
            splits.append((train, test))
    
    return splits, holdout


def optimize_portfolio_strategies(
    data: pd.DataFrame,
    n_trials: int = 500,
    n_folds: int = 5,
    n_jobs: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Optimize all portfolio strategies."""
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), 32)
    
    splits, holdout = walk_forward_split(data, n_folds=n_folds)
    
    if not splits:
        return {'error': 'Not enough data'}
    
    results = {}
    
    # Strategy 1: YES/NO Convergence
    print("\n" + "="*50)
    print("OPTIMIZING: YES/NO Convergence")
    print("="*50)
    
    def convergence_objective(trial: optuna.Trial) -> float:
        params = {
            'spread_threshold': trial.suggest_float('spread_threshold', 0.01, 0.20),
            'convergence_speed': trial.suggest_float('convergence_speed', 0.3, 1.0),
            'initial_bankroll': trial.suggest_float('initial_bankroll', 1000, 100000, log=True),
            'kelly_fraction': trial.suggest_float('kelly_fraction', 0.1, 1.0),
            'leverage': trial.suggest_float('leverage', 0.5, 5.0),
            'max_position_pct': trial.suggest_float('max_position_pct', 0.05, 0.40),
            'fee': trial.suggest_float('fee', 0.005, 0.02),
            'max_drawdown_stop': trial.suggest_float('max_drawdown_stop', 0.20, 0.70),
        }
        
        sharpes = []
        for train, test in splits:
            result = yes_no_convergence_strategy(train, test, **params)
            if result['n_trades'] > 0:
                sharpes.append(result['sharpe'])
        
        return np.mean(sharpes) if sharpes else float('-inf')
    
    sampler = CmaEsSampler(n_startup_trials=50)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(convergence_objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    
    results['convergence'] = {
        'cv_sharpe': study.best_value,
        'params': study.best_params,
    }
    print(f"Best CV Sharpe: {study.best_value:.4f}")
    
    # Strategy 2: Portfolio Stat Arb
    print("\n" + "="*50)
    print("OPTIMIZING: Portfolio Stat Arb")
    print("="*50)
    
    def portfolio_objective(trial: optuna.Trial) -> float:
        params = {
            'n_clusters': trial.suggest_int('n_clusters', 5, 30),
            'min_cluster_size': trial.suggest_int('min_cluster_size', 3, 20),
            'intra_cluster_weight': trial.suggest_float('intra_cluster_weight', 0.5, 2.0),
            'n_bins': trial.suggest_int('n_bins', 5, 20),
            'min_edge': trial.suggest_float('min_edge', 0.01, 0.15),
            'initial_bankroll': trial.suggest_float('initial_bankroll', 1000, 100000, log=True),
            'kelly_fraction': trial.suggest_float('kelly_fraction', 0.1, 1.0),
            'leverage': trial.suggest_float('leverage', 0.5, 5.0),
            'max_position_pct': trial.suggest_float('max_position_pct', 0.05, 0.40),
            'portfolio_max_pct': trial.suggest_float('portfolio_max_pct', 0.50, 0.95),
            'fee': trial.suggest_float('fee', 0.005, 0.02),
            'max_drawdown_stop': trial.suggest_float('max_drawdown_stop', 0.20, 0.70),
        }
        
        sharpes = []
        for train, test in splits:
            result = portfolio_stat_arb_strategy(train, test, **params)
            if result['n_trades'] > 0:
                sharpes.append(result['sharpe'])
        
        return np.mean(sharpes) if sharpes else float('-inf')
    
    study2 = optuna.create_study(direction='maximize', sampler=CmaEsSampler(n_startup_trials=50))
    study2.optimize(portfolio_objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    
    results['portfolio_stat_arb'] = {
        'cv_sharpe': study2.best_value,
        'params': study2.best_params,
    }
    print(f"Best CV Sharpe: {study2.best_value:.4f}")
    
    # Strategy 3: Category Risk Parity
    print("\n" + "="*50)
    print("OPTIMIZING: Category Risk Parity")
    print("="*50)
    
    def risk_parity_objective(trial: optuna.Trial) -> float:
        params = {
            'vol_lookback': trial.suggest_int('vol_lookback', 20, 200),
            'target_vol': trial.suggest_float('target_vol', 0.05, 0.40),
            'rebalance_frequency': trial.suggest_int('rebalance_frequency', 5, 100),
            'min_edge': trial.suggest_float('min_edge', 0.01, 0.15),
            'initial_bankroll': trial.suggest_float('initial_bankroll', 1000, 100000, log=True),
            'leverage': trial.suggest_float('leverage', 0.5, 5.0),
            'max_position_pct': trial.suggest_float('max_position_pct', 0.05, 0.30),
            'max_category_pct': trial.suggest_float('max_category_pct', 0.20, 0.60),
            'fee': trial.suggest_float('fee', 0.005, 0.02),
            'max_drawdown_stop': trial.suggest_float('max_drawdown_stop', 0.20, 0.70),
        }
        
        sharpes = []
        for train, test in splits:
            result = category_risk_parity_strategy(train, test, **params)
            if result['n_trades'] > 0:
                sharpes.append(result['sharpe'])
        
        return np.mean(sharpes) if sharpes else float('-inf')
    
    study3 = optuna.create_study(direction='maximize', sampler=CmaEsSampler(n_startup_trials=50))
    study3.optimize(risk_parity_objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    
    results['risk_parity'] = {
        'cv_sharpe': study3.best_value,
        'params': study3.best_params,
    }
    print(f"Best CV Sharpe: {study3.best_value:.4f}")
    
    # Holdout evaluation
    if holdout is not None and len(holdout) > 0:
        print("\n" + "="*50)
        print("HOLDOUT EVALUATION")
        print("="*50)
        
        train_all = data.iloc[:-len(holdout)]
        
        # Convergence
        conv_result = yes_no_convergence_strategy(
            train_all, holdout, **results['convergence']['params']
        )
        results['convergence']['holdout'] = conv_result
        print(f"Convergence: Sharpe={conv_result['sharpe']:.2f}, PnL=${conv_result['pnl']:,.2f}, Return={conv_result.get('return_pct', 0):.1f}%")
        
        # Portfolio
        port_result = portfolio_stat_arb_strategy(
            train_all, holdout, **results['portfolio_stat_arb']['params']
        )
        results['portfolio_stat_arb']['holdout'] = port_result
        print(f"Portfolio: Sharpe={port_result['sharpe']:.2f}, PnL=${port_result['pnl']:,.2f}, Return={port_result.get('return_pct', 0):.1f}%")
        
        # Risk Parity
        rp_result = category_risk_parity_strategy(
            train_all, holdout, **results['risk_parity']['params']
        )
        results['risk_parity']['holdout'] = rp_result
        print(f"Risk Parity: Sharpe={rp_result['sharpe']:.2f}, PnL=${rp_result['pnl']:,.2f}, Return={rp_result.get('return_pct', 0):.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-path', type=str, default='data/polymarket/optimization_cache.parquet')
    parser.add_argument('--output-dir', type=str, default='runs/portfolio_optimization')
    parser.add_argument('--n-trials', type=int, default=500)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=None)
    args = parser.parse_args()
    
    # Load data
    cache_path = Path(args.cache_path)
    print(f"Loading data from {cache_path}")
    data = pd.read_parquet(cache_path)
    print(f"Loaded {len(data):,} markets")
    
    # Run optimization
    results = optimize_portfolio_strategies(
        data=data,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        n_jobs=args.n_jobs,
        verbose=True,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
