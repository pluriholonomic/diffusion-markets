"""
Dispersion and Pairs Trading Strategies for Prediction Markets

This module implements:
1. Dispersion Strategy - trade the spread between category-level and individual market vol
2. Lead-Lag Pairs Trading - trade cointegrated market pairs

Key Insight:
When category-level implied volatility differs from the realized dispersion of 
individual markets within that category, there's an arbitrage opportunity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DispersionSignal:
    """Container for dispersion signal"""
    direction: int  # +1 = buy dispersion (buy individuals, sell index)
                    # -1 = sell dispersion (sell individuals, buy index)
    magnitude: float  # Size of mispricing
    category: str
    implied_vol: float
    realized_dispersion: float


@dataclass  
class PairsSignal:
    """Container for pairs trading signal"""
    direction: int  # +1 = long lead, short lag; -1 = opposite
    spread_zscore: float
    lead_market: str
    lag_market: str
    correlation: float


def compute_implied_volatility(
    prices: np.ndarray,
    method: str = 'binary'
) -> float:
    """
    Compute implied volatility from prediction market prices.
    
    For binary markets: IV ≈ sqrt(p * (1-p))
    This is the standard deviation of a Bernoulli random variable.
    """
    if len(prices) == 0:
        return 0.0
    
    # Clip prices to valid range
    prices = np.clip(prices, 0.01, 0.99)
    
    if method == 'binary':
        # For binary outcomes, IV is sqrt(p(1-p))
        iv = np.mean(np.sqrt(prices * (1 - prices)))
    else:
        # Historical vol from price changes
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices)
        iv = np.std(returns) * np.sqrt(252)  # Annualized
    
    return iv


def compute_realized_dispersion(
    market_prices: Dict[str, np.ndarray],
    category_price: np.ndarray = None
) -> float:
    """
    Compute realized dispersion of individual markets within a category.
    
    Dispersion = std of individual market returns relative to category average.
    """
    if len(market_prices) < 2:
        return 0.0
    
    # Compute returns for each market
    market_returns = []
    for market_id, prices in market_prices.items():
        if len(prices) >= 2:
            returns = np.diff(prices)
            market_returns.append(np.std(returns))
    
    if not market_returns:
        return 0.0
    
    # Dispersion is the cross-sectional std of individual vols
    dispersion = np.std(market_returns)
    
    return dispersion


def dispersion_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Dispersion params
    lookback: int = 30,
    dispersion_threshold: float = 0.02,
    reversion_speed: float = 0.5,  # Expected speed of vol convergence
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.20,
    max_position_pct: float = 0.15,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
) -> Dict[str, Any]:
    """
    Dispersion Strategy
    
    Trade the spread between category implied volatility and individual market 
    realized dispersion.
    
    Logic:
    1. Compute implied vol for category (average of individual IVs)
    2. Compute realized dispersion (cross-sectional vol of individual markets)
    3. If implied > realized → sell dispersion (markets are too volatile)
    4. If implied < realized → buy dispersion (markets are too calm)
    
    Implementation in prediction markets:
    - "Buy dispersion" = bet on extreme outcomes (YES or NO, not middle)
    - "Sell dispersion" = bet on middle prices staying middle
    """
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    category_col = 'category' if 'category' in train.columns else None
    market_col = 'conditionId' if 'conditionId' in train.columns else 'market_id'
    
    if category_col is None or category_col not in train.columns:
        return {
            'strategy': 'dispersion',
            'total_pnl': 0,
            'final_bankroll': initial_bankroll,
            'sharpe': 0,
            'error': 'No category column found',
        }
    
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    
    # Compute dispersion signals per category
    category_signals = {}
    
    for category, cat_data in train_copy.groupby(category_col):
        if len(cat_data) < lookback:
            continue
        
        # Get implied vol for the category
        prices = cat_data['_price'].values
        implied_vol = compute_implied_volatility(prices)
        
        # Get individual market prices
        market_prices = {}
        if market_col in cat_data.columns:
            for market_id, market_data in cat_data.groupby(market_col):
                market_prices[market_id] = market_data['_price'].values
        
        # Compute realized dispersion
        realized_disp = compute_realized_dispersion(market_prices)
        
        # Signal: difference between implied and realized
        dispersion_gap = implied_vol - realized_disp
        
        if abs(dispersion_gap) >= dispersion_threshold:
            direction = -1 if dispersion_gap > 0 else 1  # Sell if implied > realized
            magnitude = abs(dispersion_gap)
            
            category_signals[category] = DispersionSignal(
                direction=direction,
                magnitude=magnitude,
                category=category,
                implied_vol=implied_vol,
                realized_dispersion=realized_disp,
            )
    
    # Apply to test data
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    pnl_list = []
    trades = 0
    wins = 0
    
    for idx, row in test_copy.iterrows():
        price = row['_price']
        outcome = row['_outcome']
        category = row.get(category_col, None)
        
        if category is None or category not in category_signals:
            pnl_list.append(0)
            continue
        
        signal = category_signals[category]
        
        # Check drawdown
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            pnl_list.append(0)
            continue
        
        trades += 1
        
        # Position sizing based on signal magnitude
        edge = signal.magnitude * reversion_speed
        position_frac = kelly_fraction * min(edge / 0.10, 1.0)
        position_frac = min(position_frac, max_position_pct)
        position = bankroll * position_frac
        
        # Dispersion trade implementation:
        # If sell dispersion (direction=-1): expect prices to stay near 0.5
        # If buy dispersion (direction=+1): expect prices to move to extremes
        
        if signal.direction < 0:  # Sell dispersion - expect calm
            # Bet that extreme prices will move toward 0.5
            if price > 0.7:  # High price → expect it to fall
                pnl = position * ((1 - outcome) - (1 - price)) - fee * position
            elif price < 0.3:  # Low price → expect it to rise
                pnl = position * (outcome - price) - fee * position
            else:  # Already near 0.5 → skip
                pnl_list.append(0)
                trades -= 1
                continue
        else:  # Buy dispersion - expect volatility
            # Bet that middle prices will move to extremes
            if 0.4 < price < 0.6:  # Near 0.5 → expect movement
                # This is harder to trade directly; bet on the more likely extreme
                if outcome > 0.5:
                    pnl = position * (outcome - price) - fee * position
                else:
                    pnl = position * ((1 - outcome) - (1 - price)) - fee * position
            else:
                pnl_list.append(0)
                trades -= 1
                continue
        
        pnl_list.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        
        if pnl > 0:
            wins += 1
    
    pnl_array = np.array(pnl_list)
    total_pnl = np.sum(pnl_array)
    sharpe = np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252)
    max_dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
    
    return {
        'strategy': 'dispersion',
        'total_pnl': total_pnl,
        'final_bankroll': bankroll,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': wins / trades if trades > 0 else 0,
        'pnl_series': pnl_array,
        'category_signals': {k: v.__dict__ for k, v in category_signals.items()},
    }


def compute_cointegration(
    series1: np.ndarray,
    series2: np.ndarray,
) -> Tuple[float, float, bool]:
    """
    Simple cointegration test using OLS regression.
    
    Returns:
        (correlation, spread_std, is_cointegrated)
    """
    if len(series1) != len(series2) or len(series1) < 20:
        return 0.0, 0.0, False
    
    # Correlation
    corr = np.corrcoef(series1, series2)[0, 1]
    
    # Spread (residual from regression)
    # y = beta * x + alpha + epsilon
    mean1, mean2 = np.mean(series1), np.mean(series2)
    cov = np.mean((series1 - mean1) * (series2 - mean2))
    var2 = np.var(series2)
    
    if var2 < 1e-10:
        return corr, 0.0, False
    
    beta = cov / var2
    alpha = mean1 - beta * mean2
    
    # Spread (residual)
    spread = series1 - (beta * series2 + alpha)
    spread_std = np.std(spread)
    
    # Simple cointegration check: is spread mean-reverting?
    # Use autocorrelation of spread
    if len(spread) > 1:
        ac1 = np.corrcoef(spread[:-1], spread[1:])[0, 1]
        # Negative autocorrelation suggests mean reversion
        is_cointegrated = ac1 < -0.1 and abs(corr) > 0.5
    else:
        is_cointegrated = False
    
    return corr, spread_std, is_cointegrated


def find_cointegrated_pairs(
    df: pd.DataFrame,
    price_col: str = 'avg_price',
    market_col: str = 'conditionId',
    min_samples: int = 30,
    min_correlation: float = 0.5,
) -> List[Tuple[str, str, float, float]]:
    """
    Find cointegrated market pairs.
    
    Returns:
        List of (market1, market2, correlation, spread_std) tuples
    """
    if market_col not in df.columns:
        return []
    
    # Get price series for each market
    market_series = {}
    for market_id, market_data in df.groupby(market_col):
        if len(market_data) >= min_samples:
            if 'timestamp' in market_data.columns:
                market_data = market_data.sort_values('timestamp')
            market_series[market_id] = market_data[price_col].values
    
    # Find pairs
    pairs = []
    market_ids = list(market_series.keys())
    
    for i, m1 in enumerate(market_ids):
        for m2 in market_ids[i+1:]:
            s1, s2 = market_series[m1], market_series[m2]
            
            # Align series (use shorter length)
            min_len = min(len(s1), len(s2))
            s1, s2 = s1[:min_len], s2[:min_len]
            
            corr, spread_std, is_coint = compute_cointegration(s1, s2)
            
            if is_coint and abs(corr) >= min_correlation:
                pairs.append((m1, m2, corr, spread_std))
    
    # Sort by correlation
    pairs.sort(key=lambda x: -abs(x[2]))
    
    return pairs


def lead_lag_pairs_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Pairs params
    correlation_threshold: float = 0.6,
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
    lookback: int = 30,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.20,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
) -> Dict[str, Any]:
    """
    Lead-Lag Pairs Trading Strategy
    
    Trade cointegrated market pairs when the spread deviates from equilibrium.
    
    Logic:
    1. Find cointegrated pairs in training data
    2. Track spread z-score
    3. When |zscore| > entry_threshold, bet on mean reversion
    4. Exit when |zscore| < exit_threshold
    """
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    market_col = 'conditionId' if 'conditionId' in train.columns else 'market_id'
    
    if market_col not in train.columns:
        return {
            'strategy': 'lead_lag_pairs',
            'total_pnl': 0,
            'final_bankroll': initial_bankroll,
            'sharpe': 0,
            'error': 'No market column found',
        }
    
    # Find cointegrated pairs
    pairs = find_cointegrated_pairs(
        train, price_col, market_col,
        min_correlation=correlation_threshold
    )
    
    if not pairs:
        return {
            'strategy': 'lead_lag_pairs',
            'total_pnl': 0,
            'final_bankroll': initial_bankroll,
            'sharpe': 0,
            'pairs_found': 0,
            'error': 'No cointegrated pairs found',
        }
    
    # Build pair statistics from training data
    pair_stats = {}
    for m1, m2, corr, spread_std in pairs[:10]:  # Top 10 pairs
        pair_stats[(m1, m2)] = {
            'correlation': corr,
            'spread_std': spread_std,
            'spread_mean': 0.0,  # Will compute from spread
        }
    
    # Apply to test data
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    
    # Track current prices per market
    market_prices = {}
    
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    pnl_list = []
    trades = 0
    wins = 0
    
    # Active positions
    active_positions = {}  # (m1, m2) -> {'direction': +/-1, 'entry_price1': p1, 'entry_price2': p2}
    
    for idx, row in test_copy.iterrows():
        price = row['_price']
        outcome = row['_outcome']
        market_id = row.get(market_col, None)
        
        if market_id is None:
            pnl_list.append(0)
            continue
        
        market_prices[market_id] = price
        
        # Check each pair
        row_pnl = 0
        
        for (m1, m2), stats in pair_stats.items():
            if m1 not in market_prices or m2 not in market_prices:
                continue
            
            p1, p2 = market_prices[m1], market_prices[m2]
            
            # Compute spread z-score
            spread = p1 - p2 * (1 if stats['correlation'] > 0 else -1)
            if stats['spread_std'] > 0:
                zscore = (spread - stats['spread_mean']) / stats['spread_std']
            else:
                continue
            
            pair_key = (m1, m2)
            
            # Check for entry
            if pair_key not in active_positions and abs(zscore) > zscore_entry:
                # Check drawdown
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                if dd >= max_drawdown_stop:
                    continue
                
                direction = -1 if zscore > 0 else 1  # Mean reversion
                active_positions[pair_key] = {
                    'direction': direction,
                    'entry_p1': p1,
                    'entry_p2': p2,
                    'entry_zscore': zscore,
                }
                trades += 1
            
            # Check for exit (if we have a position)
            elif pair_key in active_positions:
                pos = active_positions[pair_key]
                
                if abs(zscore) < zscore_exit or np.sign(zscore) != np.sign(pos['entry_zscore']):
                    # Exit position
                    # PnL from spread convergence
                    entry_spread = pos['entry_p1'] - pos['entry_p2']
                    exit_spread = p1 - p2
                    spread_change = exit_spread - entry_spread
                    
                    position_frac = min(kelly_fraction, max_position_pct)
                    position = bankroll * position_frac
                    
                    # If we bet on spread narrowing (direction=-1), profit from spread decrease
                    pnl = pos['direction'] * spread_change * position - 2 * fee * position
                    row_pnl += pnl
                    
                    if pnl > 0:
                        wins += 1
                    
                    del active_positions[pair_key]
        
        pnl_list.append(row_pnl)
        bankroll += row_pnl
        peak_bankroll = max(peak_bankroll, bankroll)
    
    pnl_array = np.array(pnl_list)
    total_pnl = np.sum(pnl_array)
    sharpe = np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252)
    max_dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
    
    return {
        'strategy': 'lead_lag_pairs',
        'total_pnl': total_pnl,
        'final_bankroll': bankroll,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': wins / trades if trades > 0 else 0,
        'pnl_series': pnl_array,
        'pairs_found': len(pairs),
        'pairs_traded': len(pair_stats),
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dispersion and Pairs Strategies')
    parser.add_argument('--data', type=str, default='optimization_cache.parquet')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['all', 'dispersion', 'pairs'])
    parser.add_argument('--train-ratio', type=float, default=0.7)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Split
    n = len(df)
    train_idx = int(n * args.train_ratio)
    train = df.iloc[:train_idx].copy()
    test = df.iloc[train_idx:].copy()
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    if args.strategy in ['all', 'dispersion']:
        print(f"\n{'='*60}")
        print("Running Dispersion Strategy...")
        result = dispersion_strategy(train, test)
        print(f"  PnL: ${result['total_pnl']:,.0f}")
        print(f"  Sharpe: {result['sharpe']:.2f}")
        print(f"  Trades: {result['trades']}")
        print(f"  Win Rate: {result.get('win_rate', 0):.1%}")
    
    if args.strategy in ['all', 'pairs']:
        print(f"\n{'='*60}")
        print("Running Lead-Lag Pairs Strategy...")
        result = lead_lag_pairs_strategy(train, test)
        print(f"  PnL: ${result['total_pnl']:,.0f}")
        print(f"  Sharpe: {result['sharpe']:.2f}")
        print(f"  Trades: {result.get('trades', 0)}")
        print(f"  Pairs Found: {result.get('pairs_found', 0)}")
        if 'error' in result:
            print(f"  Note: {result['error']}")
