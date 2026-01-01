"""
Momentum and Trend-Following Strategies for Prediction Markets

This module implements momentum-based strategies that complement the existing
mean-reversion/calibration approaches. The key insight is that when markets
are persistently miscalibrated, momentum (not mean-reversion) is the right approach.

Strategies:
1. Calibration Momentum - trade WITH persistent miscalibration
2. Trend Following - classic dual-MA and breakout strategies
3. Cross-Category Momentum - spillover effects between categories
4. Event Momentum - short-term momentum after price jumps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class MomentumSignal:
    """Container for momentum signal information"""
    direction: int  # +1 for long, -1 for short, 0 for no position
    strength: float  # Signal strength (0-1)
    signal_type: str  # e.g., 'calibration_momentum', 'trend', 'event'
    confidence: float  # Confidence in the signal


def compute_rolling_calibration_error(
    df: pd.DataFrame,
    price_col: str = 'avg_price',
    outcome_col: str = 'y',
    window: int = 30,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling calibration error by price bin.
    
    Returns DataFrame with columns:
    - price_bin: the price bin
    - rolling_cal_error: the rolling calibration error (outcome_rate - price)
    - error_persistence: how many consecutive periods with same-sign error
    """
    df = df.copy()
    df['_price'] = df[price_col].clip(0.01, 0.99)
    df['_outcome'] = df[outcome_col]
    df['price_bin'] = pd.cut(df['_price'], bins=n_bins, labels=False)
    
    # Group by price bin and compute rolling stats
    results = []
    
    for bin_id in range(n_bins):
        bin_data = df[df['price_bin'] == bin_id].copy()
        if len(bin_data) < window:
            continue
            
        # Rolling mean of outcomes and prices
        bin_data['rolling_outcome'] = bin_data['_outcome'].rolling(window, min_periods=10).mean()
        bin_data['rolling_price'] = bin_data['_price'].rolling(window, min_periods=10).mean()
        bin_data['rolling_cal_error'] = bin_data['rolling_outcome'] - bin_data['rolling_price']
        
        # Compute error persistence (consecutive periods with same sign)
        error_sign = np.sign(bin_data['rolling_cal_error'])
        persistence = []
        count = 0
        prev_sign = 0
        for s in error_sign:
            if np.isnan(s):
                persistence.append(0)
                count = 0
            elif s == prev_sign:
                count += 1
                persistence.append(count)
            else:
                count = 1
                persistence.append(count)
                prev_sign = s
        bin_data['error_persistence'] = persistence
        
        results.append(bin_data)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return df


def calibration_momentum_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Momentum params
    persistence_window: int = 30,
    momentum_threshold: float = 0.10,
    decay_halflife: float = 20.0,
    min_persistence_days: int = 20,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.20,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calibration Momentum Strategy
    
    Instead of betting AGAINST miscalibration (mean reversion), this strategy
    bets WITH persistent miscalibration. If markets at 0.70 keep resolving
    at 0.60 for 30+ days, bet that the next market at 0.70 will also resolve ~0.60.
    
    Logic:
    1. Compute rolling calibration error per price bin
    2. If error persists > min_persistence_days with same sign → momentum trade
    3. Bet WITH the direction of miscalibration (not against it)
    4. Apply exponential decay to reduce position as error may mean-revert
    """
    # Detect column names
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    
    # Learn momentum signals from training data
    train_with_cal = compute_rolling_calibration_error(
        train, price_col, outcome_col, persistence_window, n_bins
    )
    
    # Build momentum model: for each bin, track if there's persistent miscalibration
    bin_momentum = {}
    for bin_id in range(n_bins):
        bin_data = train_with_cal[train_with_cal['price_bin'] == bin_id]
        if len(bin_data) < persistence_window:
            bin_momentum[bin_id] = {'direction': 0, 'strength': 0, 'persistence': 0}
            continue
        
        # Get last calibration error and persistence
        last_valid = bin_data[bin_data['rolling_cal_error'].notna()].tail(1)
        if len(last_valid) == 0:
            bin_momentum[bin_id] = {'direction': 0, 'strength': 0, 'persistence': 0}
            continue
            
        cal_error = last_valid['rolling_cal_error'].values[0]
        persistence = last_valid['error_persistence'].values[0]
        
        # Only trade if error is large and persistent
        if abs(cal_error) >= momentum_threshold and persistence >= min_persistence_days:
            direction = 1 if cal_error > 0 else -1  # Bet WITH the miscalibration
            strength = min(abs(cal_error) / 0.20, 1.0)  # Scale by error magnitude
            bin_momentum[bin_id] = {
                'direction': direction,
                'strength': strength,
                'persistence': persistence,
                'cal_error': cal_error
            }
        else:
            bin_momentum[bin_id] = {'direction': 0, 'strength': 0, 'persistence': 0}
    
    # Apply strategy to test data
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    test_copy['price_bin'] = pd.cut(test_copy['_price'], bins=n_bins, labels=False)
    
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    pnl_list = []
    trades = 0
    wins = 0
    
    for idx, row in test_copy.iterrows():
        price = row['_price']
        outcome = row['_outcome']
        bin_id = row['price_bin']
        
        if pd.isna(bin_id) or bin_id not in bin_momentum:
            pnl_list.append(0)
            continue
        
        momentum = bin_momentum[bin_id]
        if momentum['direction'] == 0 or momentum['strength'] < 0.1:
            pnl_list.append(0)
            continue
        
        # Check drawdown
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            pnl_list.append(0)
            continue
        
        trades += 1
        
        # Compute expected edge based on momentum
        # If cal_error > 0, outcomes are higher than prices → bet YES
        # If cal_error < 0, outcomes are lower than prices → bet NO
        direction = momentum['direction']
        
        # Position sizing: Kelly-based with momentum strength (fixed sizing, not compounding)
        edge = abs(momentum.get('cal_error', momentum_threshold))
        position_frac = kelly_fraction * momentum['strength'] * min(edge / 0.10, 1.0)
        position_frac = min(position_frac, max_position_pct)
        
        # Use fixed position size based on initial bankroll (not compounding)
        position = initial_bankroll * position_frac
        
        # PnL calculation
        if direction > 0:  # Long YES
            pnl = position * (outcome - price) - fee * position
        else:  # Long NO (short YES)
            pnl = position * (price - outcome) - fee * position
        
        pnl_list.append(pnl)
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        
        if pnl > 0:
            wins += 1
    
    # Compute metrics
    pnl_array = np.array(pnl_list)
    total_pnl = np.sum(pnl_array)
    sharpe = np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252)
    max_dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
    win_rate = wins / trades if trades > 0 else 0
    
    return {
        'strategy': 'calibration_momentum',
        'total_pnl': total_pnl,
        'final_bankroll': bankroll,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'pnl_series': pnl_array,
        'bin_momentum': bin_momentum,
    }


def trend_following_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Trend params
    fast_ma: int = 5,
    slow_ma: int = 20,
    trend_strength_threshold: float = 0.02,
    breakout_lookback: int = 10,
    breakout_percentile: float = 0.8,
    use_breakout: bool = True,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.20,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
) -> Dict[str, Any]:
    """
    Trend Following Strategy for Prediction Markets
    
    Implements classic trend-following signals adapted to prediction markets:
    1. Dual Moving Average Crossover - fast MA above slow MA → bullish
    2. Donchian Channel Breakout - price breaks above/below recent range
    
    Note: This requires time-series data per market. If data is cross-sectional,
    we approximate using recent observation ordering.
    """
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    market_col = 'conditionId' if 'conditionId' in train.columns else 'market_id'
    
    # For trend following, we need per-market time series
    # Group by market and compute moving averages
    
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    
    # Learn trend signals from training data
    market_trends = {}
    
    if market_col in train_copy.columns:
        for market_id, market_data in train_copy.groupby(market_col):
            if len(market_data) < slow_ma:
                continue
            
            # Sort by time if available
            if 'timestamp' in market_data.columns:
                market_data = market_data.sort_values('timestamp')
            
            prices = market_data['_price'].values
            
            # Compute MAs
            fast_ma_vals = pd.Series(prices).rolling(fast_ma, min_periods=1).mean().values
            slow_ma_vals = pd.Series(prices).rolling(slow_ma, min_periods=1).mean().values
            
            # Last values
            fast_last = fast_ma_vals[-1]
            slow_last = slow_ma_vals[-1]
            price_last = prices[-1]
            
            # Trend signal: fast above slow → bullish
            trend_diff = fast_last - slow_last
            
            # Breakout signal
            breakout_signal = 0
            if use_breakout and len(prices) >= breakout_lookback:
                high = np.percentile(prices[-breakout_lookback:], breakout_percentile * 100)
                low = np.percentile(prices[-breakout_lookback:], (1 - breakout_percentile) * 100)
                if price_last > high:
                    breakout_signal = 1
                elif price_last < low:
                    breakout_signal = -1
            
            # Combined signal
            if abs(trend_diff) >= trend_strength_threshold:
                direction = 1 if trend_diff > 0 else -1
                strength = min(abs(trend_diff) / 0.10, 1.0)
                
                # Boost if breakout confirms
                if breakout_signal == direction:
                    strength = min(strength * 1.5, 1.0)
                
                market_trends[market_id] = {
                    'direction': direction,
                    'strength': strength,
                    'trend_diff': trend_diff,
                    'breakout': breakout_signal,
                }
            else:
                market_trends[market_id] = {'direction': 0, 'strength': 0}
    
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
        market_id = row.get(market_col, None)
        
        if market_id is None or market_id not in market_trends:
            pnl_list.append(0)
            continue
        
        trend = market_trends[market_id]
        if trend['direction'] == 0:
            pnl_list.append(0)
            continue
        
        # Check drawdown
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            pnl_list.append(0)
            continue
        
        trades += 1
        direction = trend['direction']
        
        # Position sizing
        position_frac = kelly_fraction * trend['strength']
        position_frac = min(position_frac, max_position_pct)
        position = bankroll * position_frac
        
        # PnL
        if direction > 0:  # Bullish → expect outcome = 1
            pnl = position * (outcome - price) - fee * position
        else:  # Bearish → expect outcome = 0
            pnl = position * (price - outcome) - fee * position
        
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
        'strategy': 'trend_following',
        'total_pnl': total_pnl,
        'final_bankroll': bankroll,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': wins / trades if trades > 0 else 0,
        'pnl_series': pnl_array,
        'market_trends': market_trends,
    }


def event_momentum_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Event detection params
    event_detection_threshold: float = 0.05,
    momentum_duration: int = 3,
    reversal_after: int = 7,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.30,
    max_position_pct: float = 0.25,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
) -> Dict[str, Any]:
    """
    Event Momentum Strategy
    
    After major events (detected via price jumps), momentum persists
    for a short period before mean reversion kicks in.
    
    Logic:
    1. Detect "events" as price jumps > threshold
    2. For momentum_duration days after event, trade WITH the jump
    3. After reversal_after days, switch to mean reversion (or skip)
    """
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    market_col = 'conditionId' if 'conditionId' in train.columns else 'market_id'
    
    # Detect events in training data
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    
    # Learn event patterns: what happens after large price moves?
    event_outcomes = []
    
    if market_col in train_copy.columns:
        for market_id, market_data in train_copy.groupby(market_col):
            if len(market_data) < 2:
                continue
            
            if 'timestamp' in market_data.columns:
                market_data = market_data.sort_values('timestamp')
            
            prices = market_data['_price'].values
            outcomes = market_data['y' if 'y' in market_data.columns else 'outcome'].values
            
            # Detect price jumps
            price_changes = np.diff(prices)
            for i, change in enumerate(price_changes):
                if abs(change) >= event_detection_threshold:
                    # This is an event
                    direction = 1 if change > 0 else -1
                    
                    # Look at subsequent outcomes
                    if i + 1 < len(outcomes):
                        actual_outcome = outcomes[i + 1]
                        event_outcomes.append({
                            'jump_direction': direction,
                            'jump_magnitude': abs(change),
                            'outcome': actual_outcome,
                            'price_at_event': prices[i + 1],
                        })
    
    # Learn: does momentum predict outcomes after events?
    if event_outcomes:
        momentum_wins = sum(
            1 for e in event_outcomes
            if (e['jump_direction'] > 0 and e['outcome'] > e['price_at_event']) or
               (e['jump_direction'] < 0 and e['outcome'] < e['price_at_event'])
        )
        momentum_edge = momentum_wins / len(event_outcomes) - 0.5 if event_outcomes else 0
    else:
        momentum_edge = 0
    
    # Apply to test data - detect events and trade momentum
    test_copy = test.copy()
    test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
    test_copy['_outcome'] = test_copy[outcome_col]
    
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    pnl_list = []
    trades = 0
    wins = 0
    
    # Track active events per market
    active_events = {}  # market_id -> {direction, strength, days_since}
    
    prev_prices = {}  # market_id -> previous price
    
    for idx, row in test_copy.iterrows():
        price = row['_price']
        outcome = row['_outcome']
        market_id = row.get(market_col, idx)
        
        # Check for event
        if market_id in prev_prices:
            price_change = price - prev_prices[market_id]
            if abs(price_change) >= event_detection_threshold:
                # New event detected
                active_events[market_id] = {
                    'direction': 1 if price_change > 0 else -1,
                    'magnitude': abs(price_change),
                    'days_since': 0,
                }
        
        prev_prices[market_id] = price
        
        # Check if we have an active event for this market
        if market_id not in active_events:
            pnl_list.append(0)
            continue
        
        event = active_events[market_id]
        event['days_since'] += 1
        
        # Only trade during momentum phase
        if event['days_since'] > momentum_duration:
            # Could switch to mean reversion here, but for now just skip
            pnl_list.append(0)
            if event['days_since'] > reversal_after:
                del active_events[market_id]
            continue
        
        # Check drawdown
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            pnl_list.append(0)
            continue
        
        trades += 1
        direction = event['direction']
        
        # Strength decays with time since event
        decay = 1.0 - (event['days_since'] / momentum_duration) * 0.5
        strength = min(event['magnitude'] / 0.10, 1.0) * decay
        
        # Position sizing
        position_frac = kelly_fraction * strength * (0.5 + momentum_edge)
        position_frac = min(position_frac, max_position_pct)
        position = bankroll * position_frac
        
        # PnL
        if direction > 0:
            pnl = position * (outcome - price) - fee * position
        else:
            pnl = position * (price - outcome) - fee * position
        
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
        'strategy': 'event_momentum',
        'total_pnl': total_pnl,
        'final_bankroll': bankroll,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': wins / trades if trades > 0 else 0,
        'pnl_series': pnl_array,
        'momentum_edge': momentum_edge,
        'events_detected': len(event_outcomes),
    }


def cross_category_momentum_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    # Momentum params
    momentum_lookback: int = 14,
    lead_lag_window: int = 3,
    correlation_threshold: float = 0.3,
    spillover_decay: float = 0.5,
    # Sizing params
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.20,
    fee: float = 0.01,
    # Risk management
    max_drawdown_stop: float = 0.50,
) -> Dict[str, Any]:
    """
    Cross-Category Momentum Strategy
    
    Momentum in one category often spills over to related categories.
    E.g., positive momentum in "Trump wins X" may predict positive
    momentum in other Trump-related markets.
    
    Logic:
    1. Compute category-level momentum (avg price change)
    2. Identify lead-lag relationships between categories
    3. Trade lagging categories in direction of leading category momentum
    """
    price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
    outcome_col = 'y' if 'y' in train.columns else 'outcome'
    category_col = 'category' if 'category' in train.columns else None
    
    if category_col is None or category_col not in train.columns:
        # Can't run without categories
        return {
            'strategy': 'cross_category_momentum',
            'total_pnl': 0,
            'final_bankroll': initial_bankroll,
            'sharpe': 0,
            'max_drawdown': 0,
            'trades': 0,
            'win_rate': 0,
            'error': 'No category column found',
        }
    
    train_copy = train.copy()
    train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
    train_copy['_outcome'] = train_copy[outcome_col]
    
    # Compute category-level momentum
    category_momentum = {}
    category_outcomes = {}
    
    for category, cat_data in train_copy.groupby(category_col):
        if len(cat_data) < momentum_lookback:
            continue
        
        # Average price and outcome for category
        avg_price = cat_data['_price'].mean()
        avg_outcome = cat_data['_outcome'].mean()
        
        # Simple momentum: recent avg - older avg
        recent = cat_data.tail(momentum_lookback // 2)['_price'].mean()
        older = cat_data.head(momentum_lookback // 2)['_price'].mean()
        momentum = recent - older
        
        category_momentum[category] = {
            'momentum': momentum,
            'avg_price': avg_price,
            'direction': 1 if momentum > 0 else -1 if momentum < 0 else 0,
            'strength': min(abs(momentum) / 0.10, 1.0),
        }
        category_outcomes[category] = avg_outcome
    
    # Find lead-lag relationships (simplified: use correlation of momentum directions)
    # For now, assume all categories can influence each other with decay
    
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
        
        if category is None or category not in category_momentum:
            pnl_list.append(0)
            continue
        
        own_momentum = category_momentum[category]
        
        # Compute spillover from other categories
        spillover_signal = 0
        for other_cat, other_mom in category_momentum.items():
            if other_cat == category:
                continue
            spillover_signal += other_mom['direction'] * other_mom['strength'] * spillover_decay
        
        # Combine own momentum with spillover
        combined_direction = own_momentum['direction']
        if own_momentum['strength'] < 0.3 and abs(spillover_signal) > 0.5:
            # Weak own signal, strong spillover → use spillover
            combined_direction = 1 if spillover_signal > 0 else -1
        
        combined_strength = max(own_momentum['strength'], abs(spillover_signal))
        
        if combined_strength < 0.2:
            pnl_list.append(0)
            continue
        
        # Check drawdown
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd >= max_drawdown_stop:
            pnl_list.append(0)
            continue
        
        trades += 1
        
        # Position sizing
        position_frac = kelly_fraction * combined_strength
        position_frac = min(position_frac, max_position_pct)
        position = bankroll * position_frac
        
        # PnL
        if combined_direction > 0:
            pnl = position * (outcome - price) - fee * position
        else:
            pnl = position * (price - outcome) - fee * position
        
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
        'strategy': 'cross_category_momentum',
        'total_pnl': total_pnl,
        'final_bankroll': bankroll,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': wins / trades if trades > 0 else 0,
        'pnl_series': pnl_array,
        'category_momentum': category_momentum,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run momentum strategies')
    parser.add_argument('--data', type=str, default='optimization_cache.parquet')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['all', 'calibration_momentum', 'trend_following', 
                                 'event_momentum', 'cross_category_momentum'])
    parser.add_argument('--train-ratio', type=float, default=0.7)
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Split train/test
    n = len(df)
    train_idx = int(n * args.train_ratio)
    train = df.iloc[:train_idx].copy()
    test = df.iloc[train_idx:].copy()
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    strategies = {
        'calibration_momentum': calibration_momentum_strategy,
        'trend_following': trend_following_strategy,
        'event_momentum': event_momentum_strategy,
        'cross_category_momentum': cross_category_momentum_strategy,
    }
    
    if args.strategy == 'all':
        for name, func in strategies.items():
            print(f"\n{'='*60}")
            print(f"Running {name}...")
            result = func(train, test)
            print(f"  PnL: ${result['total_pnl']:,.0f}")
            print(f"  Sharpe: {result['sharpe']:.2f}")
            print(f"  Trades: {result['trades']}")
            print(f"  Win Rate: {result['win_rate']:.1%}")
            print(f"  Max DD: {result['max_drawdown']:.1%}")
    else:
        result = strategies[args.strategy](train, test)
        print(f"\n{args.strategy} Results:")
        for k, v in result.items():
            if not isinstance(v, (np.ndarray, dict)):
                print(f"  {k}: {v}")
