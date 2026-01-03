#!/usr/bin/env python3
"""
Fast vectorized hyperparameter optimization on 2025 OHLCV data.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 60, flush=True)
print("FAST 2025 OHLCV OPTIMIZATION", flush=True)
print("=" * 60, flush=True)
print(flush=True)

# Load data
print("Loading data...", flush=True)
df = pd.read_parquet("data/candles/polymarket_2025_1min.parquet")

# Convert to hourly
df['ts_h'] = df['timestamp'].dt.floor('1h')
hourly = df.groupby(['market_id', 'ts_h']).agg({
    'close': 'last', 'high': 'max', 'low': 'min'
}).reset_index().rename(columns={'ts_h': 'timestamp'})

# Add next hour's price for quick PnL calc
hourly = hourly.sort_values(['market_id', 'timestamp'])
hourly['next_close'] = hourly.groupby('market_id')['close'].shift(-1)
hourly = hourly.dropna()

print(f"Data: {len(hourly):,} hourly candles", flush=True)
print(flush=True)


def fast_backtest(hi_thresh, lo_thresh, profit_take, stop_loss, max_pos_pct):
    """Vectorized backtest - much faster than row iteration."""
    
    # Entry signals
    no_entry = hourly['close'] > hi_thresh
    yes_entry = hourly['close'] < lo_thresh
    
    # Calculate potential returns
    hourly['no_return'] = (hourly['close'] - hourly['next_close']) / hourly['close']
    hourly['yes_return'] = (hourly['next_close'] - hourly['close']) / (1 - hourly['close'])
    
    # Apply exit logic (simplified: 1-hour hold with profit/stop)
    no_trades = hourly[no_entry].copy()
    yes_trades = hourly[yes_entry].copy()
    
    trades = []
    
    if len(no_trades) > 0:
        no_trades['pnl_pct'] = no_trades['no_return'] * 100
        # Clip to profit take / stop loss
        no_trades['pnl_pct'] = no_trades['pnl_pct'].clip(-stop_loss, profit_take)
        no_trades['pnl'] = 500 * no_trades['pnl_pct'] / 100
        trades.append(no_trades[['timestamp', 'pnl']])
    
    if len(yes_trades) > 0:
        yes_trades['pnl_pct'] = yes_trades['yes_return'] * 100
        yes_trades['pnl_pct'] = yes_trades['pnl_pct'].clip(-stop_loss, profit_take)
        yes_trades['pnl'] = 500 * yes_trades['pnl_pct'] / 100
        trades.append(yes_trades[['timestamp', 'pnl']])
    
    if not trades:
        return {'daily_pnl': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0}
    
    all_trades = pd.concat(trades, ignore_index=True)
    
    # Daily metrics
    daily = all_trades.groupby(all_trades['timestamp'].dt.date)['pnl'].sum()
    daily_pnl = daily.mean()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    
    return {
        'daily_pnl': daily_pnl,
        'sharpe': sharpe,
        'trades': len(all_trades),
        'win_rate': (all_trades['pnl'] > 0).mean(),
        'total_pnl': all_trades['pnl'].sum()
    }


# Grid search over parameters
print("Running grid search...", flush=True)

best_result = None
best_params = None
best_score = -1e9

results = []

for hi_thresh in [0.65, 0.70, 0.75, 0.80, 0.85]:
    for lo_thresh in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for profit_take in [10, 15, 20, 25]:
            for stop_loss in [5, 10, 15]:
                
                result = fast_backtest(hi_thresh, lo_thresh, profit_take, stop_loss, 0.1)
                
                # Score: daily PnL + Sharpe bonus
                score = result['daily_pnl'] + result['sharpe'] * 5
                
                params = {
                    'hi_threshold': hi_thresh,
                    'lo_threshold': lo_thresh,
                    'profit_take_pct': profit_take,
                    'stop_loss_pct': stop_loss,
                    'max_position_pct': 0.1,
                    'max_positions': 50,
                    'min_edge': 0.01
                }
                
                results.append({**params, **result, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    print(f"  New best: hi={hi_thresh}, lo={lo_thresh}, PT={profit_take}%, SL={stop_loss}% "
                          f"-> ${result['daily_pnl']:.0f}/day, Sharpe={result['sharpe']:.1f}", flush=True)

print(flush=True)
print("=" * 60, flush=True)
print("OPTIMIZATION COMPLETE", flush=True)
print("=" * 60, flush=True)

print(f"\nBest Parameters:", flush=True)
for k, v in best_params.items():
    print(f"  {k}: {v}", flush=True)

print(f"\nBacktest Results:", flush=True)
print(f"  Daily PnL: ${best_result['daily_pnl']:.2f}", flush=True)
print(f"  Sharpe: {best_result['sharpe']:.2f}", flush=True)
print(f"  Total trades: {best_result['trades']:,}", flush=True)
print(f"  Win rate: {best_result['win_rate']*100:.1f}%", flush=True)
print(f"  Total PnL: ${best_result['total_pnl']:,.2f}", flush=True)

# Save results
output = {
    'params': best_params,
    **best_result,
    'optimized_on': '2025_ohlcv_fast',
    'timestamp': datetime.now().isoformat()
}

output_path = Path("optimization_results/best_params_2025.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {output_path}", flush=True)

# Also save all results
results_df = pd.DataFrame(results)
results_df.to_parquet("optimization_results/grid_search_2025.parquet", index=False)
print(f"All results saved to optimization_results/grid_search_2025.parquet", flush=True)
