#!/usr/bin/env python3
"""
Hyperparameter optimization on 2025 OHLCV candle data.
Uses CMA-ES to find optimal entry/exit thresholds for maximum daily PnL.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cma
import numpy as np
import pandas as pd

# Parameter bounds
PARAM_BOUNDS = {
    'hi_threshold': (0.60, 0.90),      # Fade high prices above this
    'lo_threshold': (0.10, 0.40),      # Fade low prices below this
    'profit_take_pct': (5.0, 30.0),    # Take profit %
    'stop_loss_pct': (5.0, 25.0),      # Stop loss %
    'max_position_pct': (0.05, 0.20),  # Position size as % of bankroll
    'max_positions': (10, 100),        # Max concurrent positions
    'min_edge': (0.01, 0.15),          # Minimum edge to enter
}

# Load data once
print("Loading 2025 OHLCV data...")
DATA_PATH = Path("data/candles/polymarket_2025_1min.parquet")
df = pd.read_parquet(DATA_PATH)

# Convert to hourly for speed
print("Converting to hourly candles...")
df['ts_h'] = df['timestamp'].dt.floor('1h')
df_hourly = df.groupby(['market_id', 'ts_h']).agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_count': 'sum'
}).reset_index().rename(columns={'ts_h': 'timestamp'})
df_hourly = df_hourly.sort_values('timestamp')

print(f"Data: {len(df_hourly):,} hourly candles, {df_hourly.market_id.nunique()} markets")
print(f"Range: {df_hourly.timestamp.min()} to {df_hourly.timestamp.max()}")
print()


def params_to_dict(x):
    """Convert parameter vector to dictionary."""
    keys = list(PARAM_BOUNDS.keys())
    params = {}
    for i, key in enumerate(keys):
        lo, hi = PARAM_BOUNDS[key]
        params[key] = lo + (hi - lo) * x[i]
    params['max_positions'] = int(params['max_positions'])
    return params


def run_backtest(params: dict) -> dict:
    """Run backtest with given parameters."""
    bankroll = 10000
    positions = {}
    closed_trades = []
    
    for _, row in df_hourly.iterrows():
        market_id = row['market_id']
        price = row['close']
        ts = row['timestamp']
        
        # Exit check
        if market_id in positions:
            pos = positions[market_id]
            if pos['side'] == 'NO':
                pnl_pct = (pos['entry'] - price) / pos['entry'] * 100
            else:
                pnl_pct = (price - pos['entry']) / (1 - pos['entry']) * 100
            
            # Exit conditions
            if pnl_pct >= params['profit_take_pct'] or pnl_pct <= -params['stop_loss_pct']:
                pnl = pos['size'] * pnl_pct / 100
                bankroll += pnl
                closed_trades.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'ts': ts,
                    'exit': 'profit' if pnl_pct > 0 else 'stop'
                })
                del positions[market_id]
        
        # Entry check
        if market_id not in positions and len(positions) < params['max_positions']:
            edge = 0
            side = None
            
            if price > params['hi_threshold']:
                edge = price - 0.5
                side = 'NO'
            elif price < params['lo_threshold']:
                edge = 0.5 - price
                side = 'YES'
            
            if edge > params['min_edge']:
                size = min(bankroll * params['max_position_pct'], 1000)
                if size > 10 and bankroll > size:
                    positions[market_id] = {
                        'entry': price,
                        'side': side,
                        'size': size,
                        'entry_time': ts
                    }
    
    # Calculate metrics
    if not closed_trades:
        return {'daily_pnl': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0, 'drawdown': 0}
    
    trades_df = pd.DataFrame(closed_trades)
    total_pnl = trades_df['pnl'].sum()
    
    # Daily stats
    daily = trades_df.groupby(trades_df['ts'].dt.date)['pnl'].sum()
    days = len(daily)
    daily_pnl = total_pnl / max(days, 1)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    
    # Drawdown
    cumsum = trades_df['pnl'].cumsum()
    peak = cumsum.cummax()
    drawdown = ((peak - cumsum) / (10000 + peak)).max() if len(cumsum) > 0 else 0
    
    win_rate = (trades_df['pnl'] > 0).mean()
    
    return {
        'daily_pnl': daily_pnl,
        'sharpe': sharpe,
        'trades': len(closed_trades),
        'win_rate': win_rate,
        'drawdown': drawdown,
        'total_pnl': total_pnl,
        'final_bankroll': bankroll
    }


def objective(x):
    """Objective function: maximize daily PnL with Sharpe bonus."""
    params = params_to_dict(x)
    result = run_backtest(params)
    
    # Objective: daily PnL + Sharpe bonus - drawdown penalty
    score = result['daily_pnl'] + result['sharpe'] * 10 - result['drawdown'] * 100
    
    # Penalty for too few trades
    if result['trades'] < 50:
        score -= (50 - result['trades']) * 2
    
    return -score  # CMA-ES minimizes


def main():
    print("=" * 60)
    print("2025 OHLCV HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print()
    
    # Initial guess (middle of bounds)
    x0 = [0.5] * len(PARAM_BOUNDS)
    sigma0 = 0.3
    
    # CMA-ES options
    opts = {
        'maxiter': 30,
        'popsize': 15,
        'tolfun': 1e-4,
        'verb_disp': 1,
    }
    
    print(f"Running CMA-ES optimization...")
    print(f"  Population: {opts['popsize']}")
    print(f"  Iterations: {opts['maxiter']}")
    print()
    
    # Run optimization
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    best_result = None
    best_params = None
    best_score = float('inf')
    
    while not es.stop():
        solutions = es.ask()
        fitness = []
        
        for sol in solutions:
            try:
                f = objective(sol)
                fitness.append(f)
                
                if f < best_score:
                    best_score = f
                    best_params = params_to_dict(sol)
                    best_result = run_backtest(best_params)
            except Exception as e:
                fitness.append(1e6)
        
        es.tell(solutions, fitness)
        
        if best_result:
            print(f"  Gen {es.countiter}: daily=${best_result['daily_pnl']:.0f}, "
                  f"sharpe={best_result['sharpe']:.1f}, trades={best_result['trades']}, "
                  f"WR={best_result['win_rate']*100:.0f}%")
    
    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    if best_params:
        print("\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        print("\nBacktest Results:")
        for k, v in best_result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        
        # Save results
        output = {
            'params': best_params,
            **best_result,
            'optimized_on': '2025_ohlcv',
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = Path("optimization_results/best_params_2025.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nSaved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
