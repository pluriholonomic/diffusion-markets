#!/usr/bin/env python3
"""
Polymarket Calibration Backtest using Goldsky Historical Data

Uses the full historical trade data from The Graph/Goldsky to run
a proper calibration-based statistical arbitrage backtest.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def categorize_question(q: str) -> str:
    """Categorize a market question."""
    q = str(q).lower()
    
    if any(x in q for x in ['bitcoin', 'btc', 'eth', 'crypto', 'solana', 'dogecoin']):
        return 'crypto'
    if any(x in q for x in ['trump', 'biden', 'harris', 'election', 'president', 'congress', 'senate']):
        return 'politics'
    if any(x in q for x in ['nba', 'nfl', 'mlb', 'nhl', 'win', 'beat', 'championship', 'super bowl', 'world series']):
        return 'sports'
    if any(x in q for x in ['temperature', 'weather', 'rain', 'snow', 'hurricane']):
        return 'weather'
    if any(x in q for x in ['fed', 'rate', 'inflation', 'gdp', 'unemployment']):
        return 'economics'
    
    return 'other'


def run_backtest(
    data_path: Path,
    output_dir: Path,
    train_frac: float = 0.5,
    min_trades: int = 10,
    min_date: str = None,
    verbose: bool = True
):
    """Run the calibration-weighted backtest."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("POLYMARKET CALIBRATION BACKTEST")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_parquet(data_path)
    print(f"   Total markets: {len(df):,}")
    
    # Filter to markets with prices and outcomes
    df = df[df['first_price'].notna()].copy()
    print(f"   With prices: {len(df):,}")
    
    # Parse dates - handle various formats
    if 'closedTime' in df.columns:
        # Parse with mixed formats and handle timezone
        df['resolution_time'] = pd.to_datetime(df['closedTime'], format='mixed', utc=True, errors='coerce')
    
    before_filter = len(df)
    df = df[df['resolution_time'].notna()].copy()
    print(f"   With resolution time: {len(df):,} (dropped {before_filter - len(df):,})")
    
    # Filter by date
    if min_date:
        min_dt = pd.to_datetime(min_date, utc=True)
        # Make resolution_time tz-aware if needed
        if df['resolution_time'].dt.tz is None:
            df['resolution_time'] = df['resolution_time'].dt.tz_localize('UTC')
        df = df[df['resolution_time'] >= min_dt]
        print(f"   After {min_date}: {len(df):,}")
    
    # Filter by trade count
    if 'n_trades' in df.columns:
        df = df[df['n_trades'] >= min_trades]
        print(f"   With >= {min_trades} trades: {len(df):,}")
    
    # Determine outcome (YES = 1, NO = 0)
    # For Polymarket, we need to infer from final price
    if 'last_price' in df.columns:
        # If last price is close to 1, YES won; close to 0, NO won
        df['outcome'] = (df['last_price'] > 0.5).astype(int)
    
    print(f"   Outcome distribution: YES={df['outcome'].mean():.1%}")
    
    # Categorize
    print("\n2. Categorizing markets...")
    df['category'] = df['question'].apply(categorize_question)
    print(df['category'].value_counts().to_string())
    
    # Sort by time and split
    print("\n3. Train/test split...")
    df = df.sort_values('resolution_time')
    
    train_size = int(len(df) * train_frac)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    print(f"   Train: {len(train):,} ({train['resolution_time'].min().strftime('%Y-%m-%d')} to {train['resolution_time'].max().strftime('%Y-%m-%d')})")
    print(f"   Test: {len(test):,} ({test['resolution_time'].min().strftime('%Y-%m-%d')} to {test['resolution_time'].max().strftime('%Y-%m-%d')})")
    
    # Learn calibration per category
    print("\n4. Learning calibration from training data...")
    calibrations = {}
    
    # Use lower threshold if not enough data
    min_samples = min(50, len(train) // 5)
    
    for cat in train['category'].unique():
        cat_df = train[train['category'] == cat]
        if len(cat_df) >= min_samples:
            y_rate = cat_df['outcome'].mean()
            price_mean = cat_df['first_price'].mean()
            calib = y_rate - price_mean
            
            calibrations[cat] = {
                'calibration': calib,
                'y_rate': y_rate,
                'price_mean': price_mean,
                'n': len(cat_df),
                'direction': -1 if calib < 0 else 1  # -1 = short YES, +1 = long YES
            }
            
            print(f"   {cat:12}: calib={calib:+.3f}, y_rate={y_rate:.1%}, n={len(cat_df):,}")
    
    # Run backtest on test set
    print("\n5. Running backtest on test set...")
    
    results = []
    for _, row in test.iterrows():
        cat = row['category']
        if cat not in calibrations:
            continue
        
        calib_info = calibrations[cat]
        entry_price = row['first_price']
        outcome = row['outcome']
        direction = calib_info['direction']
        calib_strength = abs(calib_info['calibration'])
        
        # Compute weight (higher for more miscalibrated)
        weight = min(calib_strength * 5 + 0.1, 2.0)
        
        # Compute PnL
        if direction == -1:  # Short YES (bet on NO)
            pnl = entry_price if outcome == 0 else -(1 - entry_price)
        else:  # Long YES
            pnl = (1 - entry_price) if outcome == 1 else -entry_price
        
        results.append({
            'market_id': row.get('id', row.get('market_id')),
            'category': cat,
            'entry_price': entry_price,
            'outcome': outcome,
            'direction': direction,
            'weight': weight,
            'pnl': pnl,
            'weighted_pnl': weight * pnl,
            'resolution_time': row['resolution_time']
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n⚠️ No trades executed - check calibrations and data")
        print(f"Calibrations learned: {list(calibrations.keys())}")
        print(f"Test categories: {test['category'].value_counts().to_dict()}")
        return {'error': 'No trades'}
    
    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    n_trades = len(results_df)
    win_rate = (results_df['pnl'] > 0).mean()
    total_pnl = results_df['pnl'].sum()
    weighted_pnl = results_df['weighted_pnl'].sum()
    
    # Time metrics
    test_days = (test['resolution_time'].max() - test['resolution_time'].min()).days
    daily_pnl = weighted_pnl / max(test_days, 1)
    
    # Sharpe (approximate)
    daily_returns = results_df.groupby(results_df['resolution_time'].dt.date)['weighted_pnl'].sum()
    if len(daily_returns) > 1:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    print(f"""
Test Period: {test['resolution_time'].min().strftime('%Y-%m-%d')} to {test['resolution_time'].max().strftime('%Y-%m-%d')}
Duration: {test_days} days

Trades: {n_trades:,}
Win Rate: {win_rate:.1%}

Total PnL: ${total_pnl:,.2f}
Weighted PnL: ${weighted_pnl:,.2f}
Daily PnL: ${daily_pnl:,.2f}/day
Annualized: ${daily_pnl * 365:,.0f}/year
Sharpe Ratio: {sharpe:.2f}
""")
    
    # By category
    print("\nBy Category:")
    cat_summary = results_df.groupby('category').agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).mean()],
        'weighted_pnl': 'sum'
    })
    cat_summary.columns = ['trades', 'pnl', 'win_rate', 'weighted_pnl']
    print(cat_summary.round(2).to_string())
    
    # Monthly breakdown
    print("\nMonthly PnL:")
    results_df['month'] = results_df['resolution_time'].dt.to_period('M')
    monthly = results_df.groupby('month').agg({
        'pnl': ['count', 'sum'],
        'weighted_pnl': 'sum'
    })
    monthly.columns = ['trades', 'pnl', 'weighted_pnl']
    print(monthly.tail(12).to_string())
    
    # Save results
    print("\n6. Saving results...")
    
    results_df.to_parquet(output_dir / 'backtest_trades.parquet')
    print(f"   Saved: {output_dir / 'backtest_trades.parquet'}")
    
    summary = {
        'run_time': datetime.now().isoformat(),
        'data_path': str(data_path),
        'train_size': len(train),
        'test_size': len(test),
        'test_start': str(test['resolution_time'].min()),
        'test_end': str(test['resolution_time'].max()),
        'test_days': test_days,
        'n_trades': n_trades,
        'win_rate': float(win_rate),
        'total_pnl': float(total_pnl),
        'weighted_pnl': float(weighted_pnl),
        'daily_pnl': float(daily_pnl),
        'annualized_pnl': float(daily_pnl * 365),
        'sharpe': float(sharpe),
        'calibrations': calibrations,
        'by_category': cat_summary.to_dict()
    }
    
    with open(output_dir / 'backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   Saved: {output_dir / 'backtest_summary.json'}")
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Polymarket backtest")
    parser.add_argument("--data-path", type=str, 
                        default="data/polymarket_goldsky_processed/markets_with_prices.parquet",
                        help="Path to markets with prices parquet")
    parser.add_argument("--output-dir", type=str, default="runs/polymarket_backtest",
                        help="Output directory")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="Fraction of data for training")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum trades per market")
    parser.add_argument("--min-date", type=str, default=None,
                        help="Minimum resolution date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    run_backtest(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        train_frac=args.train_frac,
        min_trades=args.min_trades,
        min_date=args.min_date
    )


if __name__ == "__main__":
    main()
