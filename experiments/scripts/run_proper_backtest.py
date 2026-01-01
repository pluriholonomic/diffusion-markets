#!/usr/bin/env python3
"""
Proper Polymarket Backtest with Real Outcomes

Uses:
- Gamma API data for actual resolution outcomes (y column)
- Goldsky trade data for price history
- Matches via token IDs for correct mapping
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


def load_and_merge_data(
    gamma_path: Path,
    goldsky_markets_path: Path,
    goldsky_trades_path: Path,
    verbose: bool = True
):
    """Load and merge all data sources correctly."""
    
    print("1. Loading data sources...")
    
    # Load Gamma resolved markets (has actual outcomes)
    gamma = pd.read_parquet(gamma_path)
    print(f"   Gamma resolved markets: {len(gamma):,}")
    print(f"   Outcome distribution: YES={gamma['y'].mean():.1%}")
    
    # Load Goldsky markets (has token IDs)
    goldsky_markets = pd.read_csv(goldsky_markets_path)
    print(f"   Goldsky markets: {len(goldsky_markets):,}")
    
    # Create token → goldsky_id mapping
    # token1 corresponds to YES token
    token_to_goldsky = {}
    for _, row in goldsky_markets.iterrows():
        token_to_goldsky[str(row['token1'])] = row['id']
        token_to_goldsky[str(row['token2'])] = row['id']
    
    print(f"   Token mappings: {len(token_to_goldsky):,}")
    
    # Map Gamma markets to Goldsky IDs via yes_token_id
    gamma['goldsky_id'] = gamma['yes_token_id'].astype(str).map(token_to_goldsky)
    matched_gamma = gamma[gamma['goldsky_id'].notna()].copy()
    print(f"   Gamma markets matched to Goldsky: {len(matched_gamma):,}")
    
    # Load trade aggregations
    print("\n2. Loading trade aggregations...")
    
    # Read trades in chunks and aggregate by market
    chunk_size = 5_000_000
    agg_list = []
    total_trades = 0
    
    for i, chunk in enumerate(pd.read_csv(goldsky_trades_path, chunksize=chunk_size)):
        total_trades += len(chunk)
        
        # Aggregate by market_id
        agg = chunk.groupby('market_id').agg({
            'price': ['first', 'last', 'mean', 'count'],
            'usd_amount': 'sum',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        agg.columns = ['market_id', 'first_price', 'last_price', 'avg_price', 
                       'n_trades', 'total_volume', 'first_time', 'last_time']
        agg_list.append(agg)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"   Processed {total_trades:,} trades...")
    
    print(f"   Total trades processed: {total_trades:,}")
    
    # Combine aggregations
    all_aggs = pd.concat(agg_list)
    
    # Re-aggregate to combine chunks
    final_agg = all_aggs.groupby('market_id').agg({
        'first_price': 'first',  # Take first chunk's first price
        'last_price': 'last',    # Take last chunk's last price
        'avg_price': 'mean',
        'n_trades': 'sum',
        'total_volume': 'sum',
        'first_time': 'min',
        'last_time': 'max'
    }).reset_index()
    
    print(f"   Unique markets in trades: {len(final_agg):,}")
    
    # Now the tricky part: trades.csv market_id needs to be mapped
    # The trades market_id is the Goldsky id, but stored as float
    final_agg['goldsky_id'] = final_agg['market_id'].apply(
        lambda x: int(float(x)) if pd.notna(x) else None
    )
    
    print("\n3. Merging with outcomes...")
    
    # Merge: matched_gamma (has outcomes) + final_agg (has prices)
    matched_gamma['goldsky_id'] = matched_gamma['goldsky_id'].astype(int)
    
    merged = matched_gamma.merge(
        final_agg,
        on='goldsky_id',
        how='inner'
    )
    
    print(f"   Final merged dataset: {len(merged):,} markets with outcomes + prices")
    
    return merged


def run_backtest(
    data: pd.DataFrame,
    output_dir: Path,
    train_frac: float = 0.5,
    min_trades: int = 10,
    verbose: bool = True
):
    """Run the calibration-weighted backtest on properly merged data."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("BACKTEST WITH REAL OUTCOMES")
    print("="*70)
    
    # Filter by trade count
    df = data[data['n_trades'] >= min_trades].copy()
    print(f"\nMarkets with >= {min_trades} trades: {len(df):,}")
    
    # Parse resolution time
    df['resolution_time'] = pd.to_datetime(df['closedTime'], format='mixed', utc=True, errors='coerce')
    df = df[df['resolution_time'].notna()]
    print(f"With valid resolution time: {len(df):,}")
    
    # Categorize
    if 'category' not in df.columns or df['category'].isna().all():
        df['category'] = df['question'].apply(categorize_question)
    
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().to_string())
    
    print(f"\nOutcome distribution (REAL): YES={df['y'].mean():.1%}")
    
    # Sort by time and split
    df = df.sort_values('resolution_time')
    
    train_size = int(len(df) * train_frac)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    print(f"\nTrain: {len(train):,} ({train['resolution_time'].min().strftime('%Y-%m-%d')} to {train['resolution_time'].max().strftime('%Y-%m-%d')})")
    print(f"Test: {len(test):,} ({test['resolution_time'].min().strftime('%Y-%m-%d')} to {test['resolution_time'].max().strftime('%Y-%m-%d')})")
    
    # Learn calibration per category
    print("\nLearning calibration from training data...")
    calibrations = {}
    min_samples = max(20, len(train) // 20)
    
    for cat in train['category'].unique():
        cat_df = train[train['category'] == cat]
        if len(cat_df) >= min_samples:
            y_rate = cat_df['y'].mean()  # REAL outcome rate!
            price_mean = cat_df['first_price'].mean()
            calib = y_rate - price_mean
            
            calibrations[cat] = {
                'calibration': calib,
                'y_rate': y_rate,
                'price_mean': price_mean,
                'n': len(cat_df),
                'direction': 1 if calib > 0 else -1  # +1 = long YES, -1 = short YES
            }
            
            print(f"  {cat:12}: calib={calib:+.3f}, y_rate={y_rate:.1%}, price={price_mean:.1%}, n={len(cat_df):,}")
    
    # Run backtest on test set
    print("\nRunning backtest on test set...")
    
    results = []
    for _, row in test.iterrows():
        cat = row['category']
        if cat not in calibrations:
            continue
        
        calib_info = calibrations[cat]
        entry_price = row['first_price']
        outcome = row['y']  # REAL outcome!
        direction = calib_info['direction']
        calib_strength = abs(calib_info['calibration'])
        
        # Skip if price is extreme
        if entry_price < 0.05 or entry_price > 0.95:
            continue
        
        # Compute weight
        weight = min(calib_strength * 5 + 0.1, 2.0)
        
        # Compute PnL
        if direction == 1:  # Long YES (bet price goes to 1)
            pnl = (1 - entry_price) if outcome == 1 else -entry_price
        else:  # Short YES / Long NO (bet price goes to 0)
            pnl = entry_price if outcome == 0 else -(1 - entry_price)
        
        results.append({
            'market_id': row.get('id', row.get('goldsky_id')),
            'question': row['question'][:80],
            'category': cat,
            'entry_price': entry_price,
            'outcome': outcome,
            'direction': direction,
            'calibration': calib_info['calibration'],
            'weight': weight,
            'pnl': pnl,
            'weighted_pnl': weight * pnl,
            'resolution_time': row['resolution_time']
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("⚠️ No trades executed!")
        return {'error': 'No trades'}
    
    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS (WITH REAL OUTCOMES)")
    print("="*70)
    
    n_trades = len(results_df)
    win_rate = (results_df['pnl'] > 0).mean()
    total_pnl = results_df['pnl'].sum()
    weighted_pnl = results_df['weighted_pnl'].sum()
    
    # Time metrics
    test_days = (test['resolution_time'].max() - test['resolution_time'].min()).days
    daily_pnl = weighted_pnl / max(test_days, 1)
    
    # Sharpe
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
    print("By Category:")
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
    print(monthly.to_string())
    
    # Save results
    print("\nSaving results...")
    
    results_df.to_parquet(output_dir / 'backtest_trades.parquet')
    print(f"  Saved: {output_dir / 'backtest_trades.parquet'}")
    
    summary = {
        'run_time': datetime.now().isoformat(),
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
    print(f"  Saved: {output_dir / 'backtest_summary.json'}")
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run proper Polymarket backtest")
    parser.add_argument("--gamma-path", type=str, 
                        default="data/polymarket/gamma_yesno_resolved.parquet")
    parser.add_argument("--goldsky-markets", type=str, 
                        default="data/polymarket_goldsky/markets.csv")
    parser.add_argument("--goldsky-trades", type=str, 
                        default="data/polymarket_goldsky/processed/trades.csv")
    parser.add_argument("--output-dir", type=str, default="runs/proper_backtest")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--min-trades", type=int, default=10)
    args = parser.parse_args()
    
    # Load and merge data
    data = load_and_merge_data(
        gamma_path=Path(args.gamma_path),
        goldsky_markets_path=Path(args.goldsky_markets),
        goldsky_trades_path=Path(args.goldsky_trades)
    )
    
    # Run backtest
    run_backtest(
        data=data,
        output_dir=Path(args.output_dir),
        train_frac=args.train_frac,
        min_trades=args.min_trades
    )


if __name__ == "__main__":
    main()
