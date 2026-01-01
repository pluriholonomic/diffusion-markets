#!/usr/bin/env python3
"""
Build Unified Backtest Database from All Sources

Combines:
1. Polymarket data from poly_data (Goldsky/The Graph historical trades)
2. Kalshi data from API fetch
3. Our existing Polymarket resolved markets

Outputs unified parquet and SQLite for backtesting.

Usage:
    python scripts/build_unified_backtest_db.py \
        --poly-data-dir data/polymarket_goldsky \
        --kalshi-dir data/kalshi \
        --output-dir data/unified_backtest
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_poly_data_markets(poly_data_dir: Path) -> pd.DataFrame:
    """Load markets from poly_data snapshot."""
    print("Loading Polymarket markets from poly_data...")
    
    markets_path = poly_data_dir / "markets.csv"
    if not markets_path.exists():
        print(f"  Not found: {markets_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(markets_path)
    print(f"  Loaded {len(df):,} markets")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def load_poly_data_trades(poly_data_dir: Path) -> pd.DataFrame:
    """Load processed trades from poly_data."""
    print("Loading Polymarket trades from poly_data...")
    
    trades_path = poly_data_dir / "processed" / "trades.csv"
    if not trades_path.exists():
        # Try goldsky raw data
        trades_path = poly_data_dir / "goldsky" / "orderFilled.csv"
    
    if not trades_path.exists():
        print(f"  Not found: {trades_path}")
        return pd.DataFrame()
    
    # Use chunked reading for large files
    print(f"  Reading from {trades_path}...")
    chunks = []
    for chunk in pd.read_csv(trades_path, chunksize=1_000_000):
        chunks.append(chunk)
        print(f"    Loaded {sum(len(c) for c in chunks):,} rows...")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total: {len(df):,} trades")
    
    return df


def load_kalshi_data(kalshi_dir: Path) -> tuple:
    """Load Kalshi markets and prices."""
    print("Loading Kalshi data...")
    
    markets = pd.DataFrame()
    prices = pd.DataFrame()
    
    markets_path = kalshi_dir / "kalshi_markets.parquet"
    if markets_path.exists():
        markets = pd.read_parquet(markets_path)
        print(f"  Markets: {len(markets):,}")
    
    prices_path = kalshi_dir / "kalshi_prices.parquet"
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        print(f"  Prices: {len(prices):,}")
    
    backtest_path = kalshi_dir / "kalshi_backtest.parquet"
    if backtest_path.exists():
        backtest = pd.read_parquet(backtest_path)
        print(f"  Backtest-ready: {len(backtest):,}")
        return markets, prices, backtest
    
    return markets, prices, pd.DataFrame()


def load_existing_resolved(data_dir: Path) -> pd.DataFrame:
    """Load our existing resolved Polymarket markets."""
    print("Loading existing Polymarket resolved markets...")
    
    resolved_path = data_dir / "polymarket_backups/pm_suite_derived/gamma_yesno_resolved.parquet"
    if not resolved_path.exists():
        print(f"  Not found: {resolved_path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(resolved_path)
    print(f"  Loaded {len(df):,} resolved markets")
    
    return df


def compute_polymarket_prices_from_trades(
    trades: pd.DataFrame,
    markets: pd.DataFrame,
    min_trades: int = 5
) -> pd.DataFrame:
    """Compute entry prices from historical trades."""
    print("Computing Polymarket prices from trades...")
    
    if len(trades) == 0 or len(markets) == 0:
        return pd.DataFrame()
    
    # Parse timestamps
    if 'timestamp' in trades.columns:
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='s', errors='coerce')
    
    # Group by market and get first/last prices
    trades = trades.sort_values('timestamp')
    
    # Get price from trades (USD amount / token amount)
    if 'price' in trades.columns:
        price_col = 'price'
    elif 'usd_amount' in trades.columns and 'token_amount' in trades.columns:
        trades['price'] = trades['usd_amount'] / trades['token_amount'].replace(0, np.nan)
        price_col = 'price'
    else:
        print("  Cannot compute prices - missing columns")
        return pd.DataFrame()
    
    # Aggregate by market
    agg = trades.groupby('market_id').agg({
        price_col: ['first', 'last', 'mean', 'count'],
        'timestamp': ['min', 'max']
    })
    agg.columns = ['first_price', 'last_price', 'avg_price', 'n_trades', 'first_time', 'last_time']
    agg = agg.reset_index()
    
    # Filter to markets with enough trades
    agg = agg[agg['n_trades'] >= min_trades]
    print(f"  Markets with >={min_trades} trades: {len(agg):,}")
    
    return agg


def build_unified_dataset(
    poly_data_dir: Optional[Path],
    kalshi_dir: Optional[Path],
    existing_data_dir: Optional[Path],
    output_dir: Path,
    min_date: str = None
):
    """Build unified backtest dataset."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("BUILDING UNIFIED BACKTEST DATABASE")
    print("="*70)
    
    all_markets = []
    all_prices = []
    
    # 1. Load Polymarket from poly_data
    if poly_data_dir and poly_data_dir.exists():
        pm_markets = load_poly_data_markets(poly_data_dir)
        pm_trades = load_poly_data_trades(poly_data_dir)
        
        if len(pm_markets) > 0:
            pm_markets['platform'] = 'polymarket'
            all_markets.append(pm_markets)
        
        if len(pm_trades) > 0:
            pm_prices = compute_polymarket_prices_from_trades(pm_trades, pm_markets)
            if len(pm_prices) > 0:
                pm_prices['platform'] = 'polymarket'
                pm_prices['source'] = 'goldsky_trades'
                all_prices.append(pm_prices)
    
    # 2. Load existing Polymarket resolved
    if existing_data_dir and existing_data_dir.exists():
        resolved = load_existing_resolved(existing_data_dir)
        if len(resolved) > 0:
            resolved['platform'] = 'polymarket'
            # Merge with poly_data markets if we have them
            if len(all_markets) > 0 and 'id' in resolved.columns:
                # Add outcome info to all_markets
                existing_ids = set(all_markets[0].get('id', pd.Series()).tolist())
                new_resolved = resolved[~resolved['id'].isin(existing_ids)]
                if len(new_resolved) > 0:
                    all_markets.append(new_resolved)
                    print(f"  Added {len(new_resolved):,} markets from existing resolved")
            else:
                all_markets.append(resolved)
    
    # 3. Load Kalshi
    if kalshi_dir and kalshi_dir.exists():
        kalshi_markets, kalshi_prices, kalshi_backtest = load_kalshi_data(kalshi_dir)
        
        if len(kalshi_markets) > 0:
            kalshi_markets['platform'] = 'kalshi'
            all_markets.append(kalshi_markets)
        
        if len(kalshi_prices) > 0:
            kalshi_prices['platform'] = 'kalshi'
            kalshi_prices['source'] = 'kalshi_api'
            all_prices.append(kalshi_prices)
    
    # Combine all data
    print("\n" + "="*70)
    print("COMBINING DATA")
    print("="*70)
    
    if all_markets:
        combined_markets = pd.concat(all_markets, ignore_index=True)
        print(f"Total markets: {len(combined_markets):,}")
        print(f"  By platform: {combined_markets['platform'].value_counts().to_dict()}")
    else:
        print("No markets loaded!")
        return
    
    if all_prices:
        combined_prices = pd.concat(all_prices, ignore_index=True)
        print(f"Total price records: {len(combined_prices):,}")
    else:
        combined_prices = pd.DataFrame()
        print("No prices loaded!")
    
    # Filter by date if specified
    if min_date:
        min_dt = pd.to_datetime(min_date)
        for date_col in ['closedTime', 'close_time', 'resolution_time', 'createdAt']:
            if date_col in combined_markets.columns:
                combined_markets[date_col] = pd.to_datetime(combined_markets[date_col], errors='coerce')
                before = len(combined_markets)
                combined_markets = combined_markets[
                    combined_markets[date_col].isna() | (combined_markets[date_col] >= min_dt)
                ]
                if len(combined_markets) < before:
                    print(f"Filtered by {date_col}: {before:,} -> {len(combined_markets):,}")
                break
    
    # Save outputs
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)
    
    combined_markets.to_parquet(output_dir / "all_markets.parquet")
    print(f"Saved {output_dir / 'all_markets.parquet'}")
    
    if len(combined_prices) > 0:
        combined_prices.to_parquet(output_dir / "all_prices.parquet")
        print(f"Saved {output_dir / 'all_prices.parquet'}")
    
    # Create SQLite
    conn = sqlite3.connect(output_dir / "unified_backtest.db")
    combined_markets.to_sql('markets', conn, if_exists='replace', index=False)
    if len(combined_prices) > 0:
        combined_prices.to_sql('prices', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_platform ON markets(platform)")
    conn.commit()
    conn.close()
    print(f"Saved {output_dir / 'unified_backtest.db'}")
    
    # Save metadata
    metadata = {
        'created_at': datetime.utcnow().isoformat(),
        'total_markets': len(combined_markets),
        'total_prices': len(combined_prices),
        'platforms': combined_markets['platform'].value_counts().to_dict(),
        'sources': {
            'poly_data': str(poly_data_dir) if poly_data_dir else None,
            'kalshi': str(kalshi_dir) if kalshi_dir else None,
            'existing': str(existing_data_dir) if existing_data_dir else None
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Unified Backtest Database Created:
  Total markets: {len(combined_markets):,}
  Total prices: {len(combined_prices):,}
  
  Platform breakdown:
{combined_markets['platform'].value_counts().to_string()}

Output files:
  {output_dir / 'all_markets.parquet'}
  {output_dir / 'all_prices.parquet'}
  {output_dir / 'unified_backtest.db'}
  {output_dir / 'metadata.json'}
""")


def main():
    parser = argparse.ArgumentParser(description="Build unified backtest database")
    parser.add_argument("--poly-data-dir", type=str, default="data/polymarket_goldsky",
                        help="Directory with poly_data snapshot")
    parser.add_argument("--kalshi-dir", type=str, default="data/kalshi",
                        help="Directory with Kalshi data")
    parser.add_argument("--existing-dir", type=str, default=".",
                        help="Directory with existing Polymarket data")
    parser.add_argument("--output-dir", type=str, default="data/unified_backtest",
                        help="Output directory")
    parser.add_argument("--min-date", type=str, default=None,
                        help="Minimum date filter (YYYY-MM-DD)")
    args = parser.parse_args()
    
    build_unified_dataset(
        poly_data_dir=Path(args.poly_data_dir) if args.poly_data_dir else None,
        kalshi_dir=Path(args.kalshi_dir) if args.kalshi_dir else None,
        existing_data_dir=Path(args.existing_dir) if args.existing_dir else None,
        output_dir=Path(args.output_dir),
        min_date=args.min_date
    )


if __name__ == "__main__":
    main()
