#!/usr/bin/env python3
"""
Build Comprehensive Backtest Database

Aggregates all available prediction market data into a unified format
for backtesting. Outputs to both Parquet and SQLite.

Usage:
    python scripts/build_backtest_database.py \
        --output-dir data/backtest_db \
        --include-kalshi \
        --fetch-missing-prices

Output:
    data/backtest_db/
    ├── markets.parquet           # All markets with outcomes
    ├── prices.parquet            # Price time series
    ├── markets_with_prices.parquet  # Joined view for quick access
    ├── backtest.db               # SQLite database
    └── metadata.json             # Data summary
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_polymarket_resolved(data_dir: Path) -> pd.DataFrame:
    """Load all resolved Polymarket markets."""
    print("Loading Polymarket resolved markets...")
    
    resolved_path = data_dir / "polymarket_backups/pm_suite_derived/gamma_yesno_resolved.parquet"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved markets not found: {resolved_path}")
    
    df = pd.read_parquet(resolved_path)
    
    # Standardize columns
    df = df.rename(columns={
        'id': 'market_id',
        'closedTime': 'resolution_time',
        'createdAt': 'created_time',
        'volumeNum': 'volume',
        'yes_token_id': 'token_id',
    })
    
    # Add platform
    df['platform'] = 'polymarket'
    
    # Parse timestamps
    df['resolution_time'] = pd.to_datetime(df['resolution_time'], format='mixed', utc=True)
    df['created_time'] = pd.to_datetime(df['created_time'], format='mixed', utc=True)
    
    # Ensure outcome is int
    df['outcome'] = df['y'].astype(int)
    
    # Select and order columns
    columns = [
        'market_id', 'platform', 'question', 'category', 'outcome',
        'created_time', 'resolution_time', 'volume', 'token_id', 'slug'
    ]
    df = df[[c for c in columns if c in df.columns]]
    
    print(f"  Loaded {len(df):,} markets")
    return df


def load_polymarket_prices(data_dir: Path) -> pd.DataFrame:
    """Load all available Polymarket price data."""
    print("Loading Polymarket price data...")
    
    all_prices = []
    
    # 1. Sample CLOB prices (has timestamp)
    sample_path = data_dir / "polymarket_backups/pm_suite_derived/gamma_yesno_sample_clob_fixed.parquet"
    if sample_path.exists():
        df = pd.read_parquet(sample_path)
        if 'market_prob' in df.columns and 'market_prob_ts' in df.columns:
            prices = df[['id', 'market_prob', 'market_prob_ts']].copy()
            prices = prices.rename(columns={
                'id': 'market_id',
                'market_prob': 'price',
                'market_prob_ts': 'timestamp'
            })
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='s', utc=True)
            prices['source'] = 'sample_clob'
            all_prices.append(prices)
            print(f"  sample_clob: {len(prices):,} prices")
    
    # 2. Full CLOB history (multiple price points per market)
    clob_dir = data_dir / "data/polymarket/clob_history_full"
    if clob_dir.exists():
        clob_files = list(clob_dir.glob("*.parquet"))
        print(f"  Processing {len(clob_files)} CLOB history files...")
        
        clob_data = []
        for i, f in enumerate(clob_files):
            try:
                df = pd.read_parquet(f)
                if 'market_id' in df.columns and 'p' in df.columns and 't' in df.columns:
                    # Get first and last price for each market
                    df = df.sort_values('t')
                    
                    # First price (entry)
                    first = df.groupby('market_id').first().reset_index()
                    first = first[['market_id', 'p', 't']].rename(columns={'p': 'price', 't': 'timestamp'})
                    first['price_type'] = 'entry'
                    
                    # Last price (exit)
                    last = df.groupby('market_id').last().reset_index()
                    last = last[['market_id', 'p', 't']].rename(columns={'p': 'price', 't': 'timestamp'})
                    last['price_type'] = 'exit'
                    
                    clob_data.extend([first, last])
            except Exception as e:
                pass
            
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(clob_files)} files...")
        
        if clob_data:
            clob_df = pd.concat(clob_data, ignore_index=True)
            clob_df['timestamp'] = pd.to_datetime(clob_df['timestamp'], unit='s', utc=True)
            clob_df['source'] = 'clob_full'
            all_prices.append(clob_df)
            print(f"  clob_full: {len(clob_df):,} prices")
    
    # 3. Horizon-based data
    for horizon_file in ['pm_horizon_24h.parquet', 'pm_horizon_7d.parquet']:
        horizon_path = data_dir / "data/polymarket" / horizon_file
        if horizon_path.exists():
            df = pd.read_parquet(horizon_path)
            if 'market_prob' in df.columns:
                prices = df[['id', 'market_prob']].copy() if 'id' in df.columns else None
                if prices is not None:
                    prices = prices.rename(columns={'id': 'market_id', 'market_prob': 'price'})
                    prices['source'] = horizon_file.replace('.parquet', '')
                    all_prices.append(prices)
                    print(f"  {horizon_file}: {len(prices):,} prices")
    
    if all_prices:
        combined = pd.concat(all_prices, ignore_index=True)
        # Ensure market_id is string
        combined['market_id'] = combined['market_id'].astype(str)
        return combined
    
    return pd.DataFrame()


def load_kalshi_data(themis_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kalshi data from Themis if available."""
    print("Loading Kalshi data from Themis...")
    
    # Check for Themis data
    kalshi_markets = themis_dir / "data/kalshi/markets.parquet"
    kalshi_prices = themis_dir / "data/kalshi/prices.parquet"
    
    markets_df = pd.DataFrame()
    prices_df = pd.DataFrame()
    
    if kalshi_markets.exists():
        markets_df = pd.read_parquet(kalshi_markets)
        markets_df['platform'] = 'kalshi'
        print(f"  Loaded {len(markets_df):,} Kalshi markets")
    else:
        print("  Kalshi markets not found. Run Themis download first:")
        print("    cd external/themis && cargo run --bin download -- --platform kalshi")
    
    if kalshi_prices.exists():
        prices_df = pd.read_parquet(kalshi_prices)
        prices_df['platform'] = 'kalshi'
        print(f"  Loaded {len(prices_df):,} Kalshi prices")
    
    return markets_df, prices_df


def estimate_missing_prices(
    markets: pd.DataFrame, 
    prices: pd.DataFrame
) -> pd.DataFrame:
    """Estimate prices for markets without CLOB data."""
    print("Estimating prices for markets without CLOB data...")
    
    # Find markets without prices
    markets_with_prices = set(prices['market_id'].unique())
    markets_without = markets[~markets['market_id'].isin(markets_with_prices)].copy()
    
    print(f"  Markets with prices: {len(markets_with_prices):,}")
    print(f"  Markets without prices: {len(markets_without):,}")
    
    if len(markets_without) == 0:
        return prices
    
    # Estimate price based on category calibration from markets with prices
    merged = markets.merge(
        prices[prices['source'] == 'sample_clob'][['market_id', 'price']].drop_duplicates('market_id'),
        on='market_id',
        how='left'
    )
    
    # Compute category average price for markets with data
    category_stats = merged.dropna(subset=['price']).groupby('category').agg({
        'price': 'mean',
        'outcome': 'mean'  # This is the calibration target
    }).rename(columns={'price': 'avg_price', 'outcome': 'outcome_rate'})
    
    print("  Category calibration:")
    for cat, row in category_stats.iterrows():
        print(f"    {cat}: avg_price={row['avg_price']:.3f}, outcome_rate={row['outcome_rate']:.3f}")
    
    # Assign estimated prices
    estimated = []
    for _, row in markets_without.iterrows():
        cat = row.get('category', 'unknown')
        if cat in category_stats.index:
            # Use category average price
            est_price = category_stats.loc[cat, 'avg_price']
        else:
            # Default to 0.4 (slightly above mean outcome rate)
            est_price = 0.4
        
        estimated.append({
            'market_id': row['market_id'],
            'price': est_price,
            'source': 'estimated',
            'price_type': 'entry'
        })
    
    estimated_df = pd.DataFrame(estimated)
    print(f"  Estimated {len(estimated_df):,} prices")
    
    return pd.concat([prices, estimated_df], ignore_index=True)


def create_sqlite_db(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    output_path: Path
):
    """Create SQLite database from DataFrames."""
    print(f"Creating SQLite database: {output_path}")
    
    conn = sqlite3.connect(output_path)
    
    # Create markets table
    markets.to_sql('markets', conn, if_exists='replace', index=False)
    
    # Create prices table
    prices.to_sql('prices', conn, if_exists='replace', index=False)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_id ON markets(market_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_platform ON markets(platform)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_resolution ON markets(resolution_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_market ON prices(market_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_source ON prices(source)")
    
    # Create view for easy backtesting
    conn.execute("""
        CREATE VIEW IF NOT EXISTS backtest_view AS
        SELECT 
            m.market_id,
            m.platform,
            m.question,
            m.category,
            m.outcome,
            m.resolution_time,
            m.volume,
            p.price as entry_price,
            p.timestamp as entry_time,
            p.source as price_source
        FROM markets m
        LEFT JOIN prices p ON m.market_id = p.market_id
        WHERE p.price_type = 'entry' OR p.price_type IS NULL
    """)
    
    conn.commit()
    
    # Print summary
    cursor = conn.execute("SELECT COUNT(*) FROM markets")
    n_markets = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(*) FROM prices")
    n_prices = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(DISTINCT market_id) FROM prices")
    n_markets_with_prices = cursor.fetchone()[0]
    
    print(f"  Markets: {n_markets:,}")
    print(f"  Prices: {n_prices:,}")
    print(f"  Markets with prices: {n_markets_with_prices:,}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build backtest database")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Base data directory")
    parser.add_argument("--output-dir", type=str, default="data/backtest_db",
                        help="Output directory")
    parser.add_argument("--include-kalshi", action="store_true",
                        help="Include Kalshi data from Themis")
    parser.add_argument("--estimate-missing", action="store_true",
                        help="Estimate prices for markets without CLOB data")
    parser.add_argument("--min-date", type=str, default=None,
                        help="Minimum resolution date (YYYY-MM-DD)")
    parser.add_argument("--max-date", type=str, default=None,
                        help="Maximum resolution date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    print("="*70)
    print("BUILD BACKTEST DATABASE")
    print("="*70)
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Include Kalshi: {args.include_kalshi}")
    print(f"Estimate missing: {args.estimate_missing}")
    print(f"Date range: {args.min_date or 'all'} to {args.max_date or 'now'}")
    print()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Polymarket data
    markets = load_polymarket_resolved(data_dir)
    prices = load_polymarket_prices(data_dir)
    
    # Load Kalshi if requested
    if args.include_kalshi:
        themis_dir = data_dir / "../external/themis"
        kalshi_markets, kalshi_prices = load_kalshi_data(themis_dir)
        
        if len(kalshi_markets) > 0:
            markets = pd.concat([markets, kalshi_markets], ignore_index=True)
        if len(kalshi_prices) > 0:
            prices = pd.concat([prices, kalshi_prices], ignore_index=True)
    
    # Filter by date if specified
    if args.min_date:
        min_dt = pd.to_datetime(args.min_date, utc=True)
        markets = markets[markets['resolution_time'] >= min_dt]
        print(f"Filtered to markets after {args.min_date}: {len(markets):,}")
    
    if args.max_date:
        max_dt = pd.to_datetime(args.max_date, utc=True)
        markets = markets[markets['resolution_time'] <= max_dt]
        print(f"Filtered to markets before {args.max_date}: {len(markets):,}")
    
    # Estimate missing prices if requested
    if args.estimate_missing:
        prices = estimate_missing_prices(markets, prices)
    
    # Ensure market_id is string in both
    markets['market_id'] = markets['market_id'].astype(str)
    prices['market_id'] = prices['market_id'].astype(str)
    
    # Create joined view for convenience
    print("\nCreating joined view...")
    entry_prices = prices[
        (prices.get('price_type') == 'entry') | 
        (prices.get('price_type').isna())
    ].drop_duplicates('market_id')
    
    markets_with_prices = markets.merge(
        entry_prices[['market_id', 'price', 'source']].rename(columns={
            'price': 'entry_price',
            'source': 'price_source'
        }),
        on='market_id',
        how='left'
    )
    
    # Summary by year
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    markets_with_prices['year'] = markets_with_prices['resolution_time'].dt.year
    summary = markets_with_prices.groupby(['year', 'platform']).agg({
        'market_id': 'count',
        'entry_price': lambda x: x.notna().sum(),
        'volume': 'sum'
    }).rename(columns={
        'market_id': 'total_markets',
        'entry_price': 'with_price',
        'volume': 'total_volume'
    })
    
    print(summary.to_string())
    
    # Price coverage
    has_price = markets_with_prices['entry_price'].notna().sum()
    total = len(markets_with_prices)
    print(f"\nPrice coverage: {has_price:,} / {total:,} = {has_price/total*100:.1f}%")
    
    # By source
    print("\nPrice sources:")
    print(markets_with_prices['price_source'].value_counts())
    
    # Save outputs
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)
    
    # Parquet files
    markets.to_parquet(output_dir / "markets.parquet")
    print(f"  Saved {output_dir / 'markets.parquet'}")
    
    prices.to_parquet(output_dir / "prices.parquet")
    print(f"  Saved {output_dir / 'prices.parquet'}")
    
    markets_with_prices.to_parquet(output_dir / "markets_with_prices.parquet")
    print(f"  Saved {output_dir / 'markets_with_prices.parquet'}")
    
    # SQLite database
    create_sqlite_db(markets, prices, output_dir / "backtest.db")
    
    # Metadata
    metadata = {
        'created_at': datetime.utcnow().isoformat(),
        'total_markets': len(markets),
        'total_prices': len(prices),
        'markets_with_prices': int(has_price),
        'price_coverage': float(has_price / total),
        'platforms': list(markets['platform'].unique()),
        'date_range': {
            'min': str(markets['resolution_time'].min()),
            'max': str(markets['resolution_time'].max())
        },
        'price_sources': prices['source'].value_counts().to_dict() if 'source' in prices.columns else {}
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved {output_dir / 'metadata.json'}")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"""
Output files:
  {output_dir / 'markets.parquet'}         - All markets ({len(markets):,})
  {output_dir / 'prices.parquet'}          - All prices ({len(prices):,})
  {output_dir / 'markets_with_prices.parquet'} - Joined ({len(markets_with_prices):,})
  {output_dir / 'backtest.db'}             - SQLite database
  {output_dir / 'metadata.json'}           - Summary metadata

To use in backtesting:
  import pandas as pd
  df = pd.read_parquet('{output_dir / 'markets_with_prices.parquet'}')
  # Filter to 2025
  df_2025 = df[df['resolution_time'] >= '2025-01-01']
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
