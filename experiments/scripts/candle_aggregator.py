#!/usr/bin/env python3
"""
Candle Aggregator

Converts raw tick data (t, p) to OHLCV-style candles:
- ts (timestamp)
- min_price
- median_price  
- max_price
- open_price
- close_price
- tick_count (proxy for volume)

Usage:
    python scripts/candle_aggregator.py --input data/polymarket/clob_history_yes_f1 \
        --output data/candles/polymarket_1min.parquet --interval 1min
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_market_file(args) -> pd.DataFrame:
    """Process a single market's tick data into candles."""
    file_path, interval, outcomes_map = args
    
    try:
        df = pd.read_parquet(file_path)
        
        if len(df) < 2:
            return pd.DataFrame()
        
        # Parse timestamp
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            return pd.DataFrame()
        
        # Get market info
        market_id = df['market_id'].iloc[0] if 'market_id' in df.columns else Path(file_path).stem
        token_id = df['token_id'].iloc[0] if 'token_id' in df.columns else Path(file_path).stem
        
        # Sort and set index
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # Resample to candles
        candles = df['p'].resample(interval).agg(['first', 'max', 'min', 'last', 'median', 'count'])
        candles.columns = ['open', 'high', 'low', 'close', 'median', 'tick_count']
        
        # Drop empty candles
        candles = candles.dropna(subset=['open'])
        
        if len(candles) == 0:
            return pd.DataFrame()
        
        # Add market info
        candles = candles.reset_index()
        candles['market_id'] = market_id
        candles['token_id'] = token_id
        
        # Add outcome if available
        if outcomes_map and token_id in outcomes_map:
            candles['outcome'] = outcomes_map[token_id]
        else:
            candles['outcome'] = None
        
        return candles
        
    except Exception as e:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Convert tick data to candles")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with tick parquet files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet file")
    parser.add_argument("--interval", type=str, default="1min",
                        help="Candle interval (1min, 5min, 15min, 1H, 1D)")
    parser.add_argument("--outcomes", type=str, default=None,
                        help="Path to outcomes parquet (optional)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of files to process")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CANDLE AGGREGATOR")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Interval: {args.interval}")
    print()
    
    # Load outcomes if provided
    outcomes_map = {}
    if args.outcomes and os.path.exists(args.outcomes):
        print(f"Loading outcomes from {args.outcomes}...")
        outcomes_df = pd.read_parquet(args.outcomes)
        if 'yes_token_id' in outcomes_df.columns and 'y' in outcomes_df.columns:
            outcomes_map = dict(zip(
                outcomes_df['yes_token_id'].astype(str),
                outcomes_df['y']
            ))
        print(f"Loaded {len(outcomes_map)} outcomes")
    
    # Get all parquet files
    files = list(input_dir.glob("*.parquet"))
    if args.limit:
        files = files[:args.limit]
    print(f"Files to process: {len(files)}")
    
    # Process in parallel
    all_candles = []
    processed = 0
    
    # Prepare args for workers
    work_args = [(str(f), args.interval, outcomes_map) for f in files]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_market_file, arg): arg for arg in work_args}
        
        for future in as_completed(futures):
            result = future.result()
            if len(result) > 0:
                all_candles.append(result)
            processed += 1
            
            if processed % 500 == 0:
                print(f"  Processed {processed}/{len(files)} files, "
                      f"{sum(len(c) for c in all_candles):,} candles")
    
    if not all_candles:
        print("No candles generated!")
        return 1
    
    # Combine all candles
    print("\nCombining candles...")
    combined = pd.concat(all_candles, ignore_index=True)
    combined = combined.sort_values(['market_id', 'timestamp'])
    
    # Ensure proper column order
    cols = ['timestamp', 'market_id', 'token_id', 'open', 'high', 'low', 'close', 
            'median', 'tick_count', 'outcome']
    cols = [c for c in cols if c in combined.columns]
    combined = combined[cols]
    
    # Save
    combined.to_parquet(output_path, index=False)
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total candles: {len(combined):,}")
    print(f"Markets: {combined.market_id.nunique()}")
    print(f"Date range: {combined.timestamp.min()} to {combined.timestamp.max()}")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Show sample
    print("\nSample:")
    print(combined.head(3))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
