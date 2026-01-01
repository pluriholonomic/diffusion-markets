#!/usr/bin/env python3
"""
Parallel CLOB Price Fetcher - Optimized for high-CPU servers.

Fetches historical prices from Polymarket CLOB API using parallel workers.
Designed to maximize throughput on servers with many CPU cores.

Usage:
    python scripts/fetch_clob_prices_parallel.py \
        --min-date 2024-11-06 \
        --output data/clob_prices_2025.parquet \
        --workers 40 \
        --batch-size 500
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session() -> requests.Session:
    """Create a session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_one(args: tuple) -> Dict:
    """Fetch price history for a single market."""
    market_id, token_id, session = args
    
    try:
        resp = session.get(
            "https://clob.polymarket.com/prices-history",
            params={'market': token_id, 'interval': 'max'},
            timeout=15
        )
        
        if resp.status_code == 200:
            data = resp.json()
            history = data.get('history', [])
            
            if history:
                first = history[0]
                last = history[-1]
                
                return {
                    'market_id': market_id,
                    'token_id': token_id,
                    'n_points': len(history),
                    'first_time': first.get('t'),
                    'first_price': float(first.get('p', 0)),
                    'last_time': last.get('t'),
                    'last_price': float(last.get('p', 0)),
                    'success': True
                }
            else:
                return {
                    'market_id': market_id,
                    'token_id': token_id,
                    'n_points': 0,
                    'success': True,
                    'note': 'empty'
                }
        else:
            return {
                'market_id': market_id,
                'token_id': token_id,
                'success': False,
                'error': f"http_{resp.status_code}"
            }
    except Exception as e:
        return {
            'market_id': market_id,
            'token_id': token_id,
            'success': False,
            'error': str(e)[:80]
        }


def main():
    parser = argparse.ArgumentParser(description="Parallel CLOB price fetcher")
    parser.add_argument("--data-path", type=str, 
                        default="polymarket_backups/pm_suite_derived/gamma_yesno_resolved.parquet",
                        help="Path to resolved markets parquet")
    parser.add_argument("--min-date", type=str, default="2024-11-06",
                        help="Minimum resolution date")
    parser.add_argument("--output", type=str, default="data/clob_prices_fetched.parquet",
                        help="Output parquet file")
    parser.add_argument("--workers", type=int, default=40,
                        help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Save progress every N markets")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Limit markets to fetch (for testing)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PARALLEL CLOB PRICE FETCHER")
    print("=" * 70)
    print(f"Workers: {args.workers}")
    print(f"Min date: {args.min_date}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load resolved markets
    print("Loading resolved markets...")
    resolved = pd.read_parquet(args.data_path)
    resolved['closedTime'] = pd.to_datetime(resolved['closedTime'], format='mixed', utc=True)
    print(f"  Total: {len(resolved):,}")
    
    # Filter by date
    min_dt = pd.to_datetime(args.min_date, utc=True)
    resolved = resolved[resolved['closedTime'] >= min_dt]
    print(f"  After {args.min_date}: {len(resolved):,}")
    
    # Filter to markets with token IDs
    resolved = resolved[resolved['yes_token_id'].notna() & (resolved['yes_token_id'] != '')]
    print(f"  With token IDs: {len(resolved):,}")
    
    # Check for already fetched
    output_path = Path(args.output)
    progress_file = output_path.with_suffix('.progress.json')
    
    completed = set()
    if progress_file.exists():
        with open(progress_file) as f:
            completed = set(json.load(f))
        print(f"  Already fetched: {len(completed):,}")
    
    # Filter to unfetched
    to_fetch = resolved[~resolved['id'].isin(completed)].copy()
    
    if args.max_markets:
        to_fetch = to_fetch.head(args.max_markets)
    
    print(f"  To fetch: {len(to_fetch):,}")
    
    if len(to_fetch) == 0:
        print("\nNothing to fetch!")
        return 0
    
    # Prepare market list
    markets = [
        (row['id'], row['yes_token_id'])
        for _, row in to_fetch.iterrows()
    ]
    
    # Create session pool
    sessions = [create_session() for _ in range(args.workers)]
    
    # Prepare work items with session assignment
    work_items = [
        (m[0], m[1], sessions[i % len(sessions)])
        for i, m in enumerate(markets)
    ]
    
    # Fetch in parallel
    results = []
    success = 0
    with_prices = 0
    errors = 0
    
    start_time = time.time()
    last_report = start_time
    
    print(f"\nFetching with {args.workers} workers...")
    print("-" * 70)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, item): item[0] for item in work_items}
        
        for i, future in enumerate(as_completed(futures)):
            market_id = futures[future]
            
            try:
                result = future.result()
                results.append(result)
                completed.add(result['market_id'])
                
                if result.get('success'):
                    success += 1
                    if result.get('n_points', 0) > 0:
                        with_prices += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                results.append({
                    'market_id': market_id,
                    'success': False,
                    'error': str(e)[:80]
                })
            
            # Progress report every 10 seconds
            now = time.time()
            if now - last_report >= 10:
                elapsed = now - start_time
                rate = (i + 1) / elapsed
                remaining = len(markets) - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Progress: {i+1:,}/{len(markets):,} ({100*(i+1)/len(markets):.1f}%) | "
                      f"Rate: {rate:.1f}/s | "
                      f"Success: {success:,} | "
                      f"With prices: {with_prices:,} | "
                      f"Errors: {errors:,} | "
                      f"ETA: {eta/60:.1f}min")
                last_report = now
            
            # Save progress periodically
            if (i + 1) % args.batch_size == 0:
                # Save progress
                with open(progress_file, 'w') as f:
                    json.dump(list(completed), f)
                
                # Save partial results
                partial_df = pd.DataFrame(results)
                partial_df.to_parquet(output_path.with_suffix('.partial.parquet'))
    
    # Final stats
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Rate: {len(markets)/elapsed:.1f} markets/second")
    print(f"Success: {success:,} ({100*success/len(markets):.1f}%)")
    print(f"With prices: {with_prices:,} ({100*with_prices/len(markets):.1f}%)")
    print(f"Errors: {errors:,}")
    
    # Save final results
    print("\nSaving results...")
    
    # Save progress
    with open(progress_file, 'w') as f:
        json.dump(list(completed), f)
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Convert timestamps
    for col in ['first_time', 'last_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s', utc=True, errors='coerce')
    
    # Merge with existing if any
    if output_path.exists():
        print(f"Merging with existing {output_path}...")
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates('market_id', keep='last')
    
    df.to_parquet(output_path)
    print(f"Saved {len(df):,} rows to {output_path}")
    
    # Cleanup partial
    partial = output_path.with_suffix('.partial.parquet')
    if partial.exists():
        partial.unlink()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total markets fetched: {len(df):,}")
    print(f"Markets with prices: {(df.get('n_points', pd.Series([0])) > 0).sum():,}")
    print(f"Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
