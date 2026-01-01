#!/usr/bin/env python3
"""
Fetch CLOB price history for all Polymarket markets.

Fetches historical prices from the Polymarket CLOB API and saves to parquet.
Supports resuming from where it left off.

Usage:
    python scripts/fetch_all_clob_prices.py \
        --min-date 2024-11-06 \
        --output data/clob_prices_2025.parquet \
        --batch-size 100 \
        --max-workers 5
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


def fetch_price_history(token_id: str, market_id: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch price history for a single market."""
    try:
        resp = requests.get(
            "https://clob.polymarket.com/prices-history",
            params={'market': token_id, 'interval': 'max'},
            timeout=timeout
        )
        
        if resp.status_code == 200:
            data = resp.json()
            history = data.get('history', [])
            
            if history:
                # Extract first and last prices
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
                    'note': 'empty_history'
                }
        elif resp.status_code == 429:
            # Rate limited - return None to retry
            return None
        else:
            return {
                'market_id': market_id,
                'token_id': token_id,
                'success': False,
                'error': f"status_{resp.status_code}"
            }
    except requests.exceptions.Timeout:
        return {
            'market_id': market_id,
            'token_id': token_id,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'market_id': market_id,
            'token_id': token_id,
            'success': False,
            'error': str(e)[:100]
        }


def load_progress(progress_file: Path) -> set:
    """Load already-fetched market IDs."""
    if progress_file.exists():
        with open(progress_file) as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: Path, completed: set):
    """Save progress."""
    with open(progress_file, 'w') as f:
        json.dump(list(completed), f)


def main():
    parser = argparse.ArgumentParser(description="Fetch CLOB prices")
    parser.add_argument("--min-date", type=str, default="2024-11-06",
                        help="Minimum resolution date")
    parser.add_argument("--output", type=str, default="data/clob_prices_fetched.parquet",
                        help="Output parquet file")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Save progress every N markets")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Parallel workers (be gentle on API)")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Limit number of markets to fetch")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Delay between requests (seconds)")
    args = parser.parse_args()
    
    print("="*70)
    print("FETCH CLOB PRICE HISTORY")
    print("="*70)
    print(f"Min date: {args.min_date}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.max_workers}")
    print()
    
    # Load resolved markets
    resolved = pd.read_parquet('polymarket_backups/pm_suite_derived/gamma_yesno_resolved.parquet')
    resolved['closedTime'] = pd.to_datetime(resolved['closedTime'], format='mixed', utc=True)
    
    # Filter by date
    min_dt = pd.to_datetime(args.min_date, utc=True)
    resolved = resolved[resolved['closedTime'] >= min_dt]
    print(f"Markets after {args.min_date}: {len(resolved):,}")
    
    # Filter to markets with token IDs
    resolved = resolved[resolved['yes_token_id'].notna() & (resolved['yes_token_id'] != '')]
    print(f"Markets with token IDs: {len(resolved):,}")
    
    # Load existing prices to skip already-fetched
    output_path = Path(args.output)
    progress_file = output_path.with_suffix('.progress.json')
    
    completed = load_progress(progress_file)
    print(f"Already fetched: {len(completed):,}")
    
    # Filter to unfetched
    to_fetch = resolved[~resolved['id'].isin(completed)].copy()
    
    if args.max_markets:
        to_fetch = to_fetch.head(args.max_markets)
    
    print(f"To fetch: {len(to_fetch):,}")
    
    if len(to_fetch) == 0:
        print("Nothing to fetch!")
        return 0
    
    # Prepare market list
    markets = [
        (row['id'], row['yes_token_id'])
        for _, row in to_fetch.iterrows()
    ]
    
    # Fetch with progress bar
    results = []
    errors = 0
    empty = 0
    
    print(f"\nFetching prices...")
    
    with tqdm(total=len(markets), desc="Fetching") as pbar:
        for i, (market_id, token_id) in enumerate(markets):
            result = fetch_price_history(token_id, market_id)
            
            # Handle rate limiting with retry
            retry_count = 0
            while result is None and retry_count < 3:
                time.sleep(2 ** retry_count)  # Exponential backoff
                result = fetch_price_history(token_id, market_id)
                retry_count += 1
            
            if result is None:
                result = {
                    'market_id': market_id,
                    'token_id': token_id,
                    'success': False,
                    'error': 'rate_limited'
                }
            
            results.append(result)
            completed.add(market_id)
            
            if result.get('success'):
                if result.get('n_points', 0) == 0:
                    empty += 1
            else:
                errors += 1
            
            pbar.update(1)
            pbar.set_postfix({'errors': errors, 'empty': empty})
            
            # Save progress periodically
            if (i + 1) % args.batch_size == 0:
                save_progress(progress_file, completed)
                
                # Also save partial results
                partial_df = pd.DataFrame(results)
                partial_df.to_parquet(output_path.with_suffix('.partial.parquet'))
            
            # Rate limit
            time.sleep(args.delay)
    
    # Final save
    save_progress(progress_file, completed)
    
    # Create final dataframe
    df = pd.DataFrame(results)
    
    # Convert timestamps
    if 'first_time' in df.columns:
        df['first_time'] = pd.to_datetime(df['first_time'], unit='s', utc=True, errors='coerce')
        df['last_time'] = pd.to_datetime(df['last_time'], unit='s', utc=True, errors='coerce')
    
    # Merge with existing if any
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates('market_id', keep='last')
    
    df.to_parquet(output_path)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total fetched: {len(results):,}")
    print(f"Successful: {sum(1 for r in results if r.get('success')):,}")
    print(f"With prices: {sum(1 for r in results if r.get('n_points', 0) > 0):,}")
    print(f"Empty history: {empty:,}")
    print(f"Errors: {errors:,}")
    print(f"\nSaved to: {output_path}")
    
    # Cleanup partial file
    partial = output_path.with_suffix('.partial.parquet')
    if partial.exists():
        partial.unlink()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
