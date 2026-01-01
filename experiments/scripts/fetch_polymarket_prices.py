#!/usr/bin/env python3
"""
Fetch Polymarket CLOB Price History for 2025 Markets.

This script fetches mid-price snapshots from Polymarket's CLOB API
for markets that don't already have price data.

Usage:
    python scripts/fetch_polymarket_prices.py \
        --output-dir data/clob_2025 \
        --start-date 2025-01-01 \
        --max-markets 10000 \
        --batch-size 100

Requires:
    pip install py-clob-client requests pandas
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def fetch_resolved_markets(
    start_date: str = "2025-01-01",
    end_date: Optional[str] = None,
    limit: int = 10000
) -> pd.DataFrame:
    """Fetch resolved markets from Gamma API."""
    print(f"Fetching resolved markets from {start_date}...")
    
    markets = []
    offset = 0
    batch_size = 500
    
    while len(markets) < limit:
        params = {
            "limit": batch_size,
            "offset": offset,
            "closed": "true",
            "order": "closedTime",
            "ascending": "true",
        }
        
        try:
            resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"  Error fetching markets: {e}")
            break
        
        if not batch:
            break
        
        for m in batch:
            closed_time = m.get("closedTime", "")
            if closed_time and closed_time >= start_date:
                if end_date and closed_time > end_date:
                    continue
                markets.append({
                    "id": m.get("id"),
                    "question": m.get("question"),
                    "closedTime": closed_time,
                    "clobTokenIds": m.get("clobTokenIds"),
                    "volumeNum": m.get("volumeNum", 0),
                    "category": m.get("category"),
                })
        
        offset += batch_size
        print(f"  Fetched {len(markets)} markets...")
        
        if len(batch) < batch_size:
            break
        
        time.sleep(0.5)  # Rate limiting
    
    return pd.DataFrame(markets)


def fetch_clob_price_history(
    token_id: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    interval: str = "1h"
) -> Optional[List[Dict]]:
    """
    Fetch CLOB price history for a token.
    
    Note: Polymarket's public CLOB API may have limitations.
    This uses the prices/history endpoint if available.
    """
    try:
        params = {"interval": interval}
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        
        # Try the CLOB prices endpoint
        url = f"{CLOB_API}/prices-history"
        params["tokenID"] = token_id
        
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            return data if isinstance(data, list) else None
        
        return None
        
    except Exception as e:
        return None


def fetch_current_midpoint(token_id: str) -> Optional[float]:
    """Fetch current mid-price for a token from CLOB."""
    try:
        url = f"{CLOB_API}/midpoint"
        params = {"token_id": token_id}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("mid", 0))
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket CLOB prices")
    parser.add_argument("--output-dir", type=str, default="data/clob_2025",
                        help="Output directory for price data")
    parser.add_argument("--start-date", type=str, default="2025-01-01",
                        help="Start date for markets (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for markets (YYYY-MM-DD)")
    parser.add_argument("--max-markets", type=int, default=10000,
                        help="Maximum markets to fetch")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for processing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip markets with existing price data")
    args = parser.parse_args()
    
    print("="*70)
    print("POLYMARKET CLOB PRICE FETCHER")
    print("="*70)
    print(f"Output dir: {args.output_dir}")
    print(f"Date range: {args.start_date} to {args.end_date or 'now'}")
    print(f"Max markets: {args.max_markets}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing prices if any
    existing_ids = set()
    if args.skip_existing:
        for f in output_dir.glob("*.parquet"):
            existing_ids.add(f.stem)
        print(f"Found {len(existing_ids)} existing price files")
    
    # Option 1: Use local resolved data if available
    local_resolved = Path("polymarket_backups/pm_suite_derived/gamma_yesno_resolved.parquet")
    if local_resolved.exists():
        print("Using local resolved markets data...")
        resolved = pd.read_parquet(local_resolved)
        resolved['close_ts'] = pd.to_datetime(resolved['closedTime'], format='mixed', utc=True)
        
        # Filter to date range
        start_dt = pd.to_datetime(args.start_date, utc=True)
        markets = resolved[resolved['close_ts'] >= start_dt].head(args.max_markets)
        
        print(f"Found {len(markets)} markets after {args.start_date}")
    else:
        # Fetch from API
        markets = fetch_resolved_markets(
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.max_markets
        )
    
    if len(markets) == 0:
        print("No markets found!")
        return 1
    
    # For each market, try to get price data
    print(f"\nFetching prices for {len(markets)} markets...")
    
    results = []
    errors = 0
    
    for idx, row in markets.iterrows():
        market_id = str(row.get('id', ''))
        
        if market_id in existing_ids:
            continue
        
        # Get token IDs
        token_ids_raw = row.get('clobTokenIds') or row.get('yes_token_id')
        if isinstance(token_ids_raw, str):
            try:
                token_ids = json.loads(token_ids_raw)
            except:
                token_ids = [token_ids_raw]
        elif isinstance(token_ids_raw, list):
            token_ids = token_ids_raw
        else:
            token_ids = []
        
        if not token_ids:
            continue
        
        # Fetch price for first (YES) token
        yes_token = token_ids[0] if token_ids else None
        if not yes_token:
            continue
        
        # Try to get historical prices
        history = fetch_clob_price_history(yes_token)
        
        if history and len(history) > 0:
            results.append({
                'market_id': market_id,
                'question': row.get('question', '')[:100],
                'closedTime': row.get('closedTime'),
                'token_id': yes_token,
                'price_points': len(history),
                'first_price': history[0].get('p') if history else None,
                'last_price': history[-1].get('p') if history else None,
            })
        else:
            # Try current midpoint as fallback
            mid = fetch_current_midpoint(yes_token)
            if mid:
                results.append({
                    'market_id': market_id,
                    'question': row.get('question', '')[:100],
                    'closedTime': row.get('closedTime'),
                    'token_id': yes_token,
                    'price_points': 1,
                    'first_price': mid,
                    'last_price': mid,
                })
            else:
                errors += 1
        
        if len(results) % 50 == 0:
            print(f"  Processed {idx+1}/{len(markets)}, got {len(results)} prices, {errors} errors")
        
        time.sleep(0.2)  # Rate limiting
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = output_dir / "clob_prices_2025.parquet"
        results_df.to_parquet(output_file)
        print(f"\nSaved {len(results)} market prices to {output_file}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Markets processed: {len(markets)}")
        print(f"Prices fetched: {len(results)}")
        print(f"Errors: {errors}")
        print(f"Coverage: {len(results)/len(markets)*100:.1f}%")
    else:
        print("No prices fetched!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
