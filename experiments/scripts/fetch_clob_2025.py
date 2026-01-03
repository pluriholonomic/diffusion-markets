#!/usr/bin/env python3
"""
Fetch CLOB price history for Feb 2025 - Jan 2026 from Polymarket API.
Saves tick data that can be converted to OHLCV candles.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


def create_session():
    """Create session with retry logic."""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_active_markets(session, limit=500):
    """Get active markets from Gamma API."""
    markets = []
    
    # Get open markets
    resp = session.get(
        f"{GAMMA_API}/markets",
        params={"limit": limit, "closed": "false", "order": "volume", "ascending": "false"},
        timeout=60
    )
    
    if resp.status_code == 200:
        data = resp.json()
        batch = data if isinstance(data, list) else data.get("markets", [])
        
        for m in batch:
            clob_ids = m.get("clobTokenIds", "[]")
            if isinstance(clob_ids, str):
                try:
                    tokens = json.loads(clob_ids)
                except:
                    tokens = []
            else:
                tokens = clob_ids or []
            
            if tokens and len(tokens) > 0:
                markets.append({
                    "condition_id": m.get("conditionId", ""),
                    "slug": m.get("slug", ""),
                    "question": m.get("question", "")[:100],
                    "token_id": tokens[0],  # YES token
                    "end_date": m.get("endDate", ""),
                })
    
    print(f"Found {len(markets)} active markets with token IDs")
    return markets


def fetch_price_history(session, token_id: str, start_date: datetime, end_date: datetime):
    """Fetch price history for a token from CLOB API in weekly chunks."""
    all_history = []
    
    # Fetch in 7-day chunks (API limitation)
    current = start_date
    while current < end_date:
        chunk_end = min(current + timedelta(days=7), end_date)
        start_ts = int(current.timestamp())
        end_ts = int(chunk_end.timestamp())
        
        try:
            resp = session.get(
                f"{CLOB_API}/prices-history",
                params={"market": token_id, "startTs": start_ts, "endTs": end_ts},
                timeout=30
            )
            
            if resp.status_code == 200:
                data = resp.json()
                history = data.get("history", [])
                all_history.extend(history)
            
            time.sleep(0.05)  # Rate limit
        except:
            pass
        
        current = chunk_end
    
    return all_history


def process_market(args):
    """Process a single market."""
    session, market, output_dir, start_date, end_date = args
    
    token_id = market["token_id"]
    slug = market["slug"]
    
    # Fetch price history in chunks
    history = fetch_price_history(session, token_id, start_date, end_date)
    
    if not history:
        return None
    
    # Convert to dataframe
    records = []
    for point in history:
        ts = point.get("t", 0)
        price = point.get("p", 0)
        if ts > 0:
            records.append({
                "t": ts,
                "p": price,
                "token_id": token_id,
                "slug": slug,
            })
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["t"])
    
    if len(df) == 0:
        return None
    
    # Save
    output_file = output_dir / f"{token_id[:50]}.parquet"
    df.to_parquet(output_file, index=False)
    
    return {
        "token_id": token_id[:50],
        "slug": slug,
        "rows": len(df),
        "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/clob_2025",
                        help="Output directory")
    parser.add_argument("--markets", type=int, default=500,
                        help="Number of markets to fetch")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CLOB 2025 DATA FETCHER")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Target markets: {args.markets}")
    print()
    
    session = create_session()
    
    # Get active markets
    print("Fetching active markets...")
    markets = get_active_markets(session, limit=args.markets)
    
    if not markets:
        print("No markets found!")
        return 1
    
    # Date range: Feb 1, 2025 to now
    start_date = datetime(2025, 2, 1)
    end_date = datetime.now()
    print(f"Fetching data from {start_date} to {end_date}")
    print(f"This requires ~{(end_date - start_date).days // 7} API calls per market")
    print()
    
    # Process markets in parallel
    print("Fetching price history...")
    work_args = [(session, m, output_dir, start_date, end_date) for m in markets]
    
    results = []
    total_rows = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_market, arg): arg[1]["slug"] for arg in work_args}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
                total_rows += result["rows"]
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(markets)} markets, {len(results)} with data, {total_rows:,} rows")
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Markets with 2025 data: {len(results)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output: {output_dir}")
    
    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_parquet(output_dir / "_summary.parquet", index=False)
        print(f"\nSample markets:")
        for r in results[:5]:
            print(f"  {r['slug'][:40]}: {r['rows']:,} rows")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
