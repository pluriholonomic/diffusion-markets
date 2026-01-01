#!/usr/bin/env python3
"""
Fetch historical market data from Kalshi API.

Kalshi provides free public API access for market data including:
- Market metadata
- Historical candlestick prices
- Settlement/resolution data

Usage:
    python scripts/fetch_kalshi_data.py \
        --output data/kalshi \
        --min-date 2024-01-01 \
        --max-markets 10000
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
from tqdm import tqdm


class KalshiClient:
    """Client for Kalshi public API."""
    
    # Use elections subdomain - provides access to ALL markets (not just elections)
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, rate_limit_delay: float = 0.2):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            time.sleep(self.rate_limit_delay)
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(5)
                return self._get(endpoint, params)
            else:
                return {'error': f'status_{resp.status_code}'}
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def get_markets(
        self, 
        status: str = None, 
        limit: int = 200,
        cursor: str = None
    ) -> dict:
        """Get list of markets."""
        params = {'limit': limit}
        if status:
            params['status'] = status
        if cursor:
            params['cursor'] = cursor
        return self._get('markets', params)
    
    def get_all_markets(self, status: str = None, max_markets: int = None) -> List[dict]:
        """Get all markets with pagination."""
        all_markets = []
        cursor = None
        
        while True:
            result = self.get_markets(status=status, cursor=cursor)
            
            if 'error' in result:
                print(f"Error fetching markets: {result['error']}")
                break
                
            markets = result.get('markets', [])
            all_markets.extend(markets)
            
            if max_markets and len(all_markets) >= max_markets:
                all_markets = all_markets[:max_markets]
                break
            
            cursor = result.get('cursor')
            if not cursor:
                break
                
            print(f"  Fetched {len(all_markets)} markets...")
        
        return all_markets
    
    def get_candlesticks(
        self, 
        ticker: str,
        period_interval: int = 60,  # minutes
        start_ts: int = None,
        end_ts: int = None
    ) -> List[dict]:
        """Get candlestick price history for a market."""
        params = {'period_interval': period_interval}
        if start_ts:
            params['start_ts'] = start_ts
        if end_ts:
            params['end_ts'] = end_ts
            
        result = self._get(f'markets/{ticker}/candlesticks', params)
        return result.get('candlesticks', [])
    
    def get_market(self, ticker: str) -> dict:
        """Get detailed market info."""
        return self._get(f'markets/{ticker}')


def fetch_all_kalshi_data(
    output_dir: Path,
    min_date: str = None,
    max_markets: int = None,
    fetch_prices: bool = True
):
    """Fetch all Kalshi market data."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    client = KalshiClient()
    
    print("="*70)
    print("FETCHING KALSHI DATA")
    print("="*70)
    
    # Fetch closed (resolved) markets
    print("\n1. Fetching closed markets...")
    closed_markets = client.get_all_markets(status='closed', max_markets=max_markets)
    print(f"   Got {len(closed_markets)} closed markets")
    
    # Fetch settled markets
    print("\n2. Fetching settled markets...")
    settled_markets = client.get_all_markets(status='settled', max_markets=max_markets)
    print(f"   Got {len(settled_markets)} settled markets")
    
    # Combine and deduplicate
    all_markets = {m['ticker']: m for m in closed_markets + settled_markets}
    markets_list = list(all_markets.values())
    print(f"\nTotal unique markets: {len(markets_list)}")
    
    # Convert to DataFrame
    markets_df = pd.DataFrame(markets_list)
    
    # Parse dates
    for col in ['created_time', 'close_time', 'expiration_time', 'open_time']:
        if col in markets_df.columns:
            markets_df[col] = pd.to_datetime(markets_df[col], errors='coerce')
    
    # Filter by date if specified
    if min_date:
        min_dt = pd.to_datetime(min_date)
        if 'close_time' in markets_df.columns:
            before = len(markets_df)
            markets_df = markets_df[markets_df['close_time'] >= min_dt]
            print(f"Filtered to {len(markets_df)} markets after {min_date} (was {before})")
    
    # Save markets
    markets_path = output_dir / 'kalshi_markets.parquet'
    markets_df.to_parquet(markets_path)
    print(f"\nSaved markets to {markets_path}")
    
    # Summary
    print(f"\n=== MARKET SUMMARY ===")
    print(f"Total markets: {len(markets_df)}")
    if 'close_time' in markets_df.columns:
        print(f"Date range: {markets_df['close_time'].min()} to {markets_df['close_time'].max()}")
    if 'expiration_value' in markets_df.columns:
        print(f"With settlement value: {markets_df['expiration_value'].notna().sum()}")
    
    # Fetch price history
    if fetch_prices and len(markets_df) > 0:
        print(f"\n3. Fetching price history...")
        
        prices = []
        for _, row in tqdm(markets_df.iterrows(), total=len(markets_df), desc="Fetching prices"):
            ticker = row['ticker']
            
            try:
                candles = client.get_candlesticks(ticker, period_interval=60)
                
                if candles:
                    for c in candles:
                        prices.append({
                            'ticker': ticker,
                            'timestamp': c.get('end_period_ts'),
                            'open': c.get('open'),
                            'high': c.get('high'),
                            'low': c.get('low'),
                            'close': c.get('close'),
                            'volume': c.get('volume'),
                            'yes_price': c.get('yes_price', c.get('close')),
                        })
            except Exception as e:
                pass
        
        if prices:
            prices_df = pd.DataFrame(prices)
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='s', errors='coerce')
            
            prices_path = output_dir / 'kalshi_prices.parquet'
            prices_df.to_parquet(prices_path)
            print(f"\nSaved {len(prices_df)} price points to {prices_path}")
            print(f"Markets with prices: {prices_df['ticker'].nunique()}")
        else:
            print("No price data available")
    
    # Create unified backtest format
    print("\n4. Creating backtest-ready format...")
    
    if len(markets_df) == 0:
        print("No markets to process!")
        return markets_df
    
    backtest_df = markets_df.copy()
    backtest_df['platform'] = 'kalshi'
    backtest_df['market_id'] = backtest_df['ticker'] if 'ticker' in backtest_df.columns else backtest_df.index
    
    # Map outcome
    if 'expiration_value' in backtest_df.columns:
        backtest_df['outcome'] = backtest_df['expiration_value'].apply(
            lambda x: 1 if str(x).lower() == 'yes' else (0 if str(x).lower() == 'no' else None)
        )
    
    # Map price
    if 'last_price' in backtest_df.columns:
        backtest_df['entry_price'] = backtest_df['last_price'] / 100.0  # Kalshi uses cents
    
    # Rename columns
    rename_map = {
        'title': 'question',
        'close_time': 'resolution_time',
        'created_time': 'created_time',
    }
    backtest_df = backtest_df.rename(columns={k: v for k, v in rename_map.items() if k in backtest_df.columns})
    
    # Select columns
    cols = ['market_id', 'platform', 'question', 'outcome', 'entry_price', 
            'resolution_time', 'created_time', 'ticker', 'event_ticker']
    cols = [c for c in cols if c in backtest_df.columns]
    backtest_df = backtest_df[cols]
    
    backtest_path = output_dir / 'kalshi_backtest.parquet'
    backtest_df.to_parquet(backtest_path)
    print(f"Saved backtest format to {backtest_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("KALSHI DATA FETCH COMPLETE")
    print("="*70)
    print(f"""
Output files:
  {output_dir / 'kalshi_markets.parquet'}  - Raw market data
  {output_dir / 'kalshi_prices.parquet'}   - Price history
  {output_dir / 'kalshi_backtest.parquet'} - Backtest-ready format

Summary:
  Total markets: {len(markets_df)}
  With outcomes: {backtest_df['outcome'].notna().sum() if 'outcome' in backtest_df.columns else 'N/A'}
""")
    
    return markets_df


def main():
    parser = argparse.ArgumentParser(description="Fetch Kalshi data")
    parser.add_argument("--output", type=str, default="data/kalshi",
                        help="Output directory")
    parser.add_argument("--min-date", type=str, default=None,
                        help="Minimum close date (YYYY-MM-DD)")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Max markets to fetch")
    parser.add_argument("--no-prices", action="store_true",
                        help="Skip fetching price history")
    args = parser.parse_args()
    
    fetch_all_kalshi_data(
        output_dir=Path(args.output),
        min_date=args.min_date,
        max_markets=args.max_markets,
        fetch_prices=not args.no_prices
    )


if __name__ == "__main__":
    main()
