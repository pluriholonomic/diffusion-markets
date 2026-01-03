#!/usr/bin/env python3
"""
Orderbook Scraper

Continuously captures orderbook snapshots from Polymarket CLOB API.
Saves: (ts, market_id, token_id, bid_depth, ask_depth, best_bid, best_ask, spread, mid_price)

Usage:
    python scripts/orderbook_scraper.py --output data/orderbook_snapshots \
        --interval 60 --markets 100
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


class OrderbookScraper:
    """Continuously scrape orderbook data from Polymarket."""
    
    def __init__(
        self,
        output_dir: Path,
        interval_seconds: int = 60,
        max_markets: int = 100,
    ):
        self.output_dir = output_dir
        self.interval = interval_seconds
        self.max_markets = max_markets
        self.running = True
        
        # Create session
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Markets cache
        self.markets: List[Dict] = []
        self.last_market_refresh = 0
        
        # Data buffer
        self.snapshots: List[Dict] = []
        self.snapshot_count = 0
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print("\nShutting down...")
        self.running = False
    
    def refresh_markets(self) -> List[Dict]:
        """Get active markets from Gamma API."""
        markets = []
        
        try:
            # Get active markets with high volume
            resp = self.session.get(
                f"{GAMMA_API}/markets",
                params={"limit": 100, "closed": "false", "order": "volume", "ascending": "false"},
                timeout=30
            )
            
            if resp.status_code == 200:
                data = resp.json()
                batch = data if isinstance(data, list) else data.get("markets", [])
                
                for m in batch[:self.max_markets]:
                    # Parse token IDs
                    clob_ids = m.get("clobTokenIds", "[]")
                    if isinstance(clob_ids, str):
                        try:
                            tokens = json.loads(clob_ids)
                        except:
                            tokens = []
                    else:
                        tokens = clob_ids or []
                    
                    if tokens:
                        markets.append({
                            "market_id": m.get("slug", m.get("id", "")),
                            "condition_id": m.get("conditionId", ""),
                            "yes_token_id": tokens[0] if len(tokens) > 0 else "",
                            "no_token_id": tokens[1] if len(tokens) > 1 else "",
                            "question": m.get("question", "")[:100],
                        })
        except Exception as e:
            print(f"Error refreshing markets: {e}")
        
        return markets
    
    def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Fetch orderbook for a token."""
        try:
            resp = self.session.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id},
                timeout=10
            )
            
            if resp.status_code == 200:
                return resp.json()
            return None
        except:
            return None
    
    def process_orderbook(self, market: Dict, book: Dict, timestamp: datetime) -> Dict:
        """Process orderbook into snapshot row."""
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        
        # Calculate depths (sum of sizes at each level)
        bid_depth = sum(float(b.get("size", 0)) for b in bids)
        ask_depth = sum(float(a.get("size", 0)) for a in asks)
        
        # Best bid/ask
        best_bid = float(bids[0].get("price", 0)) if bids else 0
        best_ask = float(asks[0].get("price", 0)) if asks else 0
        
        # Spread and mid
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
        mid_price = (best_bid + best_ask) / 2 if spread > 0 else 0
        
        return {
            "timestamp": timestamp,
            "market_id": market["market_id"],
            "token_id": market["yes_token_id"],
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "mid_price": mid_price,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
        }
    
    def save_snapshots(self):
        """Save buffered snapshots to parquet."""
        if not self.snapshots:
            return
        
        df = pd.DataFrame(self.snapshots)
        
        # Generate filename with date
        date_str = datetime.utcnow().strftime("%Y%m%d")
        output_file = self.output_dir / f"orderbook_{date_str}.parquet"
        
        # Append to existing file if it exists
        if output_file.exists():
            existing = pd.read_parquet(output_file)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_parquet(output_file, index=False)
        
        print(f"  Saved {len(self.snapshots)} snapshots to {output_file.name}")
        self.snapshots = []
    
    def run(self):
        """Main scraping loop."""
        print("=" * 60)
        print("ORDERBOOK SCRAPER")
        print("=" * 60)
        print(f"Output: {self.output_dir}")
        print(f"Interval: {self.interval}s")
        print(f"Max markets: {self.max_markets}")
        print()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        while self.running:
            loop_start = time.time()
            
            # Refresh markets every 5 minutes
            if time.time() - self.last_market_refresh > 300:
                print("Refreshing markets...")
                self.markets = self.refresh_markets()
                self.last_market_refresh = time.time()
                print(f"  Got {len(self.markets)} markets")
            
            if not self.markets:
                print("No markets available, waiting...")
                time.sleep(60)
                continue
            
            # Capture snapshots
            timestamp = datetime.utcnow()
            captured = 0
            
            for market in self.markets:
                if not self.running:
                    break
                
                token_id = market.get("yes_token_id", "")
                if not token_id:
                    continue
                
                book = self.get_orderbook(token_id)
                if book:
                    snapshot = self.process_orderbook(market, book, timestamp)
                    self.snapshots.append(snapshot)
                    captured += 1
                
                time.sleep(0.05)  # Rate limit
            
            self.snapshot_count += captured
            print(f"[{timestamp.strftime('%H:%M:%S')}] Captured {captured} orderbooks "
                  f"(total: {self.snapshot_count})")
            
            # Save every 100 snapshots
            if len(self.snapshots) >= 100:
                self.save_snapshots()
            
            # Wait for next interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0 and self.running:
                time.sleep(sleep_time)
        
        # Final save
        self.save_snapshots()
        print(f"\nTotal snapshots captured: {self.snapshot_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/orderbook_snapshots",
                        help="Output directory for snapshots")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between snapshot rounds")
    parser.add_argument("--markets", type=int, default=100,
                        help="Maximum markets to track")
    args = parser.parse_args()
    
    scraper = OrderbookScraper(
        output_dir=Path(args.output),
        interval_seconds=args.interval,
        max_markets=args.markets,
    )
    
    scraper.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
