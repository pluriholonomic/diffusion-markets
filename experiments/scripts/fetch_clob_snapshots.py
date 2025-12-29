#!/usr/bin/env python3
"""
Fetch and cache CLOB order book snapshots from Polymarket.

Usage:
    # Fetch current snapshots for all tokens in our CLOB data
    python scripts/fetch_clob_snapshots.py --mode current

    # Fetch snapshots for specific tokens
    python scripts/fetch_clob_snapshots.py --tokens token1,token2,token3

    # Continuous fetching (for building historical dataset)
    python scripts/fetch_clob_snapshots.py --mode continuous --interval 3600

Requirements:
    pip install py-clob-client
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

try:
    from py_clob_client.client import ClobClient
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False
    print("Warning: py-clob-client not installed. Run: pip install py-clob-client")


def get_token_ids_from_clob_data(clob_path: Path, limit: Optional[int] = None) -> List[str]:
    """Extract unique token IDs from CLOB parquet."""
    df = pd.read_parquet(clob_path)
    tokens = df['token_id'].unique().tolist()
    if limit:
        tokens = tokens[:limit]
    return tokens


def get_token_ids_from_clob_dir(clob_dir: Path, limit: Optional[int] = None) -> List[str]:
    """Extract token IDs from CLOB history directory (filenames are token IDs)."""
    tokens = [f.stem for f in clob_dir.glob("*.parquet")]
    if limit:
        tokens = tokens[:limit]
    return tokens


def fetch_order_book(client: ClobClient, token_id: str) -> Optional[Dict]:
    """Fetch order book for a single token."""
    try:
        book = client.get_order_book(token_id)
        return {
            "token_id": token_id,
            "timestamp": time.time(),
            "fetched_at": datetime.utcnow().isoformat(),
            "bids": book.get("bids", []),
            "asks": book.get("asks", []),
            "market": book.get("market", ""),
            "asset_id": book.get("asset_id", ""),
        }
    except Exception as e:
        return {"token_id": token_id, "error": str(e), "timestamp": time.time()}


def save_snapshot(snapshot: Dict, output_dir: Path, mode: str = "current"):
    """Save snapshot to file."""
    token_id = snapshot["token_id"]
    
    if mode == "current":
        # Overwrite current snapshot
        path = output_dir / f"{token_id}.json"
    else:
        # Append to historical file with timestamp
        ts = int(snapshot["timestamp"])
        date_str = datetime.fromtimestamp(ts).strftime("%Y%m%d")
        subdir = output_dir / date_str
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{token_id}_{ts}.json"
    
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    
    return path


def compute_stats(snapshot: Dict) -> Dict:
    """Compute order book statistics from snapshot."""
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    
    if not bids or not asks or "error" in snapshot:
        return {"valid": False}
    
    best_bid = float(bids[0].get("price", 0)) if bids else 0
    best_ask = float(asks[0].get("price", 1)) if asks else 1
    
    bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
    ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
    
    mid = (best_bid + best_ask) / 2 if best_bid > 0 else best_ask
    spread_bps = 10000 * (best_ask - best_bid) / mid if mid > 0 else float("inf")
    
    return {
        "valid": True,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": spread_bps,
        "bid_depth_5": bid_depth,
        "ask_depth_5": ask_depth,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
    }


def fetch_batch(
    client: ClobClient,
    token_ids: List[str],
    output_dir: Path,
    mode: str = "current",
    delay: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Fetch order books for a batch of tokens."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    stats_list = []
    errors = 0
    
    start_time = time.time()
    
    for i, token_id in enumerate(token_ids):
        snapshot = fetch_order_book(client, token_id)
        
        if snapshot:
            save_snapshot(snapshot, output_dir, mode)
            stats = compute_stats(snapshot)
            stats["token_id"] = token_id
            stats_list.append(stats)
            results[token_id] = snapshot
            
            if "error" in snapshot:
                errors += 1
        
        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(token_ids) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(token_ids)}] {rate:.1f} tok/s, ETA: {eta:.0f}s, errors: {errors}")
        
        time.sleep(delay)
    
    # Save summary stats
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_parquet(output_dir / "snapshot_stats.parquet", index=False)
        
        valid = stats_df[stats_df["valid"]]
        if len(valid) > 0:
            if verbose:
                print(f"\nSnapshot Statistics:")
                print(f"  Valid snapshots: {len(valid)}/{len(stats_list)}")
                print(f"  Mean spread: {valid['spread_bps'].mean():.1f} bps")
                print(f"  Median spread: {valid['spread_bps'].median():.1f} bps")
                print(f"  Mean bid depth (5 lvl): {valid['bid_depth_5'].mean():.1f}")
                print(f"  Mean ask depth (5 lvl): {valid['ask_depth_5'].mean():.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch CLOB order book snapshots")
    parser.add_argument(
        "--mode",
        choices=["current", "historical", "continuous"],
        default="current",
        help="Fetching mode: current (overwrite), historical (append), continuous (loop)",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        help="Comma-separated list of token IDs to fetch",
    )
    parser.add_argument(
        "--clob-data",
        type=Path,
        default=Path("data/backtest/clob_merged.parquet"),
        help="Path to CLOB parquet to extract token IDs from",
    )
    parser.add_argument(
        "--clob-dir",
        type=Path,
        default=Path("data/polymarket/clob_history_yes_f1"),
        help="Alternative: directory with per-token CLOB files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clob_snapshots"),
        help="Output directory for snapshots",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tokens to fetch",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Interval between fetch cycles in continuous mode (seconds)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()
    
    if not HAS_CLIENT:
        print("Error: py-clob-client not installed")
        print("Run: pip install py-clob-client")
        return 1
    
    verbose = not args.quiet
    
    # Get token IDs
    if args.tokens:
        token_ids = args.tokens.split(",")
    elif args.clob_data.exists():
        token_ids = get_token_ids_from_clob_data(args.clob_data, args.limit)
    elif args.clob_dir.exists():
        token_ids = get_token_ids_from_clob_dir(args.clob_dir, args.limit)
    else:
        print("Error: No token source specified")
        return 1
    
    if verbose:
        print(f"Fetching order books for {len(token_ids)} tokens")
        print(f"Output: {args.output_dir}")
        print(f"Mode: {args.mode}")
        print()
    
    # Initialize client
    client = ClobClient(host="https://clob.polymarket.com")
    
    if args.mode == "continuous":
        # Continuous fetching loop
        cycle = 0
        while True:
            cycle += 1
            if verbose:
                print(f"\n=== Cycle {cycle} at {datetime.utcnow().isoformat()} ===")
            
            fetch_batch(
                client=client,
                token_ids=token_ids,
                output_dir=args.output_dir,
                mode="historical",
                delay=args.delay,
                verbose=verbose,
            )
            
            if verbose:
                print(f"Sleeping {args.interval}s until next cycle...")
            time.sleep(args.interval)
    else:
        # Single fetch
        fetch_batch(
            client=client,
            token_ids=token_ids,
            output_dir=args.output_dir,
            mode=args.mode,
            delay=args.delay,
            verbose=verbose,
        )
    
    if verbose:
        print(f"\nDone! Snapshots saved to {args.output_dir}")
        
        # Show disk usage
        total_size = sum(f.stat().st_size for f in args.output_dir.rglob("*.json"))
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    
    return 0


if __name__ == "__main__":
    exit(main())


