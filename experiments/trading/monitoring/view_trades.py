#!/usr/bin/env python3
"""
Trade Viewer

View trades with full signal provenance from the command line.

Usage:
    python -m trading.monitoring.view_trades [--log-dir logs/paper_trading] [--limit 50]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def load_trades(log_dir: str, limit: int = 50) -> List[Dict]:
    """Load trades from log files."""
    log_path = Path(log_dir)
    trades = []
    
    # Find trade files
    trade_files = sorted(log_path.glob("trades_*.jsonl"), reverse=True)
    
    for trade_file in trade_files:
        with open(trade_file) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    trades.append(t)
                except:
                    pass
        
        if len(trades) >= limit:
            break
    
    return trades[-limit:]


def format_trade(trade: Dict) -> str:
    """Format a trade for display."""
    signal = trade.get('signal', {})
    order = trade.get('order', {})
    
    signal_id = signal.get('signal_id', 'N/A')[:8]
    timestamp = trade.get('timestamp', '')[:19]
    strategy = signal.get('strategy', 'unknown')
    platform = signal.get('platform', order.get('platform', ''))
    market_id = signal.get('market_id', order.get('market_id', ''))[:20]
    side = signal.get('side', order.get('side', ''))
    edge = signal.get('edge', 0)
    confidence = signal.get('confidence', 0)
    kelly = signal.get('kelly_fraction', 0)
    size = order.get('size', 0)
    status = order.get('status', 'unknown')
    
    metadata = signal.get('metadata', {})
    price = metadata.get('price', 0)
    spread = metadata.get('spread', 0)
    category = metadata.get('category', '')
    
    return f"""
┌─────────────────────────────────────────────────────────────────────
│ Signal ID: {signal_id}...  |  Time: {timestamp}
├─────────────────────────────────────────────────────────────────────
│ Strategy:   {strategy:<25} Platform: {platform}
│ Market:     {market_id}...
│ Category:   {category}
├─────────────────────────────────────────────────────────────────────
│ SIGNAL:
│   Side:       {side.upper() if side else 'N/A'}
│   Edge:       {edge:.2%}
│   Confidence: {confidence:.2f}
│   Kelly:      {kelly:.4f}
│   Price:      {price:.3f}
│   Spread:     {spread:.3f}
├─────────────────────────────────────────────────────────────────────
│ ORDER:
│   Size:       ${size:,.2f}
│   Status:     {status.upper() if status else 'N/A'}
└─────────────────────────────────────────────────────────────────────
"""


def print_summary(trades: List[Dict]):
    """Print summary statistics."""
    if not trades:
        print("No trades found.")
        return
    
    total = len(trades)
    filled = sum(1 for t in trades if t.get('order', {}).get('status') == 'filled')
    
    strategies = {}
    platforms = {}
    
    for t in trades:
        signal = t.get('signal', {})
        strat = signal.get('strategy', 'unknown')
        plat = signal.get('platform', 'unknown')
        
        strategies[strat] = strategies.get(strat, 0) + 1
        platforms[plat] = platforms.get(plat, 0) + 1
    
    print("\n" + "=" * 70)
    print("                         TRADE SUMMARY")
    print("=" * 70)
    print(f"  Total Trades:    {total}")
    print(f"  Filled:          {filled} ({filled/total*100:.1f}%)")
    print(f"  Pending/Other:   {total - filled}")
    print()
    print("  By Strategy:")
    for strat, count in sorted(strategies.items(), key=lambda x: -x[1]):
        print(f"    {strat:<25} {count:>5}")
    print()
    print("  By Platform:")
    for plat, count in sorted(platforms.items(), key=lambda x: -x[1]):
        print(f"    {plat:<15} {count:>5}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="View trades with provenance")
    parser.add_argument("--log-dir", default="logs/paper_trading",
                        help="Directory containing trade logs")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum number of trades to show")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show summary, not individual trades")
    parser.add_argument("--last", type=int, default=5,
                        help="Show last N trades in detail")
    args = parser.parse_args()
    
    trades = load_trades(args.log_dir, args.limit)
    
    if not trades:
        print(f"No trades found in {args.log_dir}")
        return
    
    # Summary
    print_summary(trades)
    
    if args.summary_only:
        return
    
    # Last N trades in detail
    print(f"\nLast {min(args.last, len(trades))} trades with full provenance:")
    print("=" * 70)
    
    for trade in trades[-args.last:]:
        print(format_trade(trade))


if __name__ == "__main__":
    main()
