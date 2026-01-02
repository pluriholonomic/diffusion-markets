#!/usr/bin/env python3
"""
Live Reconciliation Script

Compares hybrid (live) vs simulated trading to identify discrepancies.
Run periodically to catch bugs.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_signals(log_dir: str, since_minutes: int = 30) -> List[Dict]:
    """Load recent signals."""
    log_path = Path(log_dir)
    signals = []
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    
    for f in sorted(log_path.glob("signals_*.jsonl"), reverse=True):
        with open(f) as fp:
            for line in fp:
                try:
                    s = json.loads(line)
                    ts = datetime.fromisoformat(s.get('timestamp', '2000-01-01'))
                    if ts >= cutoff:
                        signals.append(s)
                except:
                    pass
    return signals


def load_trades(log_dir: str, since_minutes: int = 30) -> List[Dict]:
    """Load recent trades."""
    log_path = Path(log_dir)
    trades = []
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    
    for f in sorted(log_path.glob("trades_*.jsonl"), reverse=True):
        with open(f) as fp:
            for line in fp:
                try:
                    t = json.loads(line)
                    ts = datetime.fromisoformat(t.get('timestamp', '2000-01-01'))
                    if ts >= cutoff:
                        trades.append(t)
                except:
                    pass
    return trades


def analyze_signals(signals: List[Dict]) -> Dict[str, Any]:
    """Analyze signal generation."""
    by_strategy = defaultdict(list)
    by_platform = defaultdict(list)
    
    for s in signals:
        by_strategy[s.get('strategy', 'unknown')].append(s)
        by_platform[s.get('platform', 'unknown')].append(s)
    
    stats = {
        'total': len(signals),
        'by_strategy': {k: len(v) for k, v in by_strategy.items()},
        'by_platform': {k: len(v) for k, v in by_platform.items()},
        'edge_stats': {},
        'kelly_stats': {},
    }
    
    # Edge stats
    edges = [s.get('edge', 0) for s in signals]
    if edges:
        stats['edge_stats'] = {
            'min': min(edges),
            'max': max(edges),
            'mean': sum(edges) / len(edges),
            'zero_count': sum(1 for e in edges if e == 0),
        }
    
    # Kelly stats
    kellys = [s.get('kelly_fraction', 0) for s in signals]
    if kellys:
        stats['kelly_stats'] = {
            'min': min(kellys),
            'max': max(kellys),
            'mean': sum(kellys) / len(kellys),
            'zero_count': sum(1 for k in kellys if k == 0),
        }
    
    return stats


def analyze_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze trade execution."""
    by_strategy = defaultdict(list)
    by_status = defaultdict(list)
    
    for t in trades:
        signal = t.get('signal', {})
        order = t.get('order', {})
        by_strategy[signal.get('strategy', 'unknown')].append(t)
        by_status[order.get('status', 'unknown')].append(t)
    
    stats = {
        'total': len(trades),
        'by_strategy': {k: len(v) for k, v in by_strategy.items()},
        'by_status': {k: len(v) for k, v in by_status.items()},
        'size_stats': {},
    }
    
    # Size stats
    sizes = [t.get('order', {}).get('size', 0) for t in trades]
    if sizes:
        stats['size_stats'] = {
            'min': min(sizes),
            'max': max(sizes),
            'mean': sum(sizes) / len(sizes),
            'total': sum(sizes),
            'zero_count': sum(1 for s in sizes if s == 0),
        }
    
    return stats


def check_signal_to_trade_ratio(signals: List[Dict], trades: List[Dict]) -> Dict[str, Any]:
    """Check if signals are being converted to trades properly."""
    signal_ids = {s.get('signal_id') for s in signals}
    trade_signal_ids = {t.get('signal', {}).get('signal_id') for t in trades}
    
    # Find signals that didn't become trades
    orphan_signals = signal_ids - trade_signal_ids
    
    # Find why signals didn't convert
    orphan_details = []
    for s in signals:
        if s.get('signal_id') in orphan_signals:
            orphan_details.append({
                'signal_id': s.get('signal_id', '')[:8],
                'strategy': s.get('strategy'),
                'edge': s.get('edge'),
                'kelly': s.get('kelly_fraction'),
                'confidence': s.get('confidence'),
            })
    
    return {
        'total_signals': len(signals),
        'total_trades': len(trades),
        'conversion_rate': len(trades) / len(signals) if signals else 0,
        'orphan_signals': len(orphan_signals),
        'orphan_samples': orphan_details[:5],
    }


def check_for_issues(signals: List[Dict], trades: List[Dict]) -> List[str]:
    """Identify potential issues."""
    issues = []
    
    # Issue 1: No signals generated
    if len(signals) == 0:
        issues.append("⚠️  NO SIGNALS generated in the last period")
    
    # Issue 2: No trades executed
    if len(trades) == 0 and len(signals) > 0:
        issues.append("⚠️  SIGNALS but NO TRADES - check execution logic")
    
    # Issue 3: Zero Kelly fractions
    zero_kelly = sum(1 for s in signals if s.get('kelly_fraction', 0) == 0)
    if zero_kelly > 0:
        issues.append(f"⚠️  {zero_kelly} signals have kelly_fraction=0")
    
    # Issue 4: Zero size orders
    zero_size = sum(1 for t in trades if t.get('order', {}).get('size', 0) == 0)
    if zero_size > 0:
        issues.append(f"⚠️  {zero_size} trades have size=$0")
    
    # Issue 5: Low conversion rate
    if signals and trades:
        rate = len(trades) / len(signals)
        if rate < 0.1:
            issues.append(f"⚠️  Low signal→trade conversion: {rate:.1%}")
    
    # Issue 6: Only one strategy active
    strategies = {s.get('strategy') for s in signals}
    if len(strategies) == 1:
        issues.append(f"⚠️  Only 1 strategy generating signals: {list(strategies)[0]}")
    
    # Issue 7: Missing platforms
    platforms = {s.get('platform') for s in signals}
    if 'kalshi' not in platforms:
        issues.append("⚠️  No Kalshi signals - check Kalshi client")
    if 'polymarket' not in platforms:
        issues.append("⚠️  No Polymarket signals - check Polymarket client")
    
    return issues


def run_reconciliation(log_dir: str = "logs/paper_trading", since_minutes: int = 30):
    """Run full reconciliation check."""
    print("=" * 70)
    print(f"RECONCILIATION REPORT - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Looking back {since_minutes} minutes")
    print("=" * 70)
    
    # Load data
    signals = load_signals(log_dir, since_minutes)
    trades = load_trades(log_dir, since_minutes)
    
    print(f"\n📊 DATA LOADED")
    print(f"   Signals: {len(signals)}")
    print(f"   Trades:  {len(trades)}")
    
    # Signal analysis
    print(f"\n📈 SIGNAL ANALYSIS")
    sig_stats = analyze_signals(signals)
    print(f"   By Strategy: {sig_stats['by_strategy']}")
    print(f"   By Platform: {sig_stats['by_platform']}")
    if sig_stats['edge_stats']:
        print(f"   Edge: min={sig_stats['edge_stats']['min']:.2%}, max={sig_stats['edge_stats']['max']:.2%}, mean={sig_stats['edge_stats']['mean']:.2%}")
    if sig_stats['kelly_stats']:
        print(f"   Kelly: min={sig_stats['kelly_stats']['min']:.4f}, max={sig_stats['kelly_stats']['max']:.4f}, zero={sig_stats['kelly_stats']['zero_count']}")
    
    # Trade analysis
    print(f"\n📉 TRADE ANALYSIS")
    trade_stats = analyze_trades(trades)
    print(f"   By Strategy: {trade_stats['by_strategy']}")
    print(f"   By Status: {trade_stats['by_status']}")
    if trade_stats['size_stats']:
        print(f"   Size: min=${trade_stats['size_stats']['min']:.2f}, max=${trade_stats['size_stats']['max']:.2f}, total=${trade_stats['size_stats']['total']:.2f}")
    
    # Conversion check
    print(f"\n🔄 SIGNAL→TRADE CONVERSION")
    conv = check_signal_to_trade_ratio(signals, trades)
    print(f"   Conversion rate: {conv['conversion_rate']:.1%}")
    print(f"   Orphan signals: {conv['orphan_signals']}")
    if conv['orphan_samples']:
        print(f"   Sample orphans:")
        for o in conv['orphan_samples'][:3]:
            print(f"     {o['signal_id']}... strategy={o['strategy']} edge={o['edge']:.2%} kelly={o['kelly']:.4f}")
    
    # Issues
    print(f"\n🚨 ISSUES DETECTED")
    issues = check_for_issues(signals, trades)
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ No issues detected")
    
    print("\n" + "=" * 70)
    
    return {
        'signals': sig_stats,
        'trades': trade_stats,
        'conversion': conv,
        'issues': issues,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/paper_trading")
    parser.add_argument("--since", type=int, default=30, help="Minutes to look back")
    args = parser.parse_args()
    
    run_reconciliation(args.log_dir, args.since)
