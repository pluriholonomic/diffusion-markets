#!/usr/bin/env python3
"""
Analyze Live Trading Performance vs Backtest Expectations

Compares actual paper trading results with backtests on the same time period
to identify strategies that are underperforming or overperforming expectations.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import sys

# Configuration
LOG_DIR = Path("logs/paper_trading")
START_DATE = datetime(2026, 1, 2)  # When paper trading started
END_DATE = datetime.utcnow()


def load_live_trades() -> Tuple[List[Dict], List[Dict]]:
    """Load live trading data from logs."""
    closed_trades = []
    open_positions = []
    
    # Load closed trades
    closed_file = LOG_DIR / "closed_trades.jsonl"
    if closed_file.exists():
        with open(closed_file) as f:
            for line in f:
                try:
                    closed_trades.append(json.loads(line))
                except:
                    pass
    
    # Load open positions
    position_file = LOG_DIR / "position_state.json"
    if position_file.exists():
        with open(position_file) as f:
            data = json.load(f)
            open_positions = data.get('open_positions', [])
    
    return closed_trades, open_positions


def load_signal_history() -> List[Dict]:
    """Load all signals generated during live trading."""
    signals = []
    for f in sorted(LOG_DIR.glob("signals_*.jsonl")):
        with open(f) as fp:
            for line in fp:
                try:
                    signals.append(json.loads(line))
                except:
                    pass
    return signals


def load_market_data() -> pd.DataFrame:
    """Load recent market data for backtesting."""
    # Try to load from optimization cache
    pm_data = Path("data/polymarket/optimization_cache.parquet")
    if pm_data.exists():
        df = pd.read_parquet(pm_data)
        # Filter to recent period
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent = df[df['timestamp'] >= START_DATE]
            return recent
    return pd.DataFrame()


def analyze_by_strategy(closed_trades: List[Dict], open_positions: List[Dict]) -> Dict[str, Dict]:
    """Aggregate performance by strategy."""
    perf = defaultdict(lambda: {
        'realized_pnl': 0,
        'unrealized_pnl': 0,
        'total_pnl': 0,
        'closed_count': 0,
        'open_count': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'avg_return': 0,
        'max_loss': 0,
        'max_win': 0,
        'trades': [],
    })
    
    # Process closed trades
    for t in closed_trades:
        strat = t.get('strategy', 'unknown')
        pnl = t.get('pnl', 0)
        perf[strat]['realized_pnl'] += pnl
        perf[strat]['closed_count'] += 1
        perf[strat]['trades'].append(t)
        
        if pnl > 0:
            perf[strat]['wins'] += 1
            perf[strat]['max_win'] = max(perf[strat]['max_win'], pnl)
        else:
            perf[strat]['losses'] += 1
            perf[strat]['max_loss'] = min(perf[strat]['max_loss'], pnl)
    
    # Process open positions
    for p in open_positions:
        strat = p.get('strategy', 'unknown')
        pnl = p.get('unrealized_pnl', 0)
        perf[strat]['unrealized_pnl'] += pnl
        perf[strat]['open_count'] += 1
    
    # Compute totals and rates
    for strat in perf:
        perf[strat]['total_pnl'] = perf[strat]['realized_pnl'] + perf[strat]['unrealized_pnl']
        total = perf[strat]['wins'] + perf[strat]['losses']
        if total > 0:
            perf[strat]['win_rate'] = perf[strat]['wins'] / total * 100
    
    return dict(perf)


def analyze_market_types(closed_trades: List[Dict], open_positions: List[Dict]) -> Dict[str, Dict]:
    """Categorize performance by market type."""
    categories = defaultdict(lambda: {'pnl': 0, 'count': 0})
    
    all_positions = closed_trades + [
        {**p, 'pnl': p.get('unrealized_pnl', 0)} for p in open_positions
    ]
    
    for p in all_positions:
        q = p.get('market_question', '').lower()
        pnl = p.get('pnl', 0)
        
        # Categorize
        if 'bitcoin' in q or 'ethereum' in q or 'solana' in q or 'crypto' in q:
            cat = 'Crypto'
        elif 'win' in q and ('election' in q or 'president' in q or 'governor' in q):
            cat = 'Politics'
        elif any(sport in q for sport in ['nba', 'nfl', 'mlb', 'nhl', 'soccer', 'football', 'basketball']):
            cat = 'Sports'
        elif 'academy' in q or 'oscar' in q or 'emmy' in q or 'award' in q:
            cat = 'Entertainment'
        elif 'price' in q or 'above' in q or 'below' in q:
            cat = 'Price Prediction'
        else:
            cat = 'Other'
        
        categories[cat]['pnl'] += pnl
        categories[cat]['count'] += 1
    
    return dict(categories)


def backtest_on_recent_data(signals: List[Dict]) -> Dict[str, Dict]:
    """
    Simulate what backtest would have predicted for the signals we generated.
    This is a simplified backtest using signal edges and outcomes.
    """
    backtest_results = defaultdict(lambda: {
        'expected_pnl': 0,
        'signal_count': 0,
        'avg_edge': 0,
        'avg_confidence': 0,
    })
    
    for sig in signals:
        strat = sig.get('strategy', 'unknown')
        edge = sig.get('edge', 0)
        conf = sig.get('confidence', 0)
        size = sig.get('kelly_fraction', 0.1) * 10000  # Approximate sizing
        
        # Expected PnL = edge * size (simplified)
        expected = edge * size
        
        backtest_results[strat]['expected_pnl'] += expected
        backtest_results[strat]['signal_count'] += 1
        backtest_results[strat]['avg_edge'] += edge
        backtest_results[strat]['avg_confidence'] += conf
    
    # Compute averages
    for strat in backtest_results:
        n = backtest_results[strat]['signal_count']
        if n > 0:
            backtest_results[strat]['avg_edge'] /= n
            backtest_results[strat]['avg_confidence'] /= n
    
    return dict(backtest_results)


def print_analysis_report(
    strategy_perf: Dict[str, Dict],
    market_categories: Dict[str, Dict],
    backtest: Dict[str, Dict],
):
    """Print comprehensive analysis report."""
    
    print("=" * 100)
    print("LIVE TRADING PERFORMANCE ANALYSIS")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print("=" * 100)
    
    # Strategy Performance
    print("\n📊 STRATEGY PERFORMANCE (Sorted by Total PnL)")
    print("-" * 100)
    print(f"{'Strategy':<25} {'Open':<6} {'Closed':<7} {'Realized':<12} {'Unrealized':<12} {'Total PnL':<12} {'W/L':<8}")
    print("-" * 100)
    
    total_realized = 0
    total_unrealized = 0
    
    for strat, data in sorted(strategy_perf.items(), key=lambda x: x[1]['total_pnl']):
        realized = data['realized_pnl']
        unrealized = data['unrealized_pnl']
        total = data['total_pnl']
        wins = data['wins']
        losses = data['losses']
        open_c = data['open_count']
        closed_c = data['closed_count']
        
        total_realized += realized
        total_unrealized += unrealized
        
        print(f"{strat:<25} {open_c:<6} {closed_c:<7} ${realized:>9.2f}  ${unrealized:>9.2f}  ${total:>9.2f}  {wins}/{losses}")
    
    print("-" * 100)
    print(f"{'TOTAL':<25} {'':<6} {'':<7} ${total_realized:>9.2f}  ${total_unrealized:>9.2f}  ${total_realized + total_unrealized:>9.2f}")
    
    # Market Category Analysis
    print("\n\n🏷️  PERFORMANCE BY MARKET CATEGORY")
    print("-" * 60)
    print(f"{'Category':<20} {'Count':<8} {'Total PnL':<12} {'Avg PnL':<12}")
    print("-" * 60)
    
    for cat, data in sorted(market_categories.items(), key=lambda x: x[1]['pnl']):
        count = data['count']
        pnl = data['pnl']
        avg = pnl / count if count > 0 else 0
        print(f"{cat:<20} {count:<8} ${pnl:>9.2f}  ${avg:>9.2f}")
    
    # Backtest Comparison
    print("\n\n🔬 LIVE vs BACKTEST COMPARISON")
    print("-" * 100)
    print(f"{'Strategy':<25} {'Actual PnL':<12} {'Expected PnL':<14} {'Diff':<12} {'Signals':<10} {'Avg Edge':<10}")
    print("-" * 100)
    
    for strat in sorted(set(strategy_perf.keys()) | set(backtest.keys())):
        actual = strategy_perf.get(strat, {}).get('total_pnl', 0)
        expected = backtest.get(strat, {}).get('expected_pnl', 0)
        diff = actual - expected
        signals = backtest.get(strat, {}).get('signal_count', 0)
        avg_edge = backtest.get(strat, {}).get('avg_edge', 0)
        
        diff_str = f"${diff:>9.2f}"
        if diff < -50:
            diff_str = f"${diff:>9.2f} ⚠️"
        elif diff > 50:
            diff_str = f"${diff:>9.2f} ✅"
        
        print(f"{strat:<25} ${actual:>9.2f}  ${expected:>11.2f}  {diff_str}  {signals:<10} {avg_edge:.3f}")
    
    # Key Insights
    print("\n\n💡 KEY INSIGHTS")
    print("-" * 60)
    
    # Worst strategy
    worst_strat = min(strategy_perf.items(), key=lambda x: x[1]['total_pnl'])
    print(f"⚠️  Worst Performer: {worst_strat[0]} (${worst_strat[1]['total_pnl']:.2f})")
    
    # Best strategy
    best_strat = max(strategy_perf.items(), key=lambda x: x[1]['total_pnl'])
    print(f"✅ Best Performer: {best_strat[0]} (${best_strat[1]['total_pnl']:.2f})")
    
    # Worst market category
    worst_cat = min(market_categories.items(), key=lambda x: x[1]['pnl'])
    print(f"📉 Worst Category: {worst_cat[0]} (${worst_cat[1]['pnl']:.2f})")
    
    # Biggest underperformers vs backtest
    underperformers = []
    for strat in strategy_perf:
        actual = strategy_perf[strat]['total_pnl']
        expected = backtest.get(strat, {}).get('expected_pnl', 0)
        if expected > 0 and actual < expected * 0.5:
            underperformers.append((strat, actual, expected))
    
    if underperformers:
        print(f"\n⚠️  UNDERPERFORMING vs BACKTEST:")
        for strat, actual, expected in underperformers:
            print(f"   - {strat}: Expected ${expected:.0f}, Got ${actual:.0f}")
    
    # Recommendations
    print("\n\n📋 RECOMMENDATIONS")
    print("-" * 60)
    
    # Check if any strategy should be disabled
    for strat, data in strategy_perf.items():
        if data['total_pnl'] < -100 and data['closed_count'] >= 2:
            print(f"🛑 Consider disabling {strat}: ${data['total_pnl']:.2f} loss from {data['closed_count']} closed trades")
    
    # Check market categories to avoid
    for cat, data in market_categories.items():
        if data['pnl'] < -100 and data['count'] >= 5:
            print(f"⚠️  Avoid {cat} markets: ${data['pnl']:.2f} loss from {data['count']} positions")
    
    print()


def main():
    print("Loading live trading data...")
    closed_trades, open_positions = load_live_trades()
    print(f"  Closed trades: {len(closed_trades)}")
    print(f"  Open positions: {len(open_positions)}")
    
    print("\nLoading signal history...")
    signals = load_signal_history()
    print(f"  Total signals: {len(signals)}")
    
    print("\nAnalyzing performance...")
    strategy_perf = analyze_by_strategy(closed_trades, open_positions)
    market_categories = analyze_market_types(closed_trades, open_positions)
    backtest = backtest_on_recent_data(signals)
    
    print_analysis_report(strategy_perf, market_categories, backtest)


if __name__ == "__main__":
    main()
