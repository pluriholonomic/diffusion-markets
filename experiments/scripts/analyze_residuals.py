#!/usr/bin/env python3
"""
Residual Analysis for Trading Strategies

Analyzes backtest results to identify:
1. Loss patterns - what characterizes losing trades
2. Missed opportunities - signals we should have taken
3. Feature discovery - what predicts success

Run: python scripts/analyze_residuals.py --results-dir logs/continuous_optimization
"""

import sys
from pathlib import Path

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


@dataclass
class LossCluster:
    """Cluster of similar losing trades."""
    cluster_id: int
    n_trades: int
    total_loss: float
    avg_entry_price: float
    avg_volume: float
    avg_spread: float
    common_categories: Dict[str, int]
    suggested_filter: Optional[str] = None


@dataclass
class FilterSuggestion:
    """Suggested filter to avoid losses."""
    feature: str
    threshold: float
    direction: str  # '<=' or '>='
    expected_improvement: float
    trades_filtered: int
    loss_avoided: float


@dataclass
class MissedOpportunity:
    """Trade we didn't take but should have."""
    market_id: str
    signal_edge: float
    hypothetical_pnl: float
    rejection_reason: str
    timestamp: str


class LossPatternAnalyzer:
    """Analyze losing trades to find avoidable patterns."""
    
    def __init__(self, trades_df: pd.DataFrame):
        """
        Args:
            trades_df: DataFrame with columns:
                - pnl: trade profit/loss
                - entry_price: price at entry
                - volume: market volume
                - category: market category
                - spread: bid-ask spread (if available)
                - time_to_resolution: days until resolution
                - signal_edge: edge at signal generation
        """
        self.trades = trades_df
        self.losing_trades = trades_df[trades_df['pnl'] < 0].copy()
        self.winning_trades = trades_df[trades_df['pnl'] > 0].copy()
        
    def analyze_by_market_characteristics(self) -> pd.DataFrame:
        """Find market characteristics that predict losses."""
        df = self.trades.copy()
        
        # Add derived features
        df['won'] = df['pnl'] > 0
        
        # Compute correlations with winning
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != 'won' and col != 'pnl':
                try:
                    corr = df['won'].corr(df[col])
                    if not np.isnan(corr):
                        correlations[col] = corr
                except:
                    pass
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\n=== Features Correlated with Winning ===")
        for feat, corr in sorted_corrs[:10]:
            direction = "↑" if corr > 0 else "↓"
            print(f"  {feat}: {corr:+.3f} {direction}")
        
        return df
    
    def find_loss_clusters(self, n_clusters: int = 5) -> List[LossCluster]:
        """Cluster losing trades to find common patterns."""
        if len(self.losing_trades) < 10:
            print("Not enough losing trades to cluster")
            return []
        
        # Features for clustering
        features = ['entry_price', 'volume']
        if 'spread' in self.losing_trades.columns:
            features.append('spread')
        if 'time_to_resolution' in self.losing_trades.columns:
            features.append('time_to_resolution')
        
        # Get available features
        available = [f for f in features if f in self.losing_trades.columns]
        if not available:
            print("No numeric features available for clustering")
            return []
        
        X = self.losing_trades[available].fillna(0).values
        
        # Simple k-means style clustering (manual to avoid sklearn dep)
        np.random.seed(42)
        
        # Normalize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-6
        X_norm = (X - X_mean) / X_std
        
        # Initialize centroids randomly
        indices = np.random.choice(len(X_norm), min(n_clusters, len(X_norm)), replace=False)
        centroids = X_norm[indices]
        
        # Simple k-means iterations
        for _ in range(10):
            # Assign points to nearest centroid
            distances = np.sqrt(((X_norm[:, None] - centroids[None, :]) ** 2).sum(axis=2))
            labels = distances.argmin(axis=1)
            
            # Update centroids
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    centroids[k] = X_norm[mask].mean(axis=0)
        
        self.losing_trades = self.losing_trades.copy()
        self.losing_trades['cluster'] = labels
        
        # Build cluster summaries
        clusters = []
        for i in range(n_clusters):
            cluster_df = self.losing_trades[self.losing_trades['cluster'] == i]
            if len(cluster_df) == 0:
                continue
            
            category_counts = {}
            if 'category' in cluster_df.columns:
                category_counts = cluster_df['category'].value_counts().head(3).to_dict()
            
            cluster = LossCluster(
                cluster_id=i,
                n_trades=len(cluster_df),
                total_loss=float(cluster_df['pnl'].sum()),
                avg_entry_price=float(cluster_df['entry_price'].mean()) if 'entry_price' in cluster_df else 0,
                avg_volume=float(cluster_df['volume'].mean()) if 'volume' in cluster_df else 0,
                avg_spread=float(cluster_df['spread'].mean()) if 'spread' in cluster_df else 0,
                common_categories=category_counts,
            )
            clusters.append(cluster)
        
        # Sort by total loss (worst first)
        clusters.sort(key=lambda c: c.total_loss)
        
        print("\n=== Loss Clusters ===")
        for c in clusters[:5]:
            print(f"\nCluster {c.cluster_id}: {c.n_trades} trades, ${c.total_loss:.2f} loss")
            print(f"  Avg price: {c.avg_entry_price:.2f}, Avg volume: ${c.avg_volume:.0f}")
            if c.common_categories:
                cats = ", ".join(f"{k}: {v}" for k, v in list(c.common_categories.items())[:3])
                print(f"  Categories: {cats}")
        
        return clusters
    
    def suggest_filters(self) -> List[FilterSuggestion]:
        """Suggest filters to avoid loss patterns."""
        suggestions = []
        df = self.trades.copy()
        df['won'] = df['pnl'] > 0
        
        # Test various thresholds for each feature
        features_to_test = {
            'spread': (0.01, 0.10, 10, '<='),
            'volume': (100, 10000, 10, '>='),
            'entry_price': (0.05, 0.95, 10, None),  # Test both directions
        }
        
        for feature, (low, high, n_steps, default_dir) in features_to_test.items():
            if feature not in df.columns:
                continue
            
            thresholds = np.linspace(low, high, n_steps)
            
            for threshold in thresholds:
                for direction in (['<=', '>='] if default_dir is None else [default_dir]):
                    if direction == '<=':
                        mask = df[feature] <= threshold
                    else:
                        mask = df[feature] >= threshold
                    
                    included = df[mask]
                    excluded = df[~mask]
                    
                    if len(included) < 10 or len(excluded) < 10:
                        continue
                    
                    win_rate_included = included['won'].mean()
                    win_rate_excluded = excluded['won'].mean()
                    
                    improvement = win_rate_included - win_rate_excluded
                    
                    if improvement > 0.05:  # At least 5% improvement
                        loss_avoided = excluded[~excluded['won']]['pnl'].sum()
                        
                        suggestions.append(FilterSuggestion(
                            feature=feature,
                            threshold=threshold,
                            direction=direction,
                            expected_improvement=improvement,
                            trades_filtered=len(excluded),
                            loss_avoided=float(-loss_avoided),  # Convert to positive
                        ))
        
        # Sort by expected improvement
        suggestions.sort(key=lambda s: s.expected_improvement, reverse=True)
        
        print("\n=== Suggested Filters ===")
        for s in suggestions[:5]:
            print(f"\n{s.feature} {s.direction} {s.threshold:.4f}")
            print(f"  Win rate improvement: +{s.expected_improvement:.1%}")
            print(f"  Trades filtered: {s.trades_filtered}")
            print(f"  Loss avoided: ${s.loss_avoided:.2f}")
        
        return suggestions


class MissedTradeAnalyzer:
    """Analyze trades that should have been taken but weren't."""
    
    def __init__(self, signals_df: pd.DataFrame, trades_df: pd.DataFrame):
        """
        Args:
            signals_df: DataFrame of all generated signals
            trades_df: DataFrame of executed trades
        """
        self.signals = signals_df
        self.trades = trades_df
        
        # Get signal IDs that were traded
        if 'signal_id' in trades_df.columns:
            self.traded_ids = set(trades_df['signal_id'].dropna().unique())
        else:
            self.traded_ids = set()
    
    def find_profitable_missed(self, outcomes: pd.DataFrame) -> List[MissedOpportunity]:
        """Find signals that weren't traded but would have been profitable."""
        if 'signal_id' not in self.signals.columns:
            return []
        
        missed = []
        
        for _, signal in self.signals.iterrows():
            if signal.get('signal_id') in self.traded_ids:
                continue
            
            market_id = signal.get('market_id')
            if market_id is None:
                continue
            
            # Look up outcome
            outcome_row = outcomes[outcomes['market_id'] == market_id]
            if len(outcome_row) == 0:
                continue
            
            outcome = outcome_row.iloc[0].get('outcome')
            if outcome is None:
                continue
            
            # Simulate trade
            side = signal.get('side', 'yes')
            entry_price = signal.get('price', 0.5)
            size = signal.get('size', 100)
            
            if side == 'yes':
                if outcome == 1:
                    pnl = size * (1 - entry_price) / entry_price
                else:
                    pnl = -size
            else:
                if outcome == 0:
                    pnl = size * (1 - entry_price) / entry_price
                else:
                    pnl = -size
            
            if pnl > 0:
                missed.append(MissedOpportunity(
                    market_id=market_id,
                    signal_edge=signal.get('edge', 0),
                    hypothetical_pnl=pnl,
                    rejection_reason=signal.get('rejection_reason', 'unknown'),
                    timestamp=str(signal.get('timestamp', '')),
                ))
        
        # Sort by hypothetical PnL
        missed.sort(key=lambda m: m.hypothetical_pnl, reverse=True)
        
        print("\n=== Top Missed Opportunities ===")
        total_missed_pnl = sum(m.hypothetical_pnl for m in missed)
        print(f"Total missed PnL: ${total_missed_pnl:.2f} across {len(missed)} trades")
        
        for m in missed[:10]:
            print(f"\n  {m.market_id}")
            print(f"    Edge: {m.signal_edge:.1%}, Hypothetical PnL: ${m.hypothetical_pnl:.2f}")
            print(f"    Reason: {m.rejection_reason}")
        
        return missed
    
    def suggest_threshold_adjustments(self, missed: List[MissedOpportunity]) -> Dict[str, float]:
        """Suggest looser thresholds to capture missed opportunities."""
        if not missed:
            return {}
        
        # Analyze rejection reasons
        reasons = Counter(m.rejection_reason for m in missed)
        
        print("\n=== Rejection Reasons ===")
        for reason, count in reasons.most_common(5):
            print(f"  {reason}: {count}")
        
        suggestions = {}
        
        # If many were rejected for edge being too low
        if reasons.get('edge_too_low', 0) > 5:
            edges = [m.signal_edge for m in missed if m.rejection_reason == 'edge_too_low']
            if edges:
                suggestions['min_edge'] = np.percentile(edges, 10)
                print(f"\nSuggested min_edge: {suggestions['min_edge']:.3f}")
        
        if reasons.get('confidence_too_low', 0) > 5:
            # Would need confidence data to suggest
            pass
        
        if reasons.get('liquidity_too_low', 0) > 5:
            # Would need liquidity data to suggest
            pass
        
        return suggestions


def load_optimization_results(results_dir: str) -> Dict[str, Any]:
    """Load optimization results from directory."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}
    
    results = {}
    
    # Load summary
    summary_file = results_path / "optimization_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results['summary'] = json.load(f)
    
    # Load individual strategy results
    for json_file in results_path.glob("*_best.json"):
        strategy_name = json_file.stem.replace("_best", "")
        with open(json_file) as f:
            results[strategy_name] = json.load(f)
    
    return results


def create_synthetic_trades_for_analysis(results: Dict) -> pd.DataFrame:
    """Create synthetic trade data from optimization results for analysis."""
    trades = []
    
    # Generate synthetic trades based on strategy parameters
    np.random.seed(42)
    n_trades = 1000
    
    for i in range(n_trades):
        # Simulate trade characteristics
        entry_price = np.random.uniform(0.1, 0.9)
        volume = np.random.exponential(10000)
        spread = np.random.exponential(0.02)
        category = np.random.choice(['politics', 'crypto', 'sports', 'economics'])
        time_to_resolution = np.random.exponential(7)
        signal_edge = np.random.exponential(0.05)
        
        # Win probability depends on edge and other factors
        win_prob = 0.45 + signal_edge * 2 - spread * 5 - 0.1 * (entry_price < 0.1 or entry_price > 0.9)
        win_prob = np.clip(win_prob, 0.3, 0.7)
        
        won = np.random.random() < win_prob
        
        if won:
            pnl = 100 * (1 - entry_price) / entry_price * np.random.uniform(0.5, 1.5)
        else:
            pnl = -100 * np.random.uniform(0.5, 1.0)
        
        trades.append({
            'trade_id': f"trade_{i}",
            'pnl': pnl,
            'entry_price': entry_price,
            'volume': volume,
            'spread': spread,
            'category': category,
            'time_to_resolution': time_to_resolution,
            'signal_edge': signal_edge,
        })
    
    return pd.DataFrame(trades)


def main():
    parser = argparse.ArgumentParser(description="Analyze trading residuals")
    parser.add_argument("--results-dir", default="logs/continuous_optimization",
                        help="Directory with optimization results")
    parser.add_argument("--trades-file", default=None,
                        help="Parquet file with trade data")
    parser.add_argument("--output", default="logs/residual_analysis.json",
                        help="Output file for analysis results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RESIDUAL ANALYSIS")
    print("=" * 60)
    
    # Load optimization results
    results = load_optimization_results(args.results_dir)
    
    if results.get('summary'):
        summary = results['summary']
        print(f"\nLast optimization: {summary.get('last_updated', 'unknown')}")
        print(f"Tasks: {summary.get('total_tasks', 0)}")
    
    # Load or create trade data
    if args.trades_file and Path(args.trades_file).exists():
        trades_df = pd.read_parquet(args.trades_file)
        print(f"\nLoaded {len(trades_df)} trades from {args.trades_file}")
    else:
        print("\nNo trade file found, using synthetic data for demonstration")
        trades_df = create_synthetic_trades_for_analysis(results)
    
    # Run loss pattern analysis
    print("\n" + "=" * 60)
    print("LOSS PATTERN ANALYSIS")
    print("=" * 60)
    
    analyzer = LossPatternAnalyzer(trades_df)
    analyzer.analyze_by_market_characteristics()
    clusters = analyzer.find_loss_clusters()
    suggestions = analyzer.suggest_filters()
    
    # Save results
    output = {
        'timestamp': datetime.utcnow().isoformat(),
        'n_trades': len(trades_df),
        'n_winning': len(analyzer.winning_trades),
        'n_losing': len(analyzer.losing_trades),
        'win_rate': len(analyzer.winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
        'total_pnl': float(trades_df['pnl'].sum()),
        'loss_clusters': [asdict(c) for c in clusters[:5]],
        'filter_suggestions': [asdict(s) for s in suggestions[:5]],
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nAnalysis saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total trades: {output['n_trades']}")
    print(f"Win rate: {output['win_rate']:.1%}")
    print(f"Total PnL: ${output['total_pnl']:.2f}")
    print(f"Loss clusters identified: {len(clusters)}")
    print(f"Filter suggestions: {len(suggestions)}")
    
    if suggestions:
        best = suggestions[0]
        print(f"\nTop suggestion: {best.feature} {best.direction} {best.threshold:.4f}")
        print(f"  Expected improvement: +{best.expected_improvement:.1%}")


if __name__ == "__main__":
    main()
