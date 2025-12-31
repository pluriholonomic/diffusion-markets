#!/usr/bin/env python3
"""
Calibration-Weighted Portfolio Backtest

Runs the dynamic ensemble model:
    π(t) = Σᵢ aᵢ(t) πᵢ(t)

Where weights aᵢ(t) depend on:
- Calibration deviation
- Semantic features (question text embeddings)
- Market characteristics

Usage:
    python scripts/run_calibration_portfolio_backtest.py \
        --output-dir runs/calibration_portfolio \
        --use-semantic-features

Output:
    - results.json: Summary metrics
    - results.parquet: Detailed trade-by-trade results
    - calibration_summary.parquet: Category-level calibration stats
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecastbench.strategies.calibration_weights import (
    CalibrationState,
    CalibrationWeightConfig,
    CalibrationWeightTracker,
    compute_calibration_weight,
)


def categorize_question(q: str) -> str:
    """Assign category based on question text."""
    q = str(q).lower()
    if 'bitcoin' in q or 'crypto' in q or 'eth ' in q or 'btc' in q:
        return 'crypto'
    if 'trump' in q or 'biden' in q or 'harris' in q or 'election' in q or 'vote' in q:
        return 'politics'
    if 'nba' in q or 'nfl' in q or 'win ' in q or 'beat' in q or 'championship' in q:
        return 'sports'
    if 'temperature' in q or 'weather' in q or 'rain' in q:
        return 'weather'
    if 'fed ' in q or 'rate' in q or 'inflation' in q or 'gdp' in q:
        return 'finance'
    return 'other'


def extract_semantic_features(question: str) -> dict:
    """
    Extract semantic features from question text.
    
    These features help aᵢ(t) adapt to question characteristics.
    """
    q = str(question).lower()
    
    features = {
        # Question type
        'is_binary_outcome': 1 if any(w in q for w in ['will', 'does', 'is ', 'are ']) else 0,
        'is_threshold': 1 if any(w in q for w in ['above', 'below', 'over', 'under', 'more than', 'less than']) else 0,
        'is_date_specific': 1 if any(w in q for w in ['by ', 'before', 'after', 'on ', 'in 202']) else 0,
        
        # Entity type
        'has_person': 1 if any(w in q for w in ['trump', 'biden', 'harris', 'musk', 'obama']) else 0,
        'has_org': 1 if any(w in q for w in ['fed', 'sec', 'congress', 'supreme court']) else 0,
        'has_crypto': 1 if any(w in q for w in ['bitcoin', 'btc', 'ethereum', 'eth', 'solana']) else 0,
        
        # Complexity
        'question_length': len(q),
        'word_count': len(q.split()),
        'has_numbers': 1 if any(c.isdigit() for c in q) else 0,
        
        # Temporal
        'is_short_term': 1 if any(w in q for w in ['today', 'tomorrow', 'this week', 'friday']) else 0,
        'is_long_term': 1 if any(w in q for w in ['year', 'annual', '2025', '2026']) else 0,
    }
    
    return features


class SemanticCalibrationWeightTracker(CalibrationWeightTracker):
    """
    Extended weight tracker with semantic features.
    
    Weight function:
        aᵢ(t) = f(|calibration|, price_deviation, volume, semantic_features)
    """
    
    def __init__(self, cfg: CalibrationWeightConfig = CalibrationWeightConfig()):
        super().__init__(cfg)
        self.semantic_weights = {}  # Learned semantic adjustments
    
    def learn_semantic_weights(
        self, 
        df: pd.DataFrame,
        category_col: str = "cat",
        price_col: str = "market_prob",
        outcome_col: str = "y",
        question_col: str = "question"
    ):
        """Learn which semantic features improve predictions."""
        # Extract semantic features
        semantic_df = df[question_col].apply(extract_semantic_features).apply(pd.Series)
        
        # Compute per-trade "edge" (whether calibration-based trade would win)
        edges = []
        for _, row in df.iterrows():
            cat = row[category_col]
            if cat not in self.calibration_states:
                edges.append(0)
                continue
            
            direction = self.calibration_states[cat].direction
            price = row[price_col]
            y = row[outcome_col]
            
            if direction == -1:  # Short YES
                edge = 1 if y == 0 else -1
            else:  # Long YES
                edge = 1 if y == 1 else -1
            edges.append(edge)
        
        semantic_df['edge'] = edges
        
        # Learn correlations between semantic features and edge
        for col in semantic_df.columns:
            if col == 'edge':
                continue
            corr = semantic_df[col].corr(semantic_df['edge'])
            if not np.isnan(corr):
                # Convert correlation to weight multiplier
                # Positive correlation -> boost weight
                # Negative correlation -> reduce weight
                self.semantic_weights[col] = 1 + corr * 0.5  # Scale factor
        
        print("Learned semantic weights:")
        for feat, w in sorted(self.semantic_weights.items(), key=lambda x: -abs(x[1]-1)):
            print(f"  {feat:25}: {w:.3f}")
    
    def get_weight_with_semantics(
        self,
        category: str,
        market_price: float,
        volume: float,
        question: str
    ) -> float:
        """Get weight including semantic adjustments."""
        # Base weight
        base_weight = self.get_weight(category, market_price, volume)
        
        # Semantic adjustment
        if len(self.semantic_weights) == 0:
            return base_weight
        
        features = extract_semantic_features(question)
        semantic_multiplier = 1.0
        
        for feat, value in features.items():
            if feat in self.semantic_weights and value > 0:
                semantic_multiplier *= self.semantic_weights[feat]
        
        # Clip to reasonable range
        semantic_multiplier = np.clip(semantic_multiplier, 0.5, 2.0)
        
        return np.clip(base_weight * semantic_multiplier, 0.1, 5.0)


def run_backtest(
    df: pd.DataFrame,
    train_frac: float = 0.5,
    val_frac: float = 0.2,
    use_semantic: bool = False,
    verbose: bool = True
) -> dict:
    """
    Run full calibration-weighted portfolio backtest.
    
    Args:
        df: DataFrame with market_prob, y, question, volume, closedTime
        train_frac: Fraction for training
        val_frac: Fraction for validation (semantic learning)
        use_semantic: Whether to use semantic features
        verbose: Print progress
    
    Returns:
        Dictionary with all results
    """
    start_time = time.time()
    
    # Prepare data
    df = df.copy()
    df['cat'] = df['question'].apply(categorize_question)
    df['volume'] = pd.to_numeric(df['volumeNum'], errors='coerce').fillna(10000)
    df['close_ts'] = pd.to_datetime(df['closedTime'], format='mixed', utc=True)
    df = df.sort_values('close_ts').reset_index(drop=True)
    
    # Split data
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    if verbose:
        print(f"Data split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Initialize tracker
    if use_semantic:
        tracker = SemanticCalibrationWeightTracker()
    else:
        tracker = CalibrationWeightTracker()
    
    # Learn calibration from training data
    if verbose:
        print("\nLearning calibration from training data...")
    
    for cat in train['cat'].unique():
        cat_df = train[train['cat'] == cat]
        tracker.update_calibration(cat, cat_df['y'].values, cat_df['market_prob'].values)
    
    calibration_summary = tracker.summary()
    if verbose:
        print(calibration_summary.to_string())
    
    # Learn semantic weights from validation (if enabled)
    if use_semantic and isinstance(tracker, SemanticCalibrationWeightTracker):
        if verbose:
            print("\nLearning semantic weights from validation data...")
        tracker.learn_semantic_weights(val)
    
    # Run on test set
    if verbose:
        print(f"\nRunning backtest on {len(test)} test markets...")
    
    results = []
    for _, row in test.iterrows():
        cat = row['cat']
        if cat not in tracker.calibration_states:
            continue
        
        price = row['market_prob']
        y = row['y']
        volume = row['volume']
        question = row['question']
        
        direction = tracker.get_direction(cat)
        
        # Get weight
        if use_semantic and isinstance(tracker, SemanticCalibrationWeightTracker):
            weight = tracker.get_weight_with_semantics(cat, price, volume, question)
        else:
            weight = tracker.get_weight(cat, price, volume)
        
        # Compute PnL
        if direction == -1:  # Short YES
            pnl = price if y == 0 else -(1 - price)
        else:  # Long YES
            pnl = (1 - price) if y == 1 else -price
        
        # Extract semantic features for logging
        sem_features = extract_semantic_features(question) if use_semantic else {}
        
        results.append({
            'market_id': row.get('id', ''),
            'question': question[:100],
            'category': cat,
            'price': price,
            'outcome': y,
            'direction': direction,
            'weight': weight,
            'pnl': pnl,
            'weighted_pnl': weight * pnl,
            'volume': volume,
            'close_ts': str(row['close_ts']),
            **{f'sem_{k}': v for k, v in sem_features.items()}
        })
    
    results_df = pd.DataFrame(results)
    
    # Compute metrics
    elapsed = time.time() - start_time
    
    n_trades = len(results_df)
    if n_trades == 0:
        return {'error': 'No trades', 'elapsed_seconds': elapsed}
    
    pnl_std = results_df['pnl'].std()
    wpnl_std = results_df['weighted_pnl'].std()
    
    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'use_semantic_features': use_semantic,
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'n_trades': n_trades,
        'n_categories': len(tracker.calibration_states),
        
        # Unweighted metrics
        'win_rate': float((results_df['pnl'] > 0).mean()),
        'total_pnl': float(results_df['pnl'].sum()),
        'avg_pnl': float(results_df['pnl'].mean()),
        'pnl_std': float(pnl_std),
        'sharpe': float(results_df['pnl'].mean() / pnl_std * np.sqrt(n_trades)) if pnl_std > 0 else 0,
        
        # Weighted metrics
        'weighted_total_pnl': float(results_df['weighted_pnl'].sum()),
        'weighted_avg_pnl': float(results_df['weighted_pnl'].mean()),
        'weighted_pnl_std': float(wpnl_std),
        'weighted_sharpe': float(results_df['weighted_pnl'].mean() / wpnl_std * np.sqrt(n_trades)) if wpnl_std > 0 else 0,
        'avg_weight': float(results_df['weight'].mean()),
        
        # By category
        'by_category': {},
        
        'elapsed_seconds': elapsed
    }
    
    # Per-category metrics
    for cat in results_df['category'].unique():
        cat_df = results_df[results_df['category'] == cat]
        calib = tracker.calibration_states[cat].calibration if cat in tracker.calibration_states else 0
        metrics['by_category'][cat] = {
            'calibration': float(calib),
            'n_trades': len(cat_df),
            'win_rate': float((cat_df['pnl'] > 0).mean()),
            'total_pnl': float(cat_df['pnl'].sum()),
            'avg_weight': float(cat_df['weight'].mean()),
            'weighted_pnl': float(cat_df['weighted_pnl'].sum())
        }
    
    # Correlation between calibration and weight
    calib_weight_pairs = []
    for cat in tracker.calibration_states:
        cat_df = results_df[results_df['category'] == cat]
        if len(cat_df) > 0:
            calib_weight_pairs.append((
                abs(tracker.calibration_states[cat].calibration),
                cat_df['weight'].mean()
            ))
    
    if len(calib_weight_pairs) > 2:
        calibs = [x[0] for x in calib_weight_pairs]
        weights = [x[1] for x in calib_weight_pairs]
        metrics['calibration_weight_correlation'] = float(np.corrcoef(calibs, weights)[0, 1])
    
    # Semantic weights (if used)
    if use_semantic and isinstance(tracker, SemanticCalibrationWeightTracker):
        metrics['semantic_weights'] = {k: float(v) for k, v in tracker.semantic_weights.items()}
    
    return {
        'metrics': metrics,
        'calibration_summary': calibration_summary,
        'results_df': results_df
    }


def main():
    parser = argparse.ArgumentParser(description="Calibration-weighted portfolio backtest")
    parser.add_argument("--data-path", type=str, 
                        default="data/gamma_all_prices_combined.parquet",
                        help="Path to input data")
    parser.add_argument("--output-dir", type=str, 
                        default="runs/calibration_portfolio",
                        help="Output directory")
    parser.add_argument("--use-semantic-features", action="store_true",
                        help="Include semantic features in weights")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="Training fraction")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Validation fraction (for semantic learning)")
    args = parser.parse_args()
    
    print("="*70)
    print("CALIBRATION-WEIGHTED PORTFOLIO BACKTEST")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Use semantic features: {args.use_semantic_features}")
    print(f"Train/Val/Test split: {args.train_frac}/{args.val_frac}/{1-args.train_frac-args.val_frac}")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.data_path)
    df = df[df['market_prob'].notna()]
    print(f"Loaded {len(df)} markets with prices")
    
    # Run backtest
    results = run_backtest(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_semantic=args.use_semantic_features,
        verbose=True
    )
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_semantic" if args.use_semantic_features else "_baseline"
    output_dir = Path(args.output_dir) / f"{timestamp}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\nSaving results to {output_dir}/...")
    
    # Metrics JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results['metrics'], f, indent=2)
    
    # Detailed results parquet
    results['results_df'].to_parquet(output_dir / "trades.parquet")
    
    # Calibration summary
    results['calibration_summary'].to_parquet(output_dir / "calibration_summary.parquet")
    
    # Print summary
    m = results['metrics']
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Test trades: {m['n_trades']}")
    print(f"Win rate: {m['win_rate']:.1%}")
    print()
    print("Unweighted:")
    print(f"  Total PnL: {m['total_pnl']:.2f}")
    print(f"  Sharpe: {m['sharpe']:.2f}")
    print()
    print("Weighted (calibration-based aᵢ(t)):")
    print(f"  Total PnL: {m['weighted_total_pnl']:.2f}")
    print(f"  Sharpe: {m['weighted_sharpe']:.2f}")
    print(f"  Avg weight: {m['avg_weight']:.3f}")
    
    if 'calibration_weight_correlation' in m:
        print(f"\nCorrelation(|calibration|, weight) = {m['calibration_weight_correlation']:.3f}")
    
    if 'semantic_weights' in m:
        print("\nSemantic weight adjustments:")
        for feat, w in sorted(m['semantic_weights'].items(), key=lambda x: -abs(x[1]-1))[:5]:
            print(f"  {feat:25}: {w:.3f}")
    
    print(f"\nElapsed time: {m['elapsed_seconds']:.1f}s")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
