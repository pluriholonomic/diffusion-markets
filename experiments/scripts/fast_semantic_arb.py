#!/usr/bin/env python3
"""
Fast Semantic Stat Arb - uses keyword matching instead of slow embeddings.

This is a simplified version that:
1. Groups markets by entity keywords (Bitcoin, Trump, etc.)
2. Detects logical constraints (price thresholds, exclusivity)
3. Trades on constraint violations

No sentence-transformers required - just regex and heuristics.
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SemanticGroup:
    """A group of semantically related markets."""
    group_id: str
    market_ids: List[str]
    market_questions: List[str]
    group_type: str = "entity"  # "entity", "threshold", "exclusive"
    constraints: List[Dict] = field(default_factory=list)


# =============================================================================
# Fast Keyword-Based Clustering
# =============================================================================

# Entity patterns for grouping
ENTITY_PATTERNS = {
    "bitcoin": r"\b(bitcoin|btc)\b",
    "ethereum": r"\b(ethereum|eth)\b",
    "solana": r"\b(solana|sol)\b",
    "xrp": r"\b(xrp|ripple)\b",
    "trump": r"\b(trump|donald\s+trump)\b",
    "biden": r"\b(biden|joe\s+biden)\b",
    "harris": r"\b(harris|kamala)\b",
    "fed_rates": r"\b(fed|federal\s+reserve).*(rate|interest)\b",
    "nba": r"\b(nba|basketball|lakers|celtics|warriors)\b",
    "nfl": r"\b(nfl|football|chiefs|eagles|49ers)\b",
}


def extract_threshold(question: str) -> Optional[Tuple[str, float, str]]:
    """Extract price threshold from question."""
    q = question.lower()
    
    # Pattern: "above/greater/over $X" or "> X"
    match = re.search(r"(above|greater|over|>)\s*\$?([\d,]+\.?\d*)", q)
    if match:
        try:
            value = float(match.group(2).replace(",", ""))
            return ("above", value, match.group(2))
        except ValueError:
            pass
    
    # Pattern: "below/less/under $X"
    match = re.search(r"(below|less|under|<)\s*\$?([\d,]+\.?\d*)", q)
    if match:
        try:
            value = float(match.group(2).replace(",", ""))
            return ("below", value, match.group(2))
        except ValueError:
            pass
    
    return None


def cluster_by_keywords(
    market_ids: List[str],
    questions: Dict[str, str],
) -> List[SemanticGroup]:
    """Cluster markets by keyword matching."""
    groups = []
    
    # Group by entity
    entity_groups = defaultdict(list)
    
    for mid in market_ids:
        if mid not in questions:
            continue
        q = questions[mid].lower()
        
        for entity, pattern in ENTITY_PATTERNS.items():
            if re.search(pattern, q, re.IGNORECASE):
                entity_groups[entity].append((mid, questions[mid]))
    
    # Create groups with at least 2 markets
    group_id = 0
    for entity, members in entity_groups.items():
        if len(members) >= 2:
            groups.append(SemanticGroup(
                group_id=f"{entity}_{group_id}",
                market_ids=[m[0] for m in members],
                market_questions=[m[1] for m in members],
                group_type="entity",
            ))
            group_id += 1
    
    return groups


def detect_constraints(group: SemanticGroup) -> List[Dict]:
    """Detect logical constraints within a group."""
    constraints = []
    
    # Try to detect threshold-based monotonicity
    thresholds = []
    for i, q in enumerate(group.market_questions):
        thresh = extract_threshold(q)
        if thresh:
            thresholds.append((i, thresh[0], thresh[1]))
    
    # If we have multiple thresholds of same direction, create monotonicity constraints
    above_thresh = [(i, v) for i, d, v in thresholds if d == "above"]
    below_thresh = [(i, v) for i, d, v in thresholds if d == "below"]
    
    # Sort by threshold value and create pairwise constraints
    if len(above_thresh) >= 2:
        above_thresh.sort(key=lambda x: x[1])
        for j in range(len(above_thresh) - 1):
            i1, v1 = above_thresh[j]
            i2, v2 = above_thresh[j + 1]
            # P(X > v1) >= P(X > v2) when v1 < v2
            constraints.append({
                "type": "monotonic",
                "relation": "geq",
                "market_idx_1": i1,
                "market_idx_2": i2,
                "threshold_1": v1,
                "threshold_2": v2,
            })
        group.group_type = "threshold"
    
    if len(below_thresh) >= 2:
        below_thresh.sort(key=lambda x: x[1])
        for j in range(len(below_thresh) - 1):
            i1, v1 = below_thresh[j]
            i2, v2 = below_thresh[j + 1]
            # P(X < v1) <= P(X < v2) when v1 < v2
            constraints.append({
                "type": "monotonic",
                "relation": "leq",
                "market_idx_1": i1,
                "market_idx_2": i2,
                "threshold_1": v1,
                "threshold_2": v2,
            })
        group.group_type = "threshold"
    
    group.constraints = constraints
    return constraints


# =============================================================================
# Trading Strategy
# =============================================================================

def compute_positions(
    group: SemanticGroup,
    prices: Dict[str, float],
    min_mispricing: float = 0.03,
    max_position: float = 100.0,
) -> Dict[str, float]:
    """Compute positions based on constraint violations."""
    positions = {mid: 0.0 for mid in group.market_ids}
    
    # Trade on explicit constraints
    for constraint in group.constraints:
        if constraint["type"] == "monotonic":
            idx1, idx2 = constraint["market_idx_1"], constraint["market_idx_2"]
            mid1, mid2 = group.market_ids[idx1], group.market_ids[idx2]
            
            p1 = prices.get(mid1, 0.5)
            p2 = prices.get(mid2, 0.5)
            
            if constraint["relation"] == "geq":
                # p1 should be >= p2
                violation = p2 - p1
            else:
                # p1 should be <= p2
                violation = p1 - p2
            
            if violation > min_mispricing:
                size = min(violation * 100, max_position)
                if constraint["relation"] == "geq":
                    positions[mid1] = max(positions[mid1], size)
                    positions[mid2] = min(positions[mid2], -size)
                else:
                    positions[mid1] = min(positions[mid1], -size)
                    positions[mid2] = max(positions[mid2], size)
    
    # Fallback: within-group mean reversion
    if not group.constraints and len(prices) >= 2:
        valid_prices = [p for mid, p in prices.items() if mid in group.market_ids]
        if len(valid_prices) >= 2:
            group_mean = np.mean(valid_prices)
            group_std = np.std(valid_prices)
            
            if group_std > 0.02:
                for mid in group.market_ids:
                    if mid in prices:
                        deviation = prices[mid] - group_mean
                        z_score = deviation / group_std
                        if abs(z_score) > 1.0:
                            size = min(abs(z_score) * 10, max_position / 2)
                            positions[mid] = -np.sign(deviation) * size
    
    # Clip positions
    for mid in positions:
        positions[mid] = np.clip(positions[mid], -max_position, max_position)
    
    return positions


# =============================================================================
# Backtest
# =============================================================================

def load_clob_data(
    clob_dir: Path,
    market_ids: List[str],
    token_id_map: Dict[str, str],
    max_files: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Load CLOB price data."""
    data = {}
    loaded = 0
    
    for mid in market_ids:
        if loaded >= max_files:
            break
        
        token_id = token_id_map.get(mid, mid)
        fpath = clob_dir / f"{token_id}.parquet"
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath)
                if "t" in df.columns and "p" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                    df = df.set_index("timestamp")
                    # Resample to daily for speed
                    df = df.resample("D").last().dropna()
                    if len(df) > 5:  # At least 5 days of data
                        data[mid] = df
                        loaded += 1
            except Exception:
                pass
    
    return data


def run_backtest(
    groups: List[SemanticGroup],
    price_data: Dict[str, pd.DataFrame],
    min_mispricing: float = 0.03,
    transaction_cost: float = 0.01,
) -> Dict:
    """Run backtest."""
    
    # Get date range
    all_dates = set()
    for mid, df in price_data.items():
        all_dates.update(df.index.date)
    
    if not all_dates:
        return {"total_pnl": 0, "sharpe": 0, "trades": 0, "violations": 0}
    
    sorted_dates = sorted(all_dates)
    
    positions = defaultdict(float)
    daily_pnls = []
    n_trades = 0
    n_violations = 0
    prev_prices = {}
    
    for date in sorted_dates:
        # Get prices
        current_prices = {}
        for mid, df in price_data.items():
            day_data = df[df.index.date == date]
            if len(day_data) > 0:
                current_prices[mid] = day_data["p"].iloc[-1]
        
        if not current_prices:
            continue
        
        # Compute PnL from existing positions
        day_pnl = 0.0
        for mid, pos in positions.items():
            if mid in current_prices and mid in prev_prices:
                price_change = current_prices[mid] - prev_prices[mid]
                day_pnl += pos * price_change
        
        # Trade each group
        for group in groups:
            group_prices = {
                mid: current_prices.get(mid, 0.5)
                for mid in group.market_ids
                if mid in current_prices
            }
            
            if len(group_prices) < 2:
                continue
            
            # Count violations
            for constraint in group.constraints:
                idx1 = constraint["market_idx_1"]
                idx2 = constraint["market_idx_2"]
                mid1, mid2 = group.market_ids[idx1], group.market_ids[idx2]
                
                if mid1 in group_prices and mid2 in group_prices:
                    p1, p2 = group_prices[mid1], group_prices[mid2]
                    
                    if constraint["relation"] == "geq" and p2 > p1 + min_mispricing:
                        n_violations += 1
                    elif constraint["relation"] == "leq" and p1 > p2 + min_mispricing:
                        n_violations += 1
            
            # Compute new positions
            new_pos = compute_positions(group, group_prices, min_mispricing)
            
            # Update positions
            for mid, target in new_pos.items():
                old_pos = positions[mid]
                trade_size = abs(target - old_pos)
                if trade_size > 0.01:
                    day_pnl -= trade_size * transaction_cost
                    positions[mid] = target
                    n_trades += 1
        
        daily_pnls.append(day_pnl)
        prev_prices = current_prices.copy()
    
    # Compute metrics
    total_pnl = sum(daily_pnls)
    sharpe = 0.0
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)
    
    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "trades": n_trades,
        "violations": n_violations,
        "n_days": len(daily_pnls),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-markets", type=int, default=200)
    parser.add_argument("--min-mispricing", type=float, default=0.03)
    args = parser.parse_args()
    
    # Paths - try multiple locations
    clob_dir = Path("data/polymarket/clob_history_yes_f1")
    
    # Try multiple data sources in order
    exa_paths = [
        Path("/root/diffusion-markets/data/polymarket/turtel_exa_enriched.parquet"),
        Path("data/polymarket/exa_strict_matched.parquet"),
    ]
    cache_paths = [
        Path("data/polymarket/optimization_cache.parquet"),
        Path("data/polymarket/gamma_yesno_resolved.parquet"),
    ]
    
    exa_path = next((p for p in exa_paths if p.exists()), None)
    cache_path = next((p for p in cache_paths if p.exists()), None)
    
    # Load data
    if exa_path and exa_path.exists():
        print(f"Loading Exa-enriched data from {exa_path}...")
        meta = pd.read_parquet(exa_path)
    elif cache_path and cache_path.exists():
        print(f"Loading cache data from {cache_path}...")
        meta = pd.read_parquet(cache_path)
    else:
        print("No data found!")
        return
    
    meta = meta[meta["question"].notna()]
    if "volumeNum" in meta.columns:
        meta = meta.sort_values("volumeNum", ascending=False)
    meta = meta.head(args.max_markets)
    
    market_ids = meta["id"].astype(str).tolist()
    questions = dict(zip(market_ids, meta["question"].tolist()))
    
    # Build token ID map
    token_id_map = {}
    if "yes_token_id" in meta.columns:
        for _, row in meta.iterrows():
            mid = str(row.get("id", ""))
            tid = str(row.get("yes_token_id", ""))
            if mid and tid and tid != "nan":
                token_id_map[mid] = tid
    
    print(f"Loaded {len(market_ids)} markets")
    
    # Step 1: Cluster by keywords
    print("\n" + "=" * 60)
    print("STEP 1: KEYWORD CLUSTERING")
    print("=" * 60)
    
    groups = cluster_by_keywords(market_ids, questions)
    print(f"Found {len(groups)} keyword groups")
    
    for g in groups[:5]:
        print(f"  {g.group_id}: {len(g.market_ids)} markets")
    
    # Step 2: Detect constraints
    print("\n" + "=" * 60)
    print("STEP 2: CONSTRAINT DETECTION")
    print("=" * 60)
    
    total_constraints = 0
    for group in groups:
        constraints = detect_constraints(group)
        total_constraints += len(constraints)
        if constraints:
            print(f"  {group.group_id} ({group.group_type}): {len(constraints)} constraints")
    
    print(f"\nTotal constraints: {total_constraints}")
    
    # Step 3: Load prices
    print("\n" + "=" * 60)
    print("STEP 3: LOADING PRICE DATA")
    print("=" * 60)
    
    all_market_ids = set()
    for g in groups:
        all_market_ids.update(g.market_ids)
    
    price_data = load_clob_data(clob_dir, list(all_market_ids), token_id_map)
    print(f"Loaded prices for {len(price_data)}/{len(all_market_ids)} markets")
    
    # Filter groups to those with price data
    groups = [g for g in groups if sum(1 for m in g.market_ids if m in price_data) >= 2]
    print(f"Groups with price data: {len(groups)}")
    
    # Step 4: Backtest
    print("\n" + "=" * 60)
    print("STEP 4: BACKTEST")
    print("=" * 60)
    
    result = run_backtest(groups, price_data, args.min_mispricing)
    
    print(f"\nResults:")
    print(f"  Total PnL:     ${result['total_pnl']:,.2f}")
    print(f"  Sharpe Ratio:  {result['sharpe']:.3f}")
    print(f"  Total Trades:  {result['trades']}")
    print(f"  Violations:    {result['violations']}")
    print(f"  Days:          {result['n_days']}")
    
    # Baseline: random pairs
    print("\n" + "=" * 60)
    print("BASELINE: Random Pairs")
    print("=" * 60)
    
    # Shuffle and create random pairs
    np.random.seed(42)
    price_list = list(price_data.items())
    baseline_pnl = 0.0
    baseline_trades = 0
    
    if len(price_list) >= 2:
        for _ in range(50):
            i, j = np.random.choice(len(price_list), 2, replace=False)
            mid1, df1 = price_list[i]
            mid2, df2 = price_list[j]
            
            common_dates = set(df1.index.date) & set(df2.index.date)
            for date in list(common_dates)[-20:]:
                d1 = df1[df1.index.date == date]
                d2 = df2[df2.index.date == date]
                if len(d1) > 0 and len(d2) > 0:
                    spread = d1["p"].iloc[-1] - d2["p"].iloc[-1]
                    if abs(spread) > 0.1:
                        baseline_pnl -= abs(spread) * 0.01
                        baseline_trades += 1
    
    print(f"  Random pairs PnL: ${baseline_pnl:,.2f}")
    print(f"  Random trades:    {baseline_trades}")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    improvement = result["total_pnl"] - baseline_pnl
    print(f"Semantic clustering improvement: ${improvement:,.2f}")
    
    if result["sharpe"] > 0:
        print(f"✓ Positive Sharpe: {result['sharpe']:.3f}")
    else:
        print(f"✗ Negative Sharpe: {result['sharpe']:.3f}")
    
    if result["violations"] > 0:
        print(f"✓ Found {result['violations']} constraint violations (arb opportunities)")


if __name__ == "__main__":
    main()
