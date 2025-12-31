#!/usr/bin/env python3
"""
Run proper Blackwell approachability test on Polymarket data.

This implements the correct formulation from Section 4.2 of the paper:
- g_t(i) := (Y_t - q_t) * h^i(X_t, q_t)
- C_ε := [-ε, ε]^M
- AppErr_T := d_∞(g̅_T, C_ε)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.metrics.approachability import (
    BlackwellApproachabilityTracker,
    TestFunctionFamily,
    run_approachability_test,
)


def build_context(row: pd.Series) -> dict:
    """Build context dict from a data row."""
    category = str(row.get("category", "")).lower()
    question = str(row.get("question", "")).lower()
    
    return {
        "market_id": row.get("id", ""),
        "question": row.get("question", ""),
        "category": category,
        "is_crypto": "crypto" in category or "bitcoin" in question or "ethereum" in question,
        "is_sports": "sports" in category or "nfl" in question or "nba" in question,
        "is_politics": "politic" in category or "election" in question or "trump" in question,
        "has_headlines": row.get("n_headlines", 0) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Blackwell approachability test")
    parser.add_argument(
        "--data", 
        default="data/polymarket/pm_horizon_24h.parquet",
        help="Path to data parquet (must have market_prob and y columns)"
    )
    parser.add_argument(
        "--epsilon", 
        type=float, 
        default=0.05,
        help="No-arbitrage tolerance"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for results JSON"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("BLACKWELL APPROACHABILITY TEST")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Filter to resolved markets with trading prices
    price_col = "market_prob" if "market_prob" in df.columns else "final_prob"
    df = df[df["y"].notna() & df[price_col].notna()].copy()
    print(f"   Loaded {len(df):,} resolved markets")
    print(f"   Using price column: {price_col}")
    
    # Prepare arrays
    outcomes = df["y"].values.astype(float)
    prices = df[price_col].values.astype(float)
    prices = np.clip(prices, 0.001, 0.999)
    
    # Warn if prices look like resolved prices (all 0 or 1)
    if ((prices < 0.01) | (prices > 0.99)).mean() > 0.9:
        print(f"   ⚠️ WARNING: Prices look like resolved values (mostly 0/1)")
        print(f"      This suggests {price_col} is post-resolution, not trading price")
    
    contexts = [build_context(row) for _, row in df.iterrows()]
    
    print(f"\n2. Data summary:")
    print(f"   Outcome mean: {outcomes.mean():.3f}")
    print(f"   Price mean: {prices.mean():.3f}")
    print(f"   Calibration error: {abs(outcomes.mean() - prices.mean()):.3f}")
    
    # Build test function family
    print(f"\n3. Building test function family...")
    family = TestFunctionFamily()
    
    # Unconditional
    family.add("unconditional", lambda ctx, q: 1.0)
    
    # Price bins (10 bins)
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        family.add(f"bin_{lo:.1f}-{hi:.1f}", 
                   lambda ctx, q, lo=lo, hi=hi: 1.0 if lo <= q < hi else 0.0)
    
    # Category groups
    family.add("crypto", lambda ctx, q: float(ctx.get("is_crypto", False)))
    family.add("sports", lambda ctx, q: float(ctx.get("is_sports", False)))
    family.add("politics", lambda ctx, q: float(ctx.get("is_politics", False)))
    
    # Headlines available
    family.add("has_headlines", lambda ctx, q: float(ctx.get("has_headlines", False)))
    
    print(f"   {len(family)} test functions defined")
    
    # Run approachability test
    print(f"\n4. Running approachability test (ε = {args.epsilon})...")
    results = run_approachability_test(
        outcomes=outcomes,
        prices=prices,
        contexts=contexts,
        epsilon=args.epsilon,
        test_family=family,
    )
    
    # ALSO compute conditional (per-bin) calibration for trading
    print(f"\n4b. Computing CONDITIONAL calibration (for trading)...")
    conditional_g_bars = {}
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        mask = (prices >= lo) & (prices < hi)
        if mask.sum() > 0:
            g_cond = (outcomes[mask] - prices[mask]).mean()
            conditional_g_bars[f"bin_{lo:.1f}-{hi:.1f}"] = {
                "g_bar": float(g_cond),
                "n": int(mask.sum()),
                "violates": abs(g_cond) > args.epsilon,
            }
    
    results["conditional_calibration"] = conditional_g_bars
    cond_max = max(abs(v["g_bar"]) for v in conditional_g_bars.values())
    results["conditional_app_err"] = max(0, cond_max - args.epsilon)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n5. Final Approachability Error: {results['final_app_err']:.4f}")
    
    if results["final_app_err"] > 0:
        print(f"   ⚠️ ARBITRAGE EXISTS (app_err > 0)")
    else:
        print(f"   ✓ NO ARBITRAGE (app_err = 0)")
    
    print(f"\n6. Violations by test function:")
    for name, g_bar, viol_rate in zip(
        results["test_names"], 
        results["final_g_bar"],
        results["violation_rates"]
    ):
        status = "⚠️" if abs(g_bar) > args.epsilon else "✓"
        print(f"   {status} {name:20s}: g̅ = {g_bar:+.4f}, violation rate = {viol_rate:.1%}")
    
    # Decay rate analysis
    if "decay_rate" in results:
        print(f"\n7. Decay Rate Analysis:")
        print(f"   Estimated decay: app_err ~ T^(-{results['decay_rate']:.2f})")
        if results["decay_rate"] > 0.4:
            print(f"   ✓ Approaching 1/√T rate (theoretical optimum = 0.5)")
        else:
            print(f"   ⚠️ Slower than optimal (expected ~0.5)")
    
    # Show CONDITIONAL calibration (economically meaningful for trading)
    print(f"\n8. Conditional Calibration (for trading decisions):")
    print(f"   Conditional app_err: {results.get('conditional_app_err', 0):.4f}")
    if results.get('conditional_app_err', 0) > 0:
        print(f"   ⚠️ ARBITRAGE EXISTS (conditional)")
    
    print(f"\n   Per-bin E[Y-q | q ∈ bin]:")
    for name, data in sorted(results.get("conditional_calibration", {}).items()):
        status = "⚠️" if data["violates"] else "✓"
        print(f"   {status} {name}: g̅ = {data['g_bar']:+.4f} (n={data['n']})")
    
    # Identify largest arbitrage opportunities
    print(f"\n9. Largest Arbitrage Opportunities:")
    cond_cal = results.get("conditional_calibration", {})
    sorted_bins = sorted(cond_cal.items(), key=lambda x: -abs(x[1]["g_bar"]))[:3]
    for name, data in sorted_bins:
        if abs(data["g_bar"]) > args.epsilon:
            direction = "SELL" if data["g_bar"] < 0 else "BUY"
            pct = abs(data["g_bar"]) * 100
            print(f"   {name}: mispriced by {pct:.1f}% → {direction}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"runs/blackwell_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_json = convert_numpy(results)
    results_json["epsilon"] = args.epsilon
    results_json["data_path"] = args.data
    
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

