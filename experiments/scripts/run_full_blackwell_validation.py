#!/usr/bin/env python3
"""
Full Blackwell validation: compare market prices vs model predictions.

This runs approachability tests on:
1. Market prices (should be efficient)
2. Model predictions (should show miscalibration)
3. Different epsilon values
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecastbench.models.diffusion_core import ContinuousDiffusionForecaster
from backtest.metrics.approachability import (
    BlackwellApproachabilityTracker,
    TestFunctionFamily,
    run_approachability_test,
)


def build_test_family() -> TestFunctionFamily:
    """Build comprehensive test function family."""
    family = TestFunctionFamily()
    
    # Unconditional
    family.add("unconditional", lambda ctx, q: 1.0)
    
    # Price bins (10 bins)
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        family.add(f"bin_{lo:.1f}-{hi:.1f}", 
                   lambda ctx, q, lo=lo, hi=hi: 1.0 if lo <= q < hi else 0.0)
    
    # Categories
    family.add("crypto", lambda ctx, q: float(ctx.get("is_crypto", False)))
    family.add("sports", lambda ctx, q: float(ctx.get("is_sports", False)))
    family.add("politics", lambda ctx, q: float(ctx.get("is_politics", False)))
    
    # Headlines
    family.add("has_headlines", lambda ctx, q: float(ctx.get("has_headlines", False)))
    
    return family


def build_context(row: pd.Series) -> dict:
    """Build context from data row."""
    category = str(row.get("category", "")).lower()
    question = str(row.get("question", "")).lower()
    
    return {
        "market_id": row.get("id", ""),
        "is_crypto": "crypto" in category or "bitcoin" in question or "ethereum" in question,
        "is_sports": "sports" in category or "nfl" in question or "nba" in question,
        "is_politics": "politic" in category or "election" in question or "trump" in question,
        "has_headlines": row.get("n_headlines", 0) > 0,
    }


def main():
    print("=" * 70)
    print("FULL BLACKWELL VALIDATION")
    print("=" * 70)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/blackwell_validation_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_parquet("data/polymarket/turtel_exa_enriched.parquet")
    df = df[df['n_headlines'] > 0].copy()
    print(f"   Loaded {len(df):,} markets with headlines")
    
    outcomes = df['y'].values.astype(float)
    market_prices = np.clip(df['final_prob'].values, 0.001, 0.999)
    contexts = [build_context(row) for _, row in df.iterrows()]
    
    # Load model and generate predictions
    print("\n2. Loading model and generating predictions...")
    model_path = "runs/exa_diffusion_20251229_100724/model.pt"
    if Path(model_path).exists():
        model = ContinuousDiffusionForecaster.load(model_path, device="cpu")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        texts = (df['question'].fillna('') + " " + df['headlines'].str[:400].fillna('')).tolist()
        embeds = encoder.encode(texts, batch_size=64, show_progress_bar=True)
        
        model_preds = []
        for i in range(len(df)):
            logit = model.sample_x(cond=embeds[i:i+1], n_steps=32, seed=i)
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit[0, 0], -20, 20)))
            model_preds.append(prob)
        model_preds = np.array(model_preds)
        has_model = True
    else:
        print(f"   ⚠️ Model not found at {model_path}")
        model_preds = None
        has_model = False
    
    # Build test family
    family = build_test_family()
    print(f"\n3. Testing with {len(family)} test functions")
    
    # Test different epsilon values
    epsilons = [0.01, 0.02, 0.05, 0.10]
    
    results = {
        "run_id": run_id,
        "n_markets": len(df),
        "outcome_mean": float(outcomes.mean()),
        "market_price_mean": float(market_prices.mean()),
        "model_pred_mean": float(model_preds.mean()) if has_model else None,
        "test_names": family.names,
        "epsilon_tests": {},
    }
    
    for eps in epsilons:
        print(f"\n{'='*70}")
        print(f"EPSILON = {eps}")
        print(f"{'='*70}")
        
        eps_results = {}
        
        # Market prices
        print(f"\n   Market Prices:")
        market_tracker = BlackwellApproachabilityTracker(test_family=family, epsilon=eps)
        for i in range(len(df)):
            market_tracker.update(outcomes[i], market_prices[i], contexts[i])
        market_summary = market_tracker.get_summary()
        print(f"      app_err: {market_summary['final_app_err']:.4f}")
        print(f"      violations: {market_summary['final_n_violations']}")
        eps_results["market"] = {
            "app_err": market_summary["final_app_err"],
            "n_violations": market_summary["final_n_violations"],
            "g_bar": market_summary["final_g_bar"],
        }
        
        # Model predictions
        if has_model:
            print(f"\n   Model Predictions:")
            model_tracker = BlackwellApproachabilityTracker(test_family=family, epsilon=eps)
            for i in range(len(df)):
                model_tracker.update(outcomes[i], model_preds[i], contexts[i])
            model_summary = model_tracker.get_summary()
            print(f"      app_err: {model_summary['final_app_err']:.4f}")
            print(f"      violations: {model_summary['final_n_violations']}")
            eps_results["model"] = {
                "app_err": model_summary["final_app_err"],
                "n_violations": model_summary["final_n_violations"],
                "g_bar": model_summary["final_g_bar"],
            }
        
        results["epsilon_tests"][str(eps)] = eps_results
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Epsilon':<10} {'Market app_err':<18} {'Model app_err':<18}")
    print(f"{'-'*10} {'-'*18} {'-'*18}")
    for eps in epsilons:
        market_err = results["epsilon_tests"][str(eps)]["market"]["app_err"]
        model_err = results["epsilon_tests"][str(eps)].get("model", {}).get("app_err", "N/A")
        if isinstance(model_err, float):
            model_str = f"{model_err:.4f}"
        else:
            model_str = str(model_err)
        print(f"{eps:<10} {market_err:<18.4f} {model_str:<18}")
    
    # Save results
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()


