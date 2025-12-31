#!/usr/bin/env python3
"""
Analyze C_t approximation error for RLCR models.

Compares how well the RLCR-trained LLM's predictions approximate the 
true no-arbitrage set C_t vs. the diffusion model approach.

Key metrics:
1. Calibration error: |E[y - p | bin]| for each prediction bin
2. Blackwell approachability: max violation across test functions
3. Trading signal quality: PnL from model-market divergence
4. Model improvement over market: Brier(market) - Brier(model)

Usage:
    .venv/bin/python scripts/analyze_rlcr_ct_error.py \
        --model runs/ar_rlcr_20k_longrun/20251230_005945_ar_rlcr_20k_longrun/best \
        --data polymarket_backups/pm_suite_derived/gamma_yesno_ready_20k.parquet \
        --max-examples 1000
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CtApproximationMetrics:
    """Metrics for C_t approximation quality."""
    # Calibration
    calibration_error: float  # E[y] - E[p]
    calibration_by_bin: Dict[str, float]  # |E[y-p | bin]| per bin
    max_bin_calibration_error: float
    
    # Blackwell approachability
    blackwell_app_error: float  # max(|g̅| - ε) over test functions
    bin_violations: Dict[str, float]  # g̅ per bin
    
    # Brier / accuracy
    brier_score: float
    market_brier: float
    brier_improvement: float  # market - model (positive = better)
    
    # Trading signal quality
    directional_accuracy: float  # % correct on sign(p-q) == sign(y-q)
    pnl_from_edge: float  # PnL from betting on model-market divergence
    
    # Distribution
    prediction_mean: float
    prediction_std: float
    prediction_range: Tuple[float, float]


def load_rlcr_model(model_path: str, device: str = "cuda"):
    """Load RLCR-trained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Loading RLCR model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    texts: List[str],
    K: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate K predictions per input and return mean + samples."""
    import torch
    from forecastbench.models.ar_cot import _extract_prob
    
    all_means = []
    all_samples = []
    parse_failures = 0
    
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(texts)}...")
        
        # Build prompt
        prompt = f"""You are a forecaster. Given the question and context, predict the probability that the answer is YES.

Question: {text}

Think step by step, then output your probability.
OUTPUT: """
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Sample K times
        samples = []
        for k in range(K):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract probability from response
            tail = response.split("OUTPUT:", 1)[-1] if "OUTPUT:" in response else response
            p = _extract_prob(tail)
            if p is None:
                p = 0.5
                parse_failures += 1
            samples.append(float(np.clip(p, 0.001, 0.999)))
        
        all_samples.append(samples)
        all_means.append(np.mean(samples))
    
    return (
        np.array(all_means),
        np.array(all_samples),
        {"parse_failures": parse_failures, "total": len(texts) * K}
    )


def compute_ct_metrics(
    p: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 0.05,
) -> CtApproximationMetrics:
    """Compute C_t approximation error metrics."""
    n = len(p)
    
    # 1. Overall calibration error
    calibration_error = float(np.mean(y) - np.mean(p))
    
    # 2. Per-bin calibration (Blackwell test functions)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_by_bin = {}
    bin_violations = {}
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (p >= lo) & (p < hi)
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (p >= lo) & (p <= hi)
        
        if mask.sum() > 0:
            bin_id = f"{lo:.1f}-{hi:.1f}"
            g_bar = float(np.mean(y[mask] - p[mask]))  # E[y-p | p in bin]
            calibration_by_bin[bin_id] = abs(g_bar)
            bin_violations[bin_id] = g_bar  # Signed for trading
    
    max_bin_cal_error = max(calibration_by_bin.values()) if calibration_by_bin else 0.0
    
    # 3. Blackwell approachability error: max(|g̅| - ε) over bins
    blackwell_app_error = max(
        0.0,
        max(abs(v) - epsilon for v in bin_violations.values()) if bin_violations else 0.0
    )
    
    # 4. Brier scores
    brier_score = float(np.mean((p - y) ** 2))
    market_brier = float(np.mean((q - y) ** 2))
    brier_improvement = market_brier - brier_score  # Positive = model better
    
    # 5. Directional accuracy (trading signal)
    edge_pred = p - q
    edge_realized = y - q
    directional_correct = (np.sign(edge_pred) == np.sign(edge_realized))
    directional_accuracy = float(np.mean(directional_correct))
    
    # 6. PnL from edge
    # Bet size = sign(p - q), payoff = (y - q) if we bet YES, (q - y) if NO
    bet_direction = np.sign(edge_pred)
    payoff = bet_direction * edge_realized
    transaction_cost = 0.02
    pnl = payoff - transaction_cost * np.abs(bet_direction)
    pnl_from_edge = float(np.mean(pnl))
    
    # 7. Prediction distribution
    prediction_mean = float(np.mean(p))
    prediction_std = float(np.std(p))
    prediction_range = (float(np.min(p)), float(np.max(p)))
    
    return CtApproximationMetrics(
        calibration_error=calibration_error,
        calibration_by_bin=calibration_by_bin,
        max_bin_calibration_error=max_bin_cal_error,
        blackwell_app_error=blackwell_app_error,
        bin_violations=bin_violations,
        brier_score=brier_score,
        market_brier=market_brier,
        brier_improvement=brier_improvement,
        directional_accuracy=directional_accuracy,
        pnl_from_edge=pnl_from_edge,
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        prediction_range=prediction_range,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze RLCR C_t approximation error")
    parser.add_argument("--model", required=True, help="Path to RLCR model")
    parser.add_argument("--data", required=True, help="Path to data parquet")
    parser.add_argument("--max-examples", type=int, default=500, help="Max examples to evaluate")
    parser.add_argument("--K", type=int, default=3, help="Samples per input")
    parser.add_argument("--output", default="runs/rlcr_ct_analysis.json", help="Output path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*70)
    print("RLCR C_t APPROXIMATION ERROR ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Sample if needed
    if args.max_examples and len(df) > args.max_examples:
        df = df.sample(n=args.max_examples, random_state=args.seed).reset_index(drop=True)
    
    print(f"Using {len(df)} examples")
    
    # Get texts, outcomes, market prices
    texts = []
    for _, row in df.iterrows():
        parts = []
        if 'question' in df.columns and pd.notna(row.get('question')):
            parts.append(str(row['question']))
        if 'description' in df.columns and pd.notna(row.get('description')):
            parts.append(str(row['description']))
        texts.append(" ".join(parts) if parts else "unknown")
    
    y = df['y'].values.astype(float)
    
    # Get market price
    if 'market_prob' in df.columns:
        q = df['market_prob'].values.astype(float)
    elif 'final_prob' in df.columns:
        q = df['final_prob'].values.astype(float)
    else:
        print("WARNING: No market price column found, using 0.5")
        q = np.full_like(y, 0.5)
    
    print(f"Outcome rate: {y.mean():.1%}")
    print(f"Market price range: [{q.min():.3f}, {q.max():.3f}]")
    
    # Load model
    model, tokenizer = load_rlcr_model(args.model, args.device)
    
    # Generate predictions
    print(f"\nGenerating {args.K} predictions per example...")
    p_mean, p_samples, gen_meta = generate_predictions(
        model, tokenizer, texts, K=args.K
    )
    print(f"Parse failures: {gen_meta['parse_failures']}/{gen_meta['total']}")
    
    # Compute metrics
    print("\nComputing C_t approximation metrics...")
    metrics = compute_ct_metrics(p_mean, y, q)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n1. CALIBRATION")
    print(f"   Overall calibration error: {metrics.calibration_error:+.4f}")
    print(f"   Max bin calibration error: {metrics.max_bin_calibration_error:.4f}")
    print(f"   Blackwell approachability error (ε=0.05): {metrics.blackwell_app_error:.4f}")
    
    print(f"\n2. BRIER SCORE")
    print(f"   Model Brier:  {metrics.brier_score:.4f}")
    print(f"   Market Brier: {metrics.market_brier:.4f}")
    print(f"   Improvement:  {metrics.brier_improvement:+.4f} ({metrics.brier_improvement/metrics.market_brier*100:+.1f}%)")
    
    print(f"\n3. TRADING SIGNAL")
    print(f"   Directional accuracy: {metrics.directional_accuracy:.1%}")
    print(f"   PnL from edge: {metrics.pnl_from_edge:+.4f} per trade")
    
    print(f"\n4. PREDICTION DISTRIBUTION")
    print(f"   Mean: {metrics.prediction_mean:.3f}")
    print(f"   Std:  {metrics.prediction_std:.3f}")
    print(f"   Range: [{metrics.prediction_range[0]:.3f}, {metrics.prediction_range[1]:.3f}]")
    
    print(f"\n5. PER-BIN CALIBRATION (Blackwell test functions)")
    for bin_id, violation in sorted(metrics.bin_violations.items()):
        n_in_bin = np.sum((p_mean >= float(bin_id.split('-')[0])) & 
                          (p_mean < float(bin_id.split('-')[1])))
        status = "✓" if abs(violation) < 0.05 else "✗"
        print(f"   [{bin_id}]: g̅ = {violation:+.4f} (n={n_in_bin}) {status}")
    
    # Compare with market baseline
    market_metrics = compute_ct_metrics(q, y, q)
    
    print(f"\n6. COMPARISON WITH MARKET")
    print(f"   RLCR Blackwell error:   {metrics.blackwell_app_error:.4f}")
    print(f"   Market Blackwell error: {market_metrics.blackwell_app_error:.4f}")
    
    if metrics.blackwell_app_error < market_metrics.blackwell_app_error:
        print(f"   → RLCR has BETTER C_t approximation!")
    else:
        print(f"   → Market has better C_t approximation")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if metrics.pnl_from_edge > 0:
        print(f"\n✓ RLCR model generates POSITIVE trading signal: {metrics.pnl_from_edge:+.4f}/trade")
    else:
        print(f"\n✗ RLCR model generates negative trading signal: {metrics.pnl_from_edge:+.4f}/trade")
    
    if metrics.brier_improvement > 0:
        print(f"✓ RLCR BEATS market on Brier by {metrics.brier_improvement:.4f}")
    else:
        print(f"✗ Market still better on Brier by {-metrics.brier_improvement:.4f}")
    
    if metrics.blackwell_app_error < 0.05:
        print(f"✓ RLCR satisfies Blackwell ε-approachability (ε=0.05)")
    else:
        print(f"✗ RLCR violates Blackwell constraints (error={metrics.blackwell_app_error:.4f})")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": args.model,
        "data": args.data,
        "n_examples": len(df),
        "K": args.K,
        "metrics": asdict(metrics),
        "market_metrics": asdict(market_metrics),
        "gen_meta": gen_meta,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"\nArtifacts: {output_path}")


if __name__ == "__main__":
    main()
