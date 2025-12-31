#!/usr/bin/env python3
"""
Full RLCR Backtest Pipeline

1. Load the best RLCR model
2. Run inference on full dataset (50k markets)
3. Run decision tree backtest with proper rolling window
4. Compute metrics and confidence intervals

Usage (remote GPU):
    .venv/bin/python scripts/full_rlcr_backtest.py \
        --model runs/ar_rlcr_20k_longrun/20251230_005945_ar_rlcr_20k_longrun/best \
        --data polymarket_backups/pm_suite_derived/gamma_yesno_ready_20k.parquet \
        --output runs/full_rlcr_backtest.json
"""

import argparse
import json
import sys
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch


def categorize_text(text: str) -> str:
    """Derive category from question text."""
    text = str(text).lower()
    
    if any(w in text for w in ['trump', 'biden', 'election', 'president', 'congress', 
                                'senate', 'vote', 'democrat', 'republican', 'governor']):
        return 'politics'
    elif any(w in text for w in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 
                                  'token', 'defi', 'nft', 'solana', 'doge']):
        return 'crypto'
    elif any(w in text for w in ['nba', 'nfl', 'mlb', 'soccer', 'football', 'basketball', 
                                  'game', 'match', 'win', 'championship', 'super bowl',
                                  'world cup', 'playoff', 'finals']):
        return 'sports'
    elif any(w in text for w in ['price', 'stock', 'market cap', 'fed', 'interest rate', 
                                  'gdp', 'inflation', 'treasury', 's&p', 'nasdaq']):
        return 'finance'
    elif any(w in text for w in ['ai', 'gpt', 'openai', 'google', 'apple', 'microsoft', 
                                  'tech', 'meta', 'nvidia', 'chatgpt']):
        return 'tech'
    else:
        return 'other'


def load_rlcr_model(model_path: str):
    """Load the RLCR model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading RLCR model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def extract_prob(text: str) -> Optional[float]:
    """Extract probability from model output."""
    # Look for patterns like "0.75", "75%", "0.75 probability"
    patterns = [
        r'(\d+\.?\d*)%',  # 75%
        r'probability[:\s]+(\d+\.?\d*)',  # probability: 0.75
        r'(\d+\.\d+)',  # 0.75
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if val > 1:  # Percentage
                val /= 100
            return max(0.01, min(0.99, val))
    
    return None


def run_inference(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    K: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Tuple[np.ndarray, Dict]:
    """Run RLCR inference on all texts."""
    
    n = len(texts)
    predictions = np.zeros(n)
    parse_failures = 0
    
    print(f"Running inference on {n} markets with K={K} samples each...")
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        
        if (start // batch_size) % 10 == 0:
            print(f"  Processing {start}-{end} / {n}...")
        
        for i, text in enumerate(batch_texts):
            idx = start + i
            
            # Build prompt
            prompt = f"""You are a forecaster. Given the question and context, predict the probability that the answer is YES.

Question: {text}

Think step by step, then output your probability.
OUTPUT: """
            
            # Sample K times and average
            samples = []
            for k in range(K):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
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
                tail = response.split("OUTPUT:", 1)[-1] if "OUTPUT:" in response else response
                
                p = extract_prob(tail)
                if p is None:
                    p = 0.5
                    parse_failures += 1
                samples.append(p)
            
            predictions[idx] = np.mean(samples)
    
    meta = {
        "n": n,
        "K": K,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / (n * K),
    }
    
    print(f"Inference complete. Parse failures: {parse_failures}/{n*K} ({meta['parse_failure_rate']:.1%})")
    
    return predictions, meta


def get_price_bin(q: float, n_bins: int = 5) -> int:
    """Get price bin index."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        if q < bin_edges[i + 1]:
            return i
    return n_bins - 1


def run_backtest(
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    calibration_window: int = 1000,
    n_price_bins: int = 5,
    well_calibrated_threshold: float = 0.10,
    divergence_threshold: float = 0.15,
    transaction_cost: float = 0.02,
    min_samples_per_cell: int = 20,
) -> Dict:
    """Run rolling-window backtest."""
    
    n = len(p)
    window = calibration_window
    tc = transaction_cost
    
    pnl_baseline = np.zeros(n)
    pnl_decision_tree = np.zeros(n)
    strategy_used = ['warmup'] * window + [''] * (n - window)
    
    print(f"Running backtest with window={window} on {n} markets...")
    
    for i in range(window, n):
        if i % 5000 == 0:
            print(f"  Processing {i}/{n}...")
        
        # Historical data
        q_hist = q[i - window:i]
        y_hist = y[i - window:i]
        cat_hist = categories[i - window:i]
        
        # Current market
        q_i = q[i]
        p_i = p[i]
        y_i = y[i]
        cat_i = categories[i]
        bin_i = get_price_bin(q_i, n_price_bins)
        
        # Baseline
        direction_base = np.sign(p_i - q_i)
        pnl_baseline[i] = direction_base * (y_i - q_i) - tc
        
        # Estimate calibration
        cat_mask = cat_hist == cat_i
        bin_edges = np.linspace(0, 1, n_price_bins + 1)
        lo, hi = bin_edges[bin_i], bin_edges[bin_i + 1]
        price_mask = (q_hist >= lo) & (q_hist < hi)
        if bin_i == n_price_bins - 1:
            price_mask = (q_hist >= lo) & (q_hist <= hi)
        
        mask = cat_mask & price_mask
        n_samples = mask.sum()
        
        if n_samples < min_samples_per_cell:
            strategy_used[i] = 'skip_insufficient_data'
            continue
        
        cal_err = np.mean(y_hist[mask] - q_hist[mask])
        is_well_calibrated = abs(cal_err) < well_calibrated_threshold
        
        if is_well_calibrated:
            divergence = abs(p_i - q_i)
            if divergence > divergence_threshold:
                direction = np.sign(p_i - q_i)
                size = min(divergence / 0.3, 1.0)
                pnl_decision_tree[i] = direction * size * (y_i - q_i) - tc * size
                strategy_used[i] = 'divergence'
            else:
                strategy_used[i] = 'skip_low_divergence'
        else:
            if abs(cal_err) > tc:
                direction = np.sign(cal_err)
                size = min(abs(cal_err) / 0.3, 1.0)
                pnl_decision_tree[i] = direction * size * (y_i - q_i) - tc * size
                strategy_used[i] = 'momentum'
            else:
                strategy_used[i] = 'skip_low_edge'
    
    return {
        'pnl_baseline': pnl_baseline,
        'pnl_decision_tree': pnl_decision_tree,
        'strategy_used': strategy_used,
        'window': window,
    }


def compute_metrics(pnl: np.ndarray, window: int) -> Dict:
    """Compute performance metrics."""
    pnl_oos = pnl[window:]
    traded = pnl_oos != 0
    n_trades = traded.sum()
    
    if n_trades == 0:
        return {'n_trades': 0, 'sharpe': 0, 'ann_sharpe': 0, 'total_pnl': 0}
    
    pnl_traded = pnl_oos[traded]
    mean_pnl = float(pnl_traded.mean())
    std_pnl = float(pnl_traded.std())
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    
    return {
        'n_trades': int(n_trades),
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'sharpe': float(sharpe),
        'ann_sharpe': float(sharpe * np.sqrt(250)),
        'total_pnl': float(pnl_traded.sum()),
        'win_rate': float((pnl_traded > 0).mean()),
    }


def bootstrap_sharpe(pnl: np.ndarray, window: int, n_bootstrap: int = 500, seed: int = 42) -> Dict:
    """Bootstrap confidence intervals."""
    pnl_oos = pnl[window:]
    traded = pnl_oos != 0
    pnl_traded = pnl_oos[traded]
    
    if len(pnl_traded) < 10:
        return {'sharpe_ci_low': 0, 'sharpe_ci_high': 0}
    
    rng = np.random.default_rng(seed)
    sharpes = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(len(pnl_traded), size=len(pnl_traded), replace=True)
        sample = pnl_traded[idx]
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std())
    
    sharpes = np.array(sharpes)
    return {
        'sharpe_mean': float(sharpes.mean()),
        'sharpe_std': float(sharpes.std()),
        'sharpe_ci_low': float(np.percentile(sharpes, 2.5)),
        'sharpe_ci_high': float(np.percentile(sharpes, 97.5)),
    }


def main():
    parser = argparse.ArgumentParser(description="Full RLCR backtest")
    parser.add_argument("--model", required=True, help="Path to RLCR model")
    parser.add_argument("--data", required=True, help="Path to data parquet")
    parser.add_argument("--max-rows", type=int, help="Max rows to process")
    parser.add_argument("--output", default="runs/full_rlcr_backtest.json", help="Output path")
    parser.add_argument("--K", type=int, default=3, help="Samples per market")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--calibration-window", type=int, default=1000, help="Rolling window")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FULL RLCR BACKTEST PIPELINE")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
    
    # Filter to resolved
    df = df[df['y'].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} resolved markets")
    
    # Get text, market prices, outcomes
    texts = []
    for _, row in df.iterrows():
        parts = []
        if 'question' in df.columns and pd.notna(row.get('question')):
            parts.append(str(row['question']))
        if 'description' in df.columns and pd.notna(row.get('description')):
            desc = str(row['description'])[:500]  # Truncate long descriptions
            parts.append(desc)
        texts.append(" ".join(parts) if parts else "unknown")
    
    y = df['y'].values.astype(float)
    
    if 'market_prob' in df.columns:
        q = df['market_prob'].values.astype(float)
    elif 'final_prob' in df.columns:
        q = df['final_prob'].values.astype(float)
    else:
        raise ValueError("No market price column found")
    
    # Derive categories
    print("Deriving categories...")
    categories = np.array([categorize_text(t) for t in texts])
    print(f"Categories: {Counter(categories)}")
    
    # Load model and run inference
    model, tokenizer = load_rlcr_model(args.model)
    predictions, inference_meta = run_inference(
        model, tokenizer, texts, 
        batch_size=args.batch_size, 
        K=args.K
    )
    
    # Save predictions
    df['pred_prob'] = predictions
    pred_path = Path(args.output).parent / "predictions.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pred_path)
    print(f"Saved predictions to {pred_path}")
    
    # Run backtest
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST")
    print("=" * 70)
    
    backtest_results = run_backtest(
        predictions, q, y, categories,
        calibration_window=args.calibration_window,
    )
    
    # Compute metrics
    metrics_baseline = compute_metrics(backtest_results['pnl_baseline'], backtest_results['window'])
    metrics_dectree = compute_metrics(backtest_results['pnl_decision_tree'], backtest_results['window'])
    
    # Bootstrap
    print("\nComputing bootstrap confidence intervals...")
    bootstrap_baseline = bootstrap_sharpe(backtest_results['pnl_baseline'], backtest_results['window'], args.n_bootstrap, args.seed)
    bootstrap_dectree = bootstrap_sharpe(backtest_results['pnl_decision_tree'], backtest_results['window'], args.n_bootstrap, args.seed)
    
    # Strategy breakdown
    strategy_counts = Counter(backtest_results['strategy_used'][backtest_results['window']:])
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (Out-of-Sample)")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Decision Tree':<15}")
    print("-" * 50)
    for key in ['n_trades', 'mean_pnl', 'sharpe', 'ann_sharpe', 'total_pnl', 'win_rate']:
        v1 = metrics_baseline.get(key, 0)
        v2 = metrics_dectree.get(key, 0)
        if isinstance(v1, float):
            print(f"{key:<20} {v1:<15.4f} {v2:<15.4f}")
        else:
            print(f"{key:<20} {v1:<15} {v2:<15}")
    
    if metrics_baseline['sharpe'] != 0:
        improvement = (metrics_dectree['sharpe'] - metrics_baseline['sharpe']) / abs(metrics_baseline['sharpe']) * 100
        print(f"\nSharpe Improvement: {improvement:+.1f}%")
    
    print("\nBootstrap 95% CI:")
    print(f"  Baseline:      [{bootstrap_baseline['sharpe_ci_low']:.4f}, {bootstrap_baseline['sharpe_ci_high']:.4f}]")
    print(f"  Decision Tree: [{bootstrap_dectree['sharpe_ci_low']:.4f}, {bootstrap_dectree['sharpe_ci_high']:.4f}]")
    
    print("\nStrategy breakdown:")
    for s, c in strategy_counts.most_common():
        print(f"  {s}: {c}")
    
    # Save results
    output = {
        'config': {
            'model': args.model,
            'data': args.data,
            'n_markets': len(df),
            'K': args.K,
            'calibration_window': args.calibration_window,
        },
        'inference_meta': inference_meta,
        'metrics': {
            'baseline': metrics_baseline,
            'decision_tree': metrics_dectree,
        },
        'bootstrap': {
            'baseline': bootstrap_baseline,
            'decision_tree': bootstrap_dectree,
        },
        'strategy_counts': dict(strategy_counts),
        'timestamp': datetime.now().isoformat(),
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"\nArtifacts: {output_path}")


if __name__ == "__main__":
    main()
