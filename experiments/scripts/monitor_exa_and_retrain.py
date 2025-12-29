#!/usr/bin/env python3
"""
Monitor for Exa headline enrichment completion and trigger retraining.

This script:
1. Polls for turtel_exa_enriched.parquet to appear
2. When found, retrains diffusion model with proper hyperparameters
3. Runs H4 validation test

Usage:
    python scripts/monitor_exa_and_retrain.py --poll-interval 60
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def check_exa_data() -> tuple[bool, int]:
    """Check if Exa enriched data exists and return row count."""
    exa_path = Path("data/polymarket/turtel_exa_enriched.parquet")
    
    if not exa_path.exists():
        # Also check gamma_yesno_resolved as fallback
        gamma_path = Path("data/polymarket/gamma_yesno_resolved.parquet")
        if gamma_path.exists():
            df = pd.read_parquet(gamma_path)
            return False, len(df)
        return False, 0
    
    df = pd.read_parquet(exa_path)
    return True, len(df)


def train_diffusion_model(
    data_path: Path,
    output_dir: Path,
    train_steps: int = 10000,
    batch_size: int = 256,
    device: str = "cpu",
) -> Path:
    """Train diffusion model with proper hyperparameters."""
    print("\n" + "=" * 60)
    print("TRAINING DIFFUSION MODEL")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df):,}")
    
    # Filter to resolved markets
    if 'y' in df.columns:
        df = df[df['y'].notna()].copy()
        print(f"  Resolved: {len(df):,}")
    
    # Use larger subset for training
    max_rows = min(len(df), 20000)
    df = df.sample(n=max_rows, random_state=42)
    print(f"  Training on: {max_rows:,}")
    
    # Prepare text for embeddings
    text_col = 'question'
    if 'headlines' in df.columns:
        # Use headlines if available
        df['text'] = df.apply(
            lambda r: f"{r['question']} {r.get('headlines', '')[:500]}".strip(),
            axis=1
        )
        text_col = 'text'
    
    # Import model components
    from sentence_transformers import SentenceTransformer
    from forecastbench.models.diffusion_core import (
        ContinuousDiffusionForecaster,
        DiffusionModelSpec,
        DiffusionSchedule,
    )
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(
        df[text_col].tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Prepare targets (logit-transformed probabilities)
    # Use final_prob as target if available, else y
    if 'final_prob' in df.columns:
        targets = df['final_prob'].values
    else:
        targets = df['y'].values
    
    # Clip and transform to logits
    targets = np.clip(targets, 0.001, 0.999)
    logits = np.log(targets / (1 - targets)).reshape(-1, 1)
    
    print(f"\nTarget stats:")
    print(f"  Prob range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Create model
    spec = DiffusionModelSpec(
        out_dim=1,
        cond_dim=embeddings.shape[1],  # 384 for MiniLM
        time_dim=64,
        hidden_dim=256,
        depth=3,
        schedule=DiffusionSchedule(T=64, beta_start=1e-4, beta_end=2e-2),
    )
    
    model = ContinuousDiffusionForecaster(spec, device=device)
    
    # Train
    print(f"\nTraining for {train_steps} steps...")
    result = model.train_mse_eps(
        x0=logits.astype(np.float32),
        cond=embeddings.astype(np.float32),
        steps=train_steps,
        batch_size=batch_size,
        lr=2e-4,
        log_every=500,
    )
    
    print(f"\nFinal loss: {result['final_loss']:.6f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    model.save(str(model_path))
    
    # Save config
    config = {
        "data_path": str(data_path),
        "n_samples": len(df),
        "train_steps": train_steps,
        "batch_size": batch_size,
        "final_loss": result['final_loss'],
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    
    return model_path


def run_h4_test(model_path: Path, output_dir: Path) -> dict:
    """Run H4 validation test with the newly trained model."""
    print("\n" + "=" * 60)
    print("RUNNING H4 VALIDATION TEST")
    print("=" * 60)
    
    # Use the proper H4 test script
    cmd = [
        sys.executable,
        "scripts/run_h4_proper_test.py",
        "--ct-samples", "64",
        "--ddim-steps", "32",
        "--slippage-bps", "100",
        "--output", str(output_dir / "h4_results"),
    ]
    
    # Update the script to use our new model
    # For now, run inline
    import torch
    from backtest.ct_loader_legacy import LegacyDiffusionForecaster
    
    # Try to load the new model
    try:
        # Load with ContinuousDiffusionForecaster.load
        from forecastbench.models.diffusion_core import ContinuousDiffusionForecaster
        model = ContinuousDiffusionForecaster.load(str(model_path), device="cpu")
        print(f"  ✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return {"error": str(e)}
    
    # Load test data
    clob = pd.read_parquet("data/backtest/clob_merged.parquet")
    res = pd.read_parquet("data/backtest/resolutions.parquet")
    embeds_df = pd.read_parquet("data/embeddings_cache.parquet")
    
    embeddings = {
        row['market_id']: np.array(row['embedding'], dtype=np.float32)
        for _, row in embeds_df.iterrows()
    }
    
    # Get overlapping markets
    overlap = set(clob['market_id'].unique()) & set(res['market_id'].unique()) & set(embeddings.keys())
    print(f"  Testing {len(overlap)} markets")
    
    last_prices = clob.groupby('market_id').last()['mid_price'].to_dict()
    outcomes = res.set_index('market_id')['outcome'].to_dict()
    
    # Test correlation between model predictions and outcomes
    preds = []
    outs = []
    
    for market_id in list(overlap)[:100]:
        if market_id not in last_prices or market_id not in outcomes:
            continue
        outcome = outcomes[market_id]
        if pd.isna(outcome):
            continue
        
        embed = embeddings[market_id].reshape(1, -1)
        
        # Sample from model
        logits = model.sample_x(cond=embed, n_steps=32, seed=42)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits.flatten()[0], -20, 20)))
        
        preds.append(prob)
        outs.append(outcome)
    
    preds = np.array(preds)
    outs = np.array(outs)
    
    corr = np.corrcoef(preds, outs)[0, 1]
    
    print(f"\n  Results:")
    print(f"    corr(model_pred, outcome) = {corr:.4f}")
    print(f"    Model mean: {preds.mean():.3f}")
    print(f"    Outcome mean: {outs.mean():.3f}")
    
    result = {
        "n_tested": int(len(preds)),
        "model_outcome_correlation": float(corr),
        "model_mean": float(preds.mean()),
        "outcome_mean": float(outs.mean()),
        "conditioning_works": bool(corr > 0.1),
    }
    
    # Save results
    (output_dir / "h4_results").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "h4_results" / "quick_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Monitor Exa and retrain")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--max-wait", type=int, default=3600 * 24, help="Max seconds to wait")
    parser.add_argument("--train-steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--use-gamma-fallback", action="store_true", help="Use gamma_yesno_resolved if no Exa")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXA MONITOR & RETRAIN PIPELINE")
    print("=" * 60)
    print(f"\nPolling every {args.poll_interval}s for Exa data...")
    print(f"Max wait: {args.max_wait}s")
    
    start_time = time.time()
    
    while True:
        exa_ready, n_rows = check_exa_data()
        elapsed = time.time() - start_time
        
        if exa_ready:
            print(f"\n✓ Exa data found! {n_rows:,} rows")
            data_path = Path("data/polymarket/turtel_exa_enriched.parquet")
            break
        
        if args.use_gamma_fallback and n_rows > 0:
            print(f"\n→ Using gamma fallback with {n_rows:,} rows")
            data_path = Path("data/polymarket/gamma_yesno_resolved.parquet")
            break
        
        if elapsed > args.max_wait:
            print(f"\n✗ Max wait exceeded. Using gamma fallback if available.")
            if n_rows > 0:
                data_path = Path("data/polymarket/gamma_yesno_resolved.parquet")
                break
            else:
                print("No data available. Exiting.")
                return
        
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Waiting... ({elapsed:.0f}s elapsed)")
        time.sleep(args.poll_interval)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/retrained_h4_{timestamp}")
    
    # Train model
    model_path = train_diffusion_model(
        data_path=data_path,
        output_dir=output_dir,
        train_steps=args.train_steps,
        device=args.device,
    )
    
    # Run H4 test
    h4_result = run_h4_test(model_path, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"H4 conditioning works: {h4_result.get('conditioning_works', False)}")
    print(f"Model-outcome correlation: {h4_result.get('model_outcome_correlation', 'N/A')}")


if __name__ == "__main__":
    main()

