#!/usr/bin/env python3
"""
Train a diffusion model with proper hyperparameters for H4 testing.

This script trains with:
- 20,000 training steps (vs 800 in original)
- 20,000+ samples  
- Stronger conditioning signal

Run this overnight for best results.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train proper diffusion model")
    parser.add_argument("--data", type=Path, default=Path("data/polymarket/gamma_yesno_resolved.parquet"))
    parser.add_argument("--exa-data", type=Path, default=Path("data/polymarket/turtel_exa_enriched.parquet"))
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)  # Higher LR
    parser.add_argument("--hidden-dim", type=int, default=512)  # Larger model
    parser.add_argument("--depth", type=int, default=4)  # Deeper model
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROPER DIFFUSION MODEL TRAINING")
    print("=" * 60)
    
    # Check for Exa data first
    if args.exa_data.exists():
        print(f"\n✓ Using Exa-enriched data: {args.exa_data}")
        data_path = args.exa_data
    else:
        print(f"\n→ Exa data not ready, using: {args.data}")
        data_path = args.data
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df):,}")
    
    # Filter to resolved
    if 'y' in df.columns:
        df = df[df['y'].notna()].copy()
        print(f"  Resolved: {len(df):,}")
    
    # Sample for training
    n_train = min(len(df), args.max_rows)
    df = df.sample(n=n_train, random_state=42).reset_index(drop=True)
    print(f"  Training on: {n_train:,}")
    
    # Prepare text
    if 'headlines' in df.columns and df['headlines'].notna().any():
        print(f"  Using headlines for enriched conditioning")
        df['text'] = df.apply(
            lambda r: f"{r['question']} | Headlines: {str(r.get('headlines', ''))[:1000]}",
            axis=1
        )
    else:
        df['text'] = df['question'].fillna('') + ' ' + df['description'].fillna('')
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    from sentence_transformers import SentenceTransformer
    
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(
        df['text'].tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Shape: {embeddings.shape}")
    
    # Prepare targets
    if 'final_prob' in df.columns:
        targets = df['final_prob'].values
    else:
        targets = df['y'].values
    
    targets = np.clip(targets, 0.001, 0.999)
    logits = np.log(targets / (1 - targets)).reshape(-1, 1)
    
    print(f"\nTarget stats:")
    print(f"  Prob mean: {targets.mean():.3f}, std: {targets.std():.3f}")
    print(f"  Logit mean: {logits.mean():.3f}, std: {logits.std():.3f}")
    
    # Create model with larger capacity
    from forecastbench.models.diffusion_core import (
        ContinuousDiffusionForecaster,
        DiffusionModelSpec,
        DiffusionSchedule,
    )
    
    spec = DiffusionModelSpec(
        out_dim=1,
        cond_dim=embeddings.shape[1],
        time_dim=128,  # Larger time embedding
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        schedule=DiffusionSchedule(T=64, beta_start=1e-4, beta_end=2e-2),
    )
    
    print(f"\nModel spec:")
    print(f"  hidden_dim: {spec.hidden_dim}")
    print(f"  depth: {spec.depth}")
    print(f"  time_dim: {spec.time_dim}")
    
    model = ContinuousDiffusionForecaster(spec, device=args.device)
    
    # Train
    print(f"\nTraining for {args.train_steps} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Device: {args.device}")
    
    result = model.train_mse_eps(
        x0=logits.astype(np.float32),
        cond=embeddings.astype(np.float32),
        steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=1000,
    )
    
    print(f"\nFinal loss: {result['final_loss']:.6f}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(f"runs/proper_diffusion_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    model.save(str(model_path))
    
    config = {
        "data_path": str(data_path),
        "n_samples": n_train,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "final_loss": result['final_loss'],
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    
    # Quick validation
    print("\n" + "=" * 60)
    print("QUICK VALIDATION")
    print("=" * 60)
    
    # Test conditioning
    n_test = min(100, len(df))
    test_idx = np.random.choice(len(df), n_test, replace=False)
    
    preds = []
    true_y = []
    
    for idx in test_idx:
        embed = embeddings[idx:idx+1]
        samples = []
        for seed in range(32):
            logit = model.sample_x(cond=embed, n_steps=32, seed=seed)
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit[0, 0], -20, 20)))
            samples.append(prob)
        preds.append(np.mean(samples))
        true_y.append(targets[idx])
    
    preds = np.array(preds)
    true_y = np.array(true_y)
    
    corr = np.corrcoef(preds, true_y)[0, 1]
    
    print(f"\nValidation on {n_test} samples:")
    print(f"  corr(pred, true): {corr:.4f}")
    print(f"  Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  Pred std: {preds.std():.4f}")
    
    if corr > 0.3:
        print(f"\n  ✓ STRONG conditioning learned!")
    elif corr > 0.1:
        print(f"\n  ~ Weak conditioning learned")
    else:
        print(f"\n  ✗ Conditioning not learned - need more training")
    
    # Save validation results
    val_result = {
        "n_test": n_test,
        "correlation": float(corr),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "true_mean": float(true_y.mean()),
        "conditioning_strength": "strong" if corr > 0.3 else "weak" if corr > 0.1 else "none",
    }
    with open(output_dir / "validation.json", "w") as f:
        json.dump(val_result, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nTo run H4 test:")
    print(f"  PYTHONPATH=$(pwd) python scripts/run_h4_proper_test.py \\")
    print(f"    --model {model_path}")


if __name__ == "__main__":
    main()



