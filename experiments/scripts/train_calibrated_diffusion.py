#!/usr/bin/env python3
"""
Train diffusion model with proper conditioning and calibration.

This script implements the three recommendations from C_t approximation error analysis:
1. RETRAIN with proper conditioning (same embedding model for train/inference)
2. CALIBRATION POST-PROCESSING (Platt scaling)
3. HYBRID MARKET+MODEL (use model to refine market, not replace it)

Usage (remote GPU):
    .venv/bin/python scripts/train_calibrated_diffusion.py \
        --data data/polymarket/pm_horizon_24h.parquet \
        --steps 5000 --batch-size 128
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Ensure forecastbench is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class CalibrationConfig:
    """Configuration for calibrated diffusion training."""
    # Data
    data_path: str
    max_rows: Optional[int] = None
    
    # Embeddings
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    
    # Diffusion architecture
    hidden_dim: int = 256
    depth: int = 4
    time_dim: int = 64
    T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Training
    steps: int = 5000
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    seed: int = 42
    
    # Calibration
    use_platt_scaling: bool = True
    use_market_prior: bool = True
    market_weight: float = 0.5  # Blend: output = (1-w)*model + w*market
    
    # Output
    run_name: str = "calibrated_diffusion"
    output_dir: str = "runs"


def load_and_prepare_data(config: CalibrationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load data and create embeddings with consistent embedding model."""
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading data from {config.data_path}...")
    df = pd.read_parquet(config.data_path)
    
    if config.max_rows:
        df = df.head(config.max_rows).copy()
    
    # Filter to resolved markets only
    if 'y' not in df.columns:
        raise ValueError("Dataset must have 'y' column with outcomes")
    
    df = df[df['y'].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} resolved markets")
    
    # Check outcome distribution
    outcome_rate = df['y'].mean()
    print(f"Outcome rate: {outcome_rate:.1%}")
    
    # Create text for embedding
    text_cols = ['question', 'description']
    texts = []
    for _, row in df.iterrows():
        parts = []
        for c in text_cols:
            if c in df.columns and pd.notna(row.get(c)):
                parts.append(str(row[c]))
        texts.append(" ".join(parts) if parts else "unknown")
    
    # Use MiniLM consistently
    print(f"Creating embeddings with {config.embed_model}...")
    model = SentenceTransformer(config.embed_model)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype(np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Get outcomes and market prices
    y = df['y'].values.astype(np.float32)
    
    # Get market prices if available
    if 'market_prob' in df.columns:
        q = df['market_prob'].values.astype(np.float32)
    elif 'final_prob' in df.columns:
        q = df['final_prob'].values.astype(np.float32)
    else:
        q = np.full_like(y, 0.5)  # Fallback to 0.5
    
    print(f"Market price range: [{q.min():.3f}, {q.max():.3f}]")
    
    return df, embeddings, y, q


def train_diffusion_model(
    cond: np.ndarray,
    y: np.ndarray,
    config: CalibrationConfig,
    device: str = "cuda",
) -> Tuple[Any, Dict]:
    """Train diffusion model with proper conditioning."""
    from forecastbench.models.diffusion_core import (
        ContinuousDiffusionForecaster,
        DiffusionModelSpec,
        DiffusionSchedule,
    )
    from forecastbench.utils.logits import prob_to_logit
    
    print("\n" + "="*60)
    print("TRAINING DIFFUSION MODEL")
    print("="*60)
    
    # Split data
    N = len(y)
    rng = np.random.default_rng(config.seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    
    n_train = int(0.8 * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Convert outcomes to logits with label smoothing
    eps = 0.01
    p_train = y[train_idx] * (1 - 2*eps) + eps
    x0_train = prob_to_logit(p_train.reshape(-1, 1), eps=1e-9).astype(np.float32)
    cond_train = cond[train_idx].astype(np.float32)
    
    print(f"x0 range: [{x0_train.min():.3f}, {x0_train.max():.3f}]")
    print(f"cond shape: {cond_train.shape}")
    
    # Build model
    schedule = DiffusionSchedule(
        T=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    
    spec = DiffusionModelSpec(
        out_dim=1,
        cond_dim=config.embed_dim,
        time_dim=config.time_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        schedule=schedule,
    )
    
    model = ContinuousDiffusionForecaster(spec, device=device)
    
    # Train
    print(f"\nTraining for {config.steps} steps...")
    train_result = model.train_mse_eps(
        x0=x0_train,
        cond=cond_train,
        steps=config.steps,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        log_every=max(100, config.steps // 10),
    )
    
    print(f"\nTraining complete. Final loss: {train_result['final_loss']:.6f}")
    
    return model, {
        "train_result": train_result,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }


def fit_platt_scaling(
    model: Any,
    cond: np.ndarray,
    y: np.ndarray,
    n_samples: int = 32,
    seed: int = 42,
) -> Tuple[float, float]:
    """Fit Platt scaling parameters on validation data."""
    from forecastbench.utils.logits import logit_to_prob
    from scipy.optimize import minimize
    
    print("\n" + "="*60)
    print("FITTING PLATT SCALING")
    print("="*60)
    
    # Sample predictions
    print(f"Generating {n_samples} samples per market...")
    all_probs = []
    for s in range(n_samples):
        logits = model.sample_x(
            cond=cond,
            n_steps=50,
            seed=seed + s,
            eta=0.0,
        )
        probs = logit_to_prob(logits).flatten()
        all_probs.append(probs)
    
    # Mean prediction
    p_pred = np.mean(all_probs, axis=0)
    print(f"Raw prediction range: [{p_pred.min():.3f}, {p_pred.max():.3f}]")
    print(f"Raw prediction mean: {p_pred.mean():.3f}")
    print(f"Actual outcome mean: {y.mean():.3f}")
    
    # Platt scaling: p_cal = sigmoid(A * logit(p) + B)
    def neg_log_likelihood(params):
        A, B = params
        logits = np.log(p_pred / (1 - p_pred + 1e-8) + 1e-8)
        scaled_logits = A * logits + B
        p_cal = 1 / (1 + np.exp(-scaled_logits.clip(-20, 20)))
        p_cal = np.clip(p_cal, 1e-6, 1-1e-6)
        nll = -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))
        return nll
    
    # Optimize
    result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method='Nelder-Mead')
    A, B = result.x
    
    print(f"Platt parameters: A={A:.4f}, B={B:.4f}")
    
    # Verify calibration improvement
    logits = np.log(p_pred / (1 - p_pred + 1e-8) + 1e-8)
    scaled_logits = A * logits + B
    p_cal = 1 / (1 + np.exp(-scaled_logits.clip(-20, 20)))
    
    raw_brier = np.mean((p_pred - y) ** 2)
    cal_brier = np.mean((p_cal - y) ** 2)
    
    print(f"Raw Brier: {raw_brier:.4f}")
    print(f"Calibrated Brier: {cal_brier:.4f}")
    print(f"Improvement: {(raw_brier - cal_brier) / raw_brier * 100:.1f}%")
    
    return A, B


def create_hybrid_model(
    model: Any,
    platt_A: float,
    platt_B: float,
    market_weight: float,
) -> Dict:
    """Create configuration for hybrid model-market predictions."""
    print("\n" + "="*60)
    print("HYBRID MODEL CONFIGURATION")
    print("="*60)
    
    print(f"Market weight: {market_weight}")
    print(f"Model weight: {1 - market_weight}")
    print(f"Platt scaling: p_cal = sigmoid({platt_A:.4f} * logit(p) + {platt_B:.4f})")
    
    return {
        "platt_A": float(platt_A),
        "platt_B": float(platt_B),
        "market_weight": float(market_weight),
        "description": "Hybrid: output = (1-w)*calibrated_model + w*market",
    }


def evaluate_model(
    model: Any,
    cond: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    platt_A: float,
    platt_B: float,
    market_weight: float,
    n_samples: int = 32,
    seed: int = 42,
) -> Dict:
    """Evaluate the calibrated hybrid model."""
    from forecastbench.utils.logits import logit_to_prob
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Sample predictions
    all_probs = []
    for s in range(n_samples):
        logits = model.sample_x(
            cond=cond,
            n_steps=50,
            seed=seed + 1000 + s,
            eta=0.0,
        )
        probs = logit_to_prob(logits).flatten()
        all_probs.append(probs)
    
    p_raw = np.mean(all_probs, axis=0)
    
    # Apply Platt scaling
    logits = np.log(p_raw / (1 - p_raw + 1e-8) + 1e-8)
    scaled_logits = platt_A * logits + platt_B
    p_cal = 1 / (1 + np.exp(-scaled_logits.clip(-20, 20)))
    
    # Hybrid: blend with market
    p_hybrid = (1 - market_weight) * p_cal + market_weight * q
    
    # Compute metrics
    metrics = {}
    
    # Brier scores
    metrics["raw_brier"] = float(np.mean((p_raw - y) ** 2))
    metrics["calibrated_brier"] = float(np.mean((p_cal - y) ** 2))
    metrics["hybrid_brier"] = float(np.mean((p_hybrid - y) ** 2))
    metrics["market_brier"] = float(np.mean((q - y) ** 2))
    
    # Calibration error (mean prediction - mean outcome)
    metrics["raw_calibration"] = float(np.mean(p_raw) - np.mean(y))
    metrics["calibrated_calibration"] = float(np.mean(p_cal) - np.mean(y))
    metrics["hybrid_calibration"] = float(np.mean(p_hybrid) - np.mean(y))
    metrics["market_calibration"] = float(np.mean(q) - np.mean(y))
    
    # Prediction range
    metrics["raw_range"] = [float(p_raw.min()), float(p_raw.max())]
    metrics["calibrated_range"] = [float(p_cal.min()), float(p_cal.max())]
    metrics["hybrid_range"] = [float(p_hybrid.min()), float(p_hybrid.max())]
    
    print("\nBrier Scores (lower is better):")
    print(f"  Raw model:        {metrics['raw_brier']:.4f}")
    print(f"  Calibrated:       {metrics['calibrated_brier']:.4f}")
    print(f"  Hybrid:           {metrics['hybrid_brier']:.4f}")
    print(f"  Market baseline:  {metrics['market_brier']:.4f}")
    
    print("\nCalibration Error (closer to 0 is better):")
    print(f"  Raw model:        {metrics['raw_calibration']:+.4f}")
    print(f"  Calibrated:       {metrics['calibrated_calibration']:+.4f}")
    print(f"  Hybrid:           {metrics['hybrid_calibration']:+.4f}")
    print(f"  Market baseline:  {metrics['market_calibration']:+.4f}")
    
    # Does hybrid beat market?
    if metrics["hybrid_brier"] < metrics["market_brier"]:
        improvement = (metrics["market_brier"] - metrics["hybrid_brier"]) / metrics["market_brier"] * 100
        print(f"\n✓ Hybrid BEATS market by {improvement:.1f}%")
        metrics["beats_market"] = True
    else:
        degradation = (metrics["hybrid_brier"] - metrics["market_brier"]) / metrics["market_brier"] * 100
        print(f"\n✗ Hybrid WORSE than market by {degradation:.1f}%")
        metrics["beats_market"] = False
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train calibrated diffusion model")
    parser.add_argument("--data", required=True, help="Path to parquet data")
    parser.add_argument("--max-rows", type=int, help="Max rows to use")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="Network depth")
    parser.add_argument("--market-weight", type=float, default=0.5, help="Weight for market in hybrid")
    parser.add_argument("--run-name", default="calibrated_diffusion", help="Run name")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Create config
    config = CalibrationConfig(
        data_path=args.data,
        max_rows=args.max_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        market_weight=args.market_weight,
        run_name=args.run_name,
        seed=args.seed,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{timestamp}_{config.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CALIBRATED DIFFUSION TRAINING")
    print("="*60)
    print(f"Output: {run_dir}")
    print(f"Config: {asdict(config)}")
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load data
    df, embeddings, y, q = load_and_prepare_data(config)
    
    # Split for train/val/test
    N = len(y)
    rng = np.random.default_rng(config.seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    
    print(f"\nData split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Train model
    model, train_meta = train_diffusion_model(
        cond=embeddings[train_idx],
        y=y[train_idx],
        config=config,
        device=args.device,
    )
    
    # Fit Platt scaling on validation set
    platt_A, platt_B = fit_platt_scaling(
        model=model,
        cond=embeddings[val_idx],
        y=y[val_idx],
        n_samples=32,
        seed=config.seed,
    )
    
    # Create hybrid config
    hybrid_config = create_hybrid_model(
        model=model,
        platt_A=platt_A,
        platt_B=platt_B,
        market_weight=config.market_weight,
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(
        model=model,
        cond=embeddings[test_idx],
        y=y[test_idx],
        q=q[test_idx],
        platt_A=platt_A,
        platt_B=platt_B,
        market_weight=config.market_weight,
        n_samples=32,
        seed=config.seed,
    )
    
    # Save model
    import torch
    model_path = run_dir / "diffusion_model.pt"
    torch.save({
        "spec": model.spec,
        "state_dict": model.model.state_dict(),
        "platt_A": platt_A,
        "platt_B": platt_B,
        "market_weight": config.market_weight,
        "embed_model": config.embed_model,
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results
    results = {
        "config": asdict(config),
        "train_meta": train_meta,
        "hybrid_config": hybrid_config,
        "test_metrics": test_metrics,
        "data_splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Artifacts: {run_dir}")
    
    # Summary
    print("\nSUMMARY:")
    if test_metrics["beats_market"]:
        improvement = (test_metrics["market_brier"] - test_metrics["hybrid_brier"]) / test_metrics["market_brier"] * 100
        print(f"  ✓ Hybrid model BEATS market by {improvement:.1f}%")
    else:
        print(f"  ✗ Market still better than hybrid model")
    
    print(f"  Platt scaling: A={platt_A:.4f}, B={platt_B:.4f}")
    print(f"  Hybrid blend: {1-config.market_weight:.0%} model + {config.market_weight:.0%} market")


if __name__ == "__main__":
    main()
