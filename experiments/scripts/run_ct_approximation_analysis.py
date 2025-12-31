#!/usr/bin/env python
"""
Run C_t Approximation Error Analysis with Trained Diffusion Model.

This script:
1. Loads a trained diffusion model
2. Loads market data with embeddings
3. Generates probability predictions
4. Computes C_t approximation error metrics
5. Attributes mispredictions

Usage:
    python scripts/run_ct_approximation_analysis.py \
        --model runs/proper_diffusion_20251228_231929/model.pt \
        --data data/polymarket/pm_horizon_24h.parquet \
        --embeddings data/embeddings_cache.parquet \
        --output runs/ct_approximation_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.metrics.ct_approximation_error import (
    CtApproximationAnalyzer,
    CtApproximationMetrics,
)


def load_diffusion_model(model_path: Path, device: str = "cpu"):
    """Load a trained diffusion model."""
    from forecastbench.models.diffusion_core import ContinuousDiffusionForecaster
    
    print(f"Loading model from {model_path}...")
    model = ContinuousDiffusionForecaster.load(str(model_path), device=device)
    print(f"   Loaded: cond_dim={model.spec.cond_dim}, device={model.device}")
    return model


def generate_model_predictions(
    model,
    embeddings: np.ndarray,
    n_samples: int = 64,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate probability predictions from the diffusion model.
    
    Args:
        model: Trained diffusion model
        embeddings: (N, cond_dim) conditioning embeddings
        n_samples: Number of samples per market
        batch_size: Batch size for inference
        seed: Random seed
        
    Returns:
        samples: (N, n_samples) probability samples
        mean_preds: (N,) mean predictions
    """
    import torch
    
    N = len(embeddings)
    cond_dim = model.spec.cond_dim
    
    # Ensure embeddings have correct dimension
    if embeddings.shape[1] != cond_dim:
        print(f"   Projecting embeddings from {embeddings.shape[1]} to {cond_dim}")
        # Simple projection: truncate or pad
        if embeddings.shape[1] > cond_dim:
            embeddings = embeddings[:, :cond_dim]
        else:
            embeddings = np.pad(embeddings, ((0, 0), (0, cond_dim - embeddings.shape[1])))
    
    embeddings = embeddings.astype(np.float32)
    
    all_samples = []
    
    print(f"   Generating {n_samples} samples for {N} markets...")
    
    for sample_idx in range(n_samples):
        batch_preds = []
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_cond = embeddings[start:end]
            
            # Sample from diffusion model
            samples = model.sample_x(
                cond=batch_cond,
                seed=seed + sample_idx * 1000 + start,
                n_steps=50,
            )
            
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-samples.flatten()))
            batch_preds.append(probs)
        
        all_samples.append(np.concatenate(batch_preds))
    
    # Stack: (n_samples, N) -> transpose to (N, n_samples)
    samples = np.stack(all_samples, axis=1)
    mean_preds = np.mean(samples, axis=1)
    
    return samples, mean_preds


def generate_simple_embeddings(texts: List[str], dim: int = 384) -> np.ndarray:
    """Generate simple embeddings using sentence-transformers if available."""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("   Using sentence-transformers for embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Project to target dimension
        if embeddings.shape[1] != dim:
            if embeddings.shape[1] > dim:
                embeddings = embeddings[:, :dim]
            else:
                embeddings = np.pad(embeddings, ((0, 0), (0, dim - embeddings.shape[1])))
        
        return embeddings.astype(np.float32)
        
    except ImportError:
        print("   sentence-transformers not available, using random embeddings")
        return np.random.randn(len(texts), dim).astype(np.float32)


def run_analysis(
    model_path: Path,
    data_path: Path,
    embeddings_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    n_samples: int = 64,
    max_markets: int = 2000,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Run the full C_t approximation error analysis."""
    
    if verbose:
        print("=" * 70)
        print("C_t APPROXIMATION ERROR ANALYSIS (Real Model)")
        print("=" * 70)
    
    # Load model
    model = load_diffusion_model(model_path, device=device)
    cond_dim = model.spec.cond_dim
    
    # Load data
    if verbose:
        print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Limit for speed
    if len(df) > max_markets:
        if verbose:
            print(f"   Limiting to {max_markets} markets for speed")
        df = df.sample(n=max_markets, random_state=42)
    
    prices = df["market_prob"].values
    outcomes = df["y"].values
    n = len(df)
    
    if verbose:
        print(f"   Markets: {n}")
        print(f"   Outcome rate: {outcomes.mean():.1%}")
    
    # Load or generate embeddings
    if embeddings_path and embeddings_path.exists():
        if verbose:
            print(f"\nLoading embeddings from {embeddings_path}...")
        emb_df = pd.read_parquet(embeddings_path)
        
        # Match by id
        if "id" in df.columns and "id" in emb_df.columns:
            emb_df = emb_df.set_index("id")
            embeddings = []
            valid_mask = []
            
            for idx, row in df.iterrows():
                market_id = row.get("id")
                if market_id in emb_df.index:
                    emb = emb_df.loc[market_id].values
                    if isinstance(emb, pd.Series):
                        emb = emb.values
                    embeddings.append(emb[:cond_dim] if len(emb) > cond_dim else np.pad(emb, (0, max(0, cond_dim - len(emb)))))
                    valid_mask.append(True)
                else:
                    embeddings.append(np.zeros(cond_dim))
                    valid_mask.append(False)
            
            embeddings = np.array(embeddings, dtype=np.float32)
            valid_mask = np.array(valid_mask)
            
            if verbose:
                print(f"   Matched {valid_mask.sum()}/{n} embeddings")
        else:
            # Just use the embedding columns
            emb_cols = [c for c in emb_df.columns if c.startswith("emb_") or c.isdigit()]
            if emb_cols:
                embeddings = emb_df[emb_cols].values[:n].astype(np.float32)
            else:
                embeddings = emb_df.values[:n].astype(np.float32)
            valid_mask = np.ones(n, dtype=bool)
    else:
        # Generate embeddings from text
        if verbose:
            print("\nGenerating embeddings from text...")
        
        texts = []
        for idx, row in df.iterrows():
            text = f"{row.get('question', '')} {row.get('description', '')}"
            texts.append(text[:500])  # Truncate
        
        embeddings = generate_simple_embeddings(texts, dim=cond_dim)
        valid_mask = np.ones(n, dtype=bool)
    
    # Generate model predictions
    if verbose:
        print(f"\nGenerating model predictions (n_samples={n_samples})...")
    
    samples, mean_preds = generate_model_predictions(
        model,
        embeddings,
        n_samples=n_samples,
        seed=42,
    )
    
    if verbose:
        print(f"   Prediction range: [{mean_preds.min():.3f}, {mean_preds.max():.3f}]")
        print(f"   Mean prediction: {mean_preds.mean():.3f}")
    
    # Run analysis
    if verbose:
        print("\nComputing C_t approximation metrics...")
    
    analyzer = CtApproximationAnalyzer(n_bins=10)
    
    for i in range(n):
        if valid_mask[i]:
            analyzer.add_prediction(
                market_id=str(df.iloc[i].get("id", f"market_{i}")),
                samples=samples[i],
                outcome=float(outcomes[i]),
                market_price=float(prices[i]),
            )
    
    metrics = analyzer.compute_metrics()
    mispred = analyzer.get_misprediction_analysis()
    signals = analyzer.get_trading_signal_quality()
    
    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\n1. CALIBRATION")
        print(f"   Model calibration error: {metrics.model_calibration_error:+.4f}")
        print(f"   Brier score (model): {metrics.brier_score:.4f}")
        print(f"   Log loss (model): {metrics.log_loss:.4f}")
        
        print(f"\n   Calibration by prediction bin:")
        for b, err in sorted(metrics.model_calibration_by_bin.items()):
            lo, hi = b/10, (b+1)/10
            print(f"      [{lo:.1f}, {hi:.1f}): g̅ = {err:+.4f}")
        
        print(f"\n2. MODEL vs MARKET")
        print(f"   Mean |model - market|: {metrics.mean_model_market_diff:.4f}")
        print(f"   Model-market correlation: {metrics.model_market_correlation:.4f}")
        print(f"   Market Brier score: {metrics.market_error:.4f}")
        print(f"   Model Brier score: {metrics.model_error:.4f}")
        print(f"   Model improvement: {metrics.model_improvement:+.1%}")
        
        print(f"\n3. MISPREDICTION ANALYSIS")
        print(f"   Overconfident rate: {mispred['overconfident_rate']:.1%}")
        print(f"   Large error rate (>30%): {mispred['large_error_rate']:.1%}")
        print(f"   Model beats market: {mispred['model_beats_market_rate']:.1%}")
        
        print(f"\n4. TRADING SIGNAL QUALITY")
        print(f"   Buy signals: {signals['n_buy_signals']} (win rate: {signals['buy_signal_win_rate']:.1%})")
        print(f"   Sell signals: {signals['n_sell_signals']} (win rate: {signals['sell_signal_win_rate']:.1%})")
        print(f"   Total signal PnL: {signals['total_signal_pnl']:.2f}")
        
        # By-bin analysis
        print(f"\n5. ERROR ATTRIBUTION BY MARKET PRICE BIN")
        print(f"   {'Bin':>10} | {'Model MAE':>10} | {'Market MAE':>10} | {'Model Wins':>10}")
        print(f"   " + "-"*50)
        for b, stats in sorted(mispred.get("by_market_bin", {}).items()):
            lo, hi = b/10, (b+1)/10
            print(f"   [{lo:.1f},{hi:.1f}) | {stats['model_mae']:>10.4f} | {stats['market_mae']:>10.4f} | {stats['model_beats_market']:>9.1%}")
    
    # Compile results
    results = {
        "config": {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "n_samples": n_samples,
            "n_markets": n,
        },
        "metrics": {
            "calibration_error": float(metrics.model_calibration_error),
            "brier_score": float(metrics.brier_score),
            "log_loss": float(metrics.log_loss),
            "mean_sample_std": float(metrics.mean_sample_std),
            "model_market_diff": float(metrics.mean_model_market_diff),
            "model_market_corr": float(metrics.model_market_correlation),
            "market_error": float(metrics.market_error),
            "model_error": float(metrics.model_error),
            "model_improvement": float(metrics.model_improvement),
        },
        "calibration_by_bin": {str(k): float(v) for k, v in metrics.model_calibration_by_bin.items()},
        "mispredictions": {
            "overconfident_rate": float(mispred["overconfident_rate"]),
            "large_error_rate": float(mispred["large_error_rate"]),
            "model_beats_market": float(mispred["model_beats_market_rate"]),
        },
        "trading_signals": {
            "n_buy": int(signals["n_buy_signals"]),
            "n_sell": int(signals["n_sell_signals"]),
            "buy_win_rate": float(signals["buy_signal_win_rate"]),
            "sell_win_rate": float(signals["sell_signal_win_rate"]),
            "total_pnl": float(signals["total_signal_pnl"]),
        },
        "by_market_bin": {
            str(k): {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                     for kk, vv in v.items()}
            for k, v in mispred.get("by_market_bin", {}).items()
        },
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run C_t approximation error analysis")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/proper_diffusion_20251228_231929/model.pt"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/polymarket/pm_horizon_24h.parquet"),
        help="Path to market data",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to precomputed embeddings (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ct_approximation_real.json"),
        help="Path to save results",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of samples per market",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=2000,
        help="Maximum markets to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda, mps)",
    )
    
    args = parser.parse_args()
    
    run_analysis(
        model_path=args.model,
        data_path=args.data,
        embeddings_path=args.embeddings,
        output_path=args.output,
        n_samples=args.n_samples,
        max_markets=args.max_markets,
        device=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()


