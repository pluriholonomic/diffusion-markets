#!/usr/bin/env python3
"""
Test C_t generation from different model types.

Compares:
1. RLCR - fine-tuned LLM
2. AR+Diffusion Hybrid - LLM + diffusion refinement
3. Diffusion (legacy) - standalone diffusion model
4. Union - combination of all models

This script validates that each mode produces reasonable C_t samples
and compares their properties (diversity, calibration, etc.).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd


def load_test_data(data_path: str, max_rows: int = 500) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Load test data and extract texts."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if max_rows:
        df = df.head(max_rows).reset_index(drop=True)
    
    # Get market IDs (or create synthetic ones)
    if 'market_id' in df.columns:
        market_ids = df['market_id'].tolist()
    else:
        market_ids = [f"market_{i}" for i in range(len(df))]
    
    # Extract texts
    texts = {}
    text_cols = ['question', 'description']
    for i, (_, row) in enumerate(df.iterrows()):
        parts = []
        for col in text_cols:
            if col in df.columns and pd.notna(row.get(col)):
                parts.append(str(row[col]))
        if parts:
            texts[market_ids[i]] = " ".join(parts)
    
    print(f"  Loaded {len(df)} markets, {len(texts)} with text")
    return df, market_ids, texts


def test_rlcr_mode(
    market_ids: List[str],
    texts: Dict[str, str],
    rlcr_model: str,
    K: int = 5,
) -> Optional[np.ndarray]:
    """Test RLCR C_t sampling."""
    print("\n" + "=" * 60)
    print("Testing RLCR Mode")
    print("=" * 60)
    
    from backtest.model_loader import (
        CtMode, CtModel, ModelLoaderConfig, UnifiedModelLoader
    )
    
    cfg = ModelLoaderConfig(
        ct_mode=CtMode.SINGLE,
        ct_model=CtModel.RLCR,
        rlcr_model_path=rlcr_model,
        rlcr_K=K,
    )
    
    try:
        loader = UnifiedModelLoader(cfg)
        samples, valid_ids = loader.sample_ct(
            market_ids=market_ids[:10],  # Small sample for testing
            texts={k: texts[k] for k in market_ids[:10] if k in texts},
            seed=42,
        )
        
        print(f"  Samples shape: {samples.shape}")
        print(f"  Valid markets: {len(valid_ids)}")
        print(f"  Sample mean: {samples.mean():.4f}")
        print(f"  Sample std: {samples.std():.4f}")
        print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
        
        return samples
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_ar_diffusion_mode(
    market_ids: List[str],
    texts: Dict[str, str],
    ar_diffusion_path: str,
    n_samples: int = 16,
) -> Optional[np.ndarray]:
    """Test AR+Diffusion hybrid C_t sampling."""
    print("\n" + "=" * 60)
    print("Testing AR+Diffusion Mode")
    print("=" * 60)
    
    from backtest.model_loader import (
        CtMode, CtModel, ModelLoaderConfig, UnifiedModelLoader
    )
    
    cfg = ModelLoaderConfig(
        ct_mode=CtMode.SINGLE,
        ct_model=CtModel.AR_DIFFUSION,
        ar_diffusion_path=Path(ar_diffusion_path),
        ar_diffusion_samples=n_samples,
    )
    
    try:
        loader = UnifiedModelLoader(cfg)
        samples, valid_ids = loader.sample_ct(
            market_ids=market_ids[:10],
            texts={k: texts[k] for k in market_ids[:10] if k in texts},
            seed=42,
        )
        
        print(f"  Samples shape: {samples.shape}")
        print(f"  Valid markets: {len(valid_ids)}")
        print(f"  Sample mean: {samples.mean():.4f}")
        print(f"  Sample std: {samples.std():.4f}")
        print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
        
        return samples
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_bundle_mode(
    market_ids: List[str],
    texts: Dict[str, str],
    bundle_path: str,
    n_samples: int = 16,
) -> Optional[np.ndarray]:
    """Test Bundle diffusion C_t sampling."""
    print("\n" + "=" * 60)
    print("Testing Bundle Diffusion Mode")
    print("=" * 60)
    
    from backtest.model_loader import (
        CtMode, CtModel, ModelLoaderConfig, UnifiedModelLoader
    )
    
    cfg = ModelLoaderConfig(
        ct_mode=CtMode.SINGLE,
        ct_model=CtModel.BUNDLE,
        bundle_diffusion_path=Path(bundle_path),
        bundle_samples=n_samples,
    )
    
    try:
        loader = UnifiedModelLoader(cfg)
        samples, valid_ids = loader.sample_ct(
            market_ids=market_ids[:10],
            texts={k: texts[k] for k in market_ids[:10] if k in texts},
            seed=42,
        )
        
        print(f"  Samples shape: {samples.shape}")
        print(f"  Valid markets: {len(valid_ids)}")
        print(f"  Sample mean: {samples.mean():.4f}")
        print(f"  Sample std: {samples.std():.4f}")
        print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
        
        return samples
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_union_mode(
    market_ids: List[str],
    texts: Dict[str, str],
    rlcr_model: Optional[str] = None,
    ar_diffusion_path: Optional[str] = None,
    bundle_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Test Union mode C_t sampling."""
    print("\n" + "=" * 60)
    print("Testing UNION Mode")
    print("=" * 60)
    
    from backtest.model_loader import (
        CtMode, CtModel, ModelLoaderConfig, UnifiedModelLoader
    )
    
    enabled = []
    if rlcr_model:
        enabled.append("rlcr")
    if ar_diffusion_path:
        enabled.append("ar_diffusion")
    if bundle_path:
        enabled.append("bundle")
    
    if not enabled:
        print("  No models configured for union mode")
        return None
    
    cfg = ModelLoaderConfig(
        ct_mode=CtMode.UNION,
        rlcr_model_path=rlcr_model,
        ar_diffusion_path=Path(ar_diffusion_path) if ar_diffusion_path else None,
        bundle_diffusion_path=Path(bundle_path) if bundle_path else None,
        enabled_models=tuple(enabled),
        rlcr_K=5,
        ar_diffusion_samples=8,
        bundle_samples=8,
    )
    
    print(f"  Enabled models: {enabled}")
    
    try:
        loader = UnifiedModelLoader(cfg)
        samples, valid_ids = loader.sample_ct(
            market_ids=market_ids[:10],
            texts={k: texts[k] for k in market_ids[:10] if k in texts},
            seed=42,
        )
        
        print(f"  Samples shape: {samples.shape}")
        print(f"  Valid markets: {len(valid_ids)}")
        print(f"  Sample mean: {samples.mean():.4f}")
        print(f"  Sample std: {samples.std():.4f}")
        print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
        
        # Analyze diversity
        if samples.shape[0] > 1:
            per_market_std = samples.std(axis=0)
            print(f"  Per-market sample std: {per_market_std.mean():.4f} (avg)")
        
        return samples
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_legacy_mode(
    df: pd.DataFrame,
    checkpoint_dir: str,
) -> Optional[np.ndarray]:
    """Test legacy C_t loader (for comparison)."""
    print("\n" + "=" * 60)
    print("Testing Legacy Diffusion Mode")
    print("=" * 60)
    
    from backtest.ct_loader import CtCheckpointLoader, CtCheckpointSpec
    
    try:
        loader = CtCheckpointLoader(CtCheckpointSpec(
            checkpoint_dir=Path(checkpoint_dir),
            model_type="single",
            embed_dim=384,
            device="cpu",
        ))
        
        # Try to load
        loader.load_for_date("2024-01-01")
        print(f"  Loaded checkpoint from {checkpoint_dir}")
        
        # Sample
        n = min(10, len(df))
        embeddings = {f"m_{i}": np.random.randn(384).astype(np.float32) for i in range(n)}
        samples, valid_ids = loader.sample_ct_for_markets(
            market_ids=list(embeddings.keys()),
            embeddings=embeddings,
            n_samples=16,
            seed=42,
        )
        
        print(f"  Samples shape: {samples.shape}")
        print(f"  Valid markets: {len(valid_ids)}")
        
        return samples
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def compare_samples(results: Dict[str, np.ndarray]) -> None:
    """Compare C_t samples from different modes."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Mode':<20} {'Shape':<15} {'Mean':<10} {'Std':<10} {'Diversity':<10}")
    print("-" * 65)
    
    for name, samples in results.items():
        if samples is None:
            print(f"{name:<20} FAILED")
            continue
        
        # Diversity = average std across samples for each market
        diversity = samples.std(axis=0).mean() if samples.shape[0] > 1 else 0.0
        
        print(f"{name:<20} {str(samples.shape):<15} {samples.mean():<10.4f} {samples.std():<10.4f} {diversity:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test C_t generation modes")
    parser.add_argument("--data", default="data/polymarket/turtel_exa_enriched.parquet")
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--rlcr-model", default="runs/ar_rlcr_20k_highgamma/20251229_123316_ar_rlcr_20k_highgamma/best")
    parser.add_argument("--ar-diffusion", default=None, help="AR+Diffusion checkpoint path")
    parser.add_argument("--bundle", default="runs/proper_diffusion_20251228_231929", help="Bundle diffusion path")
    parser.add_argument("--legacy-checkpoint", default="runs/proper_diffusion_20251228_231929")
    parser.add_argument("--skip-rlcr", action="store_true", help="Skip RLCR test (slow)")
    parser.add_argument("--skip-ar-diffusion", action="store_true", help="Skip AR+Diffusion test")
    parser.add_argument("--skip-bundle", action="store_true", help="Skip bundle test")
    parser.add_argument("--skip-union", action="store_true", help="Skip union test")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip legacy test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("C_t MODE COMPARISON TEST")
    print("=" * 60)
    
    # Load data
    df, market_ids, texts = load_test_data(args.data, args.max_rows)
    
    results = {}
    
    # Test each mode
    if not args.skip_rlcr and args.rlcr_model:
        results["RLCR"] = test_rlcr_mode(market_ids, texts, args.rlcr_model)
    
    if not args.skip_ar_diffusion and args.ar_diffusion:
        results["AR+Diffusion"] = test_ar_diffusion_mode(market_ids, texts, args.ar_diffusion)
    
    if not args.skip_bundle and args.bundle:
        results["Bundle"] = test_bundle_mode(market_ids, texts, args.bundle)
    
    if not args.skip_legacy and args.legacy_checkpoint:
        results["Legacy"] = test_legacy_mode(df, args.legacy_checkpoint)
    
    if not args.skip_union:
        results["Union"] = test_union_mode(
            market_ids, texts,
            rlcr_model=args.rlcr_model if not args.skip_rlcr else None,
            bundle_path=args.bundle if not args.skip_bundle else None,
        )
    
    # Compare results
    compare_samples(results)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
