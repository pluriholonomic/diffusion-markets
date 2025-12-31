#!/usr/bin/env python3
"""
Generate and cache MiniLM embeddings for all markets.

Creates a parquet file with market_id -> embedding mapping
for efficient lookup during backtesting.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


def load_markets(
    clob_path: Path,
    forecast_path: Path,
    resolution_path: Path,
) -> pd.DataFrame:
    """Load and merge market metadata from all sources."""
    markets = {}
    
    # From CLOB data
    if clob_path.exists():
        df = pd.read_parquet(clob_path)
        for _, row in df[['market_id', 'slug']].drop_duplicates('market_id').iterrows():
            mid = str(row['market_id'])
            if mid not in markets:
                markets[mid] = {'market_id': mid, 'text': row.get('slug', '')}
    
    # From forecast data (has question text)
    if forecast_path.exists():
        df = pd.read_parquet(forecast_path)
        for _, row in df.iterrows():
            mid = str(row.get('id', row.get('market_id', '')))
            if mid:
                text_parts = [
                    str(row.get('question', '')),
                    str(row.get('description', ''))[:500],
                ]
                markets[mid] = {
                    'market_id': mid,
                    'text': ' '.join(t for t in text_parts if t and t != 'nan'),
                }
    
    # From resolution data
    if resolution_path.exists():
        df = pd.read_parquet(resolution_path)
        for _, row in df.iterrows():
            mid = str(row.get('market_id', row.get('id', '')))
            if mid and mid not in markets:
                text_parts = [
                    str(row.get('question', '')),
                    str(row.get('slug', '')),
                ]
                markets[mid] = {
                    'market_id': mid,
                    'text': ' '.join(t for t in text_parts if t and t != 'nan'),
                }
    
    return pd.DataFrame(list(markets.values()))


def generate_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cpu",
    dtype: str = "float16",
) -> np.ndarray:
    """Generate embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Convert to specified dtype
    if dtype == "float16":
        embeddings = embeddings.astype(np.float16)
    elif dtype == "float32":
        embeddings = embeddings.astype(np.float32)
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings cache")
    parser.add_argument(
        "--clob-data",
        type=Path,
        default=Path("data/backtest/clob_merged.parquet"),
    )
    parser.add_argument(
        "--forecast-data",
        type=Path,
        default=Path("data/polymarket/pm_horizon_24h.parquet"),
    )
    parser.add_argument(
        "--resolution-data",
        type=Path,
        default=Path("data/backtest/resolutions.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/embeddings_cache.parquet"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMBEDDINGS CACHE GENERATION")
    print("=" * 60)
    
    # Load markets
    print("\n1. Loading market metadata...")
    markets_df = load_markets(
        args.clob_data,
        args.forecast_data,
        args.resolution_data,
    )
    print(f"   Found {len(markets_df)} unique markets")
    
    # Filter to markets with text
    markets_df = markets_df[markets_df['text'].str.len() > 0].reset_index(drop=True)
    print(f"   {len(markets_df)} markets with text descriptions")
    
    # Generate embeddings
    print(f"\n2. Generating embeddings with {args.model}...")
    print(f"   Device: {args.device}, dtype: {args.dtype}")
    
    embeddings = generate_embeddings(
        texts=markets_df['text'].tolist(),
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )
    
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Create output dataframe
    print("\n3. Saving embeddings cache...")
    
    # Store embeddings as list of floats (parquet-compatible)
    markets_df['embedding'] = [emb.tolist() for emb in embeddings]
    markets_df['embed_dim'] = embeddings.shape[1]
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    markets_df[['market_id', 'embedding', 'embed_dim']].to_parquet(
        args.output, index=False
    )
    
    file_size = args.output.stat().st_size / 1024 / 1024
    print(f"   Saved to {args.output} ({file_size:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()



