#!/usr/bin/env python3
"""
Prepare data for backtesting.

Consolidates per-token CLOB files into a single parquet and 
ensures all required data is in the right format.
"""

import argparse
from pathlib import Path

import pandas as pd


def merge_clob_files(
    clob_dir: Path,
    output_path: Path,
    max_files: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge per-token CLOB parquet files into a single file.

    Args:
        clob_dir: Directory with per-token parquet files
        output_path: Output parquet path
        max_files: Maximum number of files to process
        verbose: Print progress

    Returns:
        Merged DataFrame
    """
    parquet_files = list(clob_dir.glob("*.parquet"))[:max_files]

    if verbose:
        print(f"Found {len(parquet_files)} CLOB files in {clob_dir}")

    dfs = []
    for i, f in enumerate(parquet_files):
        try:
            df = pd.read_parquet(f)
            # Standardize columns
            if "t" in df.columns and "timestamp" not in df.columns:
                df = df.rename(columns={"t": "timestamp"})
            if "p" in df.columns and "mid_price" not in df.columns:
                df = df.rename(columns={"p": "mid_price"})
            dfs.append(df)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not read {f.name}: {e}")

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(parquet_files)} files")

    if not dfs:
        raise ValueError("No CLOB files could be loaded")

    merged = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"Merged DataFrame shape: {merged.shape}")
        print(f"Date range: {pd.to_datetime(merged['timestamp'].min(), unit='s')} to "
              f"{pd.to_datetime(merged['timestamp'].max(), unit='s')}")
        print(f"Unique markets: {merged['market_id'].nunique()}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    if verbose:
        print(f"Saved to {output_path}")

    return merged


def prepare_resolution_data(
    resolution_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepare resolution data for backtesting.

    Args:
        resolution_path: Path to resolution parquet
        output_path: Output path

    Returns:
        Prepared DataFrame
    """
    df = pd.read_parquet(resolution_path)

    # Standardize columns
    col_map = {}
    if "id" in df.columns and "market_id" not in df.columns:
        col_map["id"] = "market_id"
    if "y" in df.columns and "outcome" not in df.columns:
        col_map["y"] = "outcome"
    if "closedTime" in df.columns and "timestamp" not in df.columns:
        col_map["closedTime"] = "timestamp"

    if col_map:
        df = df.rename(columns=col_map)

    # Convert timestamp
    if df["timestamp"].dtype == "object":
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True).astype(int) // 10**9

    # Select relevant columns
    cols = ["market_id", "timestamp", "outcome"]
    if "question" in df.columns:
        cols.append("question")
    if "category" in df.columns:
        cols.append("category")
    if "slug" in df.columns:
        cols.append("slug")

    df = df[cols].copy()

    if verbose:
        print(f"Resolution data shape: {df.shape}")
        print(f"Resolution outcomes: {df['outcome'].value_counts().to_dict()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    if verbose:
        print(f"Saved to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare backtest data")
    parser.add_argument(
        "--clob-dir",
        type=Path,
        default=Path("data/polymarket/clob_history_yes_f1"),
        help="Directory with per-token CLOB parquet files",
    )
    parser.add_argument(
        "--resolution-path",
        type=Path,
        default=Path("data/polymarket/gamma_yesno_resolved.parquet"),
        help="Path to resolution data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest"),
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--max-clob-files",
        type=int,
        default=500,
        help="Maximum number of CLOB files to merge",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("Preparing Backtest Data")
        print("=" * 60)

    # 1. Merge CLOB files
    if args.clob_dir.exists():
        if verbose:
            print("\n1. Merging CLOB files...")
        merge_clob_files(
            clob_dir=args.clob_dir,
            output_path=args.output_dir / "clob_merged.parquet",
            max_files=args.max_clob_files,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"\nWarning: CLOB directory not found: {args.clob_dir}")

    # 2. Prepare resolution data
    if args.resolution_path.exists():
        if verbose:
            print("\n2. Preparing resolution data...")
        prepare_resolution_data(
            resolution_path=args.resolution_path,
            output_path=args.output_dir / "resolutions.parquet",
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"\nWarning: Resolution file not found: {args.resolution_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print("=" * 60)
        print("\nTo run backtest:")
        print(f"  forecastbench backtest \\")
        print(f"    --checkpoint-dir runs/20251225_235102_pm_suite_difftrain_fixed_ready \\")
        print(f"    --clob-data {args.output_dir / 'clob_merged.parquet'} \\")
        print(f"    --resolution-data {args.output_dir / 'resolutions.parquet'} \\")
        print(f"    --start-date 2024-06-01 --end-date 2024-06-30")


if __name__ == "__main__":
    main()

