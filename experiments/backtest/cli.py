"""
CLI entry point for the backtesting framework.

Can be run standalone or integrated into the forecastbench CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def add_backtest_parser(subparsers) -> None:
    """Add backtest subparser to an existing argparse subparsers object."""
    p = subparsers.add_parser(
        "backtest",
        help="Run Blackwell arbitrage backtesting",
        description="Backtest trading strategies based on learned C_t constraint sets",
    )
    _add_common_args(p)
    p.set_defaults(func=cmd_backtest)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add common arguments for backtesting."""
    # Required arguments
    p.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory with C_t model checkpoints",
    )
    p.add_argument(
        "--clob-data",
        type=Path,
        required=True,
        help="Path to CLOB data parquet file",
    )

    # Optional data paths
    p.add_argument(
        "--resolution-data",
        type=Path,
        default=None,
        help="Path to resolution data file",
    )

    # Strategy selection
    p.add_argument(
        "--strategies",
        type=str,
        default="online_max_arb,stat_arb,conditional_graph",
        help="Comma-separated list of strategies to run",
    )

    # Group robustness
    p.add_argument(
        "--enforce-group-boundaries",
        action="store_true",
        default=True,
        help="Only trade within topic groups",
    )
    p.add_argument(
        "--no-group-boundaries",
        dest="enforce_group_boundaries",
        action="store_false",
        help="Allow cross-topic trading",
    )
    p.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated list of topics to include (e.g., crypto,sports)",
    )

    # Model configuration
    p.add_argument(
        "--model-type",
        type=str,
        default="bundle",
        choices=["bundle", "single"],
        help="Type of C_t model (bundle or single)",
    )
    p.add_argument(
        "--embed-dim",
        type=int,
        default=4096,
        help="Embedding dimension for the model",
    )
    p.add_argument(
        "--bundle-size",
        type=int,
        default=8,
        help="Number of markets per bundle",
    )
    p.add_argument(
        "--ct-samples",
        type=int,
        default=64,
        help="Number of MC samples for C_t representation",
    )

    # Execution parameters
    p.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points",
    )
    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.002,
        help="Transaction cost rate",
    )
    p.add_argument(
        "--max-position",
        type=float,
        default=1000.0,
        help="Maximum position size per market (USD)",
    )

    # Event generation
    p.add_argument(
        "--snapshot-interval",
        type=float,
        default=300.0,
        help="Snapshot interval in seconds",
    )
    p.add_argument(
        "--price-threshold",
        type=float,
        default=0.01,
        help="Minimum price change to trigger event",
    )

    # C_t validation
    p.add_argument(
        "--validate-ct",
        action="store_true",
        default=True,
        help="Validate C_t sample sufficiency (default: enabled)",
    )
    p.add_argument(
        "--no-validate-ct",
        dest="validate_ct",
        action="store_false",
        help="Disable C_t validation",
    )
    p.add_argument(
        "--ct-validation-samples",
        type=int,
        default=256,
        help="Number of samples for C_t validation tests",
    )
    p.add_argument(
        "--ct-min-coverage",
        type=float,
        default=0.90,
        help="Minimum coverage fraction for C_t sufficiency",
    )

    # Output
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("runs/backtest"),
        help="Output directory for results",
    )

    # Misc
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run the backtest command."""
    from backtest.config import BacktestConfig
    from backtest.engine import BacktestEngine

    # Parse strategies
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())

    # Parse topics
    allowed_topics = None
    if args.topics:
        allowed_topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    # Build config
    cfg = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        checkpoint_dir=args.checkpoint_dir,
        clob_data_path=args.clob_data,
        resolution_data_path=args.resolution_data,
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        bundle_size=args.bundle_size,
        ct_mc_samples=args.ct_samples,
        strategies=strategies,
        enforce_group_boundaries=args.enforce_group_boundaries,
        allowed_topics=allowed_topics,
        slippage_bps=args.slippage_bps,
        transaction_cost=args.transaction_cost,
        max_position_usd=args.max_position,
        snapshot_interval_seconds=args.snapshot_interval,
        price_change_threshold=args.price_threshold,
        output_dir=args.output,
        seed=args.seed,
        verbose=args.verbose,
        generate_plots=not args.no_plots,
        # C_t validation
        validate_ct=args.validate_ct,
        ct_validation_samples=args.ct_validation_samples,
        ct_min_coverage=args.ct_min_coverage,
    )

    # Run backtest
    engine = BacktestEngine(cfg)
    results = engine.run()

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {results.start_date} to {results.end_date}")
    print(f"Days: {results.n_days}")
    print(f"Trades: {results.n_trades}")
    print("-" * 60)
    print(f"Total PnL: {results.total_pnl:.4f}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.4f}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print("-" * 60)
    print("Per-Strategy PnL:")
    for name, pnl in results.strategy_pnl.items():
        trades = results.strategy_trades.get(name, 0)
        print(f"  {name}: {pnl:.4f} ({trades} trades)")
    print("-" * 60)
    print(f"H4 Correlation: {results.h4_correlation:.4f}")
    print("-" * 60)
    print("C_t Validation:")
    print(f"  Sufficient rate: {results.ct_sufficient_rate:.1%}")
    if results.ct_sufficient_rate < 0.9:
        print("  WARNING: Consider increasing --ct-samples for more reliable results")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Blackwell Arbitrage Backtesting Framework",
    )
    _add_common_args(parser)

    args = parser.parse_args(argv)

    try:
        cmd_backtest(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(args, "verbose") and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

