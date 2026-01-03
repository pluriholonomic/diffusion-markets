#!/usr/bin/env python3
"""
Run parallel backtests on remote box.

This script is designed to maximize CPU utilization on the remote box
by running multiple backtest workers in parallel.

Usage (local):
    python scripts/run_parallel_backtests.py --remote

Usage (on remote):
    python scripts/run_parallel_backtests.py --local --workers 80

Target: Run optimization until Sharpe > 3.0 and E[PnL/day] > $1000
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Remote box configuration
REMOTE_HOST = "root@95.133.252.72"
REMOTE_PORT = 22
REMOTE_DIR = "/root/diffusion-markets/qlt/experiments"

# Target metrics
TARGET_SHARPE = 3.0
TARGET_DAILY_PNL = 1000.0


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    strategy: str
    params: Dict
    data_file: str = "data/polymarket/optimization_cache.parquet"
    initial_bankroll: float = 10000.0


@dataclass
class BacktestResult:
    """Results from a single backtest."""
    strategy: str
    params: Dict
    sharpe: float
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    duration_seconds: float
    error: Optional[str] = None


# Strategy parameter spaces for optimization
# These parameters are used by UnifiedBacktester._compute_edge and run()
STRATEGY_PARAM_SPACES = {
    "calibration": {
        "min_edge": (0.005, 0.10),
        "kelly_fraction": (0.05, 0.5),
        "max_position_pct": (0.03, 0.15),
    },
    "mean_reversion": {
        "min_edge": (0.01, 0.15),
        "kelly_fraction": (0.05, 0.4),
        "max_position_pct": (0.03, 0.12),
    },
    "momentum": {
        "min_edge": (0.005, 0.12),
        "kelly_fraction": (0.05, 0.35),
        "max_position_pct": (0.02, 0.10),
    },
    "dispersion": {
        "min_edge": (0.01, 0.15),
        "kelly_fraction": (0.03, 0.3),
        "max_position_pct": (0.02, 0.10),
    },
    "convergence": {
        "min_edge": (0.005, 0.10),
        "kelly_fraction": (0.05, 0.4),
        "max_position_pct": (0.03, 0.12),
    },
    "risk_parity": {
        "min_edge": (0.005, 0.10),
        "kelly_fraction": (0.05, 0.35),
        "max_position_pct": (0.02, 0.10),
    },
}


def sample_params(strategy: str, seed: int = None) -> Dict:
    """Sample random parameters for a strategy."""
    if seed is not None:
        np.random.seed(seed)
    
    space = STRATEGY_PARAM_SPACES.get(strategy, {})
    params = {}
    
    for name, (low, high) in space.items():
        if isinstance(low, int) and isinstance(high, int):
            params[name] = int(np.random.randint(low, high + 1))
        else:
            params[name] = float(np.random.uniform(low, high))
    
    return params


def run_single_backtest(config: BacktestConfig) -> BacktestResult:
    """Run a single backtest with given configuration."""
    import time
    start_time = time.time()
    
    try:
        # Import backtest components
        from trading.optimization.unified_optimizer import UnifiedBacktester
        
        # Create backtester
        backtester = UnifiedBacktester(
            data_file=config.data_file,
            initial_bankroll=config.initial_bankroll,
        )
        
        # Run backtest
        metrics = backtester.run(
            strategy_name=config.strategy,
            params=config.params,
        )
        
        duration = time.time() - start_time
        
        return BacktestResult(
            strategy=config.strategy,
            params=config.params,
            sharpe=metrics.get('sharpe', 0.0),
            total_pnl=metrics.get('total_pnl', 0.0),
            daily_pnl=metrics.get('daily_pnl', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            n_trades=metrics.get('n_trades', 0),
            duration_seconds=duration,
        )
        
    except Exception as e:
        import traceback
        duration = time.time() - start_time
        return BacktestResult(
            strategy=config.strategy,
            params=config.params,
            sharpe=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            max_drawdown=1.0,
            win_rate=0.0,
            n_trades=0,
            duration_seconds=duration,
            error=f"{e}\n{traceback.format_exc()}",
        )


def run_parallel_optimization(
    strategies: List[str],
    n_workers: int,
    n_iterations: int = 100,
    output_dir: str = "logs/parallel_optimization",
) -> Dict[str, List[BacktestResult]]:
    """
    Run parallel CMA-ES style optimization.
    
    Args:
        strategies: List of strategy names to optimize
        n_workers: Number of parallel workers
        n_iterations: Number of parameter samples per strategy
        output_dir: Directory to save results
        
    Returns:
        Dictionary of strategy -> list of results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting parallel optimization with {n_workers} workers")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Iterations per strategy: {n_iterations}")
    
    # Generate all configs
    configs = []
    for strategy in strategies:
        for i in range(n_iterations):
            params = sample_params(strategy, seed=i * 1000 + hash(strategy) % 1000)
            configs.append(BacktestConfig(
                strategy=strategy,
                params=params,
            ))
    
    logger.info(f"Total backtest runs: {len(configs)}")
    
    # Run in parallel
    all_results: Dict[str, List[BacktestResult]] = {s: [] for s in strategies}
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_backtest, c): c for c in configs}
        
        for future in as_completed(futures):
            config = futures[future]
            try:
                result = future.result()
                all_results[result.strategy].append(result)
                completed += 1
                
                # Log progress every 10 completions
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(configs)} backtests")
                    
                    # Show best so far per strategy
                    for strat in strategies:
                        if all_results[strat]:
                            best = max(all_results[strat], key=lambda r: r.sharpe)
                            logger.info(f"  {strat}: best Sharpe = {best.sharpe:.2f}")
                
            except Exception as e:
                logger.error(f"Error in backtest: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"optimization_results_{timestamp}.json"
    
    results_data = {
        strategy: [
            {
                "params": r.params,
                "sharpe": r.sharpe,
                "total_pnl": r.total_pnl,
                "daily_pnl": r.daily_pnl,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "n_trades": r.n_trades,
                "error": r.error,
            }
            for r in results
        ]
        for strategy, results in all_results.items()
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for strategy in strategies:
        results = all_results[strategy]
        if not results:
            continue
        
        best = max(results, key=lambda r: r.sharpe)
        avg_sharpe = np.mean([r.sharpe for r in results if r.error is None])
        
        print(f"\n{strategy}:")
        print(f"  Best Sharpe: {best.sharpe:.2f}")
        print(f"  Best Daily PnL: ${best.daily_pnl:.2f}")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Best params: {json.dumps(best.params, indent=4)}")
    
    return all_results


def run_on_remote(
    strategies: List[str],
    n_workers: int = None,
    n_iterations: int = 50,
):
    """
    Run parallel optimization on remote box via SSH.
    """
    # Kill old processes first
    logger.info("Killing old processes on remote...")
    kill_cmd = (
        f"ssh -p {REMOTE_PORT} {REMOTE_HOST} "
        f"'pkill -f \"python.*backtest\" ; pkill -f \"python.*optim\" ; echo Done'"
    )
    subprocess.run(kill_cmd, shell=True)
    time.sleep(2)
    
    # Get CPU count if not specified
    if n_workers is None:
        result = subprocess.run(
            f"ssh -p {REMOTE_PORT} {REMOTE_HOST} 'nproc'",
            shell=True, capture_output=True, text=True
        )
        n_workers = int(result.stdout.strip()) - 4  # Leave some headroom
    
    logger.info(f"Running on remote with {n_workers} workers")
    
    # Build command
    strategies_str = " ".join(strategies)
    cmd = (
        f"ssh -p {REMOTE_PORT} {REMOTE_HOST} "
        f"'cd {REMOTE_DIR} && "
        f"python scripts/run_parallel_backtests.py "
        f"--local --workers {n_workers} --iterations {n_iterations} "
        f"--strategies {strategies_str}'"
    )
    
    logger.info(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Run parallel backtests")
    parser.add_argument("--remote", action="store_true", help="Run on remote box")
    parser.add_argument("--local", action="store_true", help="Run locally")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per strategy")
    parser.add_argument("--strategies", nargs="+", default=["all"], help="Strategies to optimize")
    parser.add_argument("--output-dir", type=str, default="logs/parallel_optimization")
    
    args = parser.parse_args()
    
    # Determine strategies
    if "all" in args.strategies:
        strategies = list(STRATEGY_PARAM_SPACES.keys())
    else:
        strategies = args.strategies
    
    if args.remote:
        run_on_remote(
            strategies=strategies,
            n_workers=args.workers,
            n_iterations=args.iterations,
        )
    elif args.local:
        n_workers = args.workers or (cpu_count() - 2)
        run_parallel_optimization(
            strategies=strategies,
            n_workers=n_workers,
            n_iterations=args.iterations,
            output_dir=args.output_dir,
        )
    else:
        # Default: run on remote
        run_on_remote(
            strategies=strategies,
            n_workers=args.workers,
            n_iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
