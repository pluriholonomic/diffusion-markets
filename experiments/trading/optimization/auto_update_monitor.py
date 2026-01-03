#!/usr/bin/env python3
"""
Auto-Update Monitor for Hyperparameter Optimization

Monitors optimization progress and automatically updates running
strategies when optimization converges.

Features:
1. Monitors optimization log files for convergence
2. Detects stationarity using rolling window statistics
3. Updates strategy configs with best parameters
4. Restarts paper trading with new parameters
5. Logs all updates for auditing
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConvergenceDetector:
    """Detects convergence from optimization progress files."""
    
    def __init__(
        self,
        window_size: int = 20,
        stationarity_threshold: float = 0.01,
        min_generations: int = 30,
    ):
        self.window_size = window_size
        self.stationarity_threshold = stationarity_threshold
        self.min_generations = min_generations
    
    def check_convergence(self, progress_file: Path) -> Tuple[bool, Dict]:
        """Check if optimization has converged."""
        if not progress_file.exists():
            return False, {'reason': 'file_not_found'}
        
        # Read progress
        objectives = []
        last_entry = None
        
        with open(progress_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    objectives.append(entry.get('best_objective', 0))
                    last_entry = entry
                except:
                    continue
        
        if len(objectives) < self.min_generations:
            return False, {
                'reason': 'insufficient_generations',
                'generations': len(objectives),
                'required': self.min_generations,
            }
        
        # Check stationarity: compare last window to previous window
        if len(objectives) >= 2 * self.window_size:
            recent = objectives[-self.window_size:]
            previous = objectives[-2*self.window_size:-self.window_size]
            
            recent_mean = np.mean(recent)
            previous_mean = np.mean(previous)
            recent_std = np.std(recent)
            
            # Check if improvement has stalled
            improvement = (recent_mean - previous_mean) / max(abs(previous_mean), 1e-6)
            
            if abs(improvement) < self.stationarity_threshold and recent_std < self.stationarity_threshold * abs(recent_mean):
                return True, {
                    'reason': 'stationarity',
                    'generations': len(objectives),
                    'recent_mean': recent_mean,
                    'improvement': improvement,
                    'best_params': last_entry.get('best_params', {}),
                    'best_metrics': last_entry.get('best_metrics', {}),
                    'best_objective': last_entry.get('best_objective', 0),
                }
        
        # Check for explicit convergence flag
        if last_entry and last_entry.get('converged', False):
            return True, {
                'reason': 'explicit_convergence',
                'generations': len(objectives),
                'best_params': last_entry.get('best_params', {}),
                'best_metrics': last_entry.get('best_metrics', {}),
                'best_objective': last_entry.get('best_objective', 0),
            }
        
        return False, {
            'reason': 'still_optimizing',
            'generations': len(objectives),
            'current_best': objectives[-1] if objectives else 0,
        }


class StrategyUpdater:
    """Updates strategy configurations with optimized parameters."""
    
    def __init__(self, trading_dir: Path, log_dir: Path):
        self.trading_dir = trading_dir
        self.log_dir = log_dir
        self.update_log = log_dir / "parameter_updates.jsonl"
        
    def update_strategy(self, strategy_name: str, params: Dict[str, float]) -> bool:
        """Update a strategy's default parameters."""
        # Map strategy to config file and class
        strategy_files = {
            'calibration': 'strategies/calibration.py',
            'stat_arb': 'strategies/stat_arb.py',
            'momentum': 'strategies/momentum.py',
            'dispersion': 'strategies/dispersion.py',
            'correlation': 'strategies/dispersion.py',
            'blackwell': 'strategies/advanced.py',
            'confidence_gated': 'strategies/advanced.py',
            'trend_following': 'strategies/advanced.py',
            'mean_reversion': 'strategies/advanced.py',
            'regime_adaptive': 'strategies/advanced.py',
            'longshot': 'strategies/longshot.py',
        }
        
        if strategy_name not in strategy_files:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return False
        
        # Log the update
        update_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'strategy': strategy_name,
            'params': params,
        }
        
        with open(self.update_log, 'a') as f:
            f.write(json.dumps(update_record) + '\n')
        
        logger.info(f"Logged parameter update for {strategy_name}")
        logger.info(f"New params: {params}")
        
        return True
    
    def restart_paper_trading(self) -> bool:
        """Restart paper trading with updated parameters."""
        try:
            # Kill existing paper trading
            subprocess.run(["pkill", "-f", "run_paper_trading"], check=False)
            time.sleep(2)
            
            # Start new paper trading
            cmd = [
                "python", "-m", "trading.run_paper_trading",
                "--mode", "hybrid",
                "--interval", "30",
                "--log-dir", "logs/paper_trading",
            ]
            
            log_file = self.log_dir / "paper_trading_restart.log"
            with open(log_file, 'a') as f:
                subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.trading_dir.parent),
                )
            
            logger.info("Paper trading restarted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart paper trading: {e}")
            return False


class AutoUpdateMonitor:
    """Main monitor class."""
    
    def __init__(
        self,
        optimization_dir: Path,
        trading_dir: Path,
        check_interval: int = 60,
        auto_restart: bool = True,
    ):
        self.optimization_dir = Path(optimization_dir)
        self.trading_dir = Path(trading_dir)
        self.check_interval = check_interval
        self.auto_restart = auto_restart
        
        self.detector = ConvergenceDetector()
        self.updater = StrategyUpdater(trading_dir, optimization_dir)
        
        # Track which strategies have been updated
        self.updated_strategies: Dict[str, datetime] = {}
        
        # Load previous updates
        self._load_update_history()
    
    def _load_update_history(self):
        """Load update history from log."""
        update_log = self.optimization_dir / "parameter_updates.jsonl"
        if update_log.exists():
            with open(update_log) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        strategy = entry.get('strategy')
                        ts = datetime.fromisoformat(entry.get('timestamp', '2000-01-01'))
                        self.updated_strategies[strategy] = ts
                    except:
                        continue
    
    def check_all_strategies(self) -> Dict[str, Dict]:
        """Check convergence status for all strategies."""
        status = {}
        
        for progress_file in self.optimization_dir.glob("*_progress.jsonl"):
            strategy = progress_file.stem.replace("_progress", "")
            converged, info = self.detector.check_convergence(progress_file)
            
            status[strategy] = {
                'converged': converged,
                'info': info,
                'last_updated': self.updated_strategies.get(strategy),
            }
            
            # If converged and not recently updated, trigger update
            if converged:
                last_update = self.updated_strategies.get(strategy)
                if last_update is None or (datetime.utcnow() - last_update) > timedelta(hours=1):
                    logger.info(f"Strategy {strategy} converged! Updating parameters...")
                    
                    if self.updater.update_strategy(strategy, info.get('best_params', {})):
                        self.updated_strategies[strategy] = datetime.utcnow()
        
        return status
    
    def run(self):
        """Run the monitoring loop."""
        logger.info("=" * 60)
        logger.info("AUTO-UPDATE MONITOR STARTED")
        logger.info("=" * 60)
        logger.info(f"Optimization dir: {self.optimization_dir}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Auto-restart: {self.auto_restart}")
        
        restart_needed = False
        
        while True:
            try:
                logger.info("\n" + "-" * 40)
                logger.info(f"Checking convergence at {datetime.utcnow().isoformat()}")
                
                status = self.check_all_strategies()
                
                converged_count = sum(1 for s in status.values() if s['converged'])
                total = len(status)
                
                logger.info(f"Convergence status: {converged_count}/{total} strategies")
                
                for strategy, info in status.items():
                    if info['converged']:
                        reason = info['info'].get('reason', 'unknown')
                        obj = info['info'].get('best_objective', 'N/A')
                        logger.info(f"  ✅ {strategy}: converged ({reason}), obj={obj:.4f}")
                        restart_needed = True
                    else:
                        reason = info['info'].get('reason', 'unknown')
                        gens = info['info'].get('generations', 0)
                        logger.info(f"  ⏳ {strategy}: {reason}, gen={gens}")
                
                # Restart if needed and enough strategies converged
                if restart_needed and self.auto_restart and converged_count >= 3:
                    logger.info("\nTriggering paper trading restart with new parameters...")
                    self.updater.restart_paper_trading()
                    restart_needed = False
                
                # Write status summary
                summary_file = self.optimization_dir / "convergence_status.json"
                with open(summary_file, 'w') as f:
                    json.dump({
                        'timestamp': datetime.utcnow().isoformat(),
                        'converged_count': converged_count,
                        'total': total,
                        'strategies': {k: {'converged': v['converged'], 'generations': v['info'].get('generations', 0)} 
                                      for k, v in status.items()},
                    }, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(description="Auto-update monitor for optimization")
    parser.add_argument("--optimization-dir", type=str, default="logs/optimization",
                        help="Directory containing optimization progress files")
    parser.add_argument("--trading-dir", type=str, default="trading",
                        help="Directory containing trading strategy files")
    parser.add_argument("--check-interval", type=int, default=60,
                        help="Seconds between convergence checks")
    parser.add_argument("--no-auto-restart", action="store_true",
                        help="Disable automatic paper trading restart")
    
    args = parser.parse_args()
    
    monitor = AutoUpdateMonitor(
        optimization_dir=Path(args.optimization_dir),
        trading_dir=Path(args.trading_dir),
        check_interval=args.check_interval,
        auto_restart=not args.no_auto_restart,
    )
    
    monitor.run()


if __name__ == "__main__":
    main()
