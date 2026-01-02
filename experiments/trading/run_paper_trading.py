#!/usr/bin/env python3
"""
Paper Trading Runner

Run this script to start paper trading on both Polymarket and Kalshi.

Usage:
    python -m trading.run_paper_trading --bankroll 10000 --interval 300
"""

import argparse
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.execution.engine import TradingEngine, EngineConfig
from trading.strategies.calibration import (
    PolymarketCalibrationStrategy,
    KalshiCalibrationStrategy,
)
from trading.strategies.longshot import LongshotStrategy
from trading.strategies.stat_arb import StatArbStrategy
from trading.utils.models import Platform, RiskLimits


def load_calibration_data(strategy, data_path: str):
    """Load historical data for calibration."""
    try:
        df = pd.read_parquet(data_path)
        strategy.update_historical_data(df)
        print(f"Loaded calibration data from {data_path}")
        return True
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--bankroll", type=float, default=10000.0,
                        help="Initial bankroll in USD")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between trading cycles")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="Maximum number of cycles (None = infinite)")
    parser.add_argument("--polymarket-data", type=str, 
                        default="data/polymarket/optimization_cache.parquet",
                        help="Path to Polymarket historical data")
    parser.add_argument("--kalshi-data", type=str,
                        default="data/kalshi/kalshi_backtest_clean.parquet",
                        help="Path to Kalshi historical data")
    parser.add_argument("--log-dir", type=str, default="logs/paper_trading",
                        help="Directory for logs")
    parser.add_argument("--mode", type=str, default="simulated",
                        choices=["simulated", "hybrid", "live"],
                        help="Data mode: simulated (historical), hybrid (live data, simulated trades), live")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PAPER TRADING SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Bankroll: ${args.bankroll:,.0f}")
    print(f"Interval: {args.interval} seconds")
    print(f"Log directory: {args.log_dir}")
    print()
    
    if args.mode == "hybrid":
        print("📡 HYBRID MODE: Fetching LIVE market data, SIMULATING trades")
    elif args.mode == "simulated":
        print("📊 SIMULATED MODE: Using historical data")
    else:
        print("⚠️  LIVE MODE: Real trading (requires credentials)")
    print()
    
    # Create engine config
    config = EngineConfig(
        paper_trading=True,
        initial_bankroll=args.bankroll,
        log_dir=args.log_dir,
        polymarket_enabled=True,
        kalshi_enabled=True,
        max_orders_per_run=20,  # Increased to utilize more signals
        min_signal_confidence=0.3,
        data_mode=args.mode,
    )
    
    # Initialize engine
    engine = TradingEngine(config)
    
    # Create and configure strategies
    risk_limits = RiskLimits(
        max_position_pct=0.10,
        max_daily_loss_pct=0.20,
        max_drawdown_pct=0.30,
        kelly_fraction=0.25,
        min_edge=0.05,
        min_liquidity=500,
    )
    
    # === POLYMARKET STRATEGIES ===
    
    # 1. Polymarket Calibration Mean-Reversion
    pm_calibration = PolymarketCalibrationStrategy(risk_limits)
    if Path(args.polymarket_data).exists():
        load_calibration_data(pm_calibration, args.polymarket_data)
    engine.add_strategy("pm_calibration", pm_calibration)
    
    # 2. Polymarket Statistical Arbitrage
    pm_stat_arb = StatArbStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    if Path(args.polymarket_data).exists():
        load_calibration_data(pm_stat_arb, args.polymarket_data)
    engine.add_strategy("pm_stat_arb", pm_stat_arb)
    
    # === KALSHI STRATEGIES ===
    
    # 3. Kalshi Calibration Mean-Reversion
    kalshi_calibration = KalshiCalibrationStrategy(risk_limits)
    if Path(args.kalshi_data).exists():
        load_calibration_data(kalshi_calibration, args.kalshi_data)
    engine.add_strategy("kalshi_calibration", kalshi_calibration)
    
    # 4. Kalshi Longshot
    kalshi_longshot = LongshotStrategy(risk_limits=risk_limits)
    if Path(args.kalshi_data).exists():
        load_calibration_data(kalshi_longshot, args.kalshi_data)
    engine.add_strategy("kalshi_longshot", kalshi_longshot)
    
    # 5. Kalshi Statistical Arbitrage
    kalshi_stat_arb = StatArbStrategy(Platform.KALSHI, risk_limits=risk_limits)
    if Path(args.kalshi_data).exists():
        load_calibration_data(kalshi_stat_arb, args.kalshi_data)
    engine.add_strategy("kalshi_stat_arb", kalshi_stat_arb)
    
    # Print calibration status
    print("\nCalibration Status:")
    for name, strategy in engine.strategies.items():
        summary = strategy.get_calibration_summary()
        print(f"  {name}: {summary['status']}")
        if summary['status'] == 'calibrated':
            print(f"    Samples: {summary['total_samples']}")
            print(f"    Mean spread: {summary['mean_spread']:.3f}")
    
    print("\n" + "=" * 60)
    print("Starting paper trading loop...")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Setup signal handler for graceful shutdown
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Main loop
    cycle = 0
    while running:
        cycle += 1
        
        if args.max_cycles and cycle > args.max_cycles:
            print(f"Reached max cycles ({args.max_cycles})")
            break
        
        print(f"\n{'='*60}")
        print(f"Cycle {cycle} - {datetime.utcnow().isoformat()}")
        print(f"{'='*60}")
        
        try:
            engine.run_cycle()
        except Exception as e:
            print(f"Error in trading cycle: {e}")
        
        # Print status
        status = engine.get_status()
        print(f"\nStatus:")
        print(f"  Bankroll: ${status['risk']['bankroll']:,.2f}")
        print(f"  Daily PnL: ${status['risk']['daily_pnl']:,.2f}")
        print(f"  Drawdown: {status['risk']['current_drawdown']}")
        print(f"  Signals: {status['signals_today']}")
        print(f"  Orders: {status['orders_today']}")
        
        if not running:
            break
        
        # Wait for next cycle
        print(f"\nSleeping {args.interval} seconds...")
        for _ in range(args.interval):
            if not running:
                break
            time.sleep(1)
    
    print("\n" + "=" * 60)
    print("Paper trading stopped")
    print("=" * 60)
    
    # Final status
    final_status = engine.get_status()
    print(f"\nFinal Status:")
    print(f"  Total cycles: {cycle}")
    print(f"  Final bankroll: ${final_status['risk']['bankroll']:,.2f}")
    print(f"  Total signals: {final_status['signals_today']}")
    print(f"  Total orders: {final_status['orders_today']}")


if __name__ == "__main__":
    main()
