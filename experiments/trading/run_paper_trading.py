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
from trading.strategies.momentum import MomentumStrategy
from trading.strategies.dispersion import DispersionStrategy, CorrelationStrategy
from trading.strategies.advanced import (
    BlackwellStrategy, ConfidenceGatedStrategy,
    TrendFollowingStrategy, MeanReversionStrategy, RegimeAdaptiveStrategy,
)
from trading.strategies.portfolio import (
    YesNoConvergenceStrategy, RelativeValueStrategy, RiskParityStrategy,
)
from trading.monitoring.strategy_logger import get_strategy_logger
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
    parser.add_argument("--interval", type=int, default=10,
                        help="Seconds between trading cycles (default: 10)")
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
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 5 second intervals with minimal logging")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable WebSocket streaming for real-time updates")
    args = parser.parse_args()
    
    # Fast mode overrides
    if args.fast:
        args.interval = 5
    
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
    
    # Load optimized params from 2025 OHLCV optimization
    import json
    optimized_params_file = Path(__file__).parent.parent / "optimization_results" / "best_params_2025.json"
    if optimized_params_file.exists():
        with open(optimized_params_file) as f:
            data = json.load(f)
            opt = data["params"]
        print(f"📊 Loaded 2025 optimized params from {optimized_params_file.name}")
        print(f"   hi_threshold={opt.get('hi_threshold', 0.65)}, lo_threshold={opt.get('lo_threshold', 0.35)}")
        print(f"   profit_take={opt['profit_take_pct']}%, stop_loss={opt['stop_loss_pct']}%")
        print(f"   Expected: ${data.get('daily_pnl', 0):.0f}/day, Sharpe={data.get('sharpe', 0):.1f}")
        print()
    else:
        # Try older params file
        older_file = Path(__file__).parent.parent / "optimization_results" / "best_params_v2.json"
        if older_file.exists():
            with open(older_file) as f:
                opt = json.load(f)["params"]
            print(f"📊 Loaded older params from {older_file.name}")
        else:
            opt = {
                "min_edge": 0.01,
                "hi_threshold": 0.65,
                "lo_threshold": 0.35,
                "kelly_fraction": 0.5,
                "max_position_pct": 0.10,
                "profit_take_pct": 25.0,
                "stop_loss_pct": 5.0,
            }
            print("⚠️  Using default optimized params (file not found)")
    
    # Create engine config with 2025 optimized params
    config = EngineConfig(
        paper_trading=True,
        initial_bankroll=args.bankroll,
        log_dir=args.log_dir,
        polymarket_enabled=True,
        kalshi_enabled=True,
        max_orders_per_run=50,  # Increased for more trades
        min_signal_confidence=0.01,  # Lower threshold for more signals
        data_mode=args.mode,
        profit_take_pct=opt.get("profit_take_pct", 25.0),
        stop_loss_pct=opt.get("stop_loss_pct", 5.0),
    )
    
    # Initialize engine
    engine = TradingEngine(config)
    
    # Create and configure strategies with optimized params
    risk_limits = RiskLimits(
        max_position_pct=opt.get("max_position_pct", 0.10),
        max_daily_loss_pct=0.20,
        max_drawdown_pct=0.30,
        kelly_fraction=opt.get("kelly_fraction", 0.5),
        min_edge=opt.get("min_edge", 0.01),
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
    
    # === NEW STRATEGIES (Momentum, Dispersion, Correlation) ===
    
    # 6. Polymarket Momentum
    pm_momentum = MomentumStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_momentum", pm_momentum)
    
    # 7. Polymarket Dispersion Trading
    pm_dispersion = DispersionStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_dispersion", pm_dispersion)
    
    # 8. Polymarket Correlation/Pairs Trading
    pm_correlation = CorrelationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_correlation", pm_correlation)
    
    # 9. Kalshi Momentum
    kalshi_momentum = MomentumStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_momentum", kalshi_momentum)
    
    # 10. Kalshi Dispersion Trading
    kalshi_dispersion = DispersionStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_dispersion", kalshi_dispersion)
    
    # 11. Kalshi Correlation/Pairs Trading
    kalshi_correlation = CorrelationStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_correlation", kalshi_correlation)
    
    # === ADVANCED STRATEGIES ===
    
    # 12. Polymarket Blackwell Calibration
    pm_blackwell = BlackwellStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_blackwell", pm_blackwell)
    
    # 13. Polymarket Confidence Gated
    pm_confidence = ConfidenceGatedStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_confidence_gated", pm_confidence)
    
    # 14. Polymarket Trend Following
    pm_trend = TrendFollowingStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_trend_following", pm_trend)
    
    # 15. Polymarket Mean Reversion
    pm_mean_rev = MeanReversionStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_mean_reversion", pm_mean_rev)
    
    # 16. Polymarket Regime Adaptive
    pm_regime = RegimeAdaptiveStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_regime_adaptive", pm_regime)
    
    # 17. Kalshi Blackwell Calibration
    kalshi_blackwell = BlackwellStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_blackwell", kalshi_blackwell)
    
    # 18. Kalshi Trend Following
    kalshi_trend = TrendFollowingStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_trend_following", kalshi_trend)
    
    # 19. Kalshi Mean Reversion
    kalshi_mean_rev = MeanReversionStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_mean_reversion", kalshi_mean_rev)
    
    # 20. Kalshi Regime Adaptive
    kalshi_regime = RegimeAdaptiveStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_regime_adaptive", kalshi_regime)
    
    # === PORTFOLIO-LEVEL STRATEGIES ===
    
    # 21. Polymarket YES/NO Convergence
    pm_yesno = YesNoConvergenceStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_yesno_convergence", pm_yesno)
    
    # 22. Polymarket Relative Value (within category groups)
    pm_relative = RelativeValueStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_relative_value", pm_relative)
    
    # 23. Polymarket Risk Parity (Markowitz-style)
    pm_riskparity = RiskParityStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
    engine.add_strategy("pm_risk_parity", pm_riskparity)
    
    # 24. Kalshi YES/NO Convergence
    kalshi_yesno = YesNoConvergenceStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_yesno_convergence", kalshi_yesno)
    
    # 25. Kalshi Relative Value
    kalshi_relative = RelativeValueStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_relative_value", kalshi_relative)
    
    # 26. Kalshi Risk Parity
    kalshi_riskparity = RiskParityStrategy(Platform.KALSHI, risk_limits=risk_limits)
    engine.add_strategy("kalshi_risk_parity", kalshi_riskparity)
    
    # Initialize strategy logger for per-minute stats
    strategy_logger = get_strategy_logger(log_dir=str(Path(args.log_dir) / "strategy_stats"))
    for name in engine.strategies.keys():
        strategy_logger.register_strategy(name)
    print(f"\n📊 Strategy logger initialized: {len(engine.strategies)} strategies tracked")
    print(f"   Stats logged every 60 seconds to: {args.log_dir}/strategy_stats/")
    
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
    
    # Initialize streaming if enabled
    stream = None
    if args.streaming:
        try:
            from trading.clients.websocket_stream import MarketPriceStream, StreamConfig
            stream = MarketPriceStream(StreamConfig())
            stream.start()
            print("📡 WebSocket streaming enabled")
        except Exception as e:
            print(f"⚠️  Streaming unavailable: {e}")
            stream = None
    
    # Main loop
    cycle = 0
    last_status_time = 0
    status_interval = 60 if args.fast else 30  # Less frequent status prints in fast mode
    
    while running:
        cycle += 1
        cycle_start = time.time()
        
        if args.max_cycles and cycle > args.max_cycles:
            print(f"Reached max cycles ({args.max_cycles})")
            break
        
        # Minimal logging in fast mode
        if not args.fast:
            print(f"\n{'='*60}")
            print(f"Cycle {cycle} - {datetime.utcnow().isoformat()}")
            print(f"{'='*60}")
        
        try:
            engine.run_cycle()
        except Exception as e:
            print(f"Error in trading cycle: {e}")
        
        # Print status (less frequently in fast mode)
        now = time.time()
        if now - last_status_time >= status_interval:
            status = engine.get_status()
            if args.fast:
                # Compact status for fast mode
                pos_count = len(engine.position_manager.open_positions) if engine.position_manager else 0
                print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                      f"Cycle {cycle} | "
                      f"Positions: {pos_count} | "
                      f"PnL: ${status['risk']['daily_pnl']:,.2f} | "
                      f"Signals: {status['signals_today']}")
            else:
                print(f"\nStatus:")
                print(f"  Bankroll: ${status['risk']['bankroll']:,.2f}")
                print(f"  Daily PnL: ${status['risk']['daily_pnl']:,.2f}")
                print(f"  Drawdown: {status['risk']['current_drawdown']}")
                print(f"  Signals: {status['signals_today']}")
                print(f"  Orders: {status['orders_today']}")
            last_status_time = now
        
        if not running:
            break
        
        # Check for streaming updates (event-driven)
        if stream and stream.has_updates():
            updates = stream.get_updates()
            if updates and not args.fast:
                print(f"  📡 {len(updates)} real-time price updates")
        
        # Wait for next cycle (use sleep intervals for fast response to signals)
        cycle_duration = time.time() - cycle_start
        sleep_time = max(0, args.interval - cycle_duration)
        
        if not args.fast and sleep_time > 0:
            print(f"\nSleeping {sleep_time:.1f} seconds...")
        
        # Sleep in small chunks for responsiveness
        sleep_chunk = 0.5 if args.fast else 1.0
        elapsed = 0
        while elapsed < sleep_time and running:
            time.sleep(min(sleep_chunk, sleep_time - elapsed))
            elapsed += sleep_chunk
            
            # Check for streaming events during sleep
            if stream and stream.has_updates():
                break  # Wake up early if we have updates
    
    # Cleanup streaming
    if stream:
        stream.stop()
    
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
