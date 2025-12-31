"""
Statistical arbitrage strategies for prediction markets.

This module implements group-based mean-reversion strategies where
calibration serves as the key indicator of mean-reverting vs momentum regimes.
"""

from forecastbench.strategies.regime_detector import (
    GroupCalibrationTracker,
    RegimeDetectorConfig,
    RegimeType,
    GroupRegimeState,
    compute_group_calibration_summary,
    detect_regime_changes,
)

from forecastbench.strategies.basket_builder import (
    Position,
    Basket,
    BasketBuilderConfig,
    CalibrationBasedBuilder,
    DollarNeutralBuilder,
    FrechetArbitrageBuilder,
    UnifiedBasketBuilder,
    aggregate_positions,
)

from forecastbench.strategies.mean_reversion_backtest import (
    GroupMeanReversionConfig,
    BacktestResult,
    TradeRecord,
    run_group_mean_reversion_backtest,
    run_walk_forward_backtest,
    compare_strategies,
)

__all__ = [
    # Regime detection
    "GroupCalibrationTracker",
    "RegimeDetectorConfig",
    "RegimeType",
    "GroupRegimeState",
    "compute_group_calibration_summary",
    "detect_regime_changes",
    # Basket construction
    "Position",
    "Basket",
    "BasketBuilderConfig",
    "CalibrationBasedBuilder",
    "DollarNeutralBuilder",
    "FrechetArbitrageBuilder",
    "UnifiedBasketBuilder",
    "aggregate_positions",
    # Backtesting
    "GroupMeanReversionConfig",
    "BacktestResult",
    "TradeRecord",
    "run_group_mean_reversion_backtest",
    "run_walk_forward_backtest",
    "compare_strategies",
]
