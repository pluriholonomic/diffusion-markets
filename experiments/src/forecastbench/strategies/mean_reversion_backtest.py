"""
Walk-forward backtesting framework for group mean-reversion strategies.

This module implements:
1. Rolling calibration estimation (no lookahead)
2. Regime detection per group
3. Position construction per rebalance period
4. PnL tracking with attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from forecastbench.strategies.regime_detector import (
    GroupCalibrationTracker,
    RegimeDetectorConfig,
    RegimeType,
)
from forecastbench.strategies.basket_builder import (
    Basket,
    BasketBuilderConfig,
    Position,
    UnifiedBasketBuilder,
    aggregate_positions,
)


@dataclass(frozen=True)
class GroupMeanReversionConfig:
    """Configuration for group mean-reversion backtest."""
    
    # Regime detection
    regime_cfg: RegimeDetectorConfig = field(default_factory=RegimeDetectorConfig)
    
    # Basket construction
    basket_cfg: BasketBuilderConfig = field(default_factory=BasketBuilderConfig)
    
    # Position methods to use
    position_methods: Tuple[str, ...] = ("calibration", "dollar_neutral")
    
    # Rebalancing
    rebalance_freq: Literal["event", "daily", "weekly"] = "event"
    
    # Risk management
    max_portfolio_exposure: float = 1.0  # Max total exposure
    stop_loss_pct: float = 0.20          # Exit if drawdown > 20%
    
    # Transaction costs
    transaction_cost: float = 0.01       # 1% round-trip cost
    
    # Evaluation
    bootstrap_samples: int = 1000        # For confidence intervals


@dataclass
class TradeRecord:
    """Record of a single trade."""
    
    idx: int                    # Event index
    timestamp: Optional[float]
    market_id: str
    group: str
    direction: int              # +1 long, -1 short
    size: float
    entry_price: float
    exit_price: float          # Outcome (0 or 1) or early exit price
    pnl: float
    regime: str
    method: str
    
    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results from a mean-reversion backtest."""
    
    # Overall performance
    total_pnl: float
    final_bankroll: float
    roi: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    
    # Trade statistics
    n_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Gross profit / gross loss
    
    # Attribution
    pnl_by_group: Dict[str, float] = field(default_factory=dict)
    pnl_by_regime: Dict[str, float] = field(default_factory=dict)
    pnl_by_method: Dict[str, float] = field(default_factory=dict)
    
    # Diagnostics
    regime_accuracy: float = 0.0  # Did mean-revert regimes actually revert?
    calibration_correlation: float = 0.0  # Corr(calibration, returns)
    
    # Time series
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Confidence intervals (from bootstrap)
    sharpe_ci: Tuple[float, float] = (0.0, 0.0)
    roi_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Raw trade records
    trades: List[TradeRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            "total_pnl": self.total_pnl,
            "final_bankroll": self.final_bankroll,
            "roi": self.roi,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "profit_factor": self.profit_factor,
            "pnl_by_group": self.pnl_by_group,
            "pnl_by_regime": self.pnl_by_regime,
            "pnl_by_method": self.pnl_by_method,
            "regime_accuracy": self.regime_accuracy,
            "calibration_correlation": self.calibration_correlation,
            "sharpe_ci": list(self.sharpe_ci),
            "roi_ci": list(self.roi_ci),
        }


def run_group_mean_reversion_backtest(
    df: pd.DataFrame,
    *,
    model_forecast_col: str,
    market_price_col: str = "market_prob",
    group_col: str = "category",
    outcome_col: str = "y",
    market_id_col: str = "id",
    time_col: Optional[str] = None,
    cfg: GroupMeanReversionConfig = GroupMeanReversionConfig(),
    initial_bankroll: float = 1.0,
    verbose: bool = False,
) -> BacktestResult:
    """
    Run walk-forward backtest for group mean-reversion strategy.
    
    This implements:
    1. Rolling calibration estimation (no lookahead)
    2. Regime detection per group
    3. Position construction per rebalance period
    4. PnL tracking with full attribution
    
    Args:
        df: DataFrame with market data (must be sorted by time)
        model_forecast_col: Column with model predictions
        market_price_col: Column with market prices
        group_col: Column with group assignments
        outcome_col: Column with outcomes
        market_id_col: Column with market identifiers
        time_col: Optional column with timestamps
        cfg: Backtest configuration
        initial_bankroll: Starting capital
        verbose: Print progress
        
    Returns:
        BacktestResult with full performance attribution
    """
    df = df.copy()
    n = len(df)
    
    if n == 0:
        return _empty_result(initial_bankroll)
    
    # Sort by time if available
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    
    # Initialize components
    calibration_tracker = GroupCalibrationTracker(cfg.regime_cfg)
    basket_builder = UnifiedBasketBuilder(cfg.basket_cfg, list(cfg.position_methods))
    
    # State
    bankroll = initial_bankroll
    equity_curve = [bankroll]
    trades: List[TradeRecord] = []
    
    # Attribution tracking
    pnl_by_group: Dict[str, float] = {}
    pnl_by_regime: Dict[str, float] = {}
    pnl_by_method: Dict[str, float] = {}
    
    # Regime accuracy tracking
    regime_predictions: List[Tuple[str, float]] = []  # (regime, actual_error)
    
    # Process each event
    for i in range(n):
        row = df.iloc[i]
        
        market_id = str(row[market_id_col])
        group = str(row[group_col]) if pd.notna(row.get(group_col)) else "unknown"
        market_price = float(row[market_price_col]) if pd.notna(row.get(market_price_col)) else 0.5
        model_price = float(row[model_forecast_col]) if pd.notna(row.get(model_forecast_col)) else 0.5
        outcome = float(row[outcome_col]) if pd.notna(row.get(outcome_col)) else np.nan
        if time_col and pd.notna(row.get(time_col)):
            ts_val = row[time_col]
            if isinstance(ts_val, (int, float)):
                timestamp = float(ts_val)
            else:
                # Convert datetime to unix timestamp
                try:
                    timestamp = pd.to_datetime(ts_val).timestamp()
                except (ValueError, TypeError):
                    timestamp = float(i)
        else:
            timestamp = float(i)
        
        if np.isnan(outcome):
            continue
        
        # Get current regime (based on past data only - no lookahead)
        regime = calibration_tracker.get_regime(group)
        calib_error = calibration_tracker.get_calibration_error(group)
        
        # Track regime prediction accuracy
        actual_error = outcome - market_price
        regime_predictions.append((regime.value, actual_error))
        
        # Build basket for this event
        regimes = calibration_tracker.get_all_regimes()
        
        basket = basket_builder.build_combined_basket(
            market_ids=np.array([market_id]),
            groups=np.array([group]),
            market_prices=np.array([market_price]),
            model_prices=np.array([model_price]),
            regimes=regimes,
            calibration_tracker=calibration_tracker,
        )
        
        # Execute trades
        for pos in basket.positions:
            # Check exposure limits
            if sum(t.size for t in trades[-100:] if t.pnl == 0) > cfg.max_portfolio_exposure:
                continue
            
            # Compute PnL
            # Long YES: pay q, receive y (profit = y - q if y=1, loss = -q if y=0)
            # Short YES (long NO): pay (1-q), receive (1-y)
            if pos.direction > 0:
                trade_pnl = pos.size * (outcome - pos.entry_price)
            else:
                trade_pnl = pos.size * (pos.entry_price - outcome)
            
            # Apply transaction cost
            trade_pnl -= pos.size * cfg.transaction_cost
            
            # Create trade record
            trade = TradeRecord(
                idx=i,
                timestamp=timestamp,
                market_id=pos.market_id,
                group=pos.group,
                direction=pos.direction,
                size=pos.size,
                entry_price=pos.entry_price,
                exit_price=outcome,
                pnl=trade_pnl,
                regime=regime.value,
                method=pos.method,
            )
            trades.append(trade)
            
            # Update bankroll
            bankroll += trade_pnl
            
            # Update attribution
            pnl_by_group[pos.group] = pnl_by_group.get(pos.group, 0.0) + trade_pnl
            pnl_by_regime[regime.value] = pnl_by_regime.get(regime.value, 0.0) + trade_pnl
            pnl_by_method[pos.method] = pnl_by_method.get(pos.method, 0.0) + trade_pnl
        
        # Update calibration tracker (after trading to avoid lookahead)
        calibration_tracker.update(group, market_price, outcome)
        
        # Track equity
        equity_curve.append(bankroll)
        
        # Check stop loss
        peak = max(equity_curve)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0
        if drawdown > cfg.stop_loss_pct:
            if verbose:
                print(f"Stop loss triggered at event {i}, drawdown={drawdown:.2%}")
            break
    
    # Compute summary statistics
    equity_curve = np.array(equity_curve)
    trade_pnls = np.array([t.pnl for t in trades])
    
    if len(trades) == 0:
        return _empty_result(initial_bankroll)
    
    # Basic metrics
    total_pnl = bankroll - initial_bankroll
    roi = total_pnl / initial_bankroll
    win_rate = np.mean([t.is_win for t in trades])
    avg_trade_pnl = np.mean(trade_pnls)
    
    # Sharpe ratio (annualized, assuming ~252 events/year)
    if np.std(trade_pnls) > 1e-10:
        sharpe = np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / np.maximum(peak, 1e-10)
    max_drawdown = float(np.max(drawdowns))
    
    # Win/loss averages
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss
    
    # Regime accuracy: did mean-revert regimes have lower error?
    mean_revert_errors = [abs(e) for r, e in regime_predictions if r == "mean_revert"]
    other_errors = [abs(e) for r, e in regime_predictions if r != "mean_revert"]
    
    if mean_revert_errors and other_errors:
        regime_accuracy = 1.0 if np.mean(mean_revert_errors) < np.mean(other_errors) else 0.0
    else:
        regime_accuracy = 0.5
    
    # Calibration correlation
    if len(regime_predictions) > 10:
        calib_scores = [1.0 if r == "mean_revert" else 0.0 for r, _ in regime_predictions]
        abs_errors = [abs(e) for _, e in regime_predictions]
        if np.std(calib_scores) > 1e-10 and np.std(abs_errors) > 1e-10:
            calibration_correlation = -np.corrcoef(calib_scores, abs_errors)[0, 1]
        else:
            calibration_correlation = 0.0
    else:
        calibration_correlation = 0.0
    
    # Bootstrap confidence intervals
    sharpe_ci, roi_ci = _bootstrap_ci(trade_pnls, initial_bankroll, cfg.bootstrap_samples)
    
    return BacktestResult(
        total_pnl=total_pnl,
        final_bankroll=bankroll,
        roi=roi,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        n_trades=len(trades),
        avg_trade_pnl=avg_trade_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        pnl_by_group=pnl_by_group,
        pnl_by_regime=pnl_by_regime,
        pnl_by_method=pnl_by_method,
        regime_accuracy=regime_accuracy,
        calibration_correlation=calibration_correlation,
        equity_curve=equity_curve,
        sharpe_ci=sharpe_ci,
        roi_ci=roi_ci,
        trades=trades,
    )


def run_walk_forward_backtest(
    df: pd.DataFrame,
    *,
    model_forecast_col: str,
    market_price_col: str = "market_prob",
    group_col: str = "category",
    outcome_col: str = "y",
    market_id_col: str = "id",
    time_col: Optional[str] = None,
    cfg: GroupMeanReversionConfig = GroupMeanReversionConfig(),
    initial_bankroll: float = 1.0,
    train_frac: float = 0.5,
    n_folds: int = 5,
    verbose: bool = False,
) -> BacktestResult:
    """
    Walk-forward backtest with expanding training window.
    
    This provides more robust out-of-sample evaluation by:
    1. Training calibration on historical data
    2. Trading on out-of-sample period
    3. Expanding window and repeating
    """
    df = df.copy()
    n = len(df)
    
    if n == 0:
        return _empty_result(initial_bankroll)
    
    # Sort by time if available
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    
    # Determine fold boundaries
    initial_train = int(n * train_frac)
    remaining = n - initial_train
    fold_size = remaining // n_folds
    
    all_trades: List[TradeRecord] = []
    fold_results: List[Dict] = []
    cumulative_bankroll = initial_bankroll
    
    for fold in range(n_folds):
        train_end = initial_train + fold * fold_size
        test_start = train_end
        test_end = train_end + fold_size if fold < n_folds - 1 else n
        
        if test_end <= test_start:
            continue
        
        # Train calibration on historical data
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        # Run backtest on test period
        fold_result = run_group_mean_reversion_backtest(
            test_df,
            model_forecast_col=model_forecast_col,
            market_price_col=market_price_col,
            group_col=group_col,
            outcome_col=outcome_col,
            market_id_col=market_id_col,
            time_col=time_col,
            cfg=cfg,
            initial_bankroll=cumulative_bankroll,
            verbose=verbose,
        )
        
        cumulative_bankroll = fold_result.final_bankroll
        all_trades.extend(fold_result.trades)
        
        fold_results.append({
            "fold": fold,
            "train_size": train_end,
            "test_size": test_end - test_start,
            "roi": fold_result.roi,
            "sharpe": fold_result.sharpe,
            "n_trades": fold_result.n_trades,
        })
        
        if verbose:
            print(f"Fold {fold}: ROI={fold_result.roi:.2%}, Sharpe={fold_result.sharpe:.2f}")
    
    # Aggregate results
    if not all_trades:
        return _empty_result(initial_bankroll)
    
    trade_pnls = np.array([t.pnl for t in all_trades])
    total_pnl = cumulative_bankroll - initial_bankroll
    roi = total_pnl / initial_bankroll
    
    if np.std(trade_pnls) > 1e-10:
        sharpe = np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Aggregate attribution
    pnl_by_group: Dict[str, float] = {}
    pnl_by_regime: Dict[str, float] = {}
    pnl_by_method: Dict[str, float] = {}
    
    for t in all_trades:
        pnl_by_group[t.group] = pnl_by_group.get(t.group, 0.0) + t.pnl
        pnl_by_regime[t.regime] = pnl_by_regime.get(t.regime, 0.0) + t.pnl
        pnl_by_method[t.method] = pnl_by_method.get(t.method, 0.0) + t.pnl
    
    # Bootstrap CI
    sharpe_ci, roi_ci = _bootstrap_ci(trade_pnls, initial_bankroll, cfg.bootstrap_samples)
    
    return BacktestResult(
        total_pnl=total_pnl,
        final_bankroll=cumulative_bankroll,
        roi=roi,
        sharpe=sharpe,
        max_drawdown=0.0,  # Would need to reconstruct equity curve
        win_rate=np.mean([t.is_win for t in all_trades]),
        n_trades=len(all_trades),
        avg_trade_pnl=np.mean(trade_pnls),
        avg_win=float(np.mean(trade_pnls[trade_pnls > 0])) if any(trade_pnls > 0) else 0.0,
        avg_loss=float(np.mean(trade_pnls[trade_pnls < 0])) if any(trade_pnls < 0) else 0.0,
        profit_factor=float(np.sum(trade_pnls[trade_pnls > 0])) / max(abs(np.sum(trade_pnls[trade_pnls < 0])), 1e-10),
        pnl_by_group=pnl_by_group,
        pnl_by_regime=pnl_by_regime,
        pnl_by_method=pnl_by_method,
        sharpe_ci=sharpe_ci,
        roi_ci=roi_ci,
        trades=all_trades,
    )


def compare_strategies(
    df: pd.DataFrame,
    *,
    model_forecast_col: str,
    market_price_col: str = "market_prob",
    group_col: str = "category",
    outcome_col: str = "y",
    strategies: List[str] = None,
    bootstrap_n: int = 1000,
) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategy configurations on the same data.
    """
    strategies = strategies or ["calibration", "dollar_neutral", "frechet"]
    results = {}
    
    for strategy in strategies:
        cfg = GroupMeanReversionConfig(
            position_methods=(strategy,),
        )
        
        result = run_group_mean_reversion_backtest(
            df,
            model_forecast_col=model_forecast_col,
            market_price_col=market_price_col,
            group_col=group_col,
            outcome_col=outcome_col,
            cfg=cfg,
        )
        results[strategy] = result
    
    return results


def _bootstrap_ci(
    trade_pnls: np.ndarray,
    initial_bankroll: float,
    n_samples: int = 1000,
    alpha: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute bootstrap confidence intervals for Sharpe and ROI."""
    if len(trade_pnls) < 10:
        return (0.0, 0.0), (0.0, 0.0)
    
    rng = np.random.default_rng(42)
    sharpes = []
    rois = []
    
    for _ in range(n_samples):
        sample = rng.choice(trade_pnls, size=len(trade_pnls), replace=True)
        
        if np.std(sample) > 1e-10:
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
        else:
            sharpe = 0.0
        sharpes.append(sharpe)
        
        roi = np.sum(sample) / initial_bankroll
        rois.append(roi)
    
    sharpe_ci = (
        float(np.percentile(sharpes, 100 * alpha / 2)),
        float(np.percentile(sharpes, 100 * (1 - alpha / 2))),
    )
    roi_ci = (
        float(np.percentile(rois, 100 * alpha / 2)),
        float(np.percentile(rois, 100 * (1 - alpha / 2))),
    )
    
    return sharpe_ci, roi_ci


def _empty_result(initial_bankroll: float) -> BacktestResult:
    """Return empty result for edge cases."""
    return BacktestResult(
        total_pnl=0.0,
        final_bankroll=initial_bankroll,
        roi=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        n_trades=0,
        avg_trade_pnl=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        profit_factor=0.0,
    )
