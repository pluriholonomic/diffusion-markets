"""
Regime detection via group-wise calibration tracking.

This module implements the core insight that in prediction markets:
- High calibration (E[Y - q | group] ≈ 0) indicates mean-reverting regime
- Low calibration with persistent bias indicates momentum regime

The GroupCalibrationTracker maintains rolling statistics per group and
classifies each group into a trading regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


class RegimeType(str, Enum):
    """Trading regime based on calibration characteristics."""
    
    MEAN_REVERT = "mean_revert"  # High calibration, trade toward group mean
    MOMENTUM = "momentum"        # Persistent bias, trade with the trend
    NEUTRAL = "neutral"          # Insufficient data or unclear signal


@dataclass
class GroupRegimeState:
    """State for a single group's calibration tracking."""
    
    # Rolling statistics (exponentially weighted)
    ema_residual: float = 0.0           # E[Y - q | group] (signed bias)
    ema_abs_residual: float = 0.0       # E[|Y - q| | group] (absolute error)
    ema_residual_sq: float = 0.0        # E[(Y - q)^2 | group] (MSE)
    
    # Counts
    n_observations: int = 0
    n_positive_residual: int = 0        # Count of Y > q
    n_negative_residual: int = 0        # Count of Y < q
    
    # Regime history for stability
    recent_regimes: List[RegimeType] = field(default_factory=list)
    
    # Tracking timestamps for decay
    last_update_idx: int = 0


@dataclass(frozen=True)
class RegimeDetectorConfig:
    """Configuration for regime detection."""
    
    # EMA decay factor (higher = more weight on recent observations)
    ema_alpha: float = 0.1
    
    # Regime thresholds
    mean_revert_threshold: float = 0.05   # |E[Y-q|g]| < threshold → mean-reverting
    momentum_threshold: float = 0.15      # |E[Y-q|g]| > threshold with persistence → momentum
    
    # Minimum observations before classifying
    min_observations: int = 10
    
    # Regime stability (require N consecutive same classifications)
    regime_stability_window: int = 3
    
    # Persistence threshold for momentum (ratio of positive/negative residuals)
    persistence_threshold: float = 0.7  # 70% same sign = persistent bias


class GroupCalibrationTracker:
    """
    Track rolling calibration statistics per group for regime detection.
    
    This implements the core insight that calibration errors reveal
    the underlying price dynamics:
    
    - Mean-reverting groups: Prices converge toward true probabilities
    - Momentum groups: Persistent systematic bias in one direction
    
    Usage:
        tracker = GroupCalibrationTracker(cfg)
        for event in stream:
            tracker.update(event.group, event.market_price, event.outcome)
            regime = tracker.get_regime(event.group)
            if regime == RegimeType.MEAN_REVERT:
                # Trade toward group mean
                pass
    """
    
    def __init__(self, cfg: RegimeDetectorConfig = RegimeDetectorConfig()):
        self.cfg = cfg
        self._states: Dict[str, GroupRegimeState] = {}
        self._global_idx: int = 0
    
    def update(
        self,
        group: str,
        price: float,
        outcome: float,
        *,
        weight: float = 1.0,
    ) -> None:
        """
        Update calibration statistics for a group.
        
        Args:
            group: Group identifier (category, topic, etc.)
            price: Market price q ∈ [0, 1]
            outcome: Realized outcome y ∈ {0, 1}
            weight: Optional weight for this observation
        """
        self._global_idx += 1
        
        if group not in self._states:
            self._states[group] = GroupRegimeState()
        
        state = self._states[group]
        alpha = self.cfg.ema_alpha * weight
        
        # Compute residual
        residual = float(outcome) - float(price)
        abs_residual = abs(residual)
        residual_sq = residual ** 2
        
        # Update EMAs
        if state.n_observations == 0:
            state.ema_residual = residual
            state.ema_abs_residual = abs_residual
            state.ema_residual_sq = residual_sq
        else:
            state.ema_residual = (1 - alpha) * state.ema_residual + alpha * residual
            state.ema_abs_residual = (1 - alpha) * state.ema_abs_residual + alpha * abs_residual
            state.ema_residual_sq = (1 - alpha) * state.ema_residual_sq + alpha * residual_sq
        
        # Update counts
        state.n_observations += 1
        if residual > 0:
            state.n_positive_residual += 1
        elif residual < 0:
            state.n_negative_residual += 1
        
        state.last_update_idx = self._global_idx
        
        # Update regime history
        regime = self._classify_regime(state)
        state.recent_regimes.append(regime)
        if len(state.recent_regimes) > self.cfg.regime_stability_window:
            state.recent_regimes.pop(0)
    
    def _classify_regime(self, state: GroupRegimeState) -> RegimeType:
        """Classify regime based on current state (internal, unstable)."""
        if state.n_observations < self.cfg.min_observations:
            return RegimeType.NEUTRAL
        
        abs_bias = abs(state.ema_residual)
        
        # Check for mean-reversion (low absolute bias)
        if abs_bias < self.cfg.mean_revert_threshold:
            return RegimeType.MEAN_REVERT
        
        # Check for momentum (high bias + persistence)
        if abs_bias > self.cfg.momentum_threshold:
            total = state.n_positive_residual + state.n_negative_residual
            if total > 0:
                # Check if residuals are persistently one-sided
                ratio = max(state.n_positive_residual, state.n_negative_residual) / total
                if ratio >= self.cfg.persistence_threshold:
                    return RegimeType.MOMENTUM
        
        return RegimeType.NEUTRAL
    
    def get_regime(self, group: str) -> RegimeType:
        """
        Get the stable regime classification for a group.
        
        Returns a stable regime only if the recent classifications are consistent.
        """
        if group not in self._states:
            return RegimeType.NEUTRAL
        
        state = self._states[group]
        
        if len(state.recent_regimes) < self.cfg.regime_stability_window:
            return RegimeType.NEUTRAL
        
        # Check for stability
        recent = state.recent_regimes[-self.cfg.regime_stability_window:]
        if all(r == recent[0] for r in recent):
            return recent[0]
        
        return RegimeType.NEUTRAL
    
    def get_calibration_error(self, group: str) -> float:
        """Get the current signed calibration error E[Y - q | group]."""
        if group not in self._states:
            return 0.0
        return self._states[group].ema_residual
    
    def get_calibration_stats(self, group: str) -> Dict[str, float]:
        """Get detailed calibration statistics for a group."""
        if group not in self._states:
            return {
                "bias": 0.0,
                "abs_error": 0.0,
                "mse": 0.0,
                "n_observations": 0,
                "positive_ratio": 0.5,
            }
        
        state = self._states[group]
        total = state.n_positive_residual + state.n_negative_residual
        positive_ratio = state.n_positive_residual / total if total > 0 else 0.5
        
        return {
            "bias": state.ema_residual,
            "abs_error": state.ema_abs_residual,
            "mse": state.ema_residual_sq,
            "n_observations": state.n_observations,
            "positive_ratio": positive_ratio,
        }
    
    def get_all_regimes(self) -> Dict[str, RegimeType]:
        """Get regime classifications for all tracked groups."""
        return {group: self.get_regime(group) for group in self._states}
    
    def get_groups_by_regime(self, regime: RegimeType) -> List[str]:
        """Get all groups classified as a specific regime."""
        return [g for g, r in self.get_all_regimes().items() if r == regime]
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self._states.clear()
        self._global_idx = 0
    
    def snapshot(self) -> Dict[str, Dict]:
        """Get a serializable snapshot of the tracker state."""
        return {
            group: {
                "regime": self.get_regime(group).value,
                **self.get_calibration_stats(group),
            }
            for group in self._states
        }


def compute_group_calibration_summary(
    groups: np.ndarray,
    prices: np.ndarray,
    outcomes: np.ndarray,
    *,
    cfg: RegimeDetectorConfig = RegimeDetectorConfig(),
) -> Dict[str, Dict]:
    """
    Compute calibration summary for all groups in a dataset.
    
    This is a batch utility for analyzing historical data before backtesting.
    
    Args:
        groups: Array of group labels (N,)
        prices: Array of market prices (N,)
        outcomes: Array of outcomes (N,)
        cfg: Regime detector configuration
        
    Returns:
        Dict mapping group -> calibration stats + regime
    """
    tracker = GroupCalibrationTracker(cfg)
    
    groups = np.asarray(groups).reshape(-1)
    prices = np.asarray(prices, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    
    for g, p, y in zip(groups, prices, outcomes):
        tracker.update(str(g), float(p), float(y))
    
    return tracker.snapshot()


def detect_regime_changes(
    groups: np.ndarray,
    prices: np.ndarray,
    outcomes: np.ndarray,
    *,
    cfg: RegimeDetectorConfig = RegimeDetectorConfig(),
) -> List[Dict]:
    """
    Detect regime changes over time for all groups.
    
    Returns a list of regime change events with timestamps.
    """
    tracker = GroupCalibrationTracker(cfg)
    
    groups = np.asarray(groups).reshape(-1)
    prices = np.asarray(prices, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    
    changes: List[Dict] = []
    prev_regimes: Dict[str, RegimeType] = {}
    
    for i, (g, p, y) in enumerate(zip(groups, prices, outcomes)):
        g_str = str(g)
        prev_regime = prev_regimes.get(g_str, RegimeType.NEUTRAL)
        
        tracker.update(g_str, float(p), float(y))
        new_regime = tracker.get_regime(g_str)
        
        if new_regime != prev_regime:
            changes.append({
                "idx": i,
                "group": g_str,
                "from_regime": prev_regime.value,
                "to_regime": new_regime.value,
                "calibration_error": tracker.get_calibration_error(g_str),
            })
            prev_regimes[g_str] = new_regime
    
    return changes
