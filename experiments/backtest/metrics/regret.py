"""
Regret tracking for online learning strategies.

Tests the Blackwell approachability hypothesis (H2):
predictions should converge at rate O(1/sqrt{T}).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RegretTracker:
    """
    Track regret over time for online learning strategies.

    Regret = (cumulative profit of best expert) - (cumulative profit of our mixture)

    For Blackwell approachability, we expect:
        Regret_T / T → 0 at rate O(1/sqrt{T})
        i.e., Regret_T = O(sqrt{T})
    """

    # Per-strategy regret curves
    strategy_regret: Dict[str, List[float]] = field(default_factory=dict)
    strategy_cum_profit: Dict[str, List[float]] = field(default_factory=dict)
    strategy_best_expert: Dict[str, List[float]] = field(default_factory=dict)

    # Global step counter
    t: int = 0

    def record(
        self,
        strategy: str,
        mixture_profit: float,
        best_expert_profit: float,
    ) -> None:
        """
        Record a single step of regret.

        Args:
            strategy: Strategy name
            mixture_profit: Cumulative profit of the mixture/algorithm
            best_expert_profit: Cumulative profit of best expert in hindsight
        """
        if strategy not in self.strategy_regret:
            self.strategy_regret[strategy] = []
            self.strategy_cum_profit[strategy] = []
            self.strategy_best_expert[strategy] = []

        regret = best_expert_profit - mixture_profit
        self.strategy_regret[strategy].append(regret)
        self.strategy_cum_profit[strategy].append(mixture_profit)
        self.strategy_best_expert[strategy].append(best_expert_profit)

        self.t += 1

    def get_regret_curve(self, strategy: str) -> List[float]:
        """Get the regret curve for a strategy."""
        return self.strategy_regret.get(strategy, [])

    def get_average_regret_curve(self, strategy: str) -> List[float]:
        """Get regret/T curve (should converge to 0)."""
        curve = self.strategy_regret.get(strategy, [])
        if not curve:
            return []
        return [r / (i + 1) for i, r in enumerate(curve)]

    def compute_regret_exponent(self, strategy: str) -> float:
        """
        Estimate the regret decay exponent.

        Theory predicts: Regret_T ~ T^{0.5}
        So: Regret_T / T ~ T^{-0.5}

        We fit log(Regret/T) ~ -α log(T) and expect α ≈ 0.5.

        Returns:
            Estimated exponent α
        """
        curve = self.strategy_regret.get(strategy, [])
        if len(curve) < 20:
            return float("nan")

        T = np.arange(1, len(curve) + 1)
        regret = np.array(curve)

        # Average regret
        avg_regret = regret / T

        # Filter to positive values for log
        mask = avg_regret > 1e-12
        if mask.sum() < 10:
            return float("nan")

        # Log-log regression
        log_T = np.log(T[mask])
        log_avg = np.log(avg_regret[mask])

        # Linear fit: log_avg = -α * log_T + c
        try:
            slope, _ = np.polyfit(log_T, log_avg, 1)
            return -float(slope)
        except Exception:
            return float("nan")

    def summary(self) -> Dict:
        """Get summary statistics."""
        result = {"t": self.t, "strategies": {}}

        for strategy in self.strategy_regret:
            curve = self.strategy_regret[strategy]
            if not curve:
                continue

            result["strategies"][strategy] = {
                "final_regret": curve[-1] if curve else 0.0,
                "final_avg_regret": curve[-1] / len(curve) if curve else 0.0,
                "regret_exponent": self.compute_regret_exponent(strategy),
                "n_steps": len(curve),
            }

        return result


def compute_regret_exponent(
    regret_curve: List[float],
    min_points: int = 20,
) -> Tuple[float, float]:
    """
    Compute the regret decay exponent from a regret curve.

    Fits: log(Regret_T / T) = -α * log(T) + c

    Args:
        regret_curve: List of cumulative regret values
        min_points: Minimum points required for fit

    Returns:
        (exponent, r_squared) tuple
    """
    if len(regret_curve) < min_points:
        return float("nan"), 0.0

    T = np.arange(1, len(regret_curve) + 1)
    regret = np.array(regret_curve)

    # Average regret
    avg_regret = regret / T

    # Filter to positive values
    mask = avg_regret > 1e-12
    if mask.sum() < min_points // 2:
        return float("nan"), 0.0

    # Log transform
    log_T = np.log(T[mask])
    log_avg = np.log(avg_regret[mask])

    # Linear regression
    try:
        coeffs = np.polyfit(log_T, log_avg, 1)
        slope, intercept = coeffs

        # Compute R²
        predicted = slope * log_T + intercept
        ss_res = np.sum((log_avg - predicted) ** 2)
        ss_tot = np.sum((log_avg - np.mean(log_avg)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return -float(slope), float(r_squared)
    except Exception:
        return float("nan"), 0.0


def theoretical_regret_bound(
    T: int,
    n_experts: int,
    B: float = 1.0,
    k: int = 1,
) -> float:
    """
    Compute the theoretical Hedge regret bound.

    Regret ≤ sqrt(2 * T * log(N)) * Range

    where Range ≤ 2 * B * k (max profit swing per round).

    Args:
        T: Number of rounds
        n_experts: Number of experts
        B: Max position size per market
        k: Number of markets per bundle

    Returns:
        Upper bound on regret
    """
    if T <= 0 or n_experts <= 1:
        return 0.0

    range_bound = 2 * B * k
    return float(np.sqrt(2 * T * np.log(n_experts)) * range_bound)


def validate_blackwell_rate(
    regret_curve: List[float],
    expected_exponent: float = 0.5,
    tolerance: float = 0.1,
) -> Dict:
    """
    Validate that regret follows the expected Blackwell rate.

    Args:
        regret_curve: Cumulative regret over time
        expected_exponent: Expected decay rate (0.5 for Blackwell)
        tolerance: Acceptable deviation from expected

    Returns:
        Dict with validation results
    """
    exponent, r_squared = compute_regret_exponent(regret_curve)

    is_valid = (
        np.isfinite(exponent)
        and abs(exponent - expected_exponent) <= tolerance
        and r_squared > 0.5
    )

    return {
        "estimated_exponent": exponent,
        "expected_exponent": expected_exponent,
        "r_squared": r_squared,
        "is_valid": is_valid,
        "deviation": abs(exponent - expected_exponent) if np.isfinite(exponent) else float("inf"),
        "n_points": len(regret_curve),
    }


