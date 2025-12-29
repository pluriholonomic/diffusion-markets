"""
H4 Hypothesis Validation.

Tests whether distance to C_t bounds extractable arbitrage profit:
    d(q, C_t) ~ realized PnL

If the diffusion model correctly learns the constraint set C_t,
then the distance from market prices to C_t should correlate
positively with realized profit from arbitrage trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class H4Validator:
    """
    Track d(q, C_t) vs realized PnL to validate H4.

    H4: The distance from market prices q to the learned constraint set C_t
        bounds extractable statistical arbitrage.

    We measure:
    1. Correlation between distance and realized profit
    2. Whether high-distance trades are more profitable
    3. Predictive power of distance for profit magnitude
    """

    # Raw data
    distances: List[float] = field(default_factory=list)
    pnls: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Per-trade attribution
    trade_distances: List[float] = field(default_factory=list)
    trade_pnls: List[float] = field(default_factory=list)

    def record(
        self,
        dist_to_ct: float,
        realized_pnl: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a distance-pnl pair.

        Args:
            dist_to_ct: Distance from market price to C_t
            realized_pnl: Realized profit from the trade
            timestamp: Optional timestamp
        """
        self.distances.append(dist_to_ct)
        self.pnls.append(realized_pnl)
        if timestamp is not None:
            self.timestamps.append(timestamp)

    def record_trade(
        self,
        dist_to_ct: float,
        realized_pnl: float,
    ) -> None:
        """Record a trade-level distance-pnl pair."""
        self.trade_distances.append(dist_to_ct)
        self.trade_pnls.append(realized_pnl)

    def compute_correlation(self) -> float:
        """
        Compute Pearson correlation between distance and PnL.

        Returns:
            Correlation coefficient (should be positive for valid H4)
        """
        if len(self.distances) < 10:
            return float("nan")

        dist = np.array(self.distances)
        pnl = np.array(self.pnls)

        # Filter out NaN/inf
        mask = np.isfinite(dist) & np.isfinite(pnl)
        if mask.sum() < 10:
            return float("nan")

        try:
            corr = np.corrcoef(dist[mask], pnl[mask])[0, 1]
            return float(corr) if np.isfinite(corr) else float("nan")
        except Exception:
            return float("nan")

    def compute_rank_correlation(self) -> float:
        """
        Compute Spearman rank correlation.

        More robust to outliers than Pearson.

        Returns:
            Rank correlation coefficient
        """
        if len(self.distances) < 10:
            return float("nan")

        dist = np.array(self.distances)
        pnl = np.array(self.pnls)

        mask = np.isfinite(dist) & np.isfinite(pnl)
        if mask.sum() < 10:
            return float("nan")

        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(dist[mask], pnl[mask])
            return float(corr) if np.isfinite(corr) else float("nan")
        except ImportError:
            # Fall back to manual rank correlation
            d_ranks = np.argsort(np.argsort(dist[mask]))
            p_ranks = np.argsort(np.argsort(pnl[mask]))
            return float(np.corrcoef(d_ranks, p_ranks)[0, 1])
        except Exception:
            return float("nan")

    def compute_quantile_pnl(self, n_quantiles: int = 5) -> Dict[str, float]:
        """
        Compute average PnL by distance quantile.

        If H4 is valid, higher distance quantiles should have higher PnL.

        Returns:
            Dict mapping quantile label to average PnL
        """
        if len(self.distances) < n_quantiles * 2:
            return {}

        dist = np.array(self.distances)
        pnl = np.array(self.pnls)

        mask = np.isfinite(dist) & np.isfinite(pnl)
        if mask.sum() < n_quantiles * 2:
            return {}

        dist = dist[mask]
        pnl = pnl[mask]

        # Compute quantile boundaries
        quantiles = np.linspace(0, 100, n_quantiles + 1)
        boundaries = np.percentile(dist, quantiles)

        result = {}
        for i in range(n_quantiles):
            lo, hi = boundaries[i], boundaries[i + 1]
            in_quantile = (dist >= lo) & (dist < hi if i < n_quantiles - 1 else dist <= hi)
            if in_quantile.sum() > 0:
                label = f"q{i+1}"
                result[label] = float(np.mean(pnl[in_quantile]))
                result[f"{label}_n"] = int(in_quantile.sum())
                result[f"{label}_dist_range"] = (float(lo), float(hi))

        return result

    def compute_regression(self) -> Tuple[float, float, float]:
        """
        Compute linear regression: PnL = α + β * distance

        Returns:
            (alpha, beta, r_squared) tuple
        """
        if len(self.distances) < 10:
            return float("nan"), float("nan"), 0.0

        dist = np.array(self.distances)
        pnl = np.array(self.pnls)

        mask = np.isfinite(dist) & np.isfinite(pnl)
        if mask.sum() < 10:
            return float("nan"), float("nan"), 0.0

        dist = dist[mask]
        pnl = pnl[mask]

        try:
            # Add constant for intercept
            X = np.column_stack([np.ones_like(dist), dist])
            coeffs, residuals, rank, s = np.linalg.lstsq(X, pnl, rcond=None)
            alpha, beta = coeffs

            # R²
            predicted = alpha + beta * dist
            ss_res = np.sum((pnl - predicted) ** 2)
            ss_tot = np.sum((pnl - np.mean(pnl)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            return float(alpha), float(beta), float(r_squared)
        except Exception:
            return float("nan"), float("nan"), 0.0

    def validate_h4(self) -> Dict:
        """
        Full H4 validation analysis.

        Returns:
            Dict with validation metrics and conclusion
        """
        n_obs = len(self.distances)
        pearson = self.compute_correlation()
        spearman = self.compute_rank_correlation()
        alpha, beta, r_squared = self.compute_regression()
        quantile_pnl = self.compute_quantile_pnl()

        # H4 is validated if:
        # 1. Positive correlation (higher distance → higher profit)
        # 2. Statistically significant (enough observations)
        # 3. Monotonic quantile relationship

        is_valid = (
            n_obs >= 100
            and np.isfinite(pearson)
            and pearson > 0.1
            and np.isfinite(beta)
            and beta > 0
        )

        # Check monotonicity of quantile PnLs
        pnl_values = [v for k, v in quantile_pnl.items() if k.startswith("q") and "_" not in k]
        if len(pnl_values) >= 3:
            # Check if roughly increasing
            increases = sum(1 for i in range(1, len(pnl_values)) if pnl_values[i] > pnl_values[i-1])
            is_monotonic = increases >= len(pnl_values) // 2

        else:
            is_monotonic = True  # Not enough data to check

        return {
            "n_observations": n_obs,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "regression_alpha": alpha,
            "regression_beta": beta,
            "r_squared": r_squared,
            "quantile_pnl": quantile_pnl,
            "is_monotonic": is_monotonic,
            "h4_validated": is_valid and is_monotonic,
            "interpretation": self._interpret(pearson, beta, is_valid),
        }

    def _interpret(self, correlation: float, beta: float, is_valid: bool) -> str:
        """Generate human-readable interpretation."""
        if not is_valid:
            if len(self.distances) < 100:
                return "Insufficient data to validate H4"
            return "H4 NOT validated: distance to C_t does not predict profit"

        if correlation > 0.3:
            strength = "strong"
        elif correlation > 0.1:
            strength = "moderate"
        else:
            strength = "weak"

        return (
            f"H4 validated with {strength} correlation ({correlation:.3f}). "
            f"Each unit of d(q, C_t) predicts {beta:.4f} additional profit."
        )

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "n_observations": len(self.distances),
            "n_trades": len(self.trade_distances),
            "correlation": self.compute_correlation(),
            "mean_distance": float(np.mean(self.distances)) if self.distances else 0.0,
            "mean_pnl": float(np.mean(self.pnls)) if self.pnls else 0.0,
        }


