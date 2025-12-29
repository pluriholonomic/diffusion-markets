"""
Statistical Arbitrage Portfolio Strategy.

Uses the C_t sample covariance as a correlation structure for
Markowitz-style portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from backtest.strategies.base import StrategyState, compute_projection_direction

if TYPE_CHECKING:
    from backtest.market_state import MarketStateManager


@dataclass
class StatArbConfig:
    """Configuration for the stat arb portfolio strategy."""

    # Portfolio optimization
    risk_aversion: float = 1.0  # Lambda in mean-variance
    regularization: float = 0.01  # Shrinkage for covariance inverse
    max_gross_exposure: float = 5.0  # Max sum of |positions|
    max_position_per_market: float = 1.0

    # Rebalancing
    min_rebalance_threshold: float = 0.05  # Min change to rebalance

    # Mean reversion parameters
    spread_halflife_hours: float = 24.0
    min_spread_to_trade: float = 0.02


class StatArbPortfolioStrategy:
    """
    Statistical arbitrage using C_t as a correlation structure.

    Key insight: C_t encodes which probability vectors are "consistent."
    The sample covariance of C_t samples gives us a correlation matrix
    that we can use for portfolio optimization.

    The "spread" (q - Π_{C_t}(q)) represents mispricing, and we construct
    a portfolio that bets on mean reversion of this spread.
    """

    name = "stat_arb"

    def __init__(self, cfg: StatArbConfig):
        self.cfg = cfg
        self.state = StrategyState()

        # Covariance structure from C_t
        self._ct_cov: Optional[np.ndarray] = None
        self._ct_cov_inv: Optional[np.ndarray] = None
        self._ct_mean: Optional[np.ndarray] = None

        # Eigenportfolios
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None

        # Current target portfolio
        self._target_positions: Dict[str, float] = {}

    def on_ct_refresh(
        self,
        ct_samples: np.ndarray,
        market_ids: List[str],
    ) -> None:
        """Update C_t representation and recompute covariance structure."""
        self.state.ct_samples = ct_samples
        self.state.market_ids = list(market_ids)

        # Compute covariance from samples
        if ct_samples.shape[0] > 1:
            self._ct_mean = np.mean(ct_samples, axis=0)
            self._ct_cov = np.cov(ct_samples.T)

            # Ensure 2D covariance
            if self._ct_cov.ndim == 0:
                self._ct_cov = np.array([[float(self._ct_cov)]])
            elif self._ct_cov.ndim == 1:
                self._ct_cov = np.diag(self._ct_cov)

            # Regularized inverse
            k = self._ct_cov.shape[0]
            reg = self.cfg.regularization * np.eye(k)
            try:
                self._ct_cov_inv = np.linalg.inv(self._ct_cov + reg)
            except np.linalg.LinAlgError:
                self._ct_cov_inv = np.eye(k)

            # Eigendecomposition for principal portfolios
            try:
                self._eigenvalues, self._eigenvectors = np.linalg.eigh(self._ct_cov)
                # Sort by eigenvalue descending
                idx = np.argsort(self._eigenvalues)[::-1]
                self._eigenvalues = self._eigenvalues[idx]
                self._eigenvectors = self._eigenvectors[:, idx]
            except np.linalg.LinAlgError:
                self._eigenvalues = None
                self._eigenvectors = None
        else:
            self._ct_cov = None
            self._ct_cov_inv = None

    def _compute_spread(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the "spread" (mispricing) relative to C_t.

        Spread = q - Π_{C_t}(q)

        This represents how far market prices are from the consistent set.
        """
        if self.state.ct_samples is None:
            return np.zeros_like(q)

        direction, distance = compute_projection_direction(q, self.state.ct_samples)
        # Spread is in the opposite direction (q is too high/low relative to C_t)
        return -direction * distance

    def _optimal_portfolio(self, spread: np.ndarray) -> np.ndarray:
        """
        Compute optimal portfolio for mean reversion on the spread.

        Uses Markowitz-style optimization:
            w* = (1/λ) Σ^{-1} μ

        where μ = -spread (we expect spread to mean-revert to 0)
        and Σ comes from C_t sample covariance.
        """
        if self._ct_cov_inv is None:
            # Fall back to simple sign trading
            return -np.sign(spread)

        # Expected "return" is the negative spread (mean reversion)
        expected_return = -spread

        # Optimal weights
        raw_weights = (1.0 / self.cfg.risk_aversion) * (self._ct_cov_inv @ expected_return)

        # Clip individual positions
        raw_weights = np.clip(
            raw_weights,
            -self.cfg.max_position_per_market,
            self.cfg.max_position_per_market,
        )

        # Scale to respect gross exposure constraint
        gross = np.sum(np.abs(raw_weights))
        if gross > self.cfg.max_gross_exposure:
            raw_weights *= self.cfg.max_gross_exposure / gross

        return raw_weights

    def on_price_update(
        self,
        market_id: str,
        old_price: float,
        new_price: float,
        state: "MarketStateManager",
    ) -> Optional[Dict[str, float]]:
        """
        This strategy uses periodic rebalancing, not event-driven trading.
        """
        return None

    def on_snapshot(
        self,
        timestamp: float,
        state: "MarketStateManager",
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio and return target positions.
        """
        if self.state.ct_samples is None or not self.state.market_ids:
            return {}

        # Get current prices
        q = state.get_bundle_prices(self.state.market_ids)

        # Filter to valid markets
        valid_mask = ~np.isnan(q)
        if not valid_mask.any():
            return {}

        valid_ids = [m for m, v in zip(self.state.market_ids, valid_mask) if v]
        valid_q = q[valid_mask]
        valid_samples = self.state.ct_samples[:, valid_mask]

        # Recompute covariance for valid subset if needed
        if valid_samples.shape[0] > 1 and valid_samples.shape[1] > 0:
            valid_cov = np.cov(valid_samples.T)
            if valid_cov.ndim == 0:
                valid_cov = np.array([[float(valid_cov)]])
            elif valid_cov.ndim == 1:
                valid_cov = np.diag(valid_cov)

            k = valid_cov.shape[0]
            reg = self.cfg.regularization * np.eye(k)
            try:
                valid_cov_inv = np.linalg.inv(valid_cov + reg)
            except np.linalg.LinAlgError:
                valid_cov_inv = np.eye(k)
        else:
            valid_cov_inv = None

        # Compute spread
        direction, distance = compute_projection_direction(valid_q, valid_samples)
        spread = -direction * distance

        # Store for diagnostics
        self.state.custom["last_spread"] = spread
        self.state.custom["last_distance"] = distance

        # Check if spread is large enough
        if distance < self.cfg.min_spread_to_trade:
            return {}

        # Compute optimal portfolio
        if valid_cov_inv is not None:
            expected_return = -spread
            raw_weights = (1.0 / self.cfg.risk_aversion) * (valid_cov_inv @ expected_return)
            raw_weights = np.clip(
                raw_weights,
                -self.cfg.max_position_per_market,
                self.cfg.max_position_per_market,
            )
            gross = np.sum(np.abs(raw_weights))
            if gross > self.cfg.max_gross_exposure:
                raw_weights *= self.cfg.max_gross_exposure / gross
            positions = raw_weights
        else:
            positions = -np.sign(spread) * min(1.0, self.cfg.max_position_per_market)

        # Check for rebalancing threshold
        new_target = {m: float(p) for m, p in zip(valid_ids, positions) if abs(p) > 1e-6}

        if self._should_rebalance(new_target):
            self._target_positions = new_target
            self.state.n_steps += 1
            return new_target

        return {}

    def _should_rebalance(self, new_target: Dict[str, float]) -> bool:
        """Check if positions have changed enough to warrant rebalancing."""
        if not self._target_positions:
            return True

        total_change = 0.0
        all_markets = set(self._target_positions.keys()) | set(new_target.keys())
        for m in all_markets:
            old = self._target_positions.get(m, 0.0)
            new = new_target.get(m, 0.0)
            total_change += abs(new - old)

        return total_change > self.cfg.min_rebalance_threshold

    def on_resolution(
        self,
        market_id: str,
        outcome: float,
    ) -> None:
        """Track resolutions for PnL attribution."""
        self.state.custom.setdefault("resolutions", []).append({
            "market_id": market_id,
            "outcome": outcome,
        })

    def get_eigenportfolios(self, n_top: int = 3) -> List[Dict[str, float]]:
        """
        Get the top eigenportfolios (principal correlation modes).

        These represent the main "factors" in the C_t correlation structure.
        """
        if self._eigenvectors is None or not self.state.market_ids:
            return []

        portfolios = []
        n = min(n_top, self._eigenvectors.shape[1])

        for i in range(n):
            weights = self._eigenvectors[:, i]
            portfolio = {
                m: float(w) for m, w in zip(self.state.market_ids, weights)
            }
            portfolios.append({
                "eigenvalue": float(self._eigenvalues[i]),
                "weights": portfolio,
            })

        return portfolios

    def get_diagnostics(self) -> Dict:
        """Return strategy diagnostics."""
        diagnostics = {
            "name": self.name,
            "n_steps": self.state.n_steps,
            "n_markets": len(self.state.market_ids),
            "n_positions": len(self._target_positions),
            "gross_exposure": sum(abs(p) for p in self._target_positions.values()),
        }

        if "last_distance" in self.state.custom:
            diagnostics["last_distance"] = self.state.custom["last_distance"]

        if self._eigenvalues is not None:
            diagnostics["top_3_eigenvalues"] = self._eigenvalues[:3].tolist()

        return diagnostics


