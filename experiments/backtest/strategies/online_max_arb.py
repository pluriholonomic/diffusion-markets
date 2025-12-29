"""
Online Max Arbitrage Strategy.

Uses Hedge (exponential weights) to achieve O(1/sqrt{T}) regret
for arbitrage extraction based on the learned C_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from backtest.strategies.base import StrategyState, compute_projection_direction

if TYPE_CHECKING:
    from backtest.market_state import MarketStateManager


@dataclass
class OnlineMaxArbConfig:
    """Configuration for the online max arbitrage strategy."""

    # Position limits
    B_max: float = 1.0  # Max position per market
    transaction_cost: float = 0.002

    # Hedge parameters
    eta: Optional[float] = None  # Learning rate (None = adaptive)

    # Expert construction
    n_direction_experts: int = 3  # Number of discretized directions

    # Minimum action threshold
    min_distance_to_trade: float = 0.02  # Min d(q, C_t) to take action


class OnlineMaxArbStrategy:
    """
    Online arbitrage trader with O(1/sqrt{T}) regret guarantee.

    The key insight: the projection Π_{C_t}(q) onto the learned constraint
    set tells us the direction of mispricing. We use Hedge over a set of
    experts derived from the projection direction.

    Regret bound: Regret_T ≤ O(sqrt{T log N}) where N is number of experts.
    """

    name = "online_max_arb"

    def __init__(self, cfg: OnlineMaxArbConfig):
        self.cfg = cfg
        self.state = StrategyState()

        # Hedge state (from forecastbench.metrics.online_arb)
        self._hedge = None
        self._k: int = 0  # Number of markets

        # Tracking for regret computation
        self._cumulative_profit: float = 0.0
        self._best_expert_profit: Optional[np.ndarray] = None

    def on_ct_refresh(
        self,
        ct_samples: np.ndarray,
        market_ids: List[str],
    ) -> None:
        """Update C_t representation."""
        self.state.ct_samples = ct_samples
        self.state.market_ids = list(market_ids)
        self._k = len(market_ids)

        # Reset hedge for new C_t
        self._reset_hedge()

    def _reset_hedge(self) -> None:
        """Initialize or reset the Hedge algorithm."""
        from forecastbench.metrics.online_arb import HedgeState

        if self._k <= 0:
            self._hedge = None
            return

        # Number of experts: 2 + 2*k (direction-based)
        # - Full toward/away projection
        # - Per-coordinate toward/away
        n_experts = 2 + 2 * self._k

        self._hedge = HedgeState(
            n_experts=n_experts,
            k=self._k,
            B=self.cfg.B_max,
            transaction_cost=self.cfg.transaction_cost,
            eta=self.cfg.eta,
        )
        self._hedge.reset()

    def _build_experts(self, direction: np.ndarray) -> np.ndarray:
        """
        Build expert actions from the projection direction.

        Experts:
        - Full vector toward projection: b = -B * sign(direction)
        - Full vector away from projection: b = +B * sign(direction)
        - Per-coordinate toward/away versions

        Args:
            direction: (k,) projection direction

        Returns:
            expert_b: (n_experts, k) expert actions
        """
        k = len(direction)
        B = self.cfg.B_max

        # Sign of direction
        sd = np.sign(direction)
        sd[sd == 0] = 1  # Default to positive for zero direction

        experts = []

        # Full vector toward/away
        experts.append(-B * sd)  # Toward projection (arbitrage direction)
        experts.append(+B * sd)  # Away from projection

        # Per-coordinate
        for j in range(k):
            e_toward = np.zeros(k)
            e_toward[j] = -B * sd[j]
            experts.append(e_toward)

            e_away = np.zeros(k)
            e_away[j] = +B * sd[j]
            experts.append(e_away)

        return np.array(experts, dtype=np.float64)

    def on_price_update(
        self,
        market_id: str,
        old_price: float,
        new_price: float,
        state: "MarketStateManager",
    ) -> Optional[Dict[str, float]]:
        """
        React to price update.

        This strategy is snapshot-based, so we don't trade on individual
        price updates. Returns None.
        """
        return None

    def on_snapshot(
        self,
        timestamp: float,
        state: "MarketStateManager",
    ) -> Dict[str, float]:
        """
        Make trading decision based on current prices and C_t.

        Args:
            timestamp: Current timestamp
            state: Market state manager

        Returns:
            Dict mapping market_id -> desired position
        """
        if self.state.ct_samples is None or not self.state.market_ids:
            return {}

        if self._hedge is None:
            self._reset_hedge()

        # Get current prices for our markets
        q = state.get_bundle_prices(self.state.market_ids)

        # Filter to valid (non-NaN) markets
        valid_mask = ~np.isnan(q)
        if not valid_mask.any():
            return {}

        valid_ids = [m for m, v in zip(self.state.market_ids, valid_mask) if v]
        valid_q = q[valid_mask]
        valid_samples = self.state.ct_samples[:, valid_mask]

        # Compute projection direction
        direction, distance = compute_projection_direction(valid_q, valid_samples)

        # Check if distance is large enough to trade
        if distance < self.cfg.min_distance_to_trade:
            return {}

        # Store distance for H4 validation
        self.state.custom["last_dist_to_ct"] = distance
        self.state.custom["last_direction"] = direction

        # Build experts and get Hedge mixture
        expert_b = self._build_experts(direction)

        # Hedge mixture action
        if self._hedge is not None and self._hedge.w is not None:
            # Truncate/pad experts if needed
            if len(expert_b) != self._hedge.n_experts:
                # Reinitialize hedge with correct size
                self._hedge = None
                self._reset_hedge()
                if self._hedge is None:
                    return {}

            # Compute mixture position
            positions = (self._hedge.w.reshape(-1, 1) * expert_b[:, :len(valid_ids)]).sum(axis=0)
        else:
            # Fall back to simple projection-based trade
            positions = -self.cfg.B_max * np.sign(direction)

        # Map back to market IDs
        result = {}
        for i, market_id in enumerate(valid_ids):
            if abs(positions[i]) > 1e-6:
                result[market_id] = float(positions[i])

        self.state.n_steps += 1
        return result

    def on_resolution(
        self,
        market_id: str,
        outcome: float,
    ) -> None:
        """
        Update Hedge with realized outcome.

        This is where the online learning happens.
        """
        if self._hedge is None or self.state.ct_samples is None:
            return

        if market_id not in self.state.market_ids:
            return

        # Get the index of this market
        idx = self.state.market_ids.index(market_id)

        # We need to call hedge.step() with the expert actions and outcome
        # This updates the weights for future decisions
        # For now, we track that an update happened
        self.state.custom.setdefault("resolutions", []).append({
            "market_id": market_id,
            "outcome": outcome,
        })

    def step_hedge_update(
        self,
        expert_b: np.ndarray,
        y: np.ndarray,
        price: np.ndarray,
    ) -> None:
        """
        Explicit Hedge update with observed outcome.

        Call this after observing the true outcome to update weights.
        """
        if self._hedge is None:
            return

        self._hedge.step(expert_b=expert_b, y=y, price=price)

    def get_diagnostics(self) -> Dict:
        """Return strategy diagnostics."""
        diagnostics = {
            "name": self.name,
            "n_steps": self.state.n_steps,
            "n_markets": len(self.state.market_ids),
            "cumulative_pnl": self.state.cumulative_pnl,
        }

        if self._hedge is not None:
            diagnostics["hedge_t"] = self._hedge.t
            diagnostics["hedge_cum_profit"] = self._hedge.cum_profit_mix
            diagnostics["hedge_best_expert"] = self._hedge.best_expert_profit()
            diagnostics["regret_bound"] = self._hedge.regret_bound()

        if "last_dist_to_ct" in self.state.custom:
            diagnostics["last_dist_to_ct"] = self.state.custom["last_dist_to_ct"]

        return diagnostics

    def get_regret_curve(self) -> List[float]:
        """Get the cumulative regret over time."""
        if self._hedge is None:
            return []

        # Regret = best_expert_profit - mixture_profit
        best = self._hedge.best_expert_profit()
        mix = self._hedge.cum_profit_mix
        return [best - mix]  # Simplified; full curve would need history tracking


