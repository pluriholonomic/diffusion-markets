"""
Base strategy protocol and common utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
    from backtest.market_state import MarketStateManager


@dataclass
class StrategyState:
    """State tracked by a strategy."""

    # C_t representation
    ct_samples: Optional[np.ndarray] = None  # (mc, k) samples
    market_ids: List[str] = field(default_factory=list)

    # Performance tracking
    cumulative_pnl: float = 0.0
    n_trades: int = 0
    n_steps: int = 0

    # Custom state
    custom: Dict = field(default_factory=dict)


class BaseStrategy(Protocol):
    """
    Protocol defining the interface for trading strategies.

    All strategies must implement these methods to integrate
    with the BacktestEngine.
    """

    name: str
    state: StrategyState

    def on_ct_refresh(
        self,
        ct_samples: np.ndarray,
        market_ids: List[str],
    ) -> None:
        """
        Called when C_t is updated (typically daily).

        Args:
            ct_samples: (mc, k) Monte Carlo samples from the diffusion model
            market_ids: List of market IDs corresponding to columns in ct_samples
        """
        ...

    def on_price_update(
        self,
        market_id: str,
        old_price: float,
        new_price: float,
        state: "MarketStateManager",
    ) -> Optional[Dict[str, float]]:
        """
        React to a significant price change.

        This is called when a market's price changes beyond the threshold.
        Useful for event-driven strategies like flip detection.

        Args:
            market_id: The market that moved
            old_price: Previous price
            new_price: New price
            state: Current market state

        Returns:
            Dict mapping market_id -> desired position, or None for no action
        """
        ...

    def on_snapshot(
        self,
        timestamp: float,
        state: "MarketStateManager",
    ) -> Dict[str, float]:
        """
        Periodic trading decision.

        Called at regular intervals (e.g., every 5 minutes).
        This is the main entry point for systematic strategies.

        Args:
            timestamp: Current timestamp
            state: Current market state

        Returns:
            Dict mapping market_id -> desired position
        """
        ...

    def on_resolution(
        self,
        market_id: str,
        outcome: float,
    ) -> None:
        """
        Update learning with realized outcome.

        Called when a market resolves. Use this to update
        online learning algorithms with the true outcome.

        Args:
            market_id: The resolved market
            outcome: Realized outcome (0 or 1)
        """
        ...

    def get_diagnostics(self) -> Dict:
        """
        Return diagnostic information for logging.

        Returns:
            Dict with strategy-specific diagnostics
        """
        ...


def compute_projection_direction(
    q: np.ndarray,
    ct_samples: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute the projection direction from q to C_t.

    Uses the convex hull projection utilities from forecastbench.

    Args:
        q: (k,) current market prices
        ct_samples: (mc, k) samples from C_t

    Returns:
        direction: (k,) unit vector pointing toward C_t
        distance: L2 distance to C_t
    """
    from forecastbench.utils.convex_hull_projection import project_point_to_convex_hull

    result = project_point_to_convex_hull(x=q, points=ct_samples)

    # Direction from q to projection
    residual = result.proj - q
    norm = np.linalg.norm(residual)
    if norm > 1e-12:
        direction = residual / norm
    else:
        direction = np.zeros_like(q)

    return direction, result.dist_l2


def sign_trader_positions(
    q: np.ndarray,
    p_model: np.ndarray,
    B: float = 1.0,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Simple sign-based trading: go long if model says higher, short if lower.

    Args:
        q: (k,) market prices
        p_model: (k,) model predictions
        B: Max position size
        threshold: Min difference to trade

    Returns:
        positions: (k,) position vector in [-B, B]
    """
    diff = p_model - q
    positions = np.zeros_like(q)
    positions[diff > threshold] = B
    positions[diff < -threshold] = -B
    return positions



