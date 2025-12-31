"""
Market State Manager.

Central state management for the backtest, handling intraday price updates,
position tracking, and event generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backtest.data.clob_loader import CLOBSnapshot, MarketEvent, iter_clob_events
from backtest.data.group_registry import GroupRegistry


@dataclass
class Position:
    """A position in a single market."""

    market_id: str
    size: float  # Positive = long (YES), negative = short (NO)
    entry_price: float
    entry_timestamp: float
    strategy: str

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        return self.size * (current_price - self.entry_price)

    def realized_pnl(self, exit_price: float) -> float:
        """Calculate realized PnL if closed at exit_price."""
        return self.size * (exit_price - self.entry_price)


@dataclass
class Trade:
    """A completed trade."""

    market_id: str
    timestamp: float
    side: str  # "buy" or "sell"
    size: float
    price: float
    cost: float  # Transaction cost
    strategy: str
    pnl: float = 0.0  # Set on close


class MarketStateManager:
    """
    Central state management for the backtest.

    Tracks:
    - Current prices for all markets
    - Price history for analysis
    - Open positions by strategy
    - Trade history

    Provides:
    - Event iteration for date ranges
    - Position management
    - PnL calculation
    """

    def __init__(
        self,
        clob_data: pd.DataFrame,
        group_registry: GroupRegistry,
        resolutions: Optional[pd.DataFrame] = None,
        snapshot_interval_seconds: float = 300.0,
        price_change_threshold: float = 0.01,
    ):
        self.clob_data = clob_data
        self.group_registry = group_registry
        self.resolutions = resolutions
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.price_change_threshold = price_change_threshold

        # Current state
        self.current_prices: Dict[str, float] = {}
        self.current_time: float = 0.0

        # History
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # market -> [(ts, price)]

        # Positions and trades
        self.positions: Dict[str, Dict[str, Position]] = {}  # strategy -> {market -> Position}
        self.trades: List[Trade] = []

        # Resolved markets
        self.resolved_markets: Set[str] = set()
        self.outcomes: Dict[str, float] = {}

        # Initialize price history structure
        for market_id in clob_data["market_id"].unique():
            self.price_history[str(market_id)] = []

    def reset(self) -> None:
        """Reset state for a new backtest run."""
        self.current_prices.clear()
        self.current_time = 0.0
        for history in self.price_history.values():
            history.clear()
        self.positions.clear()
        self.trades.clear()
        self.resolved_markets.clear()
        self.outcomes.clear()

    def get_active_markets(self) -> List[str]:
        """Get list of markets with known prices that haven't resolved."""
        return [
            m for m in self.current_prices.keys()
            if m not in self.resolved_markets
        ]

    def get_bundle_prices(self, market_ids: List[str]) -> np.ndarray:
        """
        Get current prices for a bundle of markets.

        Args:
            market_ids: List of market IDs

        Returns:
            prices: (k,) array of prices (NaN for unknown markets)
        """
        prices = []
        for m in market_ids:
            if m in self.current_prices and m not in self.resolved_markets:
                prices.append(self.current_prices[m])
            else:
                prices.append(float("nan"))
        return np.array(prices, dtype=np.float64)

    def update(self, event: MarketEvent) -> None:
        """
        Process a market event and update state.

        Args:
            event: MarketEvent to process
        """
        self.current_time = event.timestamp

        if event.type == "price_update":
            market_id = event.market_id
            new_price = event.data["new_price"]
            self.current_prices[market_id] = new_price
            if market_id in self.price_history:
                self.price_history[market_id].append((event.timestamp, new_price))

        elif event.type == "snapshot":
            for market_id, price in event.data["prices"].items():
                self.current_prices[market_id] = price
                if market_id in self.price_history:
                    self.price_history[market_id].append((event.timestamp, price))

        elif event.type == "resolution":
            market_id = event.market_id
            outcome = event.data["outcome"]
            self.resolved_markets.add(market_id)
            self.outcomes[market_id] = outcome
            # Close any open positions in this market
            self._close_positions_for_market(market_id, outcome, event.timestamp)

    def _close_positions_for_market(
        self,
        market_id: str,
        outcome: float,
        timestamp: float,
    ) -> None:
        """Close all positions in a resolved market."""
        for strategy, positions in self.positions.items():
            if market_id in positions:
                pos = positions[market_id]
                # Resolution means payout is outcome (1 for YES, 0 for NO)
                pnl = pos.size * (outcome - pos.entry_price)
                self.trades.append(Trade(
                    market_id=market_id,
                    timestamp=timestamp,
                    side="close",
                    size=abs(pos.size),
                    price=outcome,
                    cost=0.0,  # No cost on resolution
                    strategy=strategy,
                    pnl=pnl,
                ))
                del positions[market_id]

    def open_position(
        self,
        market_id: str,
        size: float,
        strategy: str,
        transaction_cost: float = 0.0,
    ) -> Optional[Trade]:
        """
        Open or adjust a position.

        Args:
            market_id: Market to trade
            size: Desired position size (positive = long YES)
            strategy: Strategy name
            transaction_cost: Transaction cost rate

        Returns:
            Trade object if executed, None if failed
        """
        if market_id not in self.current_prices:
            return None
        if market_id in self.resolved_markets:
            return None

        price = self.current_prices[market_id]

        # Initialize strategy positions dict if needed
        if strategy not in self.positions:
            self.positions[strategy] = {}

        # Check if we already have a position
        current_pos = self.positions[strategy].get(market_id)
        current_size = current_pos.size if current_pos else 0.0

        # Calculate trade size
        trade_size = size - current_size
        if abs(trade_size) < 1e-9:
            return None  # No change needed

        # Calculate cost
        cost = abs(trade_size) * transaction_cost

        # Create trade record
        trade = Trade(
            market_id=market_id,
            timestamp=self.current_time,
            side="buy" if trade_size > 0 else "sell",
            size=abs(trade_size),
            price=price,
            cost=cost,
            strategy=strategy,
        )
        self.trades.append(trade)

        # Update position
        if abs(size) < 1e-9:
            # Close position
            if market_id in self.positions[strategy]:
                del self.positions[strategy][market_id]
        else:
            # Open/adjust position
            if current_pos:
                # Adjust existing position (average entry price)
                total_size = current_size + trade_size
                if abs(total_size) > 1e-9:
                    avg_price = (
                        current_pos.entry_price * current_size + price * trade_size
                    ) / total_size
                    self.positions[strategy][market_id] = Position(
                        market_id=market_id,
                        size=total_size,
                        entry_price=avg_price,
                        entry_timestamp=self.current_time,
                        strategy=strategy,
                    )
            else:
                # New position
                self.positions[strategy][market_id] = Position(
                    market_id=market_id,
                    size=size,
                    entry_price=price,
                    entry_timestamp=self.current_time,
                    strategy=strategy,
                )

        return trade

    def get_position(self, market_id: str, strategy: str) -> float:
        """Get current position size for a market/strategy."""
        if strategy not in self.positions:
            return 0.0
        pos = self.positions[strategy].get(market_id)
        return pos.size if pos else 0.0

    def get_all_positions(self, strategy: str) -> Dict[str, float]:
        """Get all positions for a strategy."""
        if strategy not in self.positions:
            return {}
        return {m: p.size for m, p in self.positions[strategy].items()}

    def get_total_exposure(self, strategy: str) -> float:
        """Get total gross exposure for a strategy."""
        if strategy not in self.positions:
            return 0.0
        return sum(abs(p.size) for p in self.positions[strategy].values())

    def get_unrealized_pnl(self, strategy: str) -> float:
        """Get total unrealized PnL for a strategy."""
        if strategy not in self.positions:
            return 0.0
        total = 0.0
        for market_id, pos in self.positions[strategy].items():
            if market_id in self.current_prices:
                total += pos.unrealized_pnl(self.current_prices[market_id])
        return total

    def get_realized_pnl(self, strategy: str) -> float:
        """Get total realized PnL for a strategy."""
        return sum(t.pnl - t.cost for t in self.trades if t.strategy == strategy)

    def events_for_date(self, date: str) -> Iterator[MarketEvent]:
        """
        Iterate over events for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Yields:
            MarketEvent objects
        """
        # Parse date bounds
        start_ts = pd.to_datetime(f"{date} 00:00:00", utc=True).timestamp()
        end_ts = pd.to_datetime(f"{date} 23:59:59", utc=True).timestamp()

        # Filter CLOB data to this date
        mask = (self.clob_data["timestamp"] >= start_ts) & (self.clob_data["timestamp"] <= end_ts)
        day_data = self.clob_data[mask].copy()

        # Filter resolutions to this date if available
        day_resolutions = None
        if self.resolutions is not None:
            res_ts = self.resolutions["timestamp"].apply(
                lambda x: pd.to_datetime(x, utc=True).timestamp() if pd.notna(x) else float("nan")
            )
            res_mask = (res_ts >= start_ts) & (res_ts <= end_ts)
            day_resolutions = self.resolutions[res_mask].copy()
            if not day_resolutions.empty:
                day_resolutions["timestamp"] = res_ts[res_mask]

        # Generate events
        yield from iter_clob_events(
            day_data,
            snapshot_interval_seconds=self.snapshot_interval_seconds,
            price_change_threshold=self.price_change_threshold,
            resolutions=day_resolutions,
        )

    def get_price_series(self, market_id: str) -> pd.Series:
        """Get price history for a market as a pandas Series."""
        if market_id not in self.price_history:
            return pd.Series(dtype=float)
        history = self.price_history[market_id]
        if not history:
            return pd.Series(dtype=float)
        ts, prices = zip(*history)
        return pd.Series(prices, index=pd.to_datetime(ts, unit="s", utc=True))

    def detect_flip(
        self,
        market_id: str,
        lookback_seconds: float = 3600.0,
        threshold: float = 0.4,
    ) -> Optional[Dict]:
        """
        Detect if a market has "flipped" (crossed 0.5) recently.

        Args:
            market_id: Market to check
            lookback_seconds: How far back to look
            threshold: Minimum total price change to consider a flip

        Returns:
            Dict with flip info if detected, None otherwise
        """
        if market_id not in self.price_history:
            return None

        history = self.price_history[market_id]
        if len(history) < 2:
            return None

        current_ts, current_price = history[-1]
        cutoff_ts = current_ts - lookback_seconds

        # Find prices in lookback window
        window = [(ts, p) for ts, p in history if ts >= cutoff_ts]
        if len(window) < 2:
            return None

        first_price = window[0][1]
        last_price = window[-1][1]

        # Check for flip
        crossed_half = (first_price < 0.5 and last_price > 0.5) or (
            first_price > 0.5 and last_price < 0.5
        )
        large_move = abs(last_price - first_price) >= threshold

        if crossed_half or large_move:
            return {
                "market_id": market_id,
                "first_price": first_price,
                "last_price": last_price,
                "delta": last_price - first_price,
                "crossed_half": crossed_half,
                "duration_seconds": window[-1][0] - window[0][0],
            }

        return None



