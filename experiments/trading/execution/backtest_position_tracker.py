"""
Backtest Position Tracker

Tracks positions during backtesting with proper deferred outcome resolution.
Key features:
- No lookahead bias: PnL is mark-to-market until resolution
- Resolution only at actual resolution time
- Proper accounting using canonical units

This replaces the instant-resolution logic in SimulatedMarketClient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ..core.units import TradeUnits, PositionUnits

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Snapshot of position state at a point in time."""
    timestamp: datetime
    unrealized_pnl: float
    market_price: float
    contracts: float
    
    
@dataclass
class ClosedPosition:
    """Record of a closed position."""
    position: PositionUnits
    close_time: datetime
    close_reason: str  # 'resolution', 'stop_loss', 'profit_take', 'manual'
    close_price: float
    realized_pnl: float


class BacktestPositionTracker:
    """
    Track positions during backtesting with deferred outcome resolution.
    
    Key principle: We should NOT know the outcome until the market actually
    resolves. Until then, PnL is based on current market price.
    
    Usage:
        tracker = BacktestPositionTracker()
        
        # Open position
        tracker.open_position("MKT1", "yes", stake=100, entry_price=0.40, ...)
        
        # Update prices (during simulation)
        tracker.update_prices(timestamp, {"MKT1": 0.55})
        
        # Resolution (only when market actually resolves)
        tracker.resolve_market("MKT1", outcome=1, resolution_time=...)
        
        # Get PnL
        total_pnl = tracker.get_total_pnl()
    """
    
    def __init__(self):
        # Open positions by market_id
        self.open_positions: Dict[str, PositionUnits] = {}
        
        # Closed positions
        self.closed_positions: List[ClosedPosition] = []
        
        # Price history for each position (for analysis)
        self.mtm_history: Dict[str, List[PositionSnapshot]] = defaultdict(list)
        
        # Current prices
        self.current_prices: Dict[str, float] = {}
        self.current_time: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            "positions_opened": 0,
            "positions_closed": 0,
            "total_realized_pnl": 0.0,
            "resolutions": 0,
            "profit_takes": 0,
            "stop_losses": 0,
        }
    
    def open_position(
        self,
        market_id: str,
        side: str,
        stake_usd: float,
        entry_price: float,
        strategy: str = "",
        timestamp: Optional[datetime] = None,
    ) -> PositionUnits:
        """
        Open a new position.
        
        Args:
            market_id: Market identifier
            side: 'yes' or 'no'
            stake_usd: USD amount to risk
            entry_price: Entry price (side price)
            strategy: Strategy name
            timestamp: Entry time
            
        Returns:
            PositionUnits object
        """
        # Calculate contracts
        contracts = stake_usd / entry_price
        
        if market_id in self.open_positions:
            # Add to existing position
            pos = self.open_positions[market_id]
            pos.add_contracts(contracts, entry_price, timestamp)
        else:
            # New position
            pos = PositionUnits(
                market_id=market_id,
                side=side,
                strategy=strategy,
            )
            pos.add_contracts(contracts, entry_price, timestamp)
            self.open_positions[market_id] = pos
        
        self.stats["positions_opened"] += 1
        
        logger.debug(
            f"Opened position: {side.upper()} {market_id} "
            f"${stake_usd:.2f} @ {entry_price:.4f} ({contracts:.2f} contracts)"
        )
        
        return pos
    
    def close_position(
        self,
        market_id: str,
        exit_price: float,
        reason: str = "manual",
        timestamp: Optional[datetime] = None,
    ) -> Optional[ClosedPosition]:
        """
        Close a position at a given price.
        
        Args:
            market_id: Market to close
            exit_price: Exit price
            reason: Close reason ('stop_loss', 'profit_take', 'manual', etc.)
            timestamp: Close time
            
        Returns:
            ClosedPosition record, or None if no position
        """
        if market_id not in self.open_positions:
            return None
        
        pos = self.open_positions.pop(market_id)
        
        # Calculate PnL by closing all contracts
        pnl = pos.reduce_contracts(pos.total_contracts, exit_price, timestamp)
        
        closed = ClosedPosition(
            position=pos,
            close_time=timestamp or datetime.utcnow(),
            close_reason=reason,
            close_price=exit_price,
            realized_pnl=pnl,
        )
        
        self.closed_positions.append(closed)
        self.stats["positions_closed"] += 1
        self.stats["total_realized_pnl"] += pnl
        
        if reason == "profit_take":
            self.stats["profit_takes"] += 1
        elif reason == "stop_loss":
            self.stats["stop_losses"] += 1
        
        logger.debug(
            f"Closed position: {market_id} ({reason}) "
            f"@ {exit_price:.4f}, PnL=${pnl:.2f}"
        )
        
        return closed
    
    def resolve_market(
        self,
        market_id: str,
        outcome: int,
        resolution_time: Optional[datetime] = None,
    ) -> Optional[ClosedPosition]:
        """
        Close a position due to market resolution.
        
        This is the ONLY way outcomes should affect positions in backtesting.
        Called when the market actually resolves, not when we trade.
        
        Args:
            market_id: Market that resolved
            outcome: 1 if YES won, 0 if NO won
            resolution_time: Time of resolution
            
        Returns:
            ClosedPosition record, or None if no position
        """
        if market_id not in self.open_positions:
            return None
        
        pos = self.open_positions.pop(market_id)
        
        # Use close_at_resolution for proper PnL calculation
        pnl = pos.close_at_resolution(outcome)
        
        closed = ClosedPosition(
            position=pos,
            close_time=resolution_time or datetime.utcnow(),
            close_reason="resolution",
            close_price=float(outcome),
            realized_pnl=pnl,
        )
        
        self.closed_positions.append(closed)
        self.stats["positions_closed"] += 1
        self.stats["total_realized_pnl"] += pnl
        self.stats["resolutions"] += 1
        
        logger.debug(
            f"Resolved position: {market_id} (outcome={outcome}) "
            f"PnL=${pnl:.2f}"
        )
        
        return closed
    
    def update_prices(
        self,
        timestamp: datetime,
        prices: Dict[str, float],
    ) -> float:
        """
        Update current prices and compute unrealized PnL.
        
        This should be called at each simulation timestep.
        
        Args:
            timestamp: Current simulation time
            prices: Market prices {market_id: yes_price}
            
        Returns:
            Total unrealized PnL across all positions
        """
        self.current_time = timestamp
        self.current_prices.update(prices)
        
        total_unrealized = 0.0
        
        for market_id, pos in self.open_positions.items():
            if market_id in prices:
                price = prices[market_id]
                
                # For NO positions, need to convert YES price to NO price
                if pos.side == 'no':
                    side_price = 1 - price
                else:
                    side_price = price
                
                unrealized = pos.unrealized_pnl(side_price)
                total_unrealized += unrealized
                
                # Record snapshot
                self.mtm_history[market_id].append(PositionSnapshot(
                    timestamp=timestamp,
                    unrealized_pnl=unrealized,
                    market_price=price,
                    contracts=pos.total_contracts,
                ))
        
        return total_unrealized
    
    def check_exits(
        self,
        profit_take_pct: float = 20.0,
        stop_loss_pct: float = 20.0,
    ) -> List[Tuple[str, str]]:
        """
        Check positions for exit conditions.
        
        Args:
            profit_take_pct: Take profit threshold (%)
            stop_loss_pct: Stop loss threshold (%)
            
        Returns:
            List of (market_id, reason) tuples for positions to close
        """
        exits = []
        
        for market_id, pos in self.open_positions.items():
            if market_id not in self.current_prices:
                continue
            
            price = self.current_prices[market_id]
            if pos.side == 'no':
                side_price = 1 - price
            else:
                side_price = price
            
            # Calculate return percentage
            if pos.total_stake_usd > 0:
                unrealized = pos.unrealized_pnl(side_price)
                return_pct = (unrealized / pos.total_stake_usd) * 100
            else:
                return_pct = 0
            
            # Check thresholds
            if return_pct >= profit_take_pct:
                exits.append((market_id, "profit_take"))
            elif return_pct <= -stop_loss_pct:
                exits.append((market_id, "stop_loss"))
        
        return exits
    
    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        realized = self.stats["total_realized_pnl"]
        unrealized = sum(
            pos.unrealized_pnl(
                self.current_prices.get(pos.market_id, pos.avg_entry_price)
            )
            for pos in self.open_positions.values()
        )
        return realized + unrealized
    
    def get_realized_pnl(self) -> float:
        """Get total realized PnL."""
        return self.stats["total_realized_pnl"]
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(
            pos.unrealized_pnl(
                self.current_prices.get(pos.market_id, pos.avg_entry_price)
            )
            for pos in self.open_positions.values()
        )
    
    def get_open_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.open_positions)
    
    def get_summary(self) -> Dict:
        """Get tracker summary."""
        return {
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "realized_pnl": self.stats["total_realized_pnl"],
            "unrealized_pnl": self.get_unrealized_pnl(),
            "total_pnl": self.get_total_pnl(),
            "resolutions": self.stats["resolutions"],
            "profit_takes": self.stats["profit_takes"],
            "stop_losses": self.stats["stop_losses"],
        }
    
    def get_closed_trades(self) -> List[Dict]:
        """Get list of closed trades for analysis."""
        return [
            {
                "market_id": c.position.market_id,
                "side": c.position.side,
                "strategy": c.position.strategy,
                "entry_price": c.position.avg_entry_price,
                "exit_price": c.close_price,
                "stake_usd": c.position.total_stake_usd,
                "contracts": c.position.total_contracts,
                "pnl": c.realized_pnl,
                "close_reason": c.close_reason,
                "close_time": c.close_time.isoformat() if c.close_time else None,
            }
            for c in self.closed_positions
        ]
    
    def reset(self):
        """Reset tracker state."""
        self.open_positions.clear()
        self.closed_positions.clear()
        self.mtm_history.clear()
        self.current_prices.clear()
        self.current_time = None
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
