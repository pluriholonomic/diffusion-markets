"""
Risk Management for Trading System

Handles:
- Pre-trade risk checks
- Position limits
- Daily/drawdown stops
- Concentration limits
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.models import (
    Platform, Order, Fill, Position, Signal, RiskLimits
)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_balance: float
    current_balance: float
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    
    @property
    def daily_return(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return self.pnl / self.starting_balance
    
    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades


@dataclass
class RiskState:
    """Current risk state."""
    bankroll: float
    peak_bankroll: float
    daily_stats: DailyStats
    positions: Dict[str, Position] = field(default_factory=dict)
    category_exposure: Dict[str, float] = field(default_factory=dict)
    pending_orders: Dict[str, Order] = field(default_factory=dict)
    
    @property
    def current_drawdown(self) -> float:
        if self.peak_bankroll == 0:
            return 0.0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll
    
    @property
    def total_exposure(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())


class RiskManager:
    """
    Manages risk for the trading system.
    
    Performs:
    - Pre-trade risk checks
    - Position size validation
    - Daily loss limits
    - Drawdown monitoring
    - Concentration limits
    """
    
    def __init__(
        self,
        initial_bankroll: float,
        limits: Optional[RiskLimits] = None,
    ):
        self.limits = limits or RiskLimits()
        
        today = date.today()
        self.state = RiskState(
            bankroll=initial_bankroll,
            peak_bankroll=initial_bankroll,
            daily_stats=DailyStats(
                date=today,
                starting_balance=initial_bankroll,
                current_balance=initial_bankroll,
            ),
        )
        
        self._blocked_markets: Set[str] = set()
        self._trading_halted: bool = False
        self._halt_reason: str = ""
    
    def check_pre_trade(self, signal: Signal, order: Order) -> tuple[bool, str]:
        """
        Perform pre-trade risk checks.
        
        Returns (approved, reason) tuple.
        """
        # Check if trading is halted
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"
        
        # Check daily loss limit
        if not self._check_daily_loss():
            return False, "Daily loss limit exceeded"
        
        # Check drawdown limit
        if not self._check_drawdown():
            return False, "Drawdown limit exceeded"
        
        # Check position size
        if not self._check_position_size(order):
            return False, "Position size exceeds limit"
        
        # Check concentration
        category = signal.metadata.get('category', 'unknown')
        if not self._check_concentration(category, order.size):
            return False, f"Concentration limit exceeded for {category}"
        
        # Check if market is blocked
        if order.market_id in self._blocked_markets:
            return False, "Market is blocked"
        
        # Check minimum edge
        if signal.edge < self.limits.min_edge:
            return False, f"Edge {signal.edge:.2%} below minimum {self.limits.min_edge:.2%}"
        
        # Check liquidity
        liquidity = signal.metadata.get('liquidity', 0)
        if liquidity < self.limits.min_liquidity:
            return False, f"Liquidity ${liquidity:.0f} below minimum ${self.limits.min_liquidity:.0f}"
        
        return True, "Approved"
    
    def _check_daily_loss(self) -> bool:
        """Check if daily loss limit is breached."""
        daily_return = self.state.daily_stats.daily_return
        return daily_return > -self.limits.max_daily_loss_pct
    
    def _check_drawdown(self) -> bool:
        """Check if drawdown limit is breached."""
        return self.state.current_drawdown < self.limits.max_drawdown_pct
    
    def _check_position_size(self, order: Order) -> bool:
        """Check if position size is within limits."""
        max_size = self.state.bankroll * self.limits.max_position_pct
        return order.size <= max_size
    
    def _check_concentration(self, category: str, additional_size: float) -> bool:
        """Check if category concentration is within limits."""
        current_exposure = self.state.category_exposure.get(category, 0)
        new_exposure = current_exposure + additional_size
        max_exposure = self.state.bankroll * self.limits.max_concentration
        return new_exposure <= max_exposure
    
    def record_fill(self, fill: Fill, category: str = "unknown"):
        """Record a trade fill and update state."""
        # Update category exposure
        self.state.category_exposure[category] = (
            self.state.category_exposure.get(category, 0) + fill.net_cost
        )
        
        # Update daily stats
        self.state.daily_stats.trades += 1
    
    def record_fill_from_order(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
        category: str = "unknown",
    ):
        """
        Record a fill from order details (used when Fill object not available).
        
        Args:
            market_id: Market identifier
            side: 'yes' or 'no'
            size: USD size of the order
            price: Execution price
            category: Market category for concentration tracking
        """
        # Update category exposure
        self.state.category_exposure[category] = (
            self.state.category_exposure.get(category, 0) + size
        )
        
        # Update positions tracking
        if market_id not in self.state.positions:
            from ..utils.models import Position, Platform, Side
            self.state.positions[market_id] = Position(
                platform=Platform.POLYMARKET,  # Default, can be overridden
                market_id=market_id,
                side=Side.YES if side == 'yes' else Side.NO,
                size=size,
                avg_entry_price=price,
                current_price=price,
            )
        else:
            # Add to existing position
            pos = self.state.positions[market_id]
            # Update weighted average entry
            old_size = pos.size
            new_total = old_size + size
            if new_total > 0:
                pos.avg_entry_price = (pos.avg_entry_price * old_size + price * size) / new_total
            pos.size = new_total
            pos.current_price = price
        
        # Update daily stats
        self.state.daily_stats.trades += 1
    
    def sync_positions(self, open_positions: list):
        """
        Sync positions from PositionManager.
        
        Args:
            open_positions: List of position dicts from PositionManager.get_open_positions()
        """
        # Clear existing positions
        self.state.positions.clear()
        self.state.category_exposure.clear()
        
        # Rebuild from PositionManager data
        for pos_data in open_positions:
            market_id = pos_data.get('market_id', '')
            size = pos_data.get('size', 0)
            entry_price = pos_data.get('entry_price', 0)
            side = pos_data.get('side', 'yes')
            
            from ..utils.models import Position, Platform, Side
            self.state.positions[market_id] = Position(
                platform=Platform.POLYMARKET,  # Default
                market_id=market_id,
                side=Side.YES if side == 'yes' else Side.NO,
                size=size,
                avg_entry_price=entry_price,
                current_price=entry_price,
            )
            
            # Update category exposure (default category if not provided)
            category = pos_data.get('category', 'unknown')
            self.state.category_exposure[category] = (
                self.state.category_exposure.get(category, 0) + size
            )
    
    def remove_position(self, market_id: str, category: str = "unknown"):
        """
        Remove a closed position from tracking.
        
        Args:
            market_id: Market identifier
            category: Market category
        """
        if market_id in self.state.positions:
            pos = self.state.positions.pop(market_id)
            
            # Update category exposure
            if category in self.state.category_exposure:
                self.state.category_exposure[category] -= pos.size
                if self.state.category_exposure[category] <= 0:
                    del self.state.category_exposure[category]
    
    def record_pnl(self, pnl: float):
        """Record PnL from a position close."""
        self.state.bankroll += pnl
        self.state.daily_stats.pnl += pnl
        self.state.daily_stats.current_balance = self.state.bankroll
        
        if pnl > 0:
            self.state.daily_stats.wins += 1
        else:
            self.state.daily_stats.losses += 1
        
        # Update peak
        if self.state.bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = self.state.bankroll
        
        # Check for limit breaches
        self._check_limits()
    
    def _check_limits(self):
        """Check if any limits are breached and halt trading if needed."""
        # Check daily loss
        if self.state.daily_stats.daily_return <= -self.limits.max_daily_loss_pct:
            self.halt_trading(f"Daily loss limit ({self.limits.max_daily_loss_pct:.0%}) exceeded")
            return
        
        # Check drawdown
        if self.state.current_drawdown >= self.limits.max_drawdown_pct:
            self.halt_trading(f"Drawdown limit ({self.limits.max_drawdown_pct:.0%}) exceeded")
            return
    
    def halt_trading(self, reason: str):
        """Halt all trading."""
        self._trading_halted = True
        self._halt_reason = reason
    
    def resume_trading(self):
        """Resume trading after halt."""
        self._trading_halted = False
        self._halt_reason = ""
    
    def block_market(self, market_id: str):
        """Block a specific market from trading."""
        self._blocked_markets.add(market_id)
    
    def unblock_market(self, market_id: str):
        """Unblock a market."""
        self._blocked_markets.discard(market_id)
    
    def start_new_day(self):
        """Reset daily statistics for new trading day."""
        today = date.today()
        if self.state.daily_stats.date != today:
            self.state.daily_stats = DailyStats(
                date=today,
                starting_balance=self.state.bankroll,
                current_balance=self.state.bankroll,
            )
            
            # Resume trading if halted due to daily limit
            if "Daily loss" in self._halt_reason:
                self.resume_trading()
    
    def get_status(self) -> Dict:
        """Get current risk status."""
        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "bankroll": self.state.bankroll,
            "peak_bankroll": self.state.peak_bankroll,
            "current_drawdown": f"{self.state.current_drawdown:.2%}",
            "daily_pnl": self.state.daily_stats.pnl,
            "daily_return": f"{self.state.daily_stats.daily_return:.2%}",
            "daily_trades": self.state.daily_stats.trades,
            "daily_win_rate": f"{self.state.daily_stats.win_rate:.1%}",
            "total_exposure": self.state.total_exposure,
            "n_positions": len(self.state.positions),
            "blocked_markets": len(self._blocked_markets),
        }
    
    def get_available_capital(self) -> float:
        """Get capital available for new positions."""
        # Account for existing exposure
        available = self.state.bankroll - self.state.total_exposure
        
        # Reserve some capital
        available *= 0.9  # Keep 10% reserve
        
        return max(0, available)
