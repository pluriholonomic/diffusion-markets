"""
Canonical Trading Unit System

This module defines the canonical unit conventions for the trading system.
All backtest, hybrid, and live trading code MUST use these types to ensure
consistent position accounting and PnL calculations.

Key conventions:
- stake_usd: Dollar amount risked/allocated (what you pay to enter)
- entry_side_price: Price of the YES or NO side at entry (0-1 range)
- contracts: Number of contracts = stake_usd / entry_side_price
- side: 'yes' or 'no' - which outcome you bet on

PnL Calculation:
- For YES bets: contracts pay out $1 if YES wins, $0 otherwise
- For NO bets: contracts pay out $1 if NO wins, $0 otherwise
- Unrealized PnL uses mark-to-market at current side price
- Resolution PnL is computed at outcome (1.0 for win, 0.0 for loss)

Example:
    # Buy $100 of YES at 0.40 price
    trade = TradeUnits(stake_usd=100, entry_side_price=0.40, side='yes')
    # contracts = 100 / 0.40 = 250 contracts
    
    # If YES wins: payout = 250 * 1.0 = $250, PnL = $250 - $100 = $150
    # If NO wins: payout = 250 * 0.0 = $0, PnL = $0 - $100 = -$100
    
    # At current price 0.60:
    # mtm_value = 250 * 0.60 = $150
    # unrealized_pnl = $150 - $100 = $50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, Dict, Any
import math


Side = Literal['yes', 'no']


@dataclass
class TradeUnits:
    """
    Canonical trading unit conventions for a single trade.
    
    All trading code should use this class to ensure consistent
    position accounting across backtest, hybrid, and live modes.
    """
    stake_usd: float        # USD amount risked/allocated
    entry_side_price: float # Price of YES/NO side at entry (0-1)
    side: Side              # 'yes' or 'no'
    
    # Optional metadata
    market_id: str = ""
    entry_time: Optional[datetime] = None
    strategy: str = ""
    
    def __post_init__(self):
        """Validate inputs."""
        if self.stake_usd < 0:
            raise ValueError(f"stake_usd must be non-negative, got {self.stake_usd}")
        if not (0 < self.entry_side_price <= 1):
            raise ValueError(f"entry_side_price must be in (0, 1], got {self.entry_side_price}")
        if self.side not in ('yes', 'no'):
            raise ValueError(f"side must be 'yes' or 'no', got {self.side}")
    
    @property
    def contracts(self) -> float:
        """Number of contracts purchased."""
        return self.stake_usd / self.entry_side_price
    
    def mtm_value(self, current_side_price: float) -> float:
        """
        Mark-to-market value at current side price.
        
        This is what you could sell the position for at the current price.
        """
        return self.contracts * current_side_price
    
    def unrealized_pnl(self, current_side_price: float) -> float:
        """
        Unrealized PnL at current side price.
        
        This is the difference between current value and what you paid.
        """
        return self.mtm_value(current_side_price) - self.stake_usd
    
    def unrealized_return_pct(self, current_side_price: float) -> float:
        """Unrealized return as a percentage of stake."""
        if self.stake_usd == 0:
            return 0.0
        return (self.unrealized_pnl(current_side_price) / self.stake_usd) * 100
    
    def resolution_pnl(self, outcome: int) -> float:
        """
        PnL at resolution.
        
        Args:
            outcome: 1 if YES wins, 0 if NO wins
            
        Returns:
            Realized PnL in USD
        """
        # Payout per contract: 1.0 if our side wins, 0.0 if it loses
        if self.side == 'yes':
            payout_per_contract = 1.0 if outcome == 1 else 0.0
        else:
            payout_per_contract = 1.0 if outcome == 0 else 0.0
        
        payout = self.contracts * payout_per_contract
        return payout - self.stake_usd
    
    def exit_pnl(self, exit_side_price: float, exit_cost_rate: float = 0.0) -> float:
        """
        PnL if exiting at a given price.
        
        Args:
            exit_side_price: Price to exit at
            exit_cost_rate: Transaction cost as a fraction of exit value
            
        Returns:
            Realized PnL after costs
        """
        exit_value = self.contracts * exit_side_price
        exit_cost = exit_value * exit_cost_rate
        return exit_value - self.stake_usd - exit_cost
    
    @classmethod
    def from_market_order(
        cls,
        stake_usd: float,
        market_yes_price: float,
        side: Side,
        slippage_pct: float = 0.0,
        **kwargs,
    ) -> "TradeUnits":
        """
        Create TradeUnits from a market order specification.
        
        Args:
            stake_usd: Amount to risk in USD
            market_yes_price: Current YES price (0-1)
            side: 'yes' or 'no'
            slippage_pct: Slippage as percentage (e.g., 2.0 for 2%)
            **kwargs: Additional metadata
            
        Returns:
            TradeUnits with slippage-adjusted entry price
        """
        slippage = slippage_pct / 100
        
        if side == 'yes':
            # Buying YES: pay more than market
            entry_price = min(0.99, market_yes_price * (1 + slippage))
        else:
            # Buying NO: pay more than market NO price
            market_no_price = 1 - market_yes_price
            entry_price = min(0.99, market_no_price * (1 + slippage))
        
        return cls(
            stake_usd=stake_usd,
            entry_side_price=entry_price,
            side=side,
            **kwargs,
        )


@dataclass
class PositionUnits:
    """
    Tracks an open position with proper accounting for adds/reduces.
    
    Supports:
    - Adding to positions (weighted average entry)
    - Reducing positions (realize PnL on closed portion)
    - Multiple partial fills
    """
    market_id: str
    side: Side
    strategy: str = ""
    
    # Position state
    total_contracts: float = 0.0
    avg_entry_price: float = 0.0
    total_stake_usd: float = 0.0
    
    # Realized PnL from partial closes
    realized_pnl: float = 0.0
    
    # Tracking
    open_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    @property
    def is_open(self) -> bool:
        return abs(self.total_contracts) > 1e-9
    
    @property
    def mtm_value(self) -> float:
        """Mark-to-market value at average entry (for display only)."""
        return self.total_contracts * self.avg_entry_price
    
    def unrealized_pnl(self, current_side_price: float) -> float:
        """Unrealized PnL at current price."""
        if not self.is_open:
            return 0.0
        current_value = self.total_contracts * current_side_price
        return current_value - self.total_stake_usd
    
    def total_pnl(self, current_side_price: float) -> float:
        """Total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(current_side_price)
    
    def add_contracts(
        self,
        contracts: float,
        exec_price: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Add contracts to the position.
        
        Args:
            contracts: Number of contracts to add
            exec_price: Execution price per contract
            timestamp: Execution timestamp
            
        Returns:
            Cost of this addition in USD
        """
        if contracts <= 0:
            raise ValueError(f"contracts must be positive, got {contracts}")
        if not (0 < exec_price <= 1):
            raise ValueError(f"exec_price must be in (0, 1], got {exec_price}")
        
        cost = contracts * exec_price
        
        # Update weighted average entry price
        new_total = self.total_contracts + contracts
        if new_total > 0:
            self.avg_entry_price = (
                (self.total_contracts * self.avg_entry_price) + (contracts * exec_price)
            ) / new_total
        
        self.total_contracts = new_total
        self.total_stake_usd += cost
        self.last_update = timestamp or datetime.utcnow()
        
        if self.open_time is None:
            self.open_time = self.last_update
        
        return cost
    
    def reduce_contracts(
        self,
        contracts: float,
        exec_price: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Reduce the position (partial close).
        
        Args:
            contracts: Number of contracts to close
            exec_price: Execution price per contract
            timestamp: Execution timestamp
            
        Returns:
            Realized PnL on the closed portion
        """
        if contracts <= 0:
            raise ValueError(f"contracts must be positive, got {contracts}")
        if contracts > self.total_contracts + 1e-9:
            raise ValueError(f"Cannot close {contracts} contracts, only have {self.total_contracts}")
        
        # Realize PnL on closed portion
        # We bought at avg_entry_price, selling at exec_price
        closed_contracts = min(contracts, self.total_contracts)
        pnl = closed_contracts * (exec_price - self.avg_entry_price)
        
        # Update position
        self.realized_pnl += pnl
        self.total_contracts -= closed_contracts
        
        # Proportionally reduce stake
        if self.total_stake_usd > 0:
            close_fraction = closed_contracts / (self.total_contracts + closed_contracts)
            self.total_stake_usd *= (1 - close_fraction)
        
        self.last_update = timestamp or datetime.utcnow()
        
        return pnl
    
    def close_at_resolution(self, outcome: int) -> float:
        """
        Close position at market resolution.
        
        Args:
            outcome: 1 if YES wins, 0 if NO wins
            
        Returns:
            Final realized PnL
        """
        if not self.is_open:
            return 0.0
        
        # Payout per contract
        if self.side == 'yes':
            payout_per_contract = 1.0 if outcome == 1 else 0.0
        else:
            payout_per_contract = 1.0 if outcome == 0 else 0.0
        
        payout = self.total_contracts * payout_per_contract
        pnl = payout - self.total_stake_usd
        
        self.realized_pnl += pnl
        self.total_contracts = 0.0
        self.total_stake_usd = 0.0
        self.last_update = datetime.utcnow()
        
        return pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'market_id': self.market_id,
            'side': self.side,
            'strategy': self.strategy,
            'total_contracts': self.total_contracts,
            'avg_entry_price': self.avg_entry_price,
            'total_stake_usd': self.total_stake_usd,
            'realized_pnl': self.realized_pnl,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionUnits":
        """Deserialize from dictionary."""
        pos = cls(
            market_id=data['market_id'],
            side=data['side'],
            strategy=data.get('strategy', ''),
            total_contracts=data.get('total_contracts', 0.0),
            avg_entry_price=data.get('avg_entry_price', 0.0),
            total_stake_usd=data.get('total_stake_usd', 0.0),
            realized_pnl=data.get('realized_pnl', 0.0),
        )
        if data.get('open_time'):
            pos.open_time = datetime.fromisoformat(data['open_time'])
        if data.get('last_update'):
            pos.last_update = datetime.fromisoformat(data['last_update'])
        return pos


def compute_kelly_stake(
    probability: float,
    market_price: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.10,
) -> float:
    """
    Compute Kelly-optimal stake size.
    
    Args:
        probability: Estimated true probability of YES
        market_price: Current YES market price
        bankroll: Total available capital
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        max_stake_pct: Maximum stake as percentage of bankroll
        
    Returns:
        Optimal stake in USD
    """
    # Edge calculation
    if probability > market_price:
        # Bet YES
        edge = probability - market_price
        odds = (1 - market_price) / market_price  # Payout odds
    else:
        # Bet NO
        edge = market_price - probability
        odds = market_price / (1 - market_price)  # Payout odds
    
    if edge <= 0:
        return 0.0
    
    # Kelly formula: f* = edge / odds
    # For prediction markets: f* = (p * odds - (1-p)) / odds = (p * (1-q)/q - (1-p)) / ((1-q)/q)
    # Simplifies to: f* = (p - q) / (1 - q) for YES bet at price q
    if probability > market_price:
        kelly = (probability - market_price) / (1 - market_price)
    else:
        kelly = (market_price - probability) / market_price
    
    # Apply fraction and cap
    stake = bankroll * kelly * kelly_fraction
    max_stake = bankroll * max_stake_pct
    
    return min(stake, max_stake)
