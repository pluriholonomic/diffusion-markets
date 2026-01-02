"""
Core data models for the trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid


class Platform(Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class Side(Enum):
    YES = "yes"
    NO = "no"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Market:
    """Represents a prediction market."""
    market_id: str
    platform: Platform
    question: str
    category: str
    current_yes_price: float
    current_no_price: float
    volume: float
    liquidity: float
    close_time: Optional[datetime] = None
    outcome: Optional[int] = None  # 1 for YES, 0 for NO, None if unresolved
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal from a strategy."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    platform: Platform = Platform.POLYMARKET
    market_id: str = ""
    strategy: str = ""
    side: Side = Side.YES
    edge: float = 0.0  # Expected edge
    confidence: float = 0.0  # Signal confidence (0-1)
    kelly_fraction: float = 0.0  # Recommended position size
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'platform': self.platform.value,
            'market_id': self.market_id,
            'strategy': self.strategy,
            'side': self.side.value,
            'edge': self.edge,
            'confidence': self.confidence,
            'kelly_fraction': self.kelly_fraction,
            'metadata': self.metadata,
        }


@dataclass
class Order:
    """Order to be executed."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str = ""
    platform: Platform = Platform.POLYMARKET
    market_id: str = ""
    side: Side = Side.YES
    order_type: OrderType = OrderType.LIMIT
    size: float = 0.0  # In dollars
    price: float = 0.0  # Limit price
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    platform_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'signal_id': self.signal_id,
            'platform': self.platform.value,
            'market_id': self.market_id,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'size': self.size,
            'price': self.price,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'platform_order_id': self.platform_order_id,
            'metadata': self.metadata,
        }


@dataclass
class Fill:
    """Represents a trade execution."""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    platform: Platform = Platform.POLYMARKET
    market_id: str = ""
    side: Side = Side.YES
    size: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    platform_fill_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def net_cost(self) -> float:
        """Total cost including fees."""
        return self.size * self.price + self.fee
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'platform': self.platform.value,
            'market_id': self.market_id,
            'side': self.side.value,
            'size': self.size,
            'price': self.price,
            'fee': self.fee,
            'net_cost': self.net_cost,
            'timestamp': self.timestamp.isoformat(),
            'platform_fill_id': self.platform_fill_id,
            'metadata': self.metadata,
        }


@dataclass
class Position:
    """Current position in a market."""
    platform: Platform
    market_id: str
    side: Side
    size: float  # Number of shares
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.size * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.size * self.avg_entry_price


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_pct: float = 0.10  # Max position as % of bankroll
    max_daily_loss_pct: float = 0.20  # Daily loss stop
    max_drawdown_pct: float = 0.30  # Total drawdown stop
    kelly_fraction: float = 0.25  # Kelly scaling factor
    min_edge: float = 0.05  # Minimum edge to trade
    min_liquidity: float = 1000  # Minimum market liquidity
    max_concentration: float = 0.30  # Max exposure to single category
