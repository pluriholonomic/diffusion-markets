"""
Unified Execution Simulator

Provides realistic order execution simulation for both backtest and hybrid modes.
Handles:
- Fill probability (not all orders fill)
- Slippage (price impact)
- Partial fills
- Latency simulation
- Transaction costs (fees)

This ensures backtest results are comparable to hybrid/live results.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.models import Order, OrderStatus, Side

logger = logging.getLogger(__name__)


class FillMode(Enum):
    """Fill mode for execution simulator."""
    INSTANT = "instant"          # All orders fill instantly at mid
    PROBABILISTIC = "probabilistic"  # Some orders don't fill
    ORDERBOOK = "orderbook"      # Use orderbook depth for fills


@dataclass
class ExecutionSimulatorConfig:
    """Configuration for execution simulator."""
    # Fill probability
    fill_mode: FillMode = FillMode.PROBABILISTIC
    base_fill_probability: float = 0.85  # 85% of orders fill
    
    # Slippage model
    base_spread_pct: float = 0.5   # 0.5% base spread
    size_impact_per_1k: float = 0.1  # 0.1% additional impact per $1000
    volatility_multiplier: float = 2.0  # Multiply impact by recent volatility
    
    # Partial fills
    enable_partial_fills: bool = True
    partial_fill_rate: float = 0.3   # 30% of fills are partial
    min_fill_fraction: float = 0.5   # Minimum 50% fill when partial
    
    # Latency
    latency_ms: int = 100  # 100ms base latency
    latency_jitter_ms: int = 50  # +/- 50ms jitter
    
    # Fees
    maker_fee_pct: float = 0.0   # 0% maker fee (prediction markets)
    taker_fee_pct: float = 0.0   # 0% taker fee
    
    # Market conditions
    reject_if_spread_exceeds: float = 0.10  # Reject if spread > 10%


@dataclass
class OrderBook:
    """Simple orderbook representation."""
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask or 0.5
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0


@dataclass
class ExecutionResult:
    """Result of simulating order execution."""
    order: Order
    status: OrderStatus
    fill_price: float
    fill_qty: float  # Fraction filled (0-1)
    slippage: float  # Price slippage from mid
    fees: float
    latency_ms: int
    rejection_reason: Optional[str] = None
    
    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
    
    @property
    def fill_value(self) -> float:
        """Total value of fill."""
        return self.order.size * self.fill_qty * self.fill_price


class ExecutionSimulator:
    """
    Unified execution simulator for backtest and hybrid modes.
    
    Provides realistic order fills with:
    - Probabilistic fill rate
    - Size-dependent slippage
    - Partial fills
    - Latency simulation
    
    Usage:
        sim = ExecutionSimulator()
        result = sim.simulate_fill(order, orderbook, volatility)
        
        if result.is_filled:
            # Process fill
            position.add(result.fill_qty * order.size, result.fill_price)
    """
    
    def __init__(self, config: Optional[ExecutionSimulatorConfig] = None):
        self.config = config or ExecutionSimulatorConfig()
        
        # Statistics
        self.stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_partial": 0,
            "orders_rejected": 0,
            "total_slippage": 0.0,
            "total_fees": 0.0,
        }
    
    def simulate_fill(
        self,
        order: Order,
        orderbook: Optional[OrderBook] = None,
        recent_volatility: float = 0.02,
        current_mid_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Simulate order execution.
        
        Args:
            order: Order to simulate
            orderbook: Optional orderbook for realistic fills
            recent_volatility: Recent price volatility (for impact scaling)
            current_mid_price: Current mid price (if no orderbook)
            
        Returns:
            ExecutionResult with fill details
        """
        self.stats["orders_submitted"] += 1
        
        # Get mid price
        if orderbook:
            mid_price = orderbook.mid_price
            spread = orderbook.spread
        else:
            mid_price = current_mid_price or order.price
            spread = self.config.base_spread_pct / 100
        
        # Check spread condition
        if spread > self.config.reject_if_spread_exceeds:
            self.stats["orders_rejected"] += 1
            return ExecutionResult(
                order=order,
                status=OrderStatus.REJECTED,
                fill_price=0.0,
                fill_qty=0.0,
                slippage=0.0,
                fees=0.0,
                latency_ms=0,
                rejection_reason="spread_too_wide",
            )
        
        # Simulate latency
        latency = self.config.latency_ms + random.randint(
            -self.config.latency_jitter_ms, 
            self.config.latency_jitter_ms
        )
        
        # Check fill probability
        if self.config.fill_mode == FillMode.PROBABILISTIC:
            fill_prob = self._compute_fill_probability(order, spread)
            if random.random() > fill_prob:
                self.stats["orders_rejected"] += 1
                return ExecutionResult(
                    order=order,
                    status=OrderStatus.REJECTED,
                    fill_price=0.0,
                    fill_qty=0.0,
                    slippage=0.0,
                    fees=0.0,
                    latency_ms=latency,
                    rejection_reason="no_fill",
                )
        
        # Compute slippage
        slippage = self._compute_slippage(order, orderbook, recent_volatility)
        
        # Apply slippage to get fill price
        if order.side == Side.YES:
            # Buying: pay more than mid
            fill_price = min(0.99, mid_price + slippage)
        else:
            # Selling (buying NO): pay more than mid for NO
            fill_price = max(0.01, mid_price - slippage)
        
        # Determine fill quantity
        if self.config.enable_partial_fills and random.random() < self.config.partial_fill_rate:
            fill_qty = random.uniform(self.config.min_fill_fraction, 1.0)
            status = OrderStatus.PARTIAL
            self.stats["orders_partial"] += 1
        else:
            fill_qty = 1.0
            status = OrderStatus.FILLED
            self.stats["orders_filled"] += 1
        
        # Compute fees
        fees = self._compute_fees(order.size * fill_qty, fill_price)
        
        self.stats["total_slippage"] += abs(slippage)
        self.stats["total_fees"] += fees
        
        return ExecutionResult(
            order=order,
            status=status,
            fill_price=fill_price,
            fill_qty=fill_qty,
            slippage=slippage,
            fees=fees,
            latency_ms=latency,
        )
    
    def _compute_fill_probability(self, order: Order, spread: float) -> float:
        """Compute probability of order filling."""
        prob = self.config.base_fill_probability
        
        # Lower probability for larger orders
        size_penalty = (order.size / 1000) * 0.05
        prob -= size_penalty
        
        # Lower probability for wider spreads
        spread_penalty = spread * 2
        prob -= spread_penalty
        
        return max(0.1, min(1.0, prob))
    
    def _compute_slippage(
        self,
        order: Order,
        orderbook: Optional[OrderBook],
        volatility: float,
    ) -> float:
        """Compute price slippage for an order."""
        # Base spread
        base_slippage = self.config.base_spread_pct / 100 / 2  # Half spread
        
        # Size impact
        size_impact = (order.size / 1000) * (self.config.size_impact_per_1k / 100)
        
        # Volatility impact
        vol_impact = volatility * self.config.volatility_multiplier
        
        # Total slippage
        total_slippage = base_slippage + size_impact + vol_impact
        
        # Add some randomness
        total_slippage *= (1 + random.uniform(-0.2, 0.2))
        
        return total_slippage
    
    def _compute_fees(self, size: float, price: float) -> float:
        """Compute transaction fees."""
        value = size * price
        fee_pct = self.config.taker_fee_pct / 100
        return value * fee_pct
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        total = self.stats["orders_submitted"]
        if total == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            "fill_rate": (self.stats["orders_filled"] + self.stats["orders_partial"]) / total,
            "partial_rate": self.stats["orders_partial"] / total,
            "avg_slippage": self.stats["total_slippage"] / max(1, self.stats["orders_filled"]),
        }
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_partial": 0,
            "orders_rejected": 0,
            "total_slippage": 0.0,
            "total_fees": 0.0,
        }


class HybridExecutionSimulator(ExecutionSimulator):
    """
    Execution simulator that uses real orderbook data when available.
    
    Falls back to synthetic orderbook when real data is not available.
    """
    
    def __init__(
        self,
        config: Optional[ExecutionSimulatorConfig] = None,
        use_real_orderbook: bool = True,
    ):
        super().__init__(config)
        self.use_real_orderbook = use_real_orderbook
        self.orderbook_cache: Dict[str, OrderBook] = {}
    
    def update_orderbook(self, market_id: str, orderbook: OrderBook):
        """Update cached orderbook for a market."""
        self.orderbook_cache[market_id] = orderbook
    
    def simulate_fill_with_cache(
        self,
        order: Order,
        recent_volatility: float = 0.02,
    ) -> ExecutionResult:
        """Simulate fill using cached orderbook if available."""
        orderbook = self.orderbook_cache.get(order.market_id)
        return self.simulate_fill(order, orderbook, recent_volatility)


def create_execution_simulator(
    mode: str = "backtest",
    fill_probability: float = 0.85,
    slippage_pct: float = 0.5,
) -> ExecutionSimulator:
    """
    Factory function to create an execution simulator.
    
    Args:
        mode: 'backtest', 'hybrid', or 'live'
        fill_probability: Base fill probability
        slippage_pct: Base slippage in percent
        
    Returns:
        Configured ExecutionSimulator
    """
    if mode == "live":
        # Live mode: assume all orders fill (execution handled by exchange)
        config = ExecutionSimulatorConfig(
            fill_mode=FillMode.INSTANT,
            base_fill_probability=1.0,
            base_spread_pct=0.0,
        )
    elif mode == "hybrid":
        # Hybrid mode: use real orderbook when available
        config = ExecutionSimulatorConfig(
            fill_mode=FillMode.PROBABILISTIC,
            base_fill_probability=fill_probability,
            base_spread_pct=slippage_pct,
            enable_partial_fills=True,
        )
        return HybridExecutionSimulator(config, use_real_orderbook=True)
    else:
        # Backtest mode: synthetic execution
        config = ExecutionSimulatorConfig(
            fill_mode=FillMode.PROBABILISTIC,
            base_fill_probability=fill_probability,
            base_spread_pct=slippage_pct,
            enable_partial_fills=True,
        )
    
    return ExecutionSimulator(config)
