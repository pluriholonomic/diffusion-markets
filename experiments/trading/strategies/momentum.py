"""
Momentum Strategies for Prediction Markets

Adapted from backtest scripts for live/hybrid trading.

Strategies:
1. Calibration Momentum - trade WITH persistent miscalibration
2. Trend Following - dual-MA and breakout strategies  
3. Event Momentum - short-term momentum after price jumps
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.models import Platform, Side, Signal, Market, RiskLimits


@dataclass
class MomentumConfig:
    """Configuration for momentum strategies.
    
    Optimized via CMA-ES on historical data.
    """
    # Signal thresholds
    momentum_threshold: float = 0.05  # Min price change for signal
    persistence_window: int = 10  # Lookback for persistence
    min_persistence_days: int = 5  # Min consecutive days
    
    # Trend following
    fast_ma_period: int = 5
    slow_ma_period: int = 20
    breakout_percentile: float = 0.8
    
    # Sizing
    kelly_fraction: float = 0.30
    max_position_pct: float = 0.05
    
    # Risk
    profit_take_pct: float = 40.0
    stop_loss_pct: float = 25.0


class MomentumStrategy:
    """
    Momentum strategy that trades WITH persistent market movements.
    
    Key insight: When markets persistently move in one direction,
    momentum (not mean-reversion) is the right approach.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[MomentumConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or MomentumConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Price history for momentum calculation
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.price_timestamps: Dict[str, List[datetime]] = defaultdict(list)
        
    def update_price(self, market_id: str, price: float):
        """Record price observation."""
        self.price_history[market_id].append(price)
        self.price_timestamps[market_id].append(datetime.utcnow())
        
        # Keep only recent history
        max_history = self.config.slow_ma_period * 2
        if len(self.price_history[market_id]) > max_history:
            self.price_history[market_id] = self.price_history[market_id][-max_history:]
            self.price_timestamps[market_id] = self.price_timestamps[market_id][-max_history:]
    
    def _compute_momentum(self, prices: List[float]) -> float:
        """Compute momentum as rate of change."""
        if len(prices) < 2:
            return 0.0
        
        # Simple momentum: (current - past) / past
        window = min(len(prices), self.config.persistence_window)
        old_price = prices[-window]
        new_price = prices[-1]
        
        if old_price <= 0:
            return 0.0
            
        return (new_price - old_price) / old_price
    
    def _compute_trend_signal(self, prices: List[float]) -> int:
        """Compute trend signal using dual MA crossover."""
        if len(prices) < self.config.slow_ma_period:
            return 0
            
        fast_ma = np.mean(prices[-self.config.fast_ma_period:])
        slow_ma = np.mean(prices[-self.config.slow_ma_period:])
        
        if fast_ma > slow_ma * 1.02:  # 2% buffer
            return 1  # Uptrend
        elif fast_ma < slow_ma * 0.98:
            return -1  # Downtrend
        return 0
    
    def generate_signals(self, markets: List[Market]) -> List[Signal]:
        """Generate momentum signals for markets."""
        signals = []
        
        for market in markets:
            # Update price history
            self.update_price(market.market_id, market.current_yes_price)
            
            prices = self.price_history[market.market_id]
            if len(prices) < self.config.fast_ma_period:
                continue
            
            # Compute momentum
            momentum = self._compute_momentum(prices)
            trend_signal = self._compute_trend_signal(prices)
            
            # Generate signal if momentum is strong and aligned with trend
            if abs(momentum) > self.config.momentum_threshold and trend_signal != 0:
                
                # Trade in direction of momentum
                if momentum > 0 and trend_signal > 0:
                    side = Side.YES
                    edge = min(momentum, 0.15)  # Cap edge estimate
                elif momentum < 0 and trend_signal < 0:
                    side = Side.NO
                    edge = min(abs(momentum), 0.15)
                else:
                    continue
                
                # Kelly sizing
                prob = 0.5 + edge * 0.5  # Convert edge to win probability
                kelly = (prob * 2 - 1) * self.config.kelly_fraction
                kelly = max(0, min(kelly, self.config.max_position_pct))
                
                confidence = min(abs(momentum) / 0.20, 1.0)  # Scale to 0-1
                
                signal = Signal(
                    market_id=market.market_id,
                    platform=self.platform,
                    side=side,
                    edge=edge,
                    kelly_fraction=kelly,
                    confidence=confidence,
                    strategy=f"{self.platform.value}_momentum",
                    metadata={
                        'momentum': momentum,
                        'trend_signal': trend_signal,
                        'fast_ma': np.mean(prices[-self.config.fast_ma_period:]),
                        'slow_ma': np.mean(prices[-self.config.slow_ma_period:]),
                    }
                )
                signals.append(signal)
        
        return signals
    
    def get_calibration_summary(self) -> dict:
        """Return calibration summary for compatibility."""
        return {
            'status': 'online',
            'description': 'Momentum strategy - no pre-calibration needed',
            'total_samples': len(self.price_history),
            'mean_spread': 0.0,
        }
    
    def update_historical_data(self, df):
        """Compatibility method - momentum uses online learning."""
        pass
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size in dollars."""
        base_size = bankroll * signal.kelly_fraction
        max_size = bankroll * self.config.max_position_pct
        return min(base_size, max_size, bankroll * 0.5)
