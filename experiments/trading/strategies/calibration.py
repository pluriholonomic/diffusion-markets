"""
Calibration-based Mean Reversion Strategy

This strategy:
1. Learns calibration (outcome rate vs price) from historical data
2. Identifies systematically mispriced markets
3. Bets against the mispricing (e.g., bet NO when prices are too high)
4. Uses Kelly sizing with risk limits
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..utils.models import (
    Platform, Side, Signal, Market, RiskLimits
)


@dataclass
class CalibrationConfig:
    """Configuration for calibration strategy."""
    n_bins: int = 10
    spread_threshold: float = 0.05  # Minimum miscalibration to trade
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    max_position_pct: float = 0.10  # Max position as % of bankroll
    recalibrate_days: int = 7  # How often to recalibrate
    min_samples_per_bin: int = 10  # Minimum observations per bin
    categories: List[str] = field(default_factory=list)  # Focus categories


class CalibrationStrategy:
    """
    Mean-reversion strategy based on calibration errors.
    
    The key insight is that prediction markets are often miscalibrated:
    - On Polymarket: Prices systematically exceed outcome rates (overprice YES)
    - On Kalshi: Longshots are underpriced (underprice YES at low prices)
    
    This strategy bets against these systematic biases.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[CalibrationConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or CalibrationConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Calibration state
        self.calibration: Optional[pd.DataFrame] = None
        self.last_calibration_time: Optional[datetime] = None
        self.historical_data: List[Dict] = []
    
    def update_historical_data(self, data: pd.DataFrame):
        """Update historical data for calibration."""
        # Standardize columns
        df = data.copy()
        if 'avg_price' in df.columns:
            df = df.rename(columns={'avg_price': 'price'})
        if 'y' in df.columns:
            df = df.rename(columns={'y': 'outcome'})
        
        df['price'] = df['price'].clip(0.01, 0.99)
        
        # Store for calibration
        self.historical_data = df.to_dict('records')
        self._recalibrate()
    
    def _recalibrate(self):
        """Recompute calibration from historical data."""
        if len(self.historical_data) < 100:
            return
        
        df = pd.DataFrame(self.historical_data)
        
        # Create price bins
        df['price_bin'] = pd.cut(
            df['price'], 
            bins=self.config.n_bins, 
            labels=False
        )
        
        # Compute calibration per bin
        self.calibration = df.groupby('price_bin').agg({
            'price': ['mean', 'count'],
            'outcome': 'mean',
        })
        self.calibration.columns = ['bin_price', 'n_samples', 'outcome_rate']
        self.calibration['spread'] = (
            self.calibration['outcome_rate'] - self.calibration['bin_price']
        )
        
        # Filter bins with insufficient samples
        self.calibration = self.calibration[
            self.calibration['n_samples'] >= self.config.min_samples_per_bin
        ]
        
        self.last_calibration_time = datetime.utcnow()
    
    def needs_recalibration(self) -> bool:
        """Check if calibration needs to be updated."""
        if self.calibration is None:
            return True
        if self.last_calibration_time is None:
            return True
        
        age = datetime.utcnow() - self.last_calibration_time
        return age > timedelta(days=self.config.recalibrate_days)
    
    def get_spread(self, price: float) -> float:
        """Get calibration spread for a given price."""
        if self.calibration is None:
            return 0.0
        
        # Find the bin for this price
        bin_idx = int(price * self.config.n_bins)
        bin_idx = max(0, min(bin_idx, self.config.n_bins - 1))
        
        if bin_idx in self.calibration.index:
            return self.calibration.loc[bin_idx, 'spread']
        
        return 0.0
    
    def generate_signal(
        self,
        market: Market,
        bankroll: float,
    ) -> Optional[Signal]:
        """
        Generate trading signal for a market.
        
        Returns None if no trade should be made.
        """
        # Check if we should trade this category
        if self.config.categories:
            if market.category not in self.config.categories:
                return None
        
        # Check liquidity
        if market.liquidity < self.risk_limits.min_liquidity:
            return None
        
        # Get calibration spread
        price = market.current_yes_price
        spread = self.get_spread(price)
        
        # Check if spread exceeds threshold
        if abs(spread) < self.config.spread_threshold:
            return None
        
        # Determine side
        # spread > 0 means outcomes > prices (underpriced YES)
        # spread < 0 means outcomes < prices (overpriced YES, bet NO)
        if spread > 0:
            side = Side.YES
        else:
            side = Side.NO
        
        # Compute edge and Kelly
        edge = abs(spread)
        
        # Check minimum edge
        if edge < self.risk_limits.min_edge:
            return None
        
        if side == Side.YES:
            odds = (1 - price) / price if price > 0.01 else 99
        else:
            odds = price / (1 - price) if price < 0.99 else 99
        
        # Kelly formula
        kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
        kelly = max(0, min(kelly, 1)) * self.config.kelly_fraction
        
        # If Kelly is 0 or negative but we have edge, use edge-based sizing
        # This handles cases where odds are unfavorable but edge exists
        if kelly <= 0 and edge >= self.risk_limits.min_edge:
            # Simple edge-based sizing: position = kelly_fraction * edge/10%
            kelly = self.config.kelly_fraction * min(edge / 0.10, 1.0)
        
        # Apply risk limits
        kelly = min(kelly, self.risk_limits.max_position_pct)
        
        # Create signal
        signal = Signal(
            platform=self.platform,
            market_id=market.market_id,
            strategy="calibration_mean_reversion",
            side=side,
            edge=edge,
            confidence=min(edge / 0.20, 1.0),  # Scale confidence by edge
            kelly_fraction=kelly,
            metadata={
                'spread': spread,
                'price': price,
                'category': market.category,
                'liquidity': market.liquidity,
            }
        )
        
        return signal
    
    def compute_position_size(
        self,
        signal: Signal,
        bankroll: float,
    ) -> float:
        """Compute position size in dollars."""
        base_size = bankroll * signal.kelly_fraction
        
        # Apply maximum position limit
        max_size = bankroll * self.risk_limits.max_position_pct
        size = min(base_size, max_size)
        
        # Don't risk more than 50% of bankroll on one trade
        size = min(size, bankroll * 0.5)
        
        return size
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of current calibration."""
        if self.calibration is None:
            return {"status": "not_calibrated"}
        
        return {
            "status": "calibrated",
            "last_update": self.last_calibration_time.isoformat() if self.last_calibration_time else None,
            "n_bins": len(self.calibration),
            "mean_spread": float(self.calibration['spread'].mean()),
            "max_spread": float(self.calibration['spread'].abs().max()),
            "total_samples": int(self.calibration['n_samples'].sum()),
            "bins": self.calibration.to_dict('index'),
        }


class PolymarketCalibrationStrategy(CalibrationStrategy):
    """Calibration strategy optimized for Polymarket."""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        config = CalibrationConfig(
            n_bins=10,
            spread_threshold=0.05,
            kelly_fraction=0.25,
            max_position_pct=0.10,
            recalibrate_days=7,
            categories=[],  # Empty = no category filter (trade all categories)
        )
        super().__init__(Platform.POLYMARKET, config, risk_limits)


class KalshiCalibrationStrategy(CalibrationStrategy):
    """Calibration strategy optimized for Kalshi."""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        config = CalibrationConfig(
            n_bins=10,
            spread_threshold=0.10,  # Higher threshold for Kalshi
            kelly_fraction=0.15,  # More conservative
            max_position_pct=0.05,  # Smaller positions
            recalibrate_days=3,  # More frequent recalibration
            categories=[],  # All categories
        )
        
        risk_limits = risk_limits or RiskLimits(
            max_position_pct=0.05,
            kelly_fraction=0.15,
            min_edge=0.10,
            min_liquidity=100,
        )
        
        super().__init__(Platform.KALSHI, config, risk_limits)
