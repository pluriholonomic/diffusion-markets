"""
Longshot Betting Strategy

Exploits the favorite-longshot bias on Kalshi where low-probability
events are systematically underpriced.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..utils.models import Platform, Side, Signal, Market, RiskLimits


@dataclass
class LongshotConfig:
    """Configuration for longshot strategy."""
    price_threshold: float = 0.10  # Only trade markets priced below this
    min_expected_edge: float = 0.15  # Minimum edge to trade
    kelly_fraction: float = 0.10  # Conservative Kelly
    max_position_pct: float = 0.05  # Small positions
    recalibrate_days: int = 3


class LongshotStrategy:
    """
    Longshot betting strategy for Kalshi.
    
    Exploits the favorite-longshot bias where sports bettors
    systematically underprice low-probability events.
    """
    
    def __init__(
        self,
        config: Optional[LongshotConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = Platform.KALSHI
        self.config = config or LongshotConfig()
        self.risk_limits = risk_limits or RiskLimits(
            max_position_pct=0.05,
            kelly_fraction=0.10,
            min_edge=0.10,
            min_liquidity=50,
        )
        
        self.longshot_edge: float = 0.0
        self.calibration_samples: int = 0
        self.last_calibration_time: Optional[datetime] = None
    
    def update_historical_data(self, data: pd.DataFrame):
        """Update calibration from historical data."""
        df = data.copy()
        
        if 'avg_price' in df.columns:
            df = df.rename(columns={'avg_price': 'price'})
        if 'y' in df.columns:
            df = df.rename(columns={'y': 'outcome'})
        
        df['price'] = df['price'].clip(0.01, 0.99)
        
        # Focus on low-price markets
        low_price = df[df['price'] < self.config.price_threshold]
        
        if len(low_price) >= 10:
            self.longshot_edge = (
                low_price['outcome'].mean() - low_price['price'].mean()
            )
            self.calibration_samples = len(low_price)
            self.last_calibration_time = datetime.utcnow()
    
    def generate_signal(
        self,
        market: Market,
        bankroll: float,
    ) -> Optional[Signal]:
        """Generate signal for a longshot market."""
        
        # Only trade low-priced markets
        if market.current_yes_price >= self.config.price_threshold:
            return None
        
        # Check if we have edge
        if self.longshot_edge < self.config.min_expected_edge:
            return None
        
        # Check liquidity
        if market.liquidity < self.risk_limits.min_liquidity:
            return None
        
        # Compute Kelly
        price = market.current_yes_price
        edge = self.longshot_edge
        odds = (1 - price) / price if price > 0.01 else 99
        
        kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
        kelly = max(0, min(kelly, 1)) * self.config.kelly_fraction
        
        # If Kelly is 0 but we have edge, use edge-based sizing
        if kelly <= 0 and edge >= self.config.min_expected_edge:
            kelly = self.config.kelly_fraction * min(edge / 0.20, 1.0)
        
        kelly = min(kelly, self.config.max_position_pct)
        
        if kelly < 0.01:
            return None
        
        return Signal(
            platform=self.platform,
            market_id=market.market_id,
            strategy="longshot",
            side=Side.YES,  # Always bet YES on longshots
            edge=edge,
            confidence=min(edge / 0.30, 1.0),
            kelly_fraction=kelly,
            metadata={
                'price': price,
                'longshot_edge': self.longshot_edge,
                'category': market.category,
                'liquidity': market.liquidity,
            }
        )
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size."""
        size = bankroll * signal.kelly_fraction
        max_size = bankroll * self.config.max_position_pct
        return min(size, max_size, bankroll * 0.3)
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary."""
        return {
            'status': 'calibrated' if self.longshot_edge > 0 else 'not_calibrated',
            'longshot_edge': self.longshot_edge,
            'mean_spread': self.longshot_edge,  # For consistency with other strategies
            'total_samples': self.calibration_samples,
            'last_update': self.last_calibration_time.isoformat() if self.last_calibration_time else None,
        }
