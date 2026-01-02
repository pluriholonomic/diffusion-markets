"""
Statistical Arbitrage Strategy

Uses category-level calibration to identify mispriced markets
based on their category's historical performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..utils.models import Platform, Side, Signal, Market, RiskLimits


@dataclass
class StatArbConfig:
    """Configuration for stat arb strategy.
    
    Parameters optimized via CMA-ES on historical data (2026-01-02):
    - kelly_fraction: 0.40 (vs default 0.20) - more aggressive sizing
    - spread_threshold: 0.089 (vs 0.05) - require larger edge
    - min_edge: 0.086 - minimum expected edge
    - max_position_pct: 0.05 (vs 0.08) - smaller positions
    - Category weights: politics=1.76, crypto=0.51, sports=1.79
    """
    min_category_samples: int = 20  # Min samples per category
    spread_threshold: float = 0.089  # Min miscalibration (optimized)
    kelly_fraction: float = 0.40  # Fraction of Kelly (optimized)
    max_position_pct: float = 0.05  # Max position % (optimized)
    min_edge: float = 0.086  # Minimum expected edge (optimized)
    recalibrate_days: int = 5
    # Category weights (optimized)
    category_weight_politics: float = 1.76
    category_weight_crypto: float = 0.51
    category_weight_sports: float = 1.79


class StatArbStrategy:
    """
    Statistical arbitrage based on category-level calibration.
    
    Identifies categories that are systematically mispriced
    and trades accordingly.
    """
    
    def __init__(
        self,
        platform: Platform = Platform.POLYMARKET,
        config: Optional[StatArbConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or StatArbConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        self.category_calibration: Dict[str, float] = {}
        self.category_samples: Dict[str, int] = {}
        self.last_calibration_time: Optional[datetime] = None
    
    def update_historical_data(self, data: pd.DataFrame):
        """Update category-level calibration."""
        df = data.copy()
        
        if 'avg_price' in df.columns:
            df = df.rename(columns={'avg_price': 'price'})
        if 'y' in df.columns:
            df = df.rename(columns={'y': 'outcome'})
        
        df['price'] = df['price'].clip(0.01, 0.99)
        
        if 'category' not in df.columns:
            return
        
        # Compute calibration per category
        cat_stats = df.groupby('category').agg({
            'price': 'mean',
            'outcome': ['mean', 'count'],
        })
        cat_stats.columns = ['avg_price', 'outcome_rate', 'n_samples']
        cat_stats['spread'] = cat_stats['outcome_rate'] - cat_stats['avg_price']
        
        # Filter by sample size
        valid_cats = cat_stats[
            cat_stats['n_samples'] >= self.config.min_category_samples
        ]
        
        self.category_calibration = valid_cats['spread'].to_dict()
        self.category_samples = valid_cats['n_samples'].to_dict()
        self.last_calibration_time = datetime.utcnow()
    
    def generate_signal(
        self,
        market: Market,
        bankroll: float,
    ) -> Optional[Signal]:
        """Generate signal based on category calibration."""
        
        category = market.category
        
        # Map API categories to calibrated categories
        category_mapping = {
            'general': 'other',
            'US-current-affairs': 'politics',
            'us-current-affairs': 'politics',
            'US Politics': 'politics',
            'Crypto': 'crypto',
            'Sports': 'sports',
            'Economics': 'economics',
        }
        mapped_category = category_mapping.get(category, category)
        
        if mapped_category not in self.category_calibration:
            return None
        
        # Use mapped category for calibration lookup
        category = mapped_category
        
        spread = self.category_calibration[category]
        
        if abs(spread) < self.config.spread_threshold:
            return None
        
        if market.liquidity < self.risk_limits.min_liquidity:
            return None
        
        # Determine side
        side = Side.YES if spread > 0 else Side.NO
        
        # Compute Kelly
        price = market.current_yes_price
        edge = abs(spread)
        
        if side == Side.YES:
            odds = (1 - price) / price if price > 0.01 else 99
        else:
            odds = price / (1 - price) if price < 0.99 else 99
        
        kelly = (edge * odds - (1 - edge)) / odds if odds > 0 else 0
        kelly = max(0, min(kelly, 1)) * self.config.kelly_fraction
        
        # If Kelly is 0 but we have edge, use edge-based sizing
        if kelly <= 0 and edge >= self.config.spread_threshold:
            kelly = self.config.kelly_fraction * min(edge / 0.10, 1.0)
        
        kelly = min(kelly, self.config.max_position_pct)
        
        if kelly < 0.01:
            return None
        
        return Signal(
            platform=self.platform,
            market_id=market.market_id,
            strategy="stat_arb",
            side=side,
            edge=edge,
            confidence=min(edge / 0.20, 1.0),
            kelly_fraction=kelly,
            metadata={
                'category': category,
                'category_spread': spread,
                'price': price,
                'liquidity': market.liquidity,
                'category_samples': self.category_samples.get(category, 0),
            }
        )
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size."""
        size = bankroll * signal.kelly_fraction
        max_size = bankroll * self.config.max_position_pct
        return min(size, max_size, bankroll * 0.4)
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary."""
        if not self.category_calibration:
            return {'status': 'not_calibrated', 'total_samples': 0, 'mean_spread': 0}
        
        spreads = list(self.category_calibration.values())
        
        return {
            'status': 'calibrated',
            'n_categories': len(self.category_calibration),
            'total_samples': sum(self.category_samples.values()),
            'mean_spread': sum(spreads) / len(spreads) if spreads else 0,
            'categories': {
                cat: {'spread': spread, 'samples': self.category_samples.get(cat, 0)}
                for cat, spread in sorted(
                    self.category_calibration.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
            },
            'last_update': self.last_calibration_time.isoformat() if self.last_calibration_time else None,
        }
