"""
Dispersion and Correlation Trading Strategies

Implements:
1. Dispersion Strategy - trade category vol vs individual market vol
2. Correlation Trading - trade correlation breakdowns
3. Adaptive Correlation - dynamic correlation-based sizing
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.models import Platform, Side, Signal, Market, RiskLimits


@dataclass
class DispersionConfig:
    """Configuration for dispersion/correlation strategies."""
    # Dispersion thresholds
    dispersion_threshold: float = 0.10  # Min dispersion gap
    min_markets_per_category: int = 3
    lookback_periods: int = 20
    
    # Correlation params
    correlation_threshold: float = 0.6  # Min correlation for pairs
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    
    # Sizing
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.04
    
    # Category weights (optimized)
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        'politics': 1.5,
        'crypto': 0.5,
        'sports': 1.2,
        'finance': 1.0,
    })


class DispersionStrategy:
    """
    Dispersion trading: trade the spread between implied and realized dispersion.
    
    When category-level implied vol differs from realized dispersion of
    individual markets, there's an arbitrage opportunity.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[DispersionConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or DispersionConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Track prices by category
        self.category_prices: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.category_markets: Dict[str, List[str]] = defaultdict(list)
        
    def _get_category(self, market: Market) -> str:
        """Extract category from market metadata."""
        category = market.metadata.get('category', 'unknown')
        if not category:
            # Try to infer from question
            question = market.question.lower()
            if any(w in question for w in ['trump', 'biden', 'election', 'congress', 'senate']):
                return 'politics'
            elif any(w in question for w in ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth']):
                return 'crypto'
            elif any(w in question for w in ['game', 'win', 'championship', 'nfl', 'nba']):
                return 'sports'
            elif any(w in question for w in ['stock', 'price', 'market', 'fed', 'rate']):
                return 'finance'
        return category or 'unknown'
    
    def _compute_implied_vol(self, price: float) -> float:
        """Binary option implied vol: sqrt(p * (1-p))"""
        p = np.clip(price, 0.01, 0.99)
        return np.sqrt(p * (1 - p))
    
    def _compute_realized_dispersion(self, prices_by_market: Dict[str, List[float]]) -> float:
        """Compute realized dispersion across markets."""
        if len(prices_by_market) < 2:
            return 0.0
            
        # Get recent returns for each market
        returns = []
        for market_id, prices in prices_by_market.items():
            if len(prices) >= 2:
                ret = (prices[-1] - prices[-2]) / max(prices[-2], 0.01)
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
            
        return np.std(returns)
    
    def update_prices(self, markets: List[Market]):
        """Update price tracking for all markets."""
        for market in markets:
            category = self._get_category(market)
            self.category_prices[category][market.market_id].append(market.current_yes_price)
            
            # Track which markets are in each category
            if market.market_id not in self.category_markets[category]:
                self.category_markets[category].append(market.market_id)
            
            # Limit history
            if len(self.category_prices[category][market.market_id]) > self.config.lookback_periods:
                self.category_prices[category][market.market_id] = \
                    self.category_prices[category][market.market_id][-self.config.lookback_periods:]
    
    def generate_signals(self, markets: List[Market]) -> List[Signal]:
        """Generate dispersion trading signals."""
        signals = []
        
        # Update price tracking
        self.update_prices(markets)
        
        # Process each category
        for category, market_ids in self.category_markets.items():
            if len(market_ids) < self.config.min_markets_per_category:
                continue
            
            # Get current markets in this category
            category_markets = [m for m in markets if self._get_category(m) == category]
            if len(category_markets) < self.config.min_markets_per_category:
                continue
            
            # Compute category-level implied vol
            avg_price = np.mean([m.current_yes_price for m in category_markets])
            implied_vol = self._compute_implied_vol(avg_price)
            
            # Compute realized dispersion
            realized_disp = self._compute_realized_dispersion(self.category_prices[category])
            
            # Dispersion gap
            disp_gap = realized_disp - implied_vol
            
            if abs(disp_gap) < self.config.dispersion_threshold:
                continue
            
            # Generate signals
            category_weight = self.config.category_weights.get(category, 1.0)
            
            for market in category_markets:
                # If realized > implied: buy individual vol (bet on movement)
                # If realized < implied: sell individual vol (bet on stability)
                
                market_iv = self._compute_implied_vol(market.current_yes_price)
                
                if disp_gap > 0 and market_iv < implied_vol:
                    # This market has low IV, expect it to move more
                    # Bet on the more likely direction based on price
                    side = Side.YES if market.current_yes_price < 0.5 else Side.NO
                    edge = min(disp_gap * 0.5, 0.10)
                elif disp_gap < 0 and market_iv > implied_vol:
                    # This market has high IV, expect it to stabilize
                    # Bet on current price (less movement)
                    side = Side.YES if market.current_yes_price > 0.5 else Side.NO
                    edge = min(abs(disp_gap) * 0.5, 0.10)
                else:
                    continue
                
                kelly = self.config.kelly_fraction * category_weight * edge * 2
                kelly = min(kelly, self.config.max_position_pct)
                
                signal = Signal(
                    market_id=market.market_id,
                    platform=self.platform,
                    side=side,
                    edge=edge,
                    kelly_fraction=kelly,
                    confidence=min(abs(disp_gap) / 0.20, 1.0),
                    strategy=f"{self.platform.value}_dispersion",
                    metadata={
                        'category': category,
                        'implied_vol': implied_vol,
                        'realized_dispersion': realized_disp,
                        'dispersion_gap': disp_gap,
                        'market_iv': market_iv,
                    }
                )
                signals.append(signal)
        
        return signals
    
    def get_calibration_summary(self) -> dict:
        """Return calibration summary for compatibility."""
        return {
            'status': 'online',
            'description': 'Dispersion strategy - uses online learning',
            'total_samples': sum(len(v) for v in self.category_prices.values()),
            'mean_spread': 0.0,
        }
    
    def update_historical_data(self, df):
        """Compatibility method - dispersion uses online learning."""
        pass
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size in dollars."""
        base_size = bankroll * signal.kelly_fraction
        max_size = bankroll * self.config.max_position_pct
        return min(base_size, max_size, bankroll * 0.5)


class CorrelationStrategy:
    """
    Correlation-based trading: exploit correlation breakdowns and pairs.
    
    Adaptive correlation adjusts position sizes based on realized correlations.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[DispersionConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or DispersionConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Track price returns for correlation
        self.returns: Dict[str, List[float]] = defaultdict(list)
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.spread_history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
    def update_returns(self, markets: List[Market]):
        """Update return tracking."""
        for market in markets:
            prices = self.returns.get(market.market_id, [])
            if prices:
                ret = (market.current_yes_price - prices[-1]) / max(prices[-1], 0.01)
                self.returns[market.market_id].append(ret)
            else:
                self.returns[market.market_id] = [0.0]
            
            # Limit history
            if len(self.returns[market.market_id]) > self.config.lookback_periods:
                self.returns[market.market_id] = self.returns[market.market_id][-self.config.lookback_periods:]
    
    def _compute_correlation(self, returns1: List[float], returns2: List[float]) -> float:
        """Compute correlation between two return series."""
        n = min(len(returns1), len(returns2))
        if n < 5:
            return 0.0
        
        r1 = np.array(returns1[-n:])
        r2 = np.array(returns2[-n:])
        
        if np.std(r1) < 1e-6 or np.std(r2) < 1e-6:
            return 0.0
        
        return np.corrcoef(r1, r2)[0, 1]
    
    def _find_correlated_pairs(self, markets: List[Market]) -> List[Tuple[Market, Market, float]]:
        """Find highly correlated market pairs."""
        pairs = []
        
        market_ids = [m.market_id for m in markets]
        
        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                r1 = self.returns.get(m1.market_id, [])
                r2 = self.returns.get(m2.market_id, [])
                
                corr = self._compute_correlation(r1, r2)
                self.correlation_matrix[(m1.market_id, m2.market_id)] = corr
                
                if abs(corr) > self.config.correlation_threshold:
                    pairs.append((m1, m2, corr))
        
        return pairs
    
    def generate_signals(self, markets: List[Market]) -> List[Signal]:
        """Generate correlation-based signals."""
        signals = []
        
        # Update returns
        self.update_returns(markets)
        
        # Find correlated pairs
        pairs = self._find_correlated_pairs(markets)
        
        for m1, m2, corr in pairs:
            # Compute spread z-score
            spread = m1.current_yes_price - m2.current_yes_price
            pair_key = (m1.market_id, m2.market_id)
            
            self.spread_history[pair_key].append(spread)
            if len(self.spread_history[pair_key]) > self.config.lookback_periods:
                self.spread_history[pair_key] = self.spread_history[pair_key][-self.config.lookback_periods:]
            
            spreads = self.spread_history[pair_key]
            if len(spreads) < 5:
                continue
            
            spread_mean = np.mean(spreads)
            spread_std = np.std(spreads)
            
            if spread_std < 0.01:
                continue
            
            zscore = (spread - spread_mean) / spread_std
            
            # Generate signals on extreme z-scores
            if abs(zscore) > self.config.zscore_entry:
                edge = min(abs(zscore) * 0.02, 0.08)
                kelly = self.config.kelly_fraction * edge * 2
                kelly = min(kelly, self.config.max_position_pct)
                
                if zscore > self.config.zscore_entry:
                    # Spread too high: short m1, long m2
                    signals.append(Signal(
                        market_id=m1.market_id,
                        platform=self.platform,
                        side=Side.NO,
                        edge=edge,
                        kelly_fraction=kelly,
                        confidence=min(abs(zscore) / 4.0, 1.0),
                        strategy=f"{self.platform.value}_correlation",
                        metadata={
                            'pair': m2.market_id,
                            'correlation': corr,
                            'zscore': zscore,
                            'spread': spread,
                        }
                    ))
                    signals.append(Signal(
                        market_id=m2.market_id,
                        platform=self.platform,
                        side=Side.YES,
                        edge=edge,
                        kelly_fraction=kelly,
                        confidence=min(abs(zscore) / 4.0, 1.0),
                        strategy=f"{self.platform.value}_correlation",
                        metadata={
                            'pair': m1.market_id,
                            'correlation': corr,
                            'zscore': zscore,
                            'spread': spread,
                        }
                    ))
                elif zscore < -self.config.zscore_entry:
                    # Spread too low: long m1, short m2
                    signals.append(Signal(
                        market_id=m1.market_id,
                        platform=self.platform,
                        side=Side.YES,
                        edge=edge,
                        kelly_fraction=kelly,
                        confidence=min(abs(zscore) / 4.0, 1.0),
                        strategy=f"{self.platform.value}_correlation",
                        metadata={
                            'pair': m2.market_id,
                            'correlation': corr,
                            'zscore': -zscore,
                            'spread': spread,
                        }
                    ))
                    signals.append(Signal(
                        market_id=m2.market_id,
                        platform=self.platform,
                        side=Side.NO,
                        edge=edge,
                        kelly_fraction=kelly,
                        confidence=min(abs(zscore) / 4.0, 1.0),
                        strategy=f"{self.platform.value}_correlation",
                        metadata={
                            'pair': m1.market_id,
                            'correlation': corr,
                            'zscore': -zscore,
                            'spread': spread,
                        }
                    ))
        
        return signals
    
    def get_calibration_summary(self) -> dict:
        """Return calibration summary for compatibility."""
        return {
            'status': 'online',
            'description': 'Correlation strategy - uses online learning',
            'total_samples': len(self.returns),
            'mean_spread': 0.0,
        }
    
    def update_historical_data(self, df):
        """Compatibility method - correlation uses online learning."""
        pass
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size in dollars."""
        base_size = bankroll * signal.kelly_fraction
        max_size = bankroll * self.config.max_position_pct
        return min(base_size, max_size, bankroll * 0.5)
