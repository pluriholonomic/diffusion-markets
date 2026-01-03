"""
Portfolio-Level Trading Strategies

Live trading wrappers for portfolio construction strategies:
1. YesNoConvergenceStrategy - Trade YES+NO spread convergence to 1
2. RelativeValueStrategy - Trade within category groups
3. RiskParityStrategy - Markowitz-style category allocation
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.models import Platform, Side, Signal, Market, RiskLimits


@dataclass
class YesNoConvergenceConfig:
    """Configuration for YES/NO convergence strategy."""
    spread_threshold: float = 0.03  # Min deviation from 1.0 to trade
    convergence_speed: float = 0.8  # Expected convergence rate
    lookback_periods: int = 50  # History for spread estimation
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05


class YesNoConvergenceStrategy:
    """
    YES/NO Convergence Strategy.
    
    In prediction markets, YES_price + NO_price should ≈ 1.0
    Trade when the effective spread deviates, expecting convergence.
    
    This is analogous to funding rate arbitrage in perpetuals.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[YesNoConvergenceConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or YesNoConvergenceConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Track price history and calibration by price bin
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.bin_calibration: Dict[int, Dict] = {}  # bin -> {spread, n_samples}
        self.n_bins = 20
        
    def _get_bin(self, price: float) -> int:
        """Map price to bin index."""
        return min(int(price * self.n_bins), self.n_bins - 1)
    
    def _update_calibration(self, market_id: str, price: float, outcome: Optional[float] = None):
        """Update calibration estimates."""
        self.price_history[market_id].append(price)
        
        # Keep limited history
        if len(self.price_history[market_id]) > 100:
            self.price_history[market_id] = self.price_history[market_id][-100:]
        
        # If we have an outcome, update bin calibration
        if outcome is not None:
            b = self._get_bin(price)
            if b not in self.bin_calibration:
                self.bin_calibration[b] = {'prices': [], 'outcomes': []}
            self.bin_calibration[b]['prices'].append(price)
            self.bin_calibration[b]['outcomes'].append(outcome)
            
            # Keep limited samples
            if len(self.bin_calibration[b]['prices']) > 200:
                self.bin_calibration[b]['prices'] = self.bin_calibration[b]['prices'][-200:]
                self.bin_calibration[b]['outcomes'] = self.bin_calibration[b]['outcomes'][-200:]
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status."""
        total_samples = sum(len(v['prices']) for v in self.bin_calibration.values())
        return {
            'status': 'online' if total_samples < 100 else 'calibrated',
            'total_samples': total_samples,
            'mean_spread': 0.0,
            'bins_active': len(self.bin_calibration),
        }
    
    def generate_signal(self, market: Market, bankroll: float) -> Optional[Signal]:
        """Generate YES/NO convergence signal."""
        if market.platform != self.platform:
            return None
        
        price = market.current_yes_price
        self._update_calibration(market.market_id, price)
        
        b = self._get_bin(price)
        
        # Need calibration data for this bin
        if b not in self.bin_calibration or len(self.bin_calibration[b]['prices']) < 10:
            # Use heuristic for uncalibrated bins
            # Extreme prices tend to be overpriced (regression to mean)
            if price > 0.85:
                spread = -0.03  # High prices tend to be overpriced
            elif price < 0.15:
                spread = 0.02  # Low prices slightly underpriced
            else:
                return None  # No signal without calibration
        else:
            # Compute calibration spread: E[outcome] - E[price]
            outcomes = self.bin_calibration[b]['outcomes']
            prices = self.bin_calibration[b]['prices']
            spread = np.mean(outcomes) - np.mean(prices)
        
        # Only trade if spread exceeds threshold
        if abs(spread) < self.config.spread_threshold:
            return None
        
        # Trade direction: if spread > 0, YES is underpriced (buy YES)
        if spread > 0:
            side = Side.YES
            edge = spread * self.config.convergence_speed
        else:
            side = Side.NO
            edge = abs(spread) * self.config.convergence_speed
        
        # Kelly sizing
        confidence = min(abs(spread) / 0.10, 1.0)
        kelly = edge * confidence * self.config.kelly_fraction
        kelly = max(0, min(kelly, self.config.max_position_pct))
        
        if kelly < 0.01:
            return None
        
        return Signal(
            market_id=market.market_id,
            platform=market.platform,
            side=side,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            strategy=f"{self.platform.value}_yesno_convergence",
            metadata={
                'spread': spread,
                'bin': b,
                'convergence_speed': self.config.convergence_speed,
            }
        )
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size."""
        return min(bankroll * signal.kelly_fraction, bankroll * self.config.max_position_pct)


@dataclass
class RelativeValueConfig:
    """Configuration for relative value strategy."""
    min_category_markets: int = 3
    zscore_threshold: float = 1.5  # Trade when |z| > threshold
    lookback_periods: int = 30
    kelly_fraction: float = 0.20
    max_position_pct: float = 0.04


class RelativeValueStrategy:
    """
    Relative Value Strategy.
    
    Trade markets that are mispriced RELATIVE to their category peers.
    If a market's price is far from the category average, bet on convergence.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[RelativeValueConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or RelativeValueConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Track prices by category
        self.category_prices: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
    def _get_category(self, market: Market) -> str:
        """Extract category from market."""
        category = market.metadata.get('category', market.category)
        if not category or category == 'unknown':
            # Infer from question
            q = market.question.lower() if market.question else ''
            if any(w in q for w in ['trump', 'biden', 'election', 'congress']):
                return 'politics'
            elif any(w in q for w in ['bitcoin', 'ethereum', 'crypto']):
                return 'crypto'
            elif any(w in q for w in ['game', 'win', 'championship']):
                return 'sports'
        return category or 'other'
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status."""
        total = sum(len(m) for cat in self.category_prices.values() for m in cat.values())
        return {
            'status': 'online',
            'total_samples': total,
            'mean_spread': 0.0,
            'categories_tracked': len(self.category_prices),
        }
    
    def generate_signals(self, markets: List[Market]) -> List[Signal]:
        """Generate relative value signals."""
        signals = []
        
        # Update prices
        for market in markets:
            if market.platform != self.platform:
                continue
            cat = self._get_category(market)
            self.category_prices[cat][market.market_id].append(market.current_yes_price)
            
            # Limit history
            if len(self.category_prices[cat][market.market_id]) > 100:
                self.category_prices[cat][market.market_id] = \
                    self.category_prices[cat][market.market_id][-100:]
        
        # Generate signals per category
        for cat, market_prices in self.category_prices.items():
            if len(market_prices) < self.config.min_category_markets:
                continue
            
            # Get current prices for category
            current_prices = {}
            for m in markets:
                if m.market_id in market_prices and m.platform == self.platform:
                    current_prices[m.market_id] = m.current_yes_price
            
            if len(current_prices) < self.config.min_category_markets:
                continue
            
            # Compute category stats
            prices = list(current_prices.values())
            cat_mean = np.mean(prices)
            cat_std = np.std(prices)
            
            if cat_std < 0.02:  # Low dispersion
                continue
            
            # Find outliers
            for market in markets:
                if market.market_id not in current_prices:
                    continue
                
                price = current_prices[market.market_id]
                zscore = (price - cat_mean) / cat_std
                
                if abs(zscore) < self.config.zscore_threshold:
                    continue
                
                # Bet on convergence to mean
                if zscore > 0:
                    side = Side.NO  # Price too high, bet it comes down
                else:
                    side = Side.YES  # Price too low, bet it goes up
                
                edge = abs(zscore) * cat_std * 0.3  # Expected convergence
                confidence = min(abs(zscore) / 3.0, 1.0)
                kelly = edge * confidence * self.config.kelly_fraction
                kelly = max(0, min(kelly, self.config.max_position_pct))
                
                if kelly < 0.01:
                    continue
                
                signal = Signal(
                    market_id=market.market_id,
                    platform=market.platform,
                    side=side,
                    edge=edge,
                    confidence=confidence,
                    kelly_fraction=kelly,
                    strategy=f"{self.platform.value}_relative_value",
                    metadata={
                        'category': cat,
                        'zscore': zscore,
                        'category_mean': cat_mean,
                        'category_std': cat_std,
                    }
                )
                signals.append(signal)
        
        return signals
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size."""
        return min(bankroll * signal.kelly_fraction, bankroll * self.config.max_position_pct)


@dataclass
class RiskParityConfig:
    """Configuration for risk parity (Markowitz-style) strategy."""
    target_vol: float = 0.15  # Target portfolio volatility
    lookback_periods: int = 50
    min_edge: float = 0.02
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    max_category_pct: float = 0.25  # Max per category


class RiskParityStrategy:
    """
    Risk Parity / Markowitz-style Strategy.
    
    Allocate risk equally across categories, scaling positions
    by inverse volatility to target overall portfolio vol.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[RiskParityConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or RiskParityConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Track by category
        self.category_stats: Dict[str, Dict] = {}
        self.category_exposure: Dict[str, float] = defaultdict(float)
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        
    def _get_category(self, market: Market) -> str:
        """Extract category from market."""
        category = market.metadata.get('category', market.category)
        return category or 'other'
    
    def _update_category_stats(self, markets: List[Market]):
        """Update volatility estimates per category."""
        by_category = defaultdict(list)
        
        for market in markets:
            if market.platform != self.platform:
                continue
            cat = self._get_category(market)
            by_category[cat].append(market)
            
            # Update price history
            self.price_history[market.market_id].append(market.current_yes_price)
            if len(self.price_history[market.market_id]) > 100:
                self.price_history[market.market_id] = self.price_history[market.market_id][-100:]
        
        # Compute category stats
        for cat, cat_markets in by_category.items():
            if len(cat_markets) < 3:
                continue
            
            # Estimate category volatility from price movements
            vols = []
            for m in cat_markets:
                history = self.price_history.get(m.market_id, [])
                if len(history) >= 5:
                    returns = np.diff(history[-20:])
                    if len(returns) > 0:
                        vols.append(np.std(returns))
            
            if vols:
                self.category_stats[cat] = {
                    'vol': np.mean(vols) if np.mean(vols) > 0 else 0.1,
                    'n_markets': len(cat_markets),
                    'avg_price': np.mean([m.current_yes_price for m in cat_markets]),
                }
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status."""
        return {
            'status': 'online',
            'total_samples': sum(len(h) for h in self.price_history.values()),
            'mean_spread': 0.0,
            'categories': len(self.category_stats),
        }
    
    def generate_signals(self, markets: List[Market]) -> List[Signal]:
        """Generate risk parity signals."""
        signals = []
        
        # Update stats
        self._update_category_stats(markets)
        
        if not self.category_stats:
            return signals
        
        # Compute risk parity weights (inverse vol)
        total_inv_vol = sum(1 / s['vol'] for s in self.category_stats.values() if s['vol'] > 0)
        if total_inv_vol == 0:
            return signals
        
        risk_weights = {
            cat: (1 / s['vol']) / total_inv_vol 
            for cat, s in self.category_stats.items() if s['vol'] > 0
        }
        
        for market in markets:
            if market.platform != self.platform:
                continue
            
            cat = self._get_category(market)
            if cat not in self.category_stats:
                continue
            
            stats = self.category_stats[cat]
            
            # Simple edge: deviation from 0.5 (neutral price)
            price = market.current_yes_price
            deviation = price - 0.5
            
            # Only trade if price is reasonably extreme
            if abs(deviation) < 0.15:
                continue
            
            # Bet on regression to mean
            if deviation > 0:
                side = Side.NO  # High price, bet on decrease
                edge = deviation * 0.3
            else:
                side = Side.YES  # Low price, bet on increase
                edge = abs(deviation) * 0.3
            
            if edge < self.config.min_edge:
                continue
            
            # Vol-scaled sizing
            cat_vol = stats['vol']
            vol_scalar = self.config.target_vol / cat_vol if cat_vol > 0 else 1.0
            vol_scalar = min(vol_scalar, 3.0)  # Cap scaling
            
            # Risk parity weight
            weight = risk_weights.get(cat, 0.1)
            
            kelly = edge * weight * vol_scalar * self.config.kelly_fraction
            kelly = max(0, min(kelly, self.config.max_position_pct))
            
            if kelly < 0.01:
                continue
            
            confidence = min(abs(deviation) / 0.30, 1.0)
            
            signal = Signal(
                market_id=market.market_id,
                platform=market.platform,
                side=side,
                edge=edge,
                confidence=confidence,
                kelly_fraction=kelly,
                strategy=f"{self.platform.value}_risk_parity",
                metadata={
                    'category': cat,
                    'risk_weight': weight,
                    'vol_scalar': vol_scalar,
                    'category_vol': cat_vol,
                }
            )
            signals.append(signal)
        
        return signals
    
    def compute_position_size(self, signal: Signal, bankroll: float) -> float:
        """Compute position size."""
        return min(bankroll * signal.kelly_fraction, bankroll * self.config.max_position_pct)
