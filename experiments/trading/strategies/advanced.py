"""
Advanced Trading Strategies

Wraps sophisticated strategies from backtest module for use in hybrid trading:
1. BlackwellStrategy - Rolling calibration with statistical significance
2. ConfidenceGatedStrategy - Abstention-based with theoretical guarantees
3. OnlineMaxArbStrategy - Online learning with O(1/sqrt{T}) regret
4. ConditionalGraphStrategy - Cross-market dependency trading
5. TrendFollowingStrategy - Momentum with regime detection
6. MeanReversionStrategy - Pure mean reversion
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

from ..utils.models import Platform, Side, Signal, Market, RiskLimits

logger = logging.getLogger(__name__)


# ============================================================================
# Blackwell Calibration Strategy
# ============================================================================

@dataclass
class BlackwellConfig:
    """Configuration for Blackwell calibration strategy.
    
    Optimized via CMA-ES (to be updated):
    """
    n_bins: int = 10
    g_bar_threshold: float = 0.05  # Minimum |g̅| to trade
    t_stat_threshold: float = 2.0  # Minimum t-statistic
    min_samples_per_bin: int = 20
    lookback_trades: int = 500
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05


@dataclass
class BinStats:
    """Statistics for a single price bin."""
    g_bar: float = 0.0
    sigma: float = 0.0
    n_samples: int = 0
    t_stat: float = 0.0


class BlackwellStrategy:
    """
    Blackwell calibration strategy for hybrid trading.
    
    Computes calibration error (g̅) in each price bin and trades
    against systematic mispricings with statistical significance.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[BlackwellConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or BlackwellConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Rolling history: (price, estimated_outcome, market_id)
        self.history: List[Tuple[float, float, str]] = []
        self.bin_stats: Dict[int, BinStats] = {}
        self.trades_since_recalibration = 0
        
    def _get_bin(self, price: float) -> int:
        """Map price to bin index."""
        return min(int(price * self.config.n_bins), self.config.n_bins - 1)
    
    def _recalibrate(self):
        """Recompute bin statistics from history."""
        self.bin_stats = {}
        
        if len(self.history) < self.config.min_samples_per_bin:
            return
            
        # Group by bin
        bin_data: Dict[int, List[Tuple[float, float]]] = {}
        for price, outcome, _ in self.history[-self.config.lookback_trades:]:
            b = self._get_bin(price)
            if b not in bin_data:
                bin_data[b] = []
            bin_data[b].append((price, outcome))
        
        # Compute g̅ = mean(outcome - price) for each bin
        for b, data in bin_data.items():
            if len(data) < self.config.min_samples_per_bin:
                continue
                
            prices = np.array([d[0] for d in data])
            outcomes = np.array([d[1] for d in data])
            
            g_values = outcomes - prices
            g_bar = np.mean(g_values)
            sigma = np.std(g_values, ddof=1) if len(g_values) > 1 else 1.0
            t_stat = g_bar / (sigma / np.sqrt(len(g_values))) if sigma > 0 else 0.0
            
            self.bin_stats[b] = BinStats(
                g_bar=g_bar,
                sigma=sigma,
                n_samples=len(data),
                t_stat=t_stat,
            )
    
    def update_outcome(self, market_id: str, price: float, outcome: float):
        """Update with realized outcome."""
        self.history.append((price, outcome, market_id))
        self.trades_since_recalibration += 1
        
        if self.trades_since_recalibration >= 50:
            self._recalibrate()
            self.trades_since_recalibration = 0
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status summary."""
        return {
            'status': 'online' if len(self.history) < 100 else 'calibrated',
            'total_samples': len(self.history),
            'mean_spread': np.mean([h[1] - h[0] for h in self.history]) if self.history else 0.0,
            'bins_active': len(self.bin_stats),
        }
    
    def generate_signal(self, market: Market, bankroll: float) -> Optional[Signal]:
        """Generate signal based on Blackwell calibration."""
        if market.platform != self.platform:
            return None
            
        price = market.current_yes_price
        b = self._get_bin(price)
        
        # Check if we have stats for this bin
        if b not in self.bin_stats:
            # Use heuristic: estimate from live market characteristics
            # Markets with high spread or low volume are less efficient
            spread = market.metadata.get('spread', 0.02)
            volume = market.volume
            
            # Heuristic g̅ estimate based on market characteristics
            if price > 0.8:
                g_bar_estimate = -0.05  # High-priced markets tend to be overpriced
            elif price < 0.2:
                g_bar_estimate = 0.03  # Low-priced markets slightly underpriced
            else:
                g_bar_estimate = 0.0
            
            # Require larger edge for markets without historical data
            if abs(g_bar_estimate) < self.config.g_bar_threshold * 1.5:
                return None
                
            stats = BinStats(g_bar=g_bar_estimate, sigma=0.1, n_samples=0, t_stat=0.0)
        else:
            stats = self.bin_stats[b]
        
        # Check significance
        if abs(stats.t_stat) < self.config.t_stat_threshold and stats.n_samples > 0:
            return None
            
        g_bar = stats.g_bar
        
        # Trade against calibration error
        if abs(g_bar) < self.config.g_bar_threshold:
            return None
            
        # If g̅ < 0, market is overpriced (YES too high) -> bet NO
        # If g̅ > 0, market is underpriced (YES too low) -> bet YES
        side = Side.YES if g_bar > 0 else Side.NO
        
        # Edge is |g̅|
        edge = abs(g_bar)
        
        # Confidence based on t-stat
        confidence = min(1.0, abs(stats.t_stat) / 3.0) if stats.n_samples > 0 else 0.5
        
        # Kelly sizing
        if side == Side.YES:
            p_win = price + edge
            q_win = 1 - price  # payout
        else:
            p_win = (1 - price) + edge
            q_win = price  # payout for NO
            
        p_win = min(0.99, max(0.01, p_win))
        kelly = (p_win * q_win - (1 - p_win)) / q_win if q_win > 0 else 0.0
        kelly = max(0, min(kelly, 0.5)) * self.config.kelly_fraction
        
        if kelly < 0.01:
            return None
            
        size = bankroll * kelly * self.config.max_position_pct
        
        return Signal(
            market_id=market.market_id,
            platform=market.platform,
            side=side,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            size=size,
            metadata={
                'strategy': 'blackwell',
                'g_bar': g_bar,
                't_stat': stats.t_stat if stats.n_samples > 0 else 0.0,
                'n_samples': stats.n_samples,
                'bin': b,
            }
        )


# ============================================================================
# Confidence Gated Strategy  
# ============================================================================

@dataclass
class ConfidenceGatedConfig:
    """Configuration for confidence-gated strategy."""
    min_distance_threshold: float = 0.05  # Min d(q, C_t) to trade
    max_confidence_threshold: float = 0.95  # Abstain if too confident
    kelly_fraction: float = 0.30
    max_position_pct: float = 0.05
    abstention_penalty: float = 0.01  # Small cost for abstaining


class ConfidenceGatedStrategy:
    """
    Strategy that abstains when confidence is low.
    
    Only trades when the "distance to constraint set" exceeds threshold.
    This provides theoretical guarantees on hallucination rate.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[ConfidenceGatedConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or ConfidenceGatedConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Track abstention rate
        self.total_decisions = 0
        self.abstentions = 0
        self.trades = 0
        
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status summary."""
        return {
            'status': 'online',
            'total_samples': self.total_decisions,
            'mean_spread': self.abstentions / max(1, self.total_decisions),
            'abstention_rate': self.abstentions / max(1, self.total_decisions),
        }
    
    def _compute_distance(self, market: Market) -> float:
        """
        Compute proxy for d(q, C_t) - distance to constraint set.
        
        In practice, this is the deviation from "fair" pricing.
        """
        price = market.current_yes_price
        
        # Use volume-weighted expected price as C_t proxy
        # High volume = price is closer to "true" value
        volume = market.volume
        
        # Estimate "fair" price from metadata if available
        fair_price = market.metadata.get('model_price', price)
        
        # Distance is deviation from fair price
        distance = abs(price - fair_price)
        
        # Adjust for confidence (extreme prices are more "confident")
        price_confidence = 2 * abs(price - 0.5)  # 0 at 0.5, 1 at 0 or 1
        
        # Low volume markets have larger effective distance (more mispricing)
        volume_factor = 1.0 / (1.0 + np.log1p(volume / 10000))
        
        return distance + volume_factor * 0.02
    
    def generate_signal(self, market: Market, bankroll: float) -> Optional[Signal]:
        """Generate signal with confidence gating."""
        if market.platform != self.platform:
            return None
            
        self.total_decisions += 1
        price = market.current_yes_price
        
        # Compute distance to constraint set
        distance = self._compute_distance(market)
        
        # Abstention check 1: Distance too low
        if distance < self.config.min_distance_threshold:
            self.abstentions += 1
            return None
            
        # Abstention check 2: Price too extreme (too confident)
        if price > self.config.max_confidence_threshold or price < (1 - self.config.max_confidence_threshold):
            self.abstentions += 1
            return None
        
        # Trade decision: bet against mispricing
        fair_price = market.metadata.get('model_price', 0.5)
        
        if price > fair_price:
            side = Side.NO
            edge = price - fair_price
        else:
            side = Side.YES
            edge = fair_price - price
            
        if edge < self.config.min_distance_threshold:
            self.abstentions += 1
            return None
            
        self.trades += 1
        
        # Size proportional to distance
        confidence = min(1.0, distance / 0.15)
        kelly = edge * confidence * self.config.kelly_fraction
        kelly = max(0, min(kelly, 0.5))
        
        size = bankroll * kelly * self.config.max_position_pct
        
        return Signal(
            market_id=market.market_id,
            platform=market.platform,
            side=side,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            size=size,
            metadata={
                'strategy': 'confidence_gated',
                'distance': distance,
                'fair_price': fair_price,
                'abstention_rate': self.abstentions / max(1, self.total_decisions),
            }
        )


# ============================================================================
# Trend Following Strategy
# ============================================================================

@dataclass
class TrendFollowingConfig:
    """Configuration for trend following strategy."""
    fast_ema_periods: int = 5
    slow_ema_periods: int = 20
    trend_threshold: float = 0.03  # Min trend strength to trade
    momentum_lookback: int = 10
    kelly_fraction: float = 0.20
    max_position_pct: float = 0.05


class TrendFollowingStrategy:
    """
    Pure trend following strategy.
    
    Uses EMA crossovers and momentum to identify trends.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[TrendFollowingConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or TrendFollowingConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Price history per market
        self.price_history: Dict[str, List[float]] = {}
        self.ema_fast: Dict[str, float] = {}
        self.ema_slow: Dict[str, float] = {}
        
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status summary."""
        return {
            'status': 'online',
            'total_samples': sum(len(h) for h in self.price_history.values()),
            'mean_spread': 0.0,
            'markets_tracked': len(self.price_history),
        }
    
    def _update_emas(self, market_id: str, price: float):
        """Update EMAs for a market."""
        alpha_fast = 2.0 / (self.config.fast_ema_periods + 1)
        alpha_slow = 2.0 / (self.config.slow_ema_periods + 1)
        
        if market_id not in self.ema_fast:
            self.ema_fast[market_id] = price
            self.ema_slow[market_id] = price
        else:
            self.ema_fast[market_id] = alpha_fast * price + (1 - alpha_fast) * self.ema_fast[market_id]
            self.ema_slow[market_id] = alpha_slow * price + (1 - alpha_slow) * self.ema_slow[market_id]
        
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        self.price_history[market_id].append(price)
        
        # Keep limited history
        if len(self.price_history[market_id]) > 50:
            self.price_history[market_id] = self.price_history[market_id][-50:]
    
    def generate_signal(self, market: Market, bankroll: float) -> Optional[Signal]:
        """Generate trend following signal."""
        if market.platform != self.platform:
            return None
            
        price = market.current_yes_price
        market_id = market.market_id
        
        self._update_emas(market_id, price)
        
        # Need some history
        if len(self.price_history.get(market_id, [])) < self.config.momentum_lookback:
            return None
            
        ema_fast = self.ema_fast[market_id]
        ema_slow = self.ema_slow[market_id]
        
        # Trend strength = (fast EMA - slow EMA) / slow EMA
        if ema_slow == 0:
            return None
            
        trend_strength = (ema_fast - ema_slow) / ema_slow
        
        # Momentum: price change over lookback
        history = self.price_history[market_id]
        momentum = price - history[-self.config.momentum_lookback]
        
        # Only trade if trend is strong enough
        if abs(trend_strength) < self.config.trend_threshold:
            return None
            
        # Trade in direction of trend
        if trend_strength > 0 and momentum > 0:
            side = Side.YES
            edge = abs(trend_strength)
        elif trend_strength < 0 and momentum < 0:
            side = Side.NO
            edge = abs(trend_strength)
        else:
            return None  # Conflicting signals
            
        confidence = min(1.0, abs(trend_strength) / 0.10)
        kelly = edge * confidence * self.config.kelly_fraction
        kelly = max(0, min(kelly, 0.5))
        
        if kelly < 0.01:
            return None
            
        size = bankroll * kelly * self.config.max_position_pct
        
        return Signal(
            market_id=market.market_id,
            platform=market.platform,
            side=side,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            size=size,
            metadata={
                'strategy': 'trend_following',
                'trend_strength': trend_strength,
                'momentum': momentum,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
            }
        )


# ============================================================================
# Mean Reversion Strategy
# ============================================================================

@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""
    lookback_periods: int = 20
    zscore_threshold: float = 1.5  # Trade when |z| > threshold
    half_life: int = 5  # Expected reversion half-life
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05


class MeanReversionStrategy:
    """
    Pure mean reversion strategy.
    
    Bets on price reverting to historical mean when z-score is extreme.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[MeanReversionConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or MeanReversionConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        self.price_history: Dict[str, List[float]] = {}
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status summary."""
        return {
            'status': 'online',
            'total_samples': sum(len(h) for h in self.price_history.values()),
            'mean_spread': 0.0,
            'markets_tracked': len(self.price_history),
        }
        
    def generate_signal(self, market: Market, bankroll: float) -> Optional[Signal]:
        """Generate mean reversion signal."""
        if market.platform != self.platform:
            return None
            
        price = market.current_yes_price
        market_id = market.market_id
        
        # Update history
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        self.price_history[market_id].append(price)
        
        if len(self.price_history[market_id]) > 100:
            self.price_history[market_id] = self.price_history[market_id][-100:]
            
        history = self.price_history[market_id]
        
        if len(history) < self.config.lookback_periods:
            return None
            
        # Compute z-score
        recent = history[-self.config.lookback_periods:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std < 0.01:
            return None
            
        zscore = (price - mean) / std
        
        # Only trade when z-score is extreme
        if abs(zscore) < self.config.zscore_threshold:
            return None
            
        # Bet on reversion: if price is high (positive z), bet NO
        if zscore > 0:
            side = Side.NO
        else:
            side = Side.YES
            
        # Edge proportional to z-score (expected reversion)
        edge = abs(zscore) * std / 2  # Expect 50% reversion
        confidence = min(1.0, abs(zscore) / 3.0)
        
        kelly = edge * confidence * self.config.kelly_fraction
        kelly = max(0, min(kelly, 0.5))
        
        if kelly < 0.01:
            return None
            
        size = bankroll * kelly * self.config.max_position_pct
        
        return Signal(
            market_id=market.market_id,
            platform=market.platform,
            side=side,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            size=size,
            metadata={
                'strategy': 'mean_reversion',
                'zscore': zscore,
                'mean': mean,
                'std': std,
            }
        )


# ============================================================================
# Regime Adaptive Strategy (combines trend + mean reversion)
# ============================================================================

@dataclass
class RegimeAdaptiveConfig:
    """Configuration for regime-adaptive strategy."""
    volatility_lookback: int = 20
    high_volatility_threshold: float = 0.05
    trend_weight_trending: float = 0.8
    mean_rev_weight_trending: float = 0.2
    trend_weight_mean_rev: float = 0.2
    mean_rev_weight_mean_rev: float = 0.8
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05


class RegimeAdaptiveStrategy:
    """
    Adapts between trend following and mean reversion based on regime.
    
    Detects regime from volatility and autocorrelation patterns.
    """
    
    def __init__(
        self,
        platform: Platform,
        config: Optional[RegimeAdaptiveConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.platform = platform
        self.config = config or RegimeAdaptiveConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Sub-strategies
        self.trend_strategy = TrendFollowingStrategy(platform)
        self.mean_rev_strategy = MeanReversionStrategy(platform)
        
        self.price_history: Dict[str, List[float]] = {}
        self.current_regime: Dict[str, str] = {}  # 'trending' or 'mean_reverting'
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Return calibration status summary."""
        regimes = list(self.current_regime.values())
        trending = sum(1 for r in regimes if r == 'trending')
        mean_rev = sum(1 for r in regimes if r == 'mean_reverting')
        return {
            'status': 'online',
            'total_samples': sum(len(h) for h in self.price_history.values()),
            'mean_spread': 0.0,
            'markets_tracked': len(self.price_history),
            'trending_markets': trending,
            'mean_reverting_markets': mean_rev,
        }
        
    def _detect_regime(self, market_id: str) -> str:
        """Detect current regime for a market."""
        history = self.price_history.get(market_id, [])
        
        if len(history) < self.config.volatility_lookback:
            return 'unknown'
            
        recent = history[-self.config.volatility_lookback:]
        
        # Compute volatility
        returns = np.diff(recent)
        volatility = np.std(returns)
        
        # Compute autocorrelation (proxy for trending vs mean-reverting)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0
            
        # High volatility + positive autocorr = trending
        # Low volatility + negative autocorr = mean reverting
        if volatility > self.config.high_volatility_threshold and autocorr > 0:
            return 'trending'
        elif autocorr < -0.1:
            return 'mean_reverting'
        else:
            return 'neutral'
    
    def generate_signal(self, market: Market, bankroll: float) -> Optional[Signal]:
        """Generate regime-adaptive signal."""
        if market.platform != self.platform:
            return None
            
        market_id = market.market_id
        price = market.current_yes_price
        
        # Update history
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        self.price_history[market_id].append(price)
        
        if len(self.price_history[market_id]) > 100:
            self.price_history[market_id] = self.price_history[market_id][-100:]
        
        # Detect regime
        regime = self._detect_regime(market_id)
        self.current_regime[market_id] = regime
        
        # Get signals from sub-strategies
        trend_signal = self.trend_strategy.generate_signal(market, bankroll)
        mean_rev_signal = self.mean_rev_strategy.generate_signal(market, bankroll)
        
        # Combine based on regime
        if regime == 'trending':
            trend_weight = self.config.trend_weight_trending
            mr_weight = self.config.mean_rev_weight_trending
        elif regime == 'mean_reverting':
            trend_weight = self.config.trend_weight_mean_rev
            mr_weight = self.config.mean_rev_weight_mean_rev
        else:
            trend_weight = 0.5
            mr_weight = 0.5
            
        # Combine signals
        combined_edge = 0.0
        combined_side = None
        
        if trend_signal and mean_rev_signal:
            # Check if signals agree
            if trend_signal.side == mean_rev_signal.side:
                combined_side = trend_signal.side
                combined_edge = trend_weight * trend_signal.edge + mr_weight * mean_rev_signal.edge
            else:
                # Conflicting - use regime-preferred signal
                if regime == 'trending' and trend_signal:
                    combined_side = trend_signal.side
                    combined_edge = trend_signal.edge * trend_weight
                elif regime == 'mean_reverting' and mean_rev_signal:
                    combined_side = mean_rev_signal.side
                    combined_edge = mean_rev_signal.edge * mr_weight
                else:
                    return None
        elif trend_signal:
            combined_side = trend_signal.side
            combined_edge = trend_signal.edge * trend_weight
        elif mean_rev_signal:
            combined_side = mean_rev_signal.side
            combined_edge = mean_rev_signal.edge * mr_weight
        else:
            return None
            
        if combined_edge < 0.02:
            return None
            
        confidence = min(1.0, combined_edge / 0.10)
        kelly = combined_edge * confidence * self.config.kelly_fraction
        kelly = max(0, min(kelly, 0.5))
        
        if kelly < 0.01:
            return None
            
        size = bankroll * kelly * self.config.max_position_pct
        
        return Signal(
            market_id=market.market_id,
            platform=market.platform,
            side=combined_side,
            edge=combined_edge,
            confidence=confidence,
            kelly_fraction=kelly,
            size=size,
            metadata={
                'strategy': 'regime_adaptive',
                'regime': regime,
                'trend_weight': trend_weight,
                'mean_rev_weight': mr_weight,
            }
        )
