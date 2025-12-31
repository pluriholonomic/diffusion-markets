"""
Execution Cost Estimation Models.

Since we don't have full order book depth, we estimate execution costs
from observable market characteristics (volume, price, volatility).

Based on empirical prediction market microstructure research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CostModelConfig:
    """Configuration for execution cost estimation."""

    # Base spread parameters (calibrated to prediction markets)
    base_spread_bps: float = 100.0  # 1% base spread
    volume_decay: float = 0.3  # Spread decreases with log(volume)
    price_extreme_penalty: float = 2.0  # Spread widens near 0/1

    # Market impact parameters (Kyle's lambda-like)
    impact_coefficient: float = 0.001  # Price impact per unit traded
    impact_decay_seconds: float = 300.0  # Impact decays over 5 min

    # Fixed costs
    gas_cost_usd: float = 0.0  # On-chain costs (if any)
    platform_fee_bps: float = 0.0  # Platform fees

    # Conservative mode (for backtesting)
    conservative_multiplier: float = 1.5  # Safety margin


@dataclass
class ExecutionCostEstimate:
    """Estimated execution costs for a trade."""

    spread_bps: float  # Half-spread in basis points
    spread_cost: float  # $ cost from spread
    impact_cost: float  # $ cost from market impact
    fixed_cost: float  # Fixed $ costs
    total_cost: float  # Total execution cost

    effective_price: float  # Price after costs
    slippage_bps: float  # Total slippage in bps

    def __repr__(self) -> str:
        return (
            f"ExecutionCost(spread={self.spread_bps:.1f}bps, "
            f"impact=${self.impact_cost:.4f}, total=${self.total_cost:.4f})"
        )


class SpreadEstimator:
    """
    Estimates bid/ask spread from market characteristics.

    Based on empirical observations of prediction market microstructure:
    - Amihud illiquidity: spread ~ 1/sqrt(volume)
    - Price extremes: spread widens near 0 and 1
    - Time effects: spread narrows near resolution
    """

    def __init__(self, cfg: CostModelConfig = CostModelConfig()):
        self.cfg = cfg

    def estimate_spread(
        self,
        mid_price: float,
        volume_usd: float = 10000.0,
        time_to_expiry_days: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Estimate half-spread in basis points.

        Args:
            mid_price: Current mid price (0-1 for binary markets)
            volume_usd: Trading volume in USD (higher = tighter spread)
            time_to_expiry_days: Days until market resolution
            volatility: Recent price volatility (if known)

        Returns:
            Estimated half-spread in basis points
        """
        # Start with base spread
        spread = self.cfg.base_spread_bps

        # Volume adjustment: spread ~ 1/sqrt(volume)
        # Normalize to $10K baseline
        if volume_usd > 0:
            volume_factor = np.sqrt(10000.0 / max(volume_usd, 100.0))
            spread *= (1.0 + self.cfg.volume_decay * (volume_factor - 1.0))

        # Price extreme adjustment
        # Spread widens as price approaches 0 or 1
        # Using logistic-like penalty
        if 0 < mid_price < 1:
            price_dist = min(mid_price, 1 - mid_price)  # Distance to nearest extreme
            if price_dist < 0.1:
                extreme_penalty = self.cfg.price_extreme_penalty * (0.1 - price_dist) / 0.1
                spread *= (1.0 + extreme_penalty)

        # Time to expiry adjustment
        # Spread tightens as resolution approaches
        if time_to_expiry_days is not None and time_to_expiry_days > 0:
            if time_to_expiry_days < 1:
                # Very close to expiry: spread tightens significantly
                spread *= 0.5
            elif time_to_expiry_days < 7:
                # Within a week: moderate tightening
                spread *= 0.8

        # Volatility adjustment (if provided)
        if volatility is not None and volatility > 0:
            # Higher volatility = wider spread
            vol_baseline = 0.05  # 5% daily vol baseline
            spread *= (1.0 + 0.5 * (volatility / vol_baseline - 1.0))

        return max(spread, 10.0)  # Minimum 10 bps (0.1%)


class MarketImpactEstimator:
    """
    Estimates market impact (temporary and permanent price impact).

    Uses simplified Kyle's lambda model adapted for prediction markets.
    """

    def __init__(self, cfg: CostModelConfig = CostModelConfig()):
        self.cfg = cfg
        self._recent_trades: Dict[str, list] = {}  # market_id -> [(ts, size), ...]

    def estimate_impact(
        self,
        trade_size_usd: float,
        mid_price: float,
        daily_volume_usd: float = 10000.0,
        market_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Estimate price impact from a trade.

        Args:
            trade_size_usd: Trade size in USD
            mid_price: Current mid price
            daily_volume_usd: Average daily volume
            market_id: Market identifier (for tracking recent trades)
            timestamp: Trade timestamp (for decay calculation)

        Returns:
            Estimated price impact in basis points
        """
        if trade_size_usd == 0:
            return 0.0

        # Kyle's lambda: impact ~ trade_size / sqrt(volume)
        # Normalized so that trading 1% of daily volume moves price ~10 bps
        volume_normalized = max(daily_volume_usd, 100.0)
        base_impact = (
            self.cfg.impact_coefficient
            * abs(trade_size_usd)
            / np.sqrt(volume_normalized)
            * 10000  # Convert to bps
        )

        # Recent trade amplification
        # If we've traded recently, impact is higher (less liquidity replenished)
        if market_id and timestamp and market_id in self._recent_trades:
            recent = self._recent_trades[market_id]
            # Decay old trades
            recent = [
                (ts, sz) for ts, sz in recent
                if timestamp - ts < self.cfg.impact_decay_seconds
            ]
            self._recent_trades[market_id] = recent

            # Amplify impact based on recent activity
            recent_volume = sum(abs(sz) for _, sz in recent)
            if recent_volume > 0:
                amplification = 1.0 + 0.5 * recent_volume / volume_normalized
                base_impact *= amplification

        # Record this trade
        if market_id and timestamp:
            if market_id not in self._recent_trades:
                self._recent_trades[market_id] = []
            self._recent_trades[market_id].append((timestamp, trade_size_usd))

        return base_impact


class ExecutionCostModel:
    """
    Complete execution cost model combining spread and impact.

    Usage:
        model = ExecutionCostModel()
        cost = model.estimate_cost(
            side="buy",
            size_usd=100,
            mid_price=0.6,
            volume_usd=50000,
        )
        print(f"Effective price: {cost.effective_price}")
        print(f"Total cost: ${cost.total_cost}")
    """

    def __init__(self, cfg: CostModelConfig = CostModelConfig()):
        self.cfg = cfg
        self.spread_estimator = SpreadEstimator(cfg)
        self.impact_estimator = MarketImpactEstimator(cfg)

    def estimate_cost(
        self,
        side: str,  # "buy" or "sell"
        size_usd: float,
        mid_price: float,
        volume_usd: float = 10000.0,
        time_to_expiry_days: Optional[float] = None,
        volatility: Optional[float] = None,
        market_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ExecutionCostEstimate:
        """
        Estimate total execution cost for a trade.

        Args:
            side: Trade direction ("buy" or "sell")
            size_usd: Trade size in USD
            mid_price: Current mid price
            volume_usd: Market volume
            time_to_expiry_days: Time to resolution
            volatility: Price volatility
            market_id: Market identifier
            timestamp: Trade timestamp

        Returns:
            ExecutionCostEstimate with all cost components
        """
        if size_usd == 0:
            return ExecutionCostEstimate(
                spread_bps=0,
                spread_cost=0,
                impact_cost=0,
                fixed_cost=0,
                total_cost=0,
                effective_price=mid_price,
                slippage_bps=0,
            )

        # 1. Spread cost
        spread_bps = self.spread_estimator.estimate_spread(
            mid_price=mid_price,
            volume_usd=volume_usd,
            time_to_expiry_days=time_to_expiry_days,
            volatility=volatility,
        )
        spread_cost = abs(size_usd) * spread_bps / 10000

        # 2. Market impact
        impact_bps = self.impact_estimator.estimate_impact(
            trade_size_usd=size_usd,
            mid_price=mid_price,
            daily_volume_usd=volume_usd,
            market_id=market_id,
            timestamp=timestamp,
        )
        impact_cost = abs(size_usd) * impact_bps / 10000

        # 3. Fixed costs
        fixed_cost = self.cfg.gas_cost_usd + abs(size_usd) * self.cfg.platform_fee_bps / 10000

        # 4. Apply conservative multiplier for backtesting
        total_slippage_bps = (spread_bps + impact_bps) * self.cfg.conservative_multiplier
        total_cost = (spread_cost + impact_cost) * self.cfg.conservative_multiplier + fixed_cost

        # 5. Effective price
        direction = 1 if side == "buy" else -1
        price_adjustment = mid_price * total_slippage_bps / 10000 * direction
        effective_price = mid_price + price_adjustment

        return ExecutionCostEstimate(
            spread_bps=spread_bps,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            fixed_cost=fixed_cost,
            total_cost=total_cost,
            effective_price=effective_price,
            slippage_bps=total_slippage_bps,
        )


def quick_cost_estimate(
    size_usd: float,
    mid_price: float,
    volume_usd: float = 10000.0,
    conservative: bool = True,
) -> float:
    """
    Quick cost estimate for simple use cases.

    Returns total cost in USD.
    """
    cfg = CostModelConfig(
        conservative_multiplier=1.5 if conservative else 1.0
    )
    model = ExecutionCostModel(cfg)
    est = model.estimate_cost(
        side="buy",
        size_usd=size_usd,
        mid_price=mid_price,
        volume_usd=volume_usd,
    )
    return est.total_cost



