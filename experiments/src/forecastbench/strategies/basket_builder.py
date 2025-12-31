"""
Basket construction methods for group mean-reversion statistical arbitrage.

This module implements three complementary approaches:
1. Calibration-based long/short: Trade based on model vs market discrepancy
2. Dollar-neutral portfolios: Market-neutral positions within groups
3. Cross-market Frechet arbitrage: Exploit constraint violations

All methods support regime-aware trading where positions are adjusted
based on the detected calibration regime (mean-revert vs momentum).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from forecastbench.strategies.regime_detector import (
    GroupCalibrationTracker,
    RegimeType,
    RegimeDetectorConfig,
)


@dataclass
class Position:
    """A single position in a market."""
    
    market_id: str
    group: str
    direction: int  # +1 for long YES, -1 for long NO
    size: float     # Position size (fraction of portfolio)
    entry_price: float
    model_price: float
    edge: float     # Expected edge = direction * (model_price - entry_price)
    regime: RegimeType
    method: str     # Which basket method generated this position


@dataclass
class Basket:
    """A collection of positions forming a trading basket."""
    
    positions: List[Position] = field(default_factory=list)
    method: str = ""
    group: Optional[str] = None
    
    @property
    def total_long(self) -> float:
        """Total long exposure."""
        return sum(p.size for p in self.positions if p.direction > 0)
    
    @property
    def total_short(self) -> float:
        """Total short exposure (absolute value)."""
        return sum(p.size for p in self.positions if p.direction < 0)
    
    @property
    def net_exposure(self) -> float:
        """Net exposure (long - short)."""
        return self.total_long - self.total_short
    
    @property
    def gross_exposure(self) -> float:
        """Gross exposure (long + short)."""
        return self.total_long + self.total_short
    
    @property
    def n_positions(self) -> int:
        return len(self.positions)
    
    @property
    def expected_pnl(self) -> float:
        """Expected PnL based on edges."""
        return sum(p.size * p.edge for p in self.positions)


@dataclass(frozen=True)
class BasketBuilderConfig:
    """Configuration for basket construction."""
    
    # Position sizing
    kelly_fraction: float = 0.25
    max_position_size: float = 0.10   # Max 10% per position
    max_group_exposure: float = 0.30  # Max 30% per group
    
    # Edge thresholds
    min_edge: float = 0.02            # Minimum |p - q| to trade
    min_confidence: float = 0.60      # Minimum model confidence
    
    # Dollar-neutral constraints
    max_net_exposure: float = 0.10    # Max 10% net exposure for neutral
    
    # Frechet constraints
    frechet_violation_threshold: float = 0.01  # Min violation to trade
    
    # Regime adjustments
    regime_scale: Dict[str, float] = field(default_factory=lambda: {
        "mean_revert": 1.0,
        "momentum": 0.5,    # Reduce size in momentum regimes
        "neutral": 0.25,    # Minimal size when unclear
    })


class BasketBuilder(ABC):
    """Abstract base class for basket construction methods."""
    
    def __init__(self, cfg: BasketBuilderConfig = BasketBuilderConfig()):
        self.cfg = cfg
    
    @abstractmethod
    def build_basket(
        self,
        market_ids: np.ndarray,
        groups: np.ndarray,
        market_prices: np.ndarray,
        model_prices: np.ndarray,
        regimes: Dict[str, RegimeType],
        **kwargs,
    ) -> Basket:
        """Build a trading basket from market data."""
        pass
    
    def _compute_kelly_size(
        self,
        edge: float,
        price: float,
        regime: RegimeType,
    ) -> float:
        """Compute Kelly-based position size with regime adjustment."""
        if abs(edge) < self.cfg.min_edge:
            return 0.0
        
        # Kelly fraction for binary outcome
        # f* = (p - q) / (1 - q) for long, (q - p) / q for short
        if edge > 0:
            kelly = edge / max(1.0 - price, 1e-6)
        else:
            kelly = abs(edge) / max(price, 1e-6)
        
        # Apply fractional Kelly
        size = kelly * self.cfg.kelly_fraction
        
        # Apply regime scaling
        regime_scale = self.cfg.regime_scale.get(regime.value, 0.25)
        size *= regime_scale
        
        # Cap position size
        size = min(size, self.cfg.max_position_size)
        
        return float(size)


class CalibrationBasedBuilder(BasketBuilder):
    """
    Build baskets based on calibration-adjusted model/market discrepancy.
    
    Long markets where model_forecast > market_price (model says underpriced)
    Short markets where model_forecast < market_price (model says overpriced)
    Size proportional to calibration-adjusted edge.
    """
    
    def build_basket(
        self,
        market_ids: np.ndarray,
        groups: np.ndarray,
        market_prices: np.ndarray,
        model_prices: np.ndarray,
        regimes: Dict[str, RegimeType],
        *,
        calibration_tracker: Optional[GroupCalibrationTracker] = None,
        **kwargs,
    ) -> Basket:
        """
        Build calibration-based long/short basket.
        
        Args:
            market_ids: Market identifiers
            groups: Group assignments
            market_prices: Current market prices
            model_prices: Model forecast prices
            regimes: Regime classification per group
            calibration_tracker: Optional tracker for calibration adjustments
        """
        market_ids = np.asarray(market_ids).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        market_prices = np.asarray(market_prices, dtype=np.float64).reshape(-1)
        model_prices = np.asarray(model_prices, dtype=np.float64).reshape(-1)
        
        positions: List[Position] = []
        group_exposure: Dict[str, float] = {}
        
        for i, (mid, g, q, p) in enumerate(zip(market_ids, groups, market_prices, model_prices)):
            g_str = str(g)
            regime = regimes.get(g_str, RegimeType.NEUTRAL)
            
            # Skip if regime suggests avoiding
            if regime == RegimeType.NEUTRAL:
                continue
            
            # Compute raw edge
            raw_edge = p - q
            
            # Adjust for calibration bias if tracker available
            if calibration_tracker is not None:
                calib_error = calibration_tracker.get_calibration_error(g_str)
                # If model underestimates (calib_error > 0), reduce long bias
                adjusted_edge = raw_edge - calib_error * 0.5
            else:
                adjusted_edge = raw_edge
            
            if abs(adjusted_edge) < self.cfg.min_edge:
                continue
            
            # Direction based on adjusted edge
            direction = 1 if adjusted_edge > 0 else -1
            
            # Compute position size
            size = self._compute_kelly_size(adjusted_edge, q, regime)
            
            if size < 1e-6:
                continue
            
            # Check group exposure limit
            current_group_exposure = group_exposure.get(g_str, 0.0)
            if current_group_exposure + size > self.cfg.max_group_exposure:
                size = max(0.0, self.cfg.max_group_exposure - current_group_exposure)
                if size < 1e-6:
                    continue
            
            group_exposure[g_str] = current_group_exposure + size
            
            positions.append(Position(
                market_id=str(mid),
                group=g_str,
                direction=direction,
                size=size,
                entry_price=float(q),
                model_price=float(p),
                edge=abs(adjusted_edge),
                regime=regime,
                method="calibration",
            ))
        
        return Basket(positions=positions, method="calibration")


class DollarNeutralBuilder(BasketBuilder):
    """
    Build dollar-neutral portfolios within each group.
    
    For each group, construct a portfolio with approximately equal
    long and short exposure, capturing relative value.
    """
    
    def build_basket(
        self,
        market_ids: np.ndarray,
        groups: np.ndarray,
        market_prices: np.ndarray,
        model_prices: np.ndarray,
        regimes: Dict[str, RegimeType],
        **kwargs,
    ) -> Basket:
        """
        Build dollar-neutral basket with balanced long/short per group.
        """
        market_ids = np.asarray(market_ids).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        market_prices = np.asarray(market_prices, dtype=np.float64).reshape(-1)
        model_prices = np.asarray(model_prices, dtype=np.float64).reshape(-1)
        
        # Group markets
        group_markets: Dict[str, List[Tuple[str, float, float, int]]] = {}
        
        for mid, g, q, p in zip(market_ids, groups, market_prices, model_prices):
            g_str = str(g)
            if g_str not in group_markets:
                group_markets[g_str] = []
            
            edge = p - q
            if abs(edge) >= self.cfg.min_edge:
                group_markets[g_str].append((str(mid), float(q), float(p), edge))
        
        positions: List[Position] = []
        
        for g_str, markets in group_markets.items():
            regime = regimes.get(g_str, RegimeType.NEUTRAL)
            
            if regime == RegimeType.NEUTRAL or len(markets) < 2:
                continue
            
            # Split into longs and shorts
            longs = [(m, q, p, e) for m, q, p, e in markets if e > 0]
            shorts = [(m, q, p, e) for m, q, p, e in markets if e < 0]
            
            if not longs or not shorts:
                continue
            
            # Balance exposures
            total_long_edge = sum(e for _, _, _, e in longs)
            total_short_edge = sum(abs(e) for _, _, _, e in shorts)
            
            # Scale to balance
            balance_factor = min(total_long_edge, total_short_edge)
            if balance_factor < 1e-6:
                continue
            
            # Allocate group budget equally to long and short
            group_budget = self.cfg.max_group_exposure / 2
            
            regime_scale = self.cfg.regime_scale.get(regime.value, 0.25)
            group_budget *= regime_scale
            
            # Long positions
            for mid, q, p, e in longs:
                weight = e / total_long_edge
                size = min(weight * group_budget, self.cfg.max_position_size)
                
                if size > 1e-6:
                    positions.append(Position(
                        market_id=mid,
                        group=g_str,
                        direction=1,
                        size=size,
                        entry_price=q,
                        model_price=p,
                        edge=e,
                        regime=regime,
                        method="dollar_neutral",
                    ))
            
            # Short positions
            for mid, q, p, e in shorts:
                weight = abs(e) / total_short_edge
                size = min(weight * group_budget, self.cfg.max_position_size)
                
                if size > 1e-6:
                    positions.append(Position(
                        market_id=mid,
                        group=g_str,
                        direction=-1,
                        size=size,
                        entry_price=q,
                        model_price=p,
                        edge=abs(e),
                        regime=regime,
                        method="dollar_neutral",
                    ))
        
        return Basket(positions=positions, method="dollar_neutral")


class FrechetArbitrageBuilder(BasketBuilder):
    """
    Build baskets that exploit Frechet constraint violations.
    
    For related markets (A, B, A∧B), detect violations of:
    - P(A∧B) ≤ min(P(A), P(B))
    - P(A∧B) ≥ max(0, P(A) + P(B) - 1)
    
    Trade the spread when violations exceed threshold.
    """
    
    def build_basket(
        self,
        market_ids: np.ndarray,
        groups: np.ndarray,
        market_prices: np.ndarray,
        model_prices: np.ndarray,
        regimes: Dict[str, RegimeType],
        *,
        bundles: Optional[List[List[int]]] = None,
        bundle_structure: Literal["frechet", "implication"] = "frechet",
        **kwargs,
    ) -> Basket:
        """
        Build Frechet arbitrage basket.
        
        Args:
            bundles: List of index triplets [A, B, A∧B] for Frechet constraints
            bundle_structure: Type of constraint structure
        """
        if bundles is None or len(bundles) == 0:
            return Basket(positions=[], method="frechet")
        
        market_ids = np.asarray(market_ids).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        market_prices = np.asarray(market_prices, dtype=np.float64).reshape(-1)
        model_prices = np.asarray(model_prices, dtype=np.float64).reshape(-1)
        
        positions: List[Position] = []
        
        for bundle in bundles:
            if len(bundle) != 3:
                continue
            
            i_a, i_b, i_ab = bundle
            
            if max(i_a, i_b, i_ab) >= len(market_prices):
                continue
            
            q_a, q_b, q_ab = market_prices[i_a], market_prices[i_b], market_prices[i_ab]
            p_a, p_b, p_ab = model_prices[i_a], model_prices[i_b], model_prices[i_ab]
            g = str(groups[i_a])
            regime = regimes.get(g, RegimeType.NEUTRAL)
            
            # Compute Frechet violations on market prices
            frechet_lo = max(0.0, q_a + q_b - 1.0)
            frechet_hi = min(q_a, q_b)
            
            violation_lo = frechet_lo - q_ab  # Positive if q_ab too low
            violation_hi = q_ab - frechet_hi  # Positive if q_ab too high
            
            max_violation = max(violation_lo, violation_hi)
            
            if max_violation < self.cfg.frechet_violation_threshold:
                continue
            
            # Trade direction based on violation type
            if violation_lo > violation_hi:
                # q_ab is too low, buy A∧B
                direction_ab = 1
                edge = violation_lo
            else:
                # q_ab is too high, sell A∧B
                direction_ab = -1
                edge = violation_hi
            
            # Size based on violation magnitude
            size = min(
                edge * self.cfg.kelly_fraction,
                self.cfg.max_position_size
            )
            
            regime_scale = self.cfg.regime_scale.get(regime.value, 0.25)
            size *= regime_scale
            
            if size < 1e-6:
                continue
            
            # Add position on the joint market
            positions.append(Position(
                market_id=str(market_ids[i_ab]),
                group=g,
                direction=direction_ab,
                size=size,
                entry_price=q_ab,
                model_price=p_ab,
                edge=edge,
                regime=regime,
                method="frechet",
            ))
        
        return Basket(positions=positions, method="frechet")


class UnifiedBasketBuilder:
    """
    Unified basket builder that combines multiple methods.
    
    This is the main entry point for basket construction, allowing
    flexible combination of calibration, dollar-neutral, and Frechet methods.
    """
    
    def __init__(
        self,
        cfg: BasketBuilderConfig = BasketBuilderConfig(),
        methods: List[str] = None,
    ):
        self.cfg = cfg
        self.methods = methods or ["calibration", "dollar_neutral"]
        
        self._builders = {
            "calibration": CalibrationBasedBuilder(cfg),
            "dollar_neutral": DollarNeutralBuilder(cfg),
            "frechet": FrechetArbitrageBuilder(cfg),
        }
    
    def build_baskets(
        self,
        market_ids: np.ndarray,
        groups: np.ndarray,
        market_prices: np.ndarray,
        model_prices: np.ndarray,
        regimes: Dict[str, RegimeType],
        **kwargs,
    ) -> Dict[str, Basket]:
        """
        Build baskets using all configured methods.
        
        Returns a dict mapping method name to basket.
        """
        results = {}
        
        for method in self.methods:
            if method in self._builders:
                basket = self._builders[method].build_basket(
                    market_ids=market_ids,
                    groups=groups,
                    market_prices=market_prices,
                    model_prices=model_prices,
                    regimes=regimes,
                    **kwargs,
                )
                results[method] = basket
        
        return results
    
    def build_combined_basket(
        self,
        market_ids: np.ndarray,
        groups: np.ndarray,
        market_prices: np.ndarray,
        model_prices: np.ndarray,
        regimes: Dict[str, RegimeType],
        *,
        method_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Basket:
        """
        Build a single combined basket from all methods.
        
        Positions from different methods are combined with optional weighting.
        """
        method_weights = method_weights or {m: 1.0 for m in self.methods}
        
        baskets = self.build_baskets(
            market_ids=market_ids,
            groups=groups,
            market_prices=market_prices,
            model_prices=model_prices,
            regimes=regimes,
            **kwargs,
        )
        
        combined_positions: List[Position] = []
        
        for method, basket in baskets.items():
            weight = method_weights.get(method, 1.0)
            for pos in basket.positions:
                # Scale position size by method weight
                scaled_pos = Position(
                    market_id=pos.market_id,
                    group=pos.group,
                    direction=pos.direction,
                    size=pos.size * weight,
                    entry_price=pos.entry_price,
                    model_price=pos.model_price,
                    edge=pos.edge,
                    regime=pos.regime,
                    method=pos.method,
                )
                combined_positions.append(scaled_pos)
        
        return Basket(positions=combined_positions, method="combined")


def aggregate_positions(basket: Basket) -> Dict[str, Position]:
    """
    Aggregate multiple positions in the same market into a single net position.
    """
    aggregated: Dict[str, Position] = {}
    
    for pos in basket.positions:
        if pos.market_id in aggregated:
            existing = aggregated[pos.market_id]
            net_size = existing.size * existing.direction + pos.size * pos.direction
            
            if abs(net_size) < 1e-9:
                del aggregated[pos.market_id]
            else:
                aggregated[pos.market_id] = Position(
                    market_id=pos.market_id,
                    group=pos.group,
                    direction=1 if net_size > 0 else -1,
                    size=abs(net_size),
                    entry_price=pos.entry_price,
                    model_price=pos.model_price,
                    edge=(existing.edge + pos.edge) / 2,  # Average edge
                    regime=pos.regime,
                    method="aggregated",
                )
        else:
            aggregated[pos.market_id] = pos
    
    return aggregated
