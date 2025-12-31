"""
Blackwell Calibration Trading Strategy.

Implements the validated Blackwell arbitrage strategy with:
1. Rolling calibration (no lookahead bias)
2. Configurable binning granularity
3. Statistical significance filtering
4. Risk-parity position sizing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backtest.strategies.base import BaseStrategy


@dataclass
class BlackwellCalibrationConfig:
    """Configuration for Blackwell calibration strategy."""
    
    # Binning
    n_bins: int = 10  # Number of price bins (10-20 recommended)
    
    # Thresholds
    g_bar_threshold: float = 0.05  # Minimum |g̅| to trade
    t_stat_threshold: float = 2.0  # Minimum t-statistic for significance
    min_samples_per_bin: int = 20  # Minimum samples to estimate g̅
    
    # Rolling window
    lookback_trades: int = 500  # Number of historical trades for calibration
    recalibrate_freq: int = 50  # Recalibrate every N trades
    
    # Position sizing
    use_risk_parity: bool = True  # Size by inverse max loss
    target_max_loss: float = 0.2  # Target max loss per trade
    leverage: float = 1.0  # Position multiplier
    max_position: float = 1.0  # Hard cap on position size
    
    # Filters
    price_min: float = 0.0  # Minimum price to trade
    price_max: float = 1.0  # Maximum price to trade
    

@dataclass
class BinStats:
    """Statistics for a single price bin."""
    g_bar: float = 0.0
    sigma: float = 0.0
    n_samples: int = 0
    t_stat: float = 0.0
    is_significant: bool = False
    

@dataclass
class BlackwellState:
    """Internal state for the Blackwell strategy."""
    
    # Rolling history of (price, outcome) pairs
    history: List[Tuple[float, float, str]] = field(default_factory=list)  # (price, outcome, market_id)
    
    # Current bin statistics
    bin_stats: Dict[int, BinStats] = field(default_factory=dict)
    
    # Trade counter for recalibration
    trades_since_recalibration: int = 0
    
    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0


class BlackwellCalibrationStrategy(BaseStrategy):
    """
    Trading strategy based on Blackwell approachability.
    
    Identifies price bins where markets are systematically miscalibrated
    (g̅ = E[Y - q | q ∈ bin] ≠ 0) and trades accordingly.
    
    Key features:
    - Rolling calibration: only uses historical data (no lookahead bias)
    - Statistical filtering: only trades when miscalibration is significant
    - Risk-parity sizing: equal max loss per trade
    """
    
    def __init__(self, cfg: BlackwellCalibrationConfig):
        self.cfg = cfg
        self.state = BlackwellState()
        self._bin_edges = np.linspace(0, 1, cfg.n_bins + 1)
        
    def reset(self) -> None:
        """Reset strategy state."""
        self.state = BlackwellState()
        
    def _price_to_bin(self, price: float) -> int:
        """Map price to bin index."""
        bin_idx = np.digitize(price, self._bin_edges) - 1
        return int(np.clip(bin_idx, 0, self.cfg.n_bins - 1))
    
    def _recalibrate(self) -> None:
        """Recalibrate bin statistics from history."""
        if len(self.state.history) < self.cfg.min_samples_per_bin:
            return
            
        # Use only recent history
        history = self.state.history[-self.cfg.lookback_trades:]
        
        # Group by bin
        bin_data: Dict[int, List[Tuple[float, float]]] = {
            b: [] for b in range(self.cfg.n_bins)
        }
        
        for price, outcome, _ in history:
            bin_idx = self._price_to_bin(price)
            bin_data[bin_idx].append((price, outcome))
        
        # Compute statistics for each bin
        self.state.bin_stats = {}
        
        for bin_idx, data in bin_data.items():
            if len(data) < self.cfg.min_samples_per_bin:
                continue
                
            prices = np.array([p for p, _ in data])
            outcomes = np.array([o for _, o in data])
            residuals = outcomes - prices
            
            g_bar = residuals.mean()
            sigma = residuals.std()
            n = len(data)
            
            # Standard error and t-stat
            se = sigma / np.sqrt(n) if n > 0 else 0
            t_stat = g_bar / se if se > 0 else 0
            
            is_significant = (
                abs(t_stat) >= self.cfg.t_stat_threshold and
                abs(g_bar) >= self.cfg.g_bar_threshold
            )
            
            self.state.bin_stats[bin_idx] = BinStats(
                g_bar=g_bar,
                sigma=sigma,
                n_samples=n,
                t_stat=t_stat,
                is_significant=is_significant,
            )
        
        self.state.trades_since_recalibration = 0
        
    def on_resolution(self, market_id: str, outcome: float, price: float) -> None:
        """
        Record resolution for calibration.
        
        Args:
            market_id: Market identifier
            outcome: Realized outcome (0 or 1)
            price: Price at which we would have traded
        """
        self.state.history.append((price, outcome, market_id))
        
        # Recalibrate periodically
        self.state.trades_since_recalibration += 1
        if self.state.trades_since_recalibration >= self.cfg.recalibrate_freq:
            self._recalibrate()
            
    def get_position(self, market_id: str, current_price: float) -> float:
        """
        Compute target position for a market.
        
        Args:
            market_id: Market identifier
            current_price: Current market price
            
        Returns:
            Target position size (positive = long, negative = short)
        """
        # Check price filters
        if current_price < self.cfg.price_min or current_price > self.cfg.price_max:
            return 0.0
            
        # Get bin
        bin_idx = self._price_to_bin(current_price)
        
        # Check if bin is tradeable
        if bin_idx not in self.state.bin_stats:
            return 0.0
            
        stats = self.state.bin_stats[bin_idx]
        
        if not stats.is_significant:
            return 0.0
            
        # Direction: buy if underpriced (g̅ > 0), sell if overpriced (g̅ < 0)
        direction = np.sign(stats.g_bar)
        
        # Position sizing
        if self.cfg.use_risk_parity:
            # Size inversely proportional to max loss
            if direction > 0:
                max_loss = current_price  # Buy: lose price if outcome=0
            else:
                max_loss = 1 - current_price  # Sell: lose 1-price if outcome=1
                
            if max_loss <= 0:
                return 0.0
                
            size = min(self.cfg.max_position, self.cfg.target_max_loss / max_loss)
        else:
            size = 1.0
            
        # Apply leverage
        size *= self.cfg.leverage
        
        # Cap at max
        size = min(size, self.cfg.max_position)
        
        return float(direction * size)
    
    def on_price_update(
        self,
        market_id: str,
        old_price: float,
        new_price: float,
        state: any,
    ) -> Optional[Dict[str, float]]:
        """Handle price update event."""
        position = self.get_position(market_id, new_price)
        
        if abs(position) > 1e-6:
            return {market_id: position}
        return None
        
    def on_snapshot(
        self,
        timestamp: float,
        state: any,
    ) -> Optional[Dict[str, float]]:
        """Handle periodic snapshot."""
        # Could implement portfolio-level rebalancing here
        return None
        
    def on_ct_refresh(self, ct_samples: np.ndarray, market_ids: List[str]) -> None:
        """Handle C_t refresh (not used in pure calibration strategy)."""
        pass
        
    def get_stats(self) -> Dict:
        """Get current strategy statistics."""
        return {
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "total_pnl": self.state.total_pnl,
            "history_size": len(self.state.history),
            "tradeable_bins": sum(
                1 for s in self.state.bin_stats.values() if s.is_significant
            ),
            "bin_stats": {
                b: {
                    "g_bar": s.g_bar,
                    "t_stat": s.t_stat,
                    "n_samples": s.n_samples,
                    "significant": s.is_significant,
                }
                for b, s in self.state.bin_stats.items()
            },
        }


