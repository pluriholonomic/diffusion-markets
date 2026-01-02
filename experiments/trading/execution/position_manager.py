#!/usr/bin/env python3
"""
Position Manager with Learned Entry/Exit Logic

Implements:
- Profit-taking: Close position when up > x%
- Stop-loss: Close position when down > y%
- Online learning for optimal x, y thresholds
- EMA-based signals for PnL and price

The thresholds are learned via online linear regression, using:
- Features: PnL EMAs, price EMAs, volatility, time in position
- Target: Optimal exit decision in hindsight
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    position_id: str
    market_id: str
    platform: str
    side: str  # 'yes' or 'no'
    entry_price: float
    entry_time: datetime
    size: float  # Dollar amount
    strategy: str
    
    # Market info
    market_question: str = ""  # Human-readable market question
    
    # Tracking
    current_price: float = 0.0
    peak_price: float = 0.0  # Best price seen
    trough_price: float = 1.0  # Worst price seen
    
    # EMAs
    price_ema_fast: float = 0.0  # Fast EMA (short-term)
    price_ema_slow: float = 0.0  # Slow EMA (long-term)
    pnl_ema: float = 0.0
    
    # Exit info
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # 'profit_take', 'stop_loss', 'expiry', 'manual'
    realized_pnl: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL."""
        if not self.is_open:
            return self.realized_pnl or 0.0
        return self._compute_pnl(self.current_price)
    
    @property
    def unrealized_return(self) -> float:
        """Unrealized return as percentage."""
        if self.size == 0:
            return 0.0
        return (self.unrealized_pnl / self.size) * 100
    
    @property
    def time_in_position(self) -> timedelta:
        """Time since position was opened."""
        end_time = self.exit_time or datetime.utcnow()
        return end_time - self.entry_time
    
    def _compute_pnl(self, yes_price: float) -> float:
        """
        Compute PnL at a given YES price.
        
        NOTE: entry_price is stored as the SIDE price (YES price for YES bets, NO price for NO bets).
        current_price is always the YES price from the market.
        """
        if self.side == 'yes':
            # Bought YES at entry_price (which IS the YES price)
            # Current value at yes_price
            if self.entry_price <= 0:
                return 0.0
            return self.size * (yes_price - self.entry_price) / self.entry_price
        else:
            # Bought NO at entry_price (which is the NO price, i.e., 1 - yes_entry)
            # entry_price = NO price we paid
            # current NO price = 1 - yes_price
            no_current = 1 - yes_price
            if self.entry_price <= 0:
                return 0.0
            return self.size * (no_current - self.entry_price) / self.entry_price
    
    def update_price(self, new_price: float, ema_alpha_fast: float = 0.1, ema_alpha_slow: float = 0.02):
        """Update current price and EMAs."""
        self.current_price = new_price
        self.peak_price = max(self.peak_price, new_price)
        self.trough_price = min(self.trough_price, new_price)
        
        # Update price EMAs
        if self.price_ema_fast == 0:
            self.price_ema_fast = new_price
            self.price_ema_slow = new_price
        else:
            self.price_ema_fast = ema_alpha_fast * new_price + (1 - ema_alpha_fast) * self.price_ema_fast
            self.price_ema_slow = ema_alpha_slow * new_price + (1 - ema_alpha_slow) * self.price_ema_slow
        
        # Update PnL EMA
        pnl = self.unrealized_pnl
        if self.pnl_ema == 0:
            self.pnl_ema = pnl
        else:
            self.pnl_ema = ema_alpha_fast * pnl + (1 - ema_alpha_fast) * self.pnl_ema
    
    def close(self, exit_price: float, reason: str):
        """Close the position."""
        self.exit_price = exit_price
        self.exit_time = datetime.utcnow()
        self.exit_reason = reason
        self.realized_pnl = self._compute_pnl(exit_price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position_id': self.position_id,
            'market_id': self.market_id,
            'market_question': self.market_question,
            'platform': self.platform,
            'side': self.side,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'size': self.size,
            'strategy': self.strategy,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_return': self.unrealized_return,
            'is_open': self.is_open,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
        }


@dataclass
class OnlineLearnerConfig:
    """Configuration for online learner.
    
    Parameters optimized via CMA-ES on historical data (2026-01-02):
    - profit_take_pct: 60% - exit winners at 60% gain
    - stop_loss_pct: 38.5% - exit losers at 38.5% loss  
    - trailing_stop_pct: 19.6% - trail by 19.6%
    - ema_alpha_fast: 0.15 (vs 0.10) - faster reaction
    - ema_alpha_slow: 0.028 (vs 0.02) - slightly faster
    - learning_rate: 0.05 (vs 0.01) - faster adaptation
    """
    # Initial thresholds (optimized via CMA-ES)
    initial_profit_take_pct: float = 60.0  # Take profit at 60% gain (optimized)
    initial_stop_loss_pct: float = 38.5    # Stop loss at 38.5% loss (optimized)
    trailing_stop_pct: float = 19.6        # Trailing stop (optimized)
    
    # Learning rate and regularization (optimized)
    learning_rate: float = 0.05  # Faster adaptation (optimized)
    l2_regularization: float = 0.001
    
    # Feature configuration
    use_pnl_ema: bool = True
    use_price_ema: bool = True
    use_time_feature: bool = True
    use_volatility: bool = True
    
    # EMA parameters (optimized)
    ema_alpha_fast: float = 0.15  # Faster reaction (optimized)
    ema_alpha_slow: float = 0.028  # Slightly faster (optimized)
    
    # Constraints
    min_profit_take_pct: float = 5.0
    max_profit_take_pct: float = 100.0
    min_stop_loss_pct: float = 5.0
    max_stop_loss_pct: float = 50.0
    
    # History
    max_history_size: int = 1000


class OnlineLinearLearner:
    """
    Online linear regression learner for exit thresholds.
    
    Uses stochastic gradient descent to learn optimal thresholds based on
    historical position outcomes.
    """
    
    def __init__(self, config: OnlineLearnerConfig = None):
        self.config = config or OnlineLearnerConfig()
        
        # Number of features
        self.n_features = 0
        if self.config.use_pnl_ema:
            self.n_features += 2  # PnL and PnL EMA
        if self.config.use_price_ema:
            self.n_features += 3  # Price, fast EMA, slow EMA
        if self.config.use_time_feature:
            self.n_features += 1  # Time in position (hours)
        if self.config.use_volatility:
            self.n_features += 1  # Price volatility
        
        self.n_features = max(self.n_features, 4)  # Minimum features
        
        # Weights for profit-take and stop-loss predictions
        # Output: [profit_take_threshold, stop_loss_threshold]
        self.weights_profit = np.zeros(self.n_features)
        self.weights_stop = np.zeros(self.n_features)
        self.bias_profit = self.config.initial_profit_take_pct
        self.bias_stop = self.config.initial_stop_loss_pct
        
        # History for analysis
        self.history: deque = deque(maxlen=self.config.max_history_size)
        
        # Running statistics for normalization
        self.feature_means = np.zeros(self.n_features)
        self.feature_vars = np.ones(self.n_features)
        self.n_samples = 0
    
    def extract_features(self, position: Position) -> np.ndarray:
        """Extract features from a position for prediction."""
        features = []
        
        if self.config.use_pnl_ema:
            features.append(position.unrealized_return / 100)  # Normalize to [-1, 1] range
            features.append(position.pnl_ema / max(position.size, 1))  # Normalized PnL EMA
        
        if self.config.use_price_ema:
            features.append(position.current_price)
            features.append(position.price_ema_fast)
            features.append(position.price_ema_slow)
        
        if self.config.use_time_feature:
            hours = position.time_in_position.total_seconds() / 3600
            features.append(min(hours / 24, 1.0))  # Normalize to days
        
        if self.config.use_volatility:
            # Price range as volatility proxy
            volatility = (position.peak_price - position.trough_price)
            features.append(volatility)
        
        # Pad to expected size
        while len(features) < self.n_features:
            features.append(0.0)
        
        return np.array(features[:self.n_features])
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        if self.n_samples < 2:
            return features
        return (features - self.feature_means) / np.sqrt(self.feature_vars + 1e-8)
    
    def _update_statistics(self, features: np.ndarray):
        """Update running mean and variance."""
        self.n_samples += 1
        delta = features - self.feature_means
        self.feature_means += delta / self.n_samples
        delta2 = features - self.feature_means
        self.feature_vars += (delta * delta2 - self.feature_vars) / self.n_samples
    
    def predict_thresholds(self, position: Position) -> Tuple[float, float]:
        """
        Predict optimal profit-take and stop-loss thresholds for a position.
        
        Returns:
            Tuple of (profit_take_pct, stop_loss_pct)
        """
        features = self.extract_features(position)
        features_norm = self._normalize_features(features)
        
        # Linear prediction
        profit_take = np.dot(features_norm, self.weights_profit) + self.bias_profit
        stop_loss = np.dot(features_norm, self.weights_stop) + self.bias_stop
        
        # Apply constraints
        profit_take = np.clip(profit_take, 
                             self.config.min_profit_take_pct, 
                             self.config.max_profit_take_pct)
        stop_loss = np.clip(stop_loss, 
                           self.config.min_stop_loss_pct, 
                           self.config.max_stop_loss_pct)
        
        return float(profit_take), float(stop_loss)
    
    def update(self, position: Position, final_return: float, optimal_profit_take: float, optimal_stop_loss: float):
        """
        Update the learner with a closed position.
        
        Args:
            position: The closed position
            final_return: The actual return achieved (%)
            optimal_profit_take: What profit-take threshold would have been optimal (%)
            optimal_stop_loss: What stop-loss threshold would have been optimal (%)
        """
        features = self.extract_features(position)
        self._update_statistics(features)
        features_norm = self._normalize_features(features)
        
        # Current predictions
        pred_profit = np.dot(features_norm, self.weights_profit) + self.bias_profit
        pred_stop = np.dot(features_norm, self.weights_stop) + self.bias_stop
        
        # Errors
        error_profit = optimal_profit_take - pred_profit
        error_stop = optimal_stop_loss - pred_stop
        
        # SGD updates with L2 regularization
        lr = self.config.learning_rate
        reg = self.config.l2_regularization
        
        self.weights_profit += lr * (error_profit * features_norm - reg * self.weights_profit)
        self.weights_stop += lr * (error_stop * features_norm - reg * self.weights_stop)
        self.bias_profit += lr * error_profit
        self.bias_stop += lr * error_stop
        
        # Store in history
        self.history.append({
            'features': features.tolist(),
            'final_return': final_return,
            'optimal_profit_take': optimal_profit_take,
            'optimal_stop_loss': optimal_stop_loss,
            'pred_profit_take': pred_profit,
            'pred_stop_loss': pred_stop,
        })
    
    def compute_optimal_thresholds(self, position: Position, price_history: List[float]) -> Tuple[float, float]:
        """
        Compute what the optimal thresholds would have been in hindsight.
        
        Uses the price history to determine what profit-take and stop-loss
        levels would have maximized returns.
        """
        if not price_history or len(price_history) < 2:
            return self.config.initial_profit_take_pct, self.config.initial_stop_loss_pct
        
        best_profit_take = self.config.initial_profit_take_pct
        best_stop_loss = self.config.initial_stop_loss_pct
        best_return = float('-inf')
        
        # Sweep thresholds to find optimal
        for profit_take_pct in np.arange(5, 100, 5):
            for stop_loss_pct in np.arange(5, 50, 5):
                simulated_return = self._simulate_exit(
                    position, price_history, profit_take_pct, stop_loss_pct
                )
                if simulated_return > best_return:
                    best_return = simulated_return
                    best_profit_take = profit_take_pct
                    best_stop_loss = stop_loss_pct
        
        return best_profit_take, best_stop_loss
    
    def _simulate_exit(self, position: Position, price_history: List[float], 
                       profit_take_pct: float, stop_loss_pct: float) -> float:
        """Simulate exit with given thresholds."""
        entry_price = position.entry_price
        side = position.side
        
        for price in price_history:
            if side == 'yes':
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                no_entry = 1 - entry_price
                no_current = 1 - price
                pnl_pct = (no_current - no_entry) / no_entry * 100
            
            if pnl_pct >= profit_take_pct:
                return pnl_pct
            if pnl_pct <= -stop_loss_pct:
                return pnl_pct
        
        # Held to end - use final price
        final_price = price_history[-1]
        if side == 'yes':
            return (final_price - entry_price) / entry_price * 100
        else:
            no_entry = 1 - entry_price
            no_final = 1 - final_price
            return (no_final - no_entry) / no_entry * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get learner summary."""
        return {
            'n_samples': self.n_samples,
            'current_profit_take_bias': float(self.bias_profit),
            'current_stop_loss_bias': float(self.bias_stop),
            'weights_profit_norm': float(np.linalg.norm(self.weights_profit)),
            'weights_stop_norm': float(np.linalg.norm(self.weights_stop)),
            'history_size': len(self.history),
        }


@dataclass 
class PositionManagerConfig:
    """Configuration for position manager."""
    # Default thresholds (before learning)
    default_profit_take_pct: float = 20.0
    default_stop_loss_pct: float = 15.0
    
    # Whether to use online learning
    use_online_learning: bool = True
    
    # EMA parameters
    ema_alpha_fast: float = 0.1
    ema_alpha_slow: float = 0.02
    
    # Position limits
    max_positions: int = 50
    max_position_per_market: int = 1
    max_position_size: float = 1000.0
    
    # Time limits
    max_hold_time_hours: float = 72.0  # Force close after 72 hours


class PositionManager:
    """
    Manages open positions with learned entry/exit logic.
    
    Features:
    - Tracks all open and closed positions
    - Applies profit-taking and stop-loss rules
    - Uses online learning to optimize thresholds
    - EMA-based signals for exit timing
    """
    
    def __init__(self, config: PositionManagerConfig = None):
        self.config = config or PositionManagerConfig()
        
        # Positions
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Online learner
        if self.config.use_online_learning:
            learner_config = OnlineLearnerConfig(
                initial_profit_take_pct=self.config.default_profit_take_pct,
                initial_stop_loss_pct=self.config.default_stop_loss_pct,
                ema_alpha_fast=self.config.ema_alpha_fast,
                ema_alpha_slow=self.config.ema_alpha_slow,
            )
            self.learner = OnlineLinearLearner(learner_config)
        else:
            self.learner = None
        
        # Price history per market (for optimal threshold computation)
        self.price_history: Dict[str, List[float]] = {}
        
        # Statistics
        self.stats = {
            'total_opened': 0,
            'total_closed': 0,
            'profit_takes': 0,
            'stop_losses': 0,
            'expiries': 0,
            'total_pnl': 0.0,
        }
    
    def open_position(self, position_id: str, market_id: str, platform: str,
                      side: str, entry_price: float, size: float, 
                      strategy: str, current_yes_price: Optional[float] = None,
                      market_question: str = "") -> Optional[Position]:
        """Open a new position."""
        # Check limits
        if len(self.open_positions) >= self.config.max_positions:
            logger.warning("Max positions reached")
            return None
        
        market_positions = [p for p in self.open_positions.values() if p.market_id == market_id]
        if len(market_positions) >= self.config.max_position_per_market:
            logger.warning(f"Max positions for market {market_id} reached")
            return None
        
        if size > self.config.max_position_size:
            size = self.config.max_position_size
        
        # current_price should be the YES price (market price convention)
        # entry_price is the side price (YES for YES bets, NO for NO bets)
        if current_yes_price is None:
            # If not provided, compute from side price
            if side == 'yes':
                current_yes_price = entry_price
            else:
                current_yes_price = 1 - entry_price  # NO price -> YES price
        
        position = Position(
            position_id=position_id,
            market_id=market_id,
            platform=platform,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            size=size,
            strategy=strategy,
            market_question=market_question,
            current_price=current_yes_price,  # Always YES price
            peak_price=current_yes_price,
            trough_price=current_yes_price,
            price_ema_fast=current_yes_price,
            price_ema_slow=current_yes_price,
        )
        
        self.open_positions[position_id] = position
        self.stats['total_opened'] += 1
        
        # Initialize price history
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        
        logger.info(f"Opened position {position_id}: {side.upper()} {market_id} @ {entry_price:.4f}")
        return position
    
    def update_prices(self, market_prices: Dict[str, float]):
        """Update prices for all open positions."""
        for position in self.open_positions.values():
            if position.market_id in market_prices:
                new_price = market_prices[position.market_id]
                position.update_price(
                    new_price,
                    self.config.ema_alpha_fast,
                    self.config.ema_alpha_slow,
                )
                
                # Store price history
                if position.market_id not in self.price_history:
                    self.price_history[position.market_id] = []
                self.price_history[position.market_id].append(new_price)
                
                # Limit history size
                if len(self.price_history[position.market_id]) > 1000:
                    self.price_history[position.market_id] = self.price_history[position.market_id][-500:]
    
    def check_exits(self) -> List[Tuple[Position, str]]:
        """
        Check all positions for exit conditions.
        
        Returns list of (position, reason) tuples for positions to close.
        """
        to_close = []
        
        for position in list(self.open_positions.values()):
            # Get thresholds (learned or default)
            if self.learner:
                profit_take_pct, stop_loss_pct = self.learner.predict_thresholds(position)
            else:
                profit_take_pct = self.config.default_profit_take_pct
                stop_loss_pct = self.config.default_stop_loss_pct
            
            current_return = position.unrealized_return
            
            # Check profit-take
            if current_return >= profit_take_pct:
                to_close.append((position, 'profit_take'))
                continue
            
            # Check stop-loss
            if current_return <= -stop_loss_pct:
                to_close.append((position, 'stop_loss'))
                continue
            
            # Check time limit
            hours_held = position.time_in_position.total_seconds() / 3600
            if hours_held >= self.config.max_hold_time_hours:
                to_close.append((position, 'time_limit'))
                continue
        
        return to_close
    
    def close_position(self, position_id: str, exit_price: float, reason: str) -> Optional[Position]:
        """Close a position."""
        if position_id not in self.open_positions:
            logger.warning(f"Position {position_id} not found")
            return None
        
        position = self.open_positions.pop(position_id)
        position.close(exit_price, reason)
        self.closed_positions.append(position)
        
        # Update stats
        self.stats['total_closed'] += 1
        self.stats['total_pnl'] += position.realized_pnl or 0
        
        if reason == 'profit_take':
            self.stats['profit_takes'] += 1
        elif reason == 'stop_loss':
            self.stats['stop_losses'] += 1
        elif reason == 'time_limit':
            self.stats['expiries'] += 1
        
        # Update learner with hindsight-optimal thresholds
        if self.learner and position.market_id in self.price_history:
            price_hist = self.price_history[position.market_id]
            optimal_profit, optimal_stop = self.learner.compute_optimal_thresholds(position, price_hist)
            self.learner.update(
                position,
                position.unrealized_return,
                optimal_profit,
                optimal_stop,
            )
        
        logger.info(f"Closed position {position_id}: {reason}, PnL=${position.realized_pnl:.2f}")
        return position
    
    def get_summary(self) -> Dict[str, Any]:
        """Get position manager summary."""
        open_pnl = sum(p.unrealized_pnl for p in self.open_positions.values())
        
        summary = {
            'open_positions': len(self.open_positions),
            'closed_positions': len(self.closed_positions),
            'open_unrealized_pnl': open_pnl,
            'closed_realized_pnl': self.stats['total_pnl'],
            'profit_takes': self.stats['profit_takes'],
            'stop_losses': self.stats['stop_losses'],
            'time_limit_exits': self.stats['expiries'],
        }
        
        if self.learner:
            summary['learner'] = self.learner.get_summary()
        
        return summary
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        return [p.to_dict() for p in self.open_positions.values()]


def sweep_hyperparameters(
    closed_positions: List[Position],
    price_histories: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Sweep hyperparameters for the online learner.
    
    Tests different learning rates and regularization values.
    Returns best configuration.
    """
    best_config = None
    best_total_return = float('-inf')
    
    results = []
    
    for learning_rate in [0.001, 0.01, 0.05, 0.1]:
        for l2_reg in [0.0, 0.001, 0.01, 0.1]:
            for ema_alpha_fast in [0.05, 0.1, 0.2]:
                config = OnlineLearnerConfig(
                    learning_rate=learning_rate,
                    l2_regularization=l2_reg,
                    ema_alpha_fast=ema_alpha_fast,
                )
                
                # Simulate learning on historical positions
                learner = OnlineLinearLearner(config)
                total_return = 0
                
                for position in closed_positions:
                    if position.market_id not in price_histories:
                        continue
                    
                    price_hist = price_histories[position.market_id]
                    
                    # Get predictions before learning
                    profit_take, stop_loss = learner.predict_thresholds(position)
                    
                    # Simulate with predicted thresholds
                    simulated_return = learner._simulate_exit(
                        position, price_hist, profit_take, stop_loss
                    )
                    total_return += simulated_return
                    
                    # Update learner
                    optimal_profit, optimal_stop = learner.compute_optimal_thresholds(
                        position, price_hist
                    )
                    learner.update(position, simulated_return, optimal_profit, optimal_stop)
                
                results.append({
                    'learning_rate': learning_rate,
                    'l2_reg': l2_reg,
                    'ema_alpha_fast': ema_alpha_fast,
                    'total_return': total_return,
                })
                
                if total_return > best_total_return:
                    best_total_return = total_return
                    best_config = config
    
    return {
        'best_config': {
            'learning_rate': best_config.learning_rate if best_config else 0.01,
            'l2_regularization': best_config.l2_regularization if best_config else 0.001,
            'ema_alpha_fast': best_config.ema_alpha_fast if best_config else 0.1,
        },
        'best_total_return': best_total_return,
        'all_results': sorted(results, key=lambda x: x['total_return'], reverse=True)[:10],
    }
