"""
Trading Engine

Orchestrates the entire trading process:
1. Fetch market data
2. Generate signals from strategies
3. Apply risk checks
4. Execute orders
5. Track positions and PnL
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from ..utils.models import (
    Platform, Side, Order, OrderType, OrderStatus, Fill, Market, Signal, RiskLimits
)
from ..clients.polymarket import PolymarketClient, PolymarketConfig
from ..clients.kalshi import KalshiClient, KalshiConfig
from ..clients.simulated import SimulatedMarketClient, create_simulated_polymarket, create_simulated_kalshi
from ..clients.hybrid import HybridPolymarketClient, HybridKalshiClient, create_hybrid_polymarket, create_hybrid_kalshi
from ..strategies.calibration import CalibrationStrategy
from .risk_manager import RiskManager
from .position_manager import PositionManager, PositionManagerConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for trading engine."""
    paper_trading: bool = True
    initial_bankroll: float = 10000.0
    log_dir: str = "logs/trading"
    state_file: str = "state/engine_state.json"
    
    # Platform configs
    polymarket_enabled: bool = True
    kalshi_enabled: bool = True
    
    # Execution settings
    max_orders_per_run: int = 20  # Increased for better signal utilization
    min_signal_confidence: float = 0.3
    
    # Data mode: 'simulated' (historical), 'hybrid' (live data, simulated execution), 'live' (real trading)
    data_mode: str = "simulated"
    
    # Position management with learned entry/exit
    position_management_enabled: bool = True
    profit_take_pct: float = 20.0  # Take profit threshold
    stop_loss_pct: float = 15.0    # Stop loss threshold
    use_online_learning: bool = True  # Learn optimal thresholds
    
    # Market impact / slippage (applied to entry price)
    impact_cost_pct: float = 2.0  # 2% impact cost by default


class TradingEngine:
    """
    Main trading engine.
    
    Coordinates:
    - Multiple platform clients
    - Multiple strategies
    - Risk management
    - Order execution
    - State persistence
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        
        # Initialize clients based on data_mode
        self.clients: Dict[Platform, Any] = {}
        
        if self.config.polymarket_enabled:
            if self.config.data_mode == "hybrid":
                # Live data, simulated execution
                self.clients[Platform.POLYMARKET] = create_hybrid_polymarket()
                logger.info("Polymarket: HYBRID mode (live data, simulated execution)")
            elif self.config.data_mode == "simulated" or self.config.paper_trading:
                # Historical data, simulated execution
                self.clients[Platform.POLYMARKET] = create_simulated_polymarket(
                    "data/polymarket/optimization_cache.parquet"
                )
                logger.info("Polymarket: SIMULATED mode (historical data)")
            else:
                # Full live trading
                self.clients[Platform.POLYMARKET] = PolymarketClient(
                    PolymarketConfig(paper_trading=False)
                )
                logger.info("Polymarket: LIVE mode")
        
        if self.config.kalshi_enabled:
            if self.config.data_mode == "hybrid":
                # Live data, simulated execution
                self.clients[Platform.KALSHI] = create_hybrid_kalshi()
                logger.info("Kalshi: HYBRID mode (live data, simulated execution)")
            elif self.config.data_mode == "simulated" or self.config.paper_trading:
                # Historical data, simulated execution
                self.clients[Platform.KALSHI] = create_simulated_kalshi(
                    "data/kalshi/kalshi_backtest_clean.parquet"
                )
                logger.info("Kalshi: SIMULATED mode (historical data)")
            else:
                # Full live trading
                self.clients[Platform.KALSHI] = KalshiClient(
                    KalshiConfig(paper_trading=False)
                )
                logger.info("Kalshi: LIVE mode")
        
        # Initialize strategies
        self.strategies: Dict[str, CalibrationStrategy] = {}
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_bankroll=self.config.initial_bankroll,
            limits=RiskLimits(),
        )
        
        # Initialize position manager with learned entry/exit
        if self.config.position_management_enabled:
            pm_config = PositionManagerConfig(
                default_profit_take_pct=self.config.profit_take_pct,
                default_stop_loss_pct=self.config.stop_loss_pct,
                use_online_learning=self.config.use_online_learning,
                max_position_size=self.config.initial_bankroll * 0.1,  # Max 10% per position
            )
            self.position_manager = PositionManager(pm_config)
            logger.info(f"Position manager enabled: profit_take={self.config.profit_take_pct}%, stop_loss={self.config.stop_loss_pct}%, online_learning={self.config.use_online_learning}")
        else:
            self.position_manager = None
        
        # State tracking
        self.signals_generated: List[Signal] = []
        self.orders_placed: List[Order] = []
        self.fills_received: List[Fill] = []
        
        # Market question cache (market_id -> question)
        self.market_questions: Dict[str, str] = {}
        
        # Setup logging
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging(log_dir)
    
    def _setup_logging(self, log_dir: Path):
        """Setup file logging."""
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        
        # Trade log
        self.trade_log = log_dir / f"trades_{timestamp}.jsonl"
        
        # Signal log
        self.signal_log = log_dir / f"signals_{timestamp}.jsonl"
    
    def add_strategy(self, name: str, strategy: CalibrationStrategy):
        """Add a strategy to the engine."""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
    
    def fetch_markets(self, platform: Platform) -> List[Market]:
        """Fetch available markets from a platform."""
        if platform not in self.clients:
            return []
        
        client = self.clients[platform]
        try:
            markets = client.get_markets()
            
            # Cache market questions for later use
            for m in markets:
                if m.question:
                    self.market_questions[m.market_id] = m.question
            
            logger.info(f"Fetched {len(markets)} markets from {platform.value}")
            return markets
        except Exception as e:
            logger.error(f"Error fetching markets from {platform.value}: {e}")
            return []
    
    def generate_signals(self, markets: List[Market]) -> List[Signal]:
        """Generate signals for all markets using all strategies."""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            for market in markets:
                # Check if strategy applies to this platform
                if market.platform != strategy.platform:
                    continue
                
                try:
                    signal = strategy.generate_signal(
                        market=market,
                        bankroll=self.risk_manager.state.bankroll,
                    )
                    
                    if signal is not None:
                        signal.strategy = strategy_name
                        signals.append(signal)
                        
                        # Log signal
                        self._log_signal(signal)
                        
                except Exception as e:
                    logger.warning(f"Error generating signal for {market.market_id}: {e}")
        
        logger.info(f"Generated {len(signals)} signals")
        return signals
    
    def execute_signals(self, signals: List[Signal]) -> List[Order]:
        """Execute signals by placing orders."""
        orders = []
        
        # Filter signals by confidence
        valid_signals = [
            s for s in signals 
            if s.confidence >= self.config.min_signal_confidence
        ]
        
        # Fair selection: take top N from each strategy to ensure diversity
        from collections import defaultdict
        by_strategy = defaultdict(list)
        for s in valid_signals:
            by_strategy[s.strategy].append(s)
        
        # Sort each strategy's signals by confidence
        for strat in by_strategy:
            by_strategy[strat].sort(key=lambda s: s.confidence, reverse=True)
        
        # Round-robin selection across strategies
        selected = []
        max_per_strategy = max(5, self.config.max_orders_per_run // max(1, len(by_strategy)))
        
        for strat, strat_signals in by_strategy.items():
            selected.extend(strat_signals[:max_per_strategy])
        
        # Sort final selection by confidence and limit
        selected.sort(key=lambda s: s.confidence, reverse=True)
        valid_signals = selected[:self.config.max_orders_per_run]
        
        for signal in valid_signals:
            try:
                order = self._create_order(signal)
                
                # Risk check
                approved, reason = self.risk_manager.check_pre_trade(signal, order)
                
                if not approved:
                    logger.info(f"Order rejected: {reason}")
                    continue
                
                # Place order
                order = self._place_order(order)
                orders.append(order)
                
                # Log trade
                self._log_trade(signal, order)
                
            except Exception as e:
                logger.error(f"Error executing signal {signal.signal_id}: {e}")
        
        logger.info(f"Placed {len(orders)} orders")
        return orders
    
    def _create_order(self, signal: Signal) -> Order:
        """Create order from signal."""
        # Get strategy for position sizing
        # Use INITIAL bankroll for sizing to prevent compounding effects
        # This makes PnL comparisons more realistic and meaningful
        strategy = self.strategies.get(signal.strategy)
        if strategy:
            size = strategy.compute_position_size(
                signal=signal,
                bankroll=self.config.initial_bankroll,  # Fixed, not compounding
            )
        else:
            size = self.config.initial_bankroll * signal.kelly_fraction
        
        # Get current market price
        market_price = signal.metadata.get('price', 0.5)
        
        # Apply impact cost (slippage) - we pay more than market price when buying
        # This simulates realistic execution with market impact
        impact_multiplier = 1 + (self.config.impact_cost_pct / 100)
        
        if signal.side == Side.YES:
            # Buying YES: pay more than market YES price
            entry_price = min(0.99, market_price * impact_multiplier)
        else:
            # Buying NO: pay more than market NO price
            no_market_price = 1 - market_price
            entry_no_price = min(0.99, no_market_price * impact_multiplier)
            entry_price = entry_no_price  # Store as side price
        
        order = Order(
            signal_id=signal.signal_id,
            platform=signal.platform,
            market_id=signal.market_id,
            side=signal.side,
            order_type=OrderType.LIMIT,
            size=size,
            price=entry_price,  # Entry price with impact cost
            metadata={
                'strategy': signal.strategy,
                'edge': signal.edge,
                'confidence': signal.confidence,
                'market_price': market_price,  # Store original market price
                'impact_cost_pct': self.config.impact_cost_pct,
            }
        )
        
        return order
    
    def _place_order(self, order: Order) -> Order:
        """Place order via platform client."""
        if order.platform not in self.clients:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = 'Platform not enabled'
            return order
        
        client = self.clients[order.platform]
        
        try:
            order = client.place_order(order)
            
            # Log order details including PnL if available
            pnl = order.metadata.get('pnl', 0)
            result = order.metadata.get('result', 'pending')
            logger.info(
                f"Placed order: {order.order_id} on {order.platform.value} "
                f"for {order.market_id} ({order.side.value}) @ ${order.size:.2f} "
                f"[{result}, PnL=${pnl:.2f}]"
            )
            
            # Update risk manager with PnL
            if pnl != 0:
                self.risk_manager.record_pnl(pnl)
            
            # Track position for position management
            if self.position_manager and order.status == OrderStatus.FILLED:
                # Get the market YES price (before impact cost) for mark-to-market tracking
                # order.price is the entry price WITH impact cost
                # order.metadata['market_price'] is the original market YES price
                market_yes_price = order.metadata.get('market_price', order.price)
                
                # Get market question from cache
                market_question = self.market_questions.get(order.market_id, "")
                
                self.position_manager.open_position(
                    position_id=order.order_id,
                    market_id=order.market_id,
                    platform=order.platform.value,
                    side=order.side.value,
                    entry_price=order.price,  # Entry price with impact
                    size=order.size,
                    strategy=order.metadata.get('strategy', 'unknown'),
                    current_yes_price=market_yes_price,  # Current market price for tracking
                    market_question=market_question,
                )
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)
            logger.error(f"Order placement failed: {e}")
        
        return order
    
    def _log_signal(self, signal: Signal):
        """Log signal to file."""
        with open(self.signal_log, 'a') as f:
            f.write(json.dumps(signal.to_dict()) + '\n')
    
    def _log_trade(self, signal: Signal, order: Order):
        """Log trade to file."""
        trade_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'signal': signal.to_dict(),
            'order': order.to_dict(),
        }
        with open(self.trade_log, 'a') as f:
            f.write(json.dumps(trade_data) + '\n')
    
    def _log_positions(self):
        """Log current position state for dashboard."""
        if not self.position_manager:
            return
        
        log_dir = Path(self.config.log_dir)
        position_file = log_dir / "positions.json"
        
        # Compute unrealized PnL
        open_positions = self.position_manager.get_open_positions()
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in open_positions)
        
        # Get closed positions summary
        summary = self.position_manager.get_summary()
        
        position_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'open_count': len(open_positions),
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': summary.get('closed_realized_pnl', 0),
            'profit_takes': summary.get('profit_takes', 0),
            'stop_losses': summary.get('stop_losses', 0),
            'positions': open_positions[:20],  # Last 20 for display
        }
        
        with open(position_file, 'w') as f:
            json.dump(position_data, f, indent=2)
    
    def run_cycle(self):
        """Run one trading cycle."""
        logger.info("=" * 50)
        logger.info(f"Starting trading cycle at {datetime.utcnow()}")
        
        # Check if trading day changed
        self.risk_manager.start_new_day()
        
        # Check if trading is halted
        if self.risk_manager._trading_halted:
            logger.warning(f"Trading halted: {self.risk_manager._halt_reason}")
            return
        
        # Fetch markets from all platforms
        all_markets = []
        for platform in self.clients.keys():
            markets = self.fetch_markets(platform)
            all_markets.extend(markets)
        
        if not all_markets:
            logger.warning("No markets available")
            return
        
        # Manage existing positions (check exits)
        if self.position_manager:
            self._manage_positions(all_markets)
        
        # Generate signals
        signals = self.generate_signals(all_markets)
        self.signals_generated.extend(signals)
        
        if not signals:
            logger.info("No signals generated")
            return
        
        # Execute signals
        orders = self.execute_signals(signals)
        self.orders_placed.extend(orders)
        
        # Log position state for dashboard
        self._log_positions()
        
        # Log status
        status = self.risk_manager.get_status()
        logger.info(f"Risk status: {json.dumps(status, indent=2)}")
    
    def _manage_positions(self, markets: List[Market]):
        """
        Manage open positions: update prices and check for exits.
        Uses the position manager's learned thresholds.
        """
        if not self.position_manager:
            return
        
        # Build price map from current markets
        market_prices = {}
        for market in markets:
            market_prices[market.market_id] = market.current_yes_price
        
        # Update position prices
        self.position_manager.update_prices(market_prices)
        
        # Check for exits
        exits = self.position_manager.check_exits()
        
        for position, reason in exits:
            # Get current price
            current_price = market_prices.get(position.market_id, position.current_price)
            
            # Close position
            closed = self.position_manager.close_position(
                position.position_id, 
                current_price, 
                reason
            )
            
            if closed:
                logger.info(
                    f"Position exit: {position.position_id} ({reason}) "
                    f"PnL=${closed.realized_pnl:.2f}"
                )
                
                # Record PnL with risk manager
                if closed.realized_pnl:
                    self.risk_manager.record_pnl(closed.realized_pnl)
        
        # Log position summary
        if self.position_manager.open_positions:
            summary = self.position_manager.get_summary()
            logger.info(
                f"Positions: {summary['open_positions']} open, "
                f"${summary['open_unrealized_pnl']:.2f} unrealized PnL"
            )
    
    def get_status(self) -> Dict:
        """Get engine status."""
        status = {
            'config': {
                'paper_trading': self.config.paper_trading,
                'initial_bankroll': self.config.initial_bankroll,
                'data_mode': self.config.data_mode,
            },
            'risk': self.risk_manager.get_status(),
            'strategies': list(self.strategies.keys()),
            'platforms': [p.value for p in self.clients.keys()],
            'signals_today': len(self.signals_generated),
            'orders_today': len(self.orders_placed),
        }
        
        # Add position manager status if enabled
        if self.position_manager:
            status['positions'] = self.position_manager.get_summary()
        
        return status


def create_paper_trading_engine(bankroll: float = 10000.0) -> TradingEngine:
    """Create a paper trading engine with default strategies."""
    config = EngineConfig(
        paper_trading=True,
        initial_bankroll=bankroll,
    )
    
    engine = TradingEngine(config)
    
    # Add strategies
    from ..strategies.calibration import (
        PolymarketCalibrationStrategy, 
        KalshiCalibrationStrategy
    )
    
    engine.add_strategy("pm_calibration", PolymarketCalibrationStrategy())
    engine.add_strategy("kalshi_calibration", KalshiCalibrationStrategy())
    
    return engine
