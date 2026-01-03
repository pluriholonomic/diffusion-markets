"""
Strategy Statistics Logger for Reconciliation

Logs detailed per-minute statistics for each strategy to enable:
1. Backtest vs reality comparison
2. Residual analysis
3. Strategy performance tracking
4. Signal quality assessment
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class StrategySnapshot:
    """Point-in-time snapshot of strategy state."""
    timestamp: str
    strategy_name: str
    
    # Signal stats
    signals_generated: int = 0
    signals_filtered: int = 0
    avg_signal_edge: float = 0.0
    avg_signal_confidence: float = 0.0
    
    # Trade stats
    orders_placed: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    
    # Position stats
    open_positions: int = 0
    total_exposure: float = 0.0
    
    # PnL stats
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    session_pnl: float = 0.0
    
    # Risk stats
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_estimate: float = 0.0
    
    # Market stats
    markets_scanned: int = 0
    avg_market_spread: float = 0.0
    avg_market_volume: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconciliationRecord:
    """Record for reconciliation between backtest and live."""
    timestamp: str
    strategy_name: str
    market_id: str
    
    # Signal details
    signal_side: str
    signal_edge: float
    signal_confidence: float
    signal_kelly: float
    
    # Market state at signal time
    market_yes_price: float
    market_bid: float
    market_ask: float
    market_volume: float
    
    # Execution details
    order_placed: bool = False
    order_filled: bool = False
    fill_price: float = 0.0
    fill_slippage: float = 0.0
    
    # For later reconciliation
    expected_pnl: float = 0.0
    backtest_would_trade: bool = True


class StrategyLogger:
    """
    Comprehensive strategy logger for reconciliation.
    
    Writes per-minute snapshots and individual signal records.
    """
    
    def __init__(
        self,
        log_dir: str = "logs/strategy_stats",
        snapshot_interval_seconds: int = 60,
        auto_start: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshot_interval = snapshot_interval_seconds
        
        # Current session stats by strategy
        self.strategy_stats: Dict[str, Dict[str, Any]] = {}
        self.signal_buffer: List[ReconciliationRecord] = []
        self.snapshot_buffer: List[StrategySnapshot] = []
        
        # Session tracking
        self.session_start = datetime.utcnow()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # PnL tracking for Sharpe
        self.pnl_history: Dict[str, List[float]] = {}
        self.peak_pnl: Dict[str, float] = {}
        
        # Background thread for periodic snapshots
        self._stop_event = threading.Event()
        self._snapshot_thread = None
        
        if auto_start:
            self.start()
        
        logger.info(f"StrategyLogger initialized: session={self.session_id}, interval={snapshot_interval_seconds}s")
    
    def start(self):
        """Start background snapshot thread."""
        if self._snapshot_thread is None or not self._snapshot_thread.is_alive():
            self._stop_event.clear()
            self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
            self._snapshot_thread.start()
    
    def stop(self):
        """Stop background thread."""
        self._stop_event.set()
        if self._snapshot_thread:
            self._snapshot_thread.join(timeout=5)
    
    def _snapshot_loop(self):
        """Background loop for periodic snapshots."""
        while not self._stop_event.is_set():
            try:
                self._take_snapshots()
                self._flush_buffers()
            except Exception as e:
                logger.error(f"Snapshot error: {e}")
            
            self._stop_event.wait(self.snapshot_interval)
    
    def register_strategy(self, strategy_name: str):
        """Register a strategy for tracking."""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                'signals_generated': 0,
                'signals_filtered': 0,
                'orders_placed': 0,
                'orders_filled': 0,
                'orders_rejected': 0,
                'open_positions': 0,
                'total_exposure': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'markets_scanned': 0,
                'total_edge': 0.0,
                'total_confidence': 0.0,
            }
            self.pnl_history[strategy_name] = []
            self.peak_pnl[strategy_name] = 0.0
            logger.info(f"Registered strategy: {strategy_name}")
    
    def log_signal(
        self,
        strategy_name: str,
        market_id: str,
        side: str,
        edge: float,
        confidence: float,
        kelly: float,
        market_yes_price: float,
        market_bid: float = 0.0,
        market_ask: float = 0.0,
        market_volume: float = 0.0,
        filtered: bool = False,
    ):
        """Log a signal generation event."""
        self.register_strategy(strategy_name)
        
        stats = self.strategy_stats[strategy_name]
        stats['signals_generated'] += 1
        stats['total_edge'] += edge
        stats['total_confidence'] += confidence
        
        if filtered:
            stats['signals_filtered'] += 1
        
        # Create reconciliation record
        record = ReconciliationRecord(
            timestamp=datetime.utcnow().isoformat(),
            strategy_name=strategy_name,
            market_id=market_id,
            signal_side=side,
            signal_edge=edge,
            signal_confidence=confidence,
            signal_kelly=kelly,
            market_yes_price=market_yes_price,
            market_bid=market_bid,
            market_ask=market_ask,
            market_volume=market_volume,
            backtest_would_trade=not filtered,
        )
        self.signal_buffer.append(record)
    
    def log_order(
        self,
        strategy_name: str,
        market_id: str,
        placed: bool = True,
        filled: bool = False,
        fill_price: float = 0.0,
        expected_price: float = 0.0,
    ):
        """Log an order event."""
        self.register_strategy(strategy_name)
        
        stats = self.strategy_stats[strategy_name]
        if placed:
            stats['orders_placed'] += 1
        if filled:
            stats['orders_filled'] += 1
        else:
            stats['orders_rejected'] += 1
        
        # Update matching signal record
        for record in reversed(self.signal_buffer[-100:]):
            if record.market_id == market_id and record.strategy_name == strategy_name:
                record.order_placed = placed
                record.order_filled = filled
                record.fill_price = fill_price
                record.fill_slippage = fill_price - expected_price if expected_price > 0 else 0
                break
    
    def log_position_update(
        self,
        strategy_name: str,
        open_positions: int,
        total_exposure: float,
        unrealized_pnl: float,
        realized_pnl: float,
    ):
        """Log position state update."""
        self.register_strategy(strategy_name)
        
        stats = self.strategy_stats[strategy_name]
        stats['open_positions'] = open_positions
        stats['total_exposure'] = total_exposure
        stats['unrealized_pnl'] = unrealized_pnl
        stats['realized_pnl'] = realized_pnl
        
        # Track PnL for Sharpe
        total_pnl = unrealized_pnl + realized_pnl
        self.pnl_history[strategy_name].append(total_pnl)
        self.peak_pnl[strategy_name] = max(self.peak_pnl[strategy_name], total_pnl)
    
    def log_market_scan(self, strategy_name: str, markets_scanned: int):
        """Log market scan event."""
        self.register_strategy(strategy_name)
        self.strategy_stats[strategy_name]['markets_scanned'] = markets_scanned
    
    def _compute_sharpe(self, strategy_name: str) -> float:
        """Estimate Sharpe ratio from PnL history."""
        pnl = self.pnl_history.get(strategy_name, [])
        if len(pnl) < 10:
            return 0.0
        
        returns = [pnl[i] - pnl[i-1] for i in range(1, len(pnl))]
        if not returns:
            return 0.0
        
        mean_ret = sum(returns) / len(returns)
        std_ret = (sum((r - mean_ret)**2 for r in returns) / len(returns)) ** 0.5
        
        if std_ret < 1e-6:
            return 0.0
        
        # Annualize: assume 1-minute snapshots
        return mean_ret / std_ret * (525600 ** 0.5)  # sqrt(minutes per year)
    
    def _compute_drawdown(self, strategy_name: str) -> float:
        """Compute current drawdown."""
        pnl = self.pnl_history.get(strategy_name, [])
        if not pnl:
            return 0.0
        
        current = pnl[-1]
        peak = self.peak_pnl.get(strategy_name, 0)
        
        if peak <= 0:
            return 0.0
        
        return (peak - current) / peak
    
    def _take_snapshots(self):
        """Take snapshots for all strategies."""
        timestamp = datetime.utcnow().isoformat()
        
        for strategy_name, stats in self.strategy_stats.items():
            n_signals = stats['signals_generated']
            
            snapshot = StrategySnapshot(
                timestamp=timestamp,
                strategy_name=strategy_name,
                signals_generated=stats['signals_generated'],
                signals_filtered=stats['signals_filtered'],
                avg_signal_edge=stats['total_edge'] / max(n_signals, 1),
                avg_signal_confidence=stats['total_confidence'] / max(n_signals, 1),
                orders_placed=stats['orders_placed'],
                orders_filled=stats['orders_filled'],
                orders_rejected=stats['orders_rejected'],
                open_positions=stats['open_positions'],
                total_exposure=stats['total_exposure'],
                unrealized_pnl=stats['unrealized_pnl'],
                realized_pnl=stats['realized_pnl'],
                session_pnl=stats['unrealized_pnl'] + stats['realized_pnl'],
                sharpe_estimate=self._compute_sharpe(strategy_name),
                current_drawdown=self._compute_drawdown(strategy_name),
                max_drawdown=max(self._compute_drawdown(strategy_name), 0),
                markets_scanned=stats['markets_scanned'],
            )
            self.snapshot_buffer.append(snapshot)
    
    def _flush_buffers(self):
        """Write buffers to disk."""
        # Write signals
        if self.signal_buffer:
            signals_file = self.log_dir / f"signals_{self.session_id}.jsonl"
            with open(signals_file, 'a') as f:
                for record in self.signal_buffer:
                    f.write(json.dumps(asdict(record)) + '\n')
            self.signal_buffer = []
        
        # Write snapshots
        if self.snapshot_buffer:
            snapshots_file = self.log_dir / f"snapshots_{self.session_id}.jsonl"
            with open(snapshots_file, 'a') as f:
                for snapshot in self.snapshot_buffer:
                    f.write(json.dumps(asdict(snapshot)) + '\n')
            self.snapshot_buffer = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current summary of all strategies."""
        summary = {
            'session_id': self.session_id,
            'session_duration_minutes': (datetime.utcnow() - self.session_start).total_seconds() / 60,
            'strategies': {},
        }
        
        for strategy_name, stats in self.strategy_stats.items():
            n_signals = stats['signals_generated']
            summary['strategies'][strategy_name] = {
                'signals': n_signals,
                'fill_rate': stats['orders_filled'] / max(stats['orders_placed'], 1),
                'avg_edge': stats['total_edge'] / max(n_signals, 1),
                'positions': stats['open_positions'],
                'exposure': stats['total_exposure'],
                'pnl': stats['unrealized_pnl'] + stats['realized_pnl'],
                'sharpe': self._compute_sharpe(strategy_name),
                'drawdown': self._compute_drawdown(strategy_name),
            }
        
        return summary


# Global logger instance
_strategy_logger: Optional[StrategyLogger] = None


def get_strategy_logger(log_dir: str = "logs/strategy_stats") -> StrategyLogger:
    """Get or create the global strategy logger."""
    global _strategy_logger
    if _strategy_logger is None:
        _strategy_logger = StrategyLogger(log_dir=log_dir)
    return _strategy_logger
