"""
Time-Ordered Market Simulator

Replays historical market data in chronological order, ensuring:
- No lookahead bias (outcomes not visible until resolution time)
- Proper event sequencing (snapshots, price updates, resolutions)
- Deferred PnL resolution

This replaces the random-sampling SimulatedMarketClient with
a clock-driven replay system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from ..utils.models import Market, Platform, Side

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Snapshot of a market at a point in time."""
    market_id: str
    timestamp: datetime
    yes_price: float
    no_price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    
    # Metadata
    question: str = ""
    category: str = ""
    
    # Resolution info (only set after market resolves)
    resolved: bool = False
    outcome: Optional[int] = None
    resolution_time: Optional[datetime] = None
    
    def to_market(self, platform: Platform = Platform.POLYMARKET) -> Market:
        """Convert to Market object for strategy consumption."""
        return Market(
            market_id=self.market_id,
            platform=platform,
            question=self.question,
            current_yes_price=self.yes_price,
            volume=self.volume,
            metadata={
                "category": self.category,
                "bid": self.bid,
                "ask": self.ask,
                "resolved": self.resolved,
                # NOTE: outcome is intentionally NOT included here
                # to prevent lookahead bias
            },
        )


@dataclass
class SimulatorConfig:
    """Configuration for the time-ordered simulator."""
    # Time stepping
    step_interval_seconds: int = 60  # 1 minute default
    
    # Data filtering
    min_volume: float = 100.0
    exclude_resolved: bool = True
    
    # Resolution handling
    resolution_delay_hours: float = 0.0  # Delay after end time before resolution
    
    # Platforms
    platform: Platform = Platform.POLYMARKET


class TimeOrderedSimulator:
    """
    Replay historical data chronologically.
    
    Key features:
    - Clock-driven: advance_to(timestamp) to move simulation forward
    - No lookahead: outcomes not visible until resolution_time
    - Event-based: generates price updates and resolution events
    
    Usage:
        sim = TimeOrderedSimulator("data.parquet", "2025-01-01", "2025-01-31")
        
        for timestamp in sim.generate_timestamps():
            markets = sim.advance_to(timestamp)
            # Generate signals, execute trades
            
            resolutions = sim.get_resolutions()
            # Close positions for resolved markets
    """
    
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        config: Optional[SimulatorConfig] = None,
    ):
        self.config = config or SimulatorConfig()
        self.data_path = data_path
        
        # Parse date range
        self.start_time = pd.Timestamp(start_date, tz='UTC')
        self.end_time = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Load and prepare data
        self._load_data()
        
        # Simulation state
        self.current_time: pd.Timestamp = self.start_time
        self.market_state: Dict[str, MarketSnapshot] = {}
        self.pending_resolutions: Dict[str, Tuple[pd.Timestamp, int]] = {}
        self.resolved_markets: set = set()
        
        # Event tracking
        self.event_index: int = 0
        self.resolutions_this_step: List[Tuple[str, int]] = []
        
        logger.info(f"TimeOrderedSimulator initialized: {self.start_time} to {self.end_time}")
        logger.info(f"Loaded {len(self.events)} events across {len(self.market_info)} markets")
    
    def _load_data(self):
        """Load and prepare historical data."""
        df = pd.read_parquet(self.data_path)
        
        # Ensure timestamp column
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'float64' or df['timestamp'].dtype == 'int64':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Filter to date range
        df = df[(df['timestamp'] >= self.start_time) & (df['timestamp'] <= self.end_time)]
        
        # Filter by volume
        if 'volume' in df.columns and self.config.min_volume > 0:
            df = df[df['volume'] >= self.config.min_volume]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Store as list of events
        self.events = df.to_dict('records')
        
        # Extract market info
        self.market_info: Dict[str, Dict] = {}
        for record in self.events:
            mid = record['market_id']
            if mid not in self.market_info:
                self.market_info[mid] = {
                    'question': record.get('question', ''),
                    'category': record.get('category', ''),
                    'outcome': record.get('outcome'),
                    'resolution_time': record.get('resolution_time'),
                }
        
        # Pre-process resolutions
        self._prepare_resolutions(df)
    
    def _prepare_resolutions(self, df: pd.DataFrame):
        """Prepare resolution events from data."""
        if 'outcome' not in df.columns:
            return
        
        # Get unique markets with outcomes
        for mid, info in self.market_info.items():
            if info.get('outcome') is not None:
                # Get resolution time
                res_time = info.get('resolution_time')
                if res_time is None:
                    # Use last price update + delay
                    market_events = [e for e in self.events if e['market_id'] == mid]
                    if market_events:
                        last_ts = market_events[-1]['timestamp']
                        if isinstance(last_ts, str):
                            last_ts = pd.Timestamp(last_ts, tz='UTC')
                        res_time = last_ts + pd.Timedelta(hours=self.config.resolution_delay_hours)
                
                if res_time is not None:
                    if isinstance(res_time, str):
                        res_time = pd.Timestamp(res_time, tz='UTC')
                    outcome = int(info['outcome'])
                    self.pending_resolutions[mid] = (res_time, outcome)
    
    def reset(self):
        """Reset simulation to start."""
        self.current_time = self.start_time
        self.market_state.clear()
        self.resolved_markets.clear()
        self.event_index = 0
        self.resolutions_this_step.clear()
    
    def generate_timestamps(
        self,
        interval_seconds: Optional[int] = None,
    ) -> Iterator[pd.Timestamp]:
        """Generate timestamps for simulation steps."""
        interval = interval_seconds or self.config.step_interval_seconds
        
        current = self.start_time
        while current <= self.end_time:
            yield current
            current += pd.Timedelta(seconds=interval)
    
    def advance_to(self, target_time: pd.Timestamp) -> List[Market]:
        """
        Advance simulation clock to target_time.
        
        Processes all events between current_time and target_time,
        updating market state and resolving any markets.
        
        Args:
            target_time: Target timestamp to advance to
            
        Returns:
            List of active (non-resolved) markets at target_time
        """
        self.resolutions_this_step.clear()
        
        # Process events up to target time
        while self.event_index < len(self.events):
            event = self.events[self.event_index]
            event_time = event['timestamp']
            
            if isinstance(event_time, str):
                event_time = pd.Timestamp(event_time, tz='UTC')
            
            if event_time > target_time:
                break
            
            self._process_event(event)
            self.event_index += 1
        
        # Check for resolutions
        self._check_resolutions(target_time)
        
        self.current_time = target_time
        
        # Return active markets
        return self._get_active_markets()
    
    def _process_event(self, event: Dict):
        """Process a single market event."""
        market_id = event['market_id']
        
        if market_id in self.resolved_markets:
            return
        
        # Extract price
        price = event.get('price')
        if price is None:
            price = event.get('yes_price', 0.5)
        
        # Create or update snapshot
        snapshot = MarketSnapshot(
            market_id=market_id,
            timestamp=event['timestamp'],
            yes_price=price,
            no_price=1 - price,
            volume=event.get('volume', 0),
            bid=event.get('bid', price - 0.01),
            ask=event.get('ask', price + 0.01),
            question=self.market_info.get(market_id, {}).get('question', ''),
            category=self.market_info.get(market_id, {}).get('category', ''),
        )
        
        self.market_state[market_id] = snapshot
    
    def _check_resolutions(self, current_time: pd.Timestamp):
        """Check for markets that have resolved by current_time."""
        for market_id, (res_time, outcome) in list(self.pending_resolutions.items()):
            if res_time <= current_time and market_id not in self.resolved_markets:
                # Market has resolved
                self.resolved_markets.add(market_id)
                self.resolutions_this_step.append((market_id, outcome))
                
                # Update snapshot
                if market_id in self.market_state:
                    self.market_state[market_id].resolved = True
                    self.market_state[market_id].outcome = outcome
                    self.market_state[market_id].resolution_time = res_time
                
                logger.debug(f"Market {market_id} resolved: outcome={outcome}")
    
    def _get_active_markets(self) -> List[Market]:
        """Get list of active (non-resolved) markets."""
        markets = []
        for market_id, snapshot in self.market_state.items():
            if market_id not in self.resolved_markets:
                markets.append(snapshot.to_market(self.config.platform))
        return markets
    
    def get_resolutions(self) -> List[Tuple[str, int]]:
        """
        Get markets that resolved in the last advance_to step.
        
        Returns:
            List of (market_id, outcome) tuples
        """
        return self.resolutions_this_step.copy()
    
    def get_market_price(self, market_id: str) -> Optional[float]:
        """Get current YES price for a market."""
        if market_id in self.market_state:
            return self.market_state[market_id].yes_price
        return None
    
    def get_outcome(self, market_id: str) -> Optional[int]:
        """
        Get outcome for a resolved market.
        
        Returns None if market hasn't resolved yet (prevents lookahead).
        """
        if market_id in self.resolved_markets:
            return self.pending_resolutions.get(market_id, (None, None))[1]
        return None
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        return {
            "current_time": str(self.current_time),
            "total_events": len(self.events),
            "events_processed": self.event_index,
            "total_markets": len(self.market_info),
            "active_markets": len(self.market_state) - len(self.resolved_markets),
            "resolved_markets": len(self.resolved_markets),
            "pending_resolutions": len(self.pending_resolutions) - len(self.resolved_markets),
        }


def create_time_ordered_simulator(
    platform: str = "polymarket",
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-31",
    data_path: Optional[str] = None,
) -> TimeOrderedSimulator:
    """
    Factory function to create a time-ordered simulator.
    
    Args:
        platform: 'polymarket' or 'kalshi'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_path: Path to data file (optional, uses default)
        
    Returns:
        TimeOrderedSimulator instance
    """
    if data_path is None:
        if platform.lower() == "polymarket":
            data_path = "data/polymarket/optimization_cache.parquet"
        else:
            data_path = "data/kalshi/kalshi_backtest_clean.parquet"
    
    config = SimulatorConfig(
        platform=Platform.POLYMARKET if platform.lower() == "polymarket" else Platform.KALSHI,
    )
    
    return TimeOrderedSimulator(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
