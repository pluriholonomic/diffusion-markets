#!/usr/bin/env python3
"""
WebSocket Streaming Client for Real-Time Market Data

Supports:
- Polymarket: wss://ws-live-data.polymarket.com
- Kalshi: wss://api.elections.kalshi.com/trade-api/ws/v2

Provides event-driven price updates instead of polling.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not installed. Run: pip install websockets")


@dataclass
class PriceUpdate:
    """A real-time price update."""
    market_id: str
    platform: str
    yes_price: float
    no_price: float
    timestamp: datetime
    volume: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class StreamConfig:
    """Configuration for WebSocket streaming."""
    polymarket_url: str = "wss://ws-live-data.polymarket.com"
    kalshi_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    # Reconnection settings
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    reconnect_multiplier: float = 2.0
    
    # Ping/pong for keepalive
    ping_interval: float = 5.0
    ping_timeout: float = 10.0
    
    # Batch updates
    batch_interval: float = 0.5  # Batch updates every 500ms


class MarketPriceStream:
    """
    Real-time market price streaming via WebSocket.
    
    Usage:
        stream = MarketPriceStream(config)
        stream.subscribe_markets(['market_id_1', 'market_id_2'])
        stream.on_price_update(callback_function)
        stream.start()  # Runs in background thread
        
        # Later...
        updates = stream.get_pending_updates()  # Get batched updates
        stream.stop()
    """
    
    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self._subscribed_markets: Set[str] = set()
        self._price_cache: Dict[str, PriceUpdate] = {}
        self._update_queue: Queue = Queue()
        self._callbacks: List[Callable[[PriceUpdate], None]] = []
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Connection state
        self._connected = False
        self._last_ping = 0
        self._reconnect_delay = self.config.reconnect_delay
        
    def subscribe_markets(self, market_ids: List[str], platform: str = "polymarket"):
        """Subscribe to price updates for specific markets."""
        for mid in market_ids:
            self._subscribed_markets.add(f"{platform}:{mid}")
        logger.info(f"Subscribed to {len(market_ids)} markets on {platform}")
    
    def on_price_update(self, callback: Callable[[PriceUpdate], None]):
        """Register a callback for price updates."""
        self._callbacks.append(callback)
    
    def get_pending_updates(self) -> List[PriceUpdate]:
        """Get all pending price updates (non-blocking)."""
        updates = []
        try:
            while True:
                update = self._update_queue.get_nowait()
                updates.append(update)
        except Empty:
            pass
        return updates
    
    def get_latest_prices(self) -> Dict[str, PriceUpdate]:
        """Get the latest cached price for each market."""
        return dict(self._price_cache)
    
    def has_updates(self) -> bool:
        """Check if there are pending updates."""
        return not self._update_queue.empty()
    
    def get_updates(self) -> List[PriceUpdate]:
        """Alias for get_pending_updates."""
        return self.get_pending_updates()
    
    def start(self):
        """Start the WebSocket streaming in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available")
            return False
            
        if self._running:
            return True
            
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WebSocket streaming started")
        return True
    
    def stop(self):
        """Stop the WebSocket streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("WebSocket streaming stopped")
    
    def _run_loop(self):
        """Run the async event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._stream_loop())
        except Exception as e:
            logger.error(f"Stream loop error: {e}")
        finally:
            self._loop.close()
    
    async def _stream_loop(self):
        """Main streaming loop with reconnection logic."""
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.warning(f"Connection lost: {e}")
                
            if self._running:
                # Exponential backoff for reconnection
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * self.config.reconnect_multiplier,
                    self.config.max_reconnect_delay
                )
    
    async def _connect_and_stream(self):
        """Connect to WebSocket and stream updates."""
        # Connect to Polymarket
        polymarket_markets = [
            m.split(":", 1)[1] for m in self._subscribed_markets 
            if m.startswith("polymarket:")
        ]
        
        if polymarket_markets:
            try:
                async with websockets.connect(
                    self.config.polymarket_url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                ) as ws:
                    self._connected = True
                    self._reconnect_delay = self.config.reconnect_delay
                    logger.info(f"Connected to Polymarket WebSocket")
                    
                    # Subscribe to market updates
                    subscribe_msg = {
                        "action": "subscribe",
                        "topics": ["market_data"],
                        "markets": polymarket_markets[:50],  # Limit subscriptions
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    # Process messages
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_polymarket_message(message)
                        
            except Exception as e:
                logger.error(f"Polymarket WebSocket error: {e}")
                self._connected = False
    
    async def _handle_polymarket_message(self, message: str):
        """Handle a message from Polymarket WebSocket."""
        try:
            data = json.loads(message)
            topic = data.get("topic", "")
            payload = data.get("payload", {})
            
            if topic == "market_data" or "price" in str(data).lower():
                market_id = payload.get("market_id") or payload.get("condition_id", "")
                
                if not market_id:
                    return
                
                # Extract prices
                yes_price = float(payload.get("yes_price", payload.get("outcome_prices", [0.5, 0.5])[0]))
                no_price = 1 - yes_price
                
                update = PriceUpdate(
                    market_id=market_id,
                    platform="polymarket",
                    yes_price=yes_price,
                    no_price=no_price,
                    timestamp=datetime.utcnow(),
                    volume=payload.get("volume"),
                    spread=payload.get("spread"),
                )
                
                # Cache and queue
                self._price_cache[market_id] = update
                self._update_queue.put(update)
                
                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(update)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Message handling error: {e}")


class HybridPriceProvider:
    """
    Combines WebSocket streaming with REST polling fallback.
    
    - Uses WebSocket for real-time updates when available
    - Falls back to REST polling when WebSocket is unavailable
    - Provides a unified interface for price data
    """
    
    def __init__(self, rest_client, stream_config: StreamConfig = None):
        self.rest_client = rest_client
        self.stream = MarketPriceStream(stream_config)
        self._use_streaming = WEBSOCKETS_AVAILABLE
        self._last_rest_fetch = 0
        self._rest_interval = 30.0  # Fallback polling interval
        
    def start(self):
        """Start the price provider."""
        if self._use_streaming:
            success = self.stream.start()
            if not success:
                logger.warning("WebSocket unavailable, using REST polling only")
                self._use_streaming = False
    
    def stop(self):
        """Stop the price provider."""
        if self._use_streaming:
            self.stream.stop()
    
    def subscribe(self, market_ids: List[str], platform: str = "polymarket"):
        """Subscribe to market price updates."""
        if self._use_streaming:
            self.stream.subscribe_markets(market_ids, platform)
    
    def get_prices(self, market_ids: List[str]) -> Dict[str, float]:
        """Get current prices for markets."""
        prices = {}
        
        # Get from stream cache first
        if self._use_streaming:
            cache = self.stream.get_latest_prices()
            for mid in market_ids:
                if mid in cache:
                    prices[mid] = cache[mid].yes_price
        
        # Fill in missing from REST (with rate limiting)
        missing = [mid for mid in market_ids if mid not in prices]
        if missing and (time.time() - self._last_rest_fetch > self._rest_interval):
            try:
                rest_prices = self.rest_client.get_market_prices(missing)
                prices.update(rest_prices)
                self._last_rest_fetch = time.time()
            except Exception as e:
                logger.debug(f"REST fallback error: {e}")
        
        return prices
    
    def has_updates(self) -> bool:
        """Check if there are pending updates."""
        if self._use_streaming:
            return not self.stream._update_queue.empty()
        return False
    
    def get_updates(self) -> List[PriceUpdate]:
        """Get pending price updates."""
        if self._use_streaming:
            return self.stream.get_pending_updates()
        return []
