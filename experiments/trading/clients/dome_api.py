"""
Dome API Client

Provides unified access to prediction market data across Polymarket and Kalshi.
https://docs.domeapi.io/

Key features:
- Historical orderbook data for realistic backtesting
- Candlestick data for momentum strategies
- Wallet PnL for reconciliation
- Cross-platform market matching for arbitrage
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class DomeConfig:
    """Configuration for Dome API client."""
    api_key: str = ""
    base_url: str = "https://api.domeapi.io/v1"  # Note: v1 API
    timeout: int = 30
    max_retries: int = 3
    rate_limit_qps: float = 1.0  # Free tier: 1 QPS
    

@dataclass
class OrderbookSnapshot:
    """Snapshot of orderbook at a point in time."""
    timestamp: datetime
    market_id: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class Candlestick:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class TradeRecord:
    """Historical trade record."""
    timestamp: datetime
    market_id: str
    side: str
    price: float
    size: float
    maker_address: Optional[str] = None


class DomeAPIClient:
    """
    Client for Dome API.
    
    Usage:
        client = DomeAPIClient(api_key="your-key")
        
        # Get historical orderbook for backtesting
        orderbooks = client.get_orderbook_history(
            token_id="...",
            start_date="2025-01-01",
            end_date="2025-01-31"
        )
        
        # Get candlesticks for momentum strategy
        candles = client.get_candlesticks(
            token_id="...",
            interval="1h",
            limit=100
        )
    """
    
    def __init__(self, config: Optional[DomeConfig] = None):
        self.config = config or DomeConfig(
            api_key=os.environ.get("DOME_API_KEY", "")
        )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        })
        self._last_request_time = 0.0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self.config.rate_limit_qps
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Make API request with retries and rate limiting."""
        self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.config.timeout,
                )
                
                if response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        
        return {}
    
    # ========================================================================
    # Polymarket Endpoints
    # ========================================================================
    
    def get_polymarket_markets(
        self,
        market_slug: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Get Polymarket markets.
        
        Args:
            market_slug: Filter by market slug
            limit: Max results
            offset: Pagination offset
        
        Returns:
            List of market data
        """
        params = {"limit": limit, "offset": offset}
        if market_slug:
            params["market_slug"] = market_slug
        
        response = self._request("GET", "/polymarket/markets", params=params)
        return response.get("markets", [])
    
    def get_polymarket_market_price(self, token_id: str) -> Dict:
        """
        Get current price for a Polymarket token.
        
        Args:
            token_id: The token ID
        
        Returns:
            Price data including bid/ask
        """
        response = self._request(
            "GET", 
            "/polymarket/markets/price",
            params={"token_id": token_id}
        )
        return response.get("data", {})
    
    def get_polymarket_orderbook_history(
        self,
        token_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1h",
    ) -> List[OrderbookSnapshot]:
        """
        Get historical orderbook snapshots for backtesting.
        
        Args:
            token_id: Token ID
            start_time: Start of range
            end_time: End of range  
            interval: Snapshot interval (1m, 5m, 1h, 1d)
        
        Returns:
            List of OrderbookSnapshot objects
        """
        params = {"token_id": token_id, "interval": interval}
        
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        response = self._request(
            "GET",
            "/polymarket/orderbook-history",
            params=params
        )
        
        snapshots = []
        for item in response.get("data", []):
            try:
                snapshots.append(OrderbookSnapshot(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    market_id=token_id,
                    bids=[(b["price"], b["size"]) for b in item.get("bids", [])],
                    asks=[(a["price"], a["size"]) for a in item.get("asks", [])],
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse orderbook snapshot: {e}")
        
        return snapshots
    
    def get_polymarket_candlesticks(
        self,
        token_id: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> List[Candlestick]:
        """
        Get candlestick data for momentum/trend analysis.
        
        Args:
            token_id: Token ID
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles
        
        Returns:
            List of Candlestick objects
        """
        response = self._request(
            "GET",
            "/polymarket/candlesticks",
            params={
                "token_id": token_id,
                "interval": interval,
                "limit": limit,
            }
        )
        
        candles = []
        for item in response.get("data", []):
            try:
                candles.append(Candlestick(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume", 0)),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse candlestick: {e}")
        
        return candles
    
    def get_polymarket_trade_history(
        self,
        token_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[TradeRecord]:
        """
        Get historical trades for a market.
        
        Args:
            token_id: Token ID
            start_time: Start of range
            end_time: End of range
            limit: Max trades
        
        Returns:
            List of TradeRecord objects
        """
        params = {"token_id": token_id, "limit": limit}
        
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        response = self._request(
            "GET",
            "/polymarket/trade-history",
            params=params
        )
        
        trades = []
        for item in response.get("data", []):
            try:
                trades.append(TradeRecord(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    market_id=token_id,
                    side=item.get("side", "unknown"),
                    price=float(item["price"]),
                    size=float(item["size"]),
                    maker_address=item.get("maker_address"),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse trade: {e}")
        
        return trades
    
    def get_polymarket_activity(
        self,
        market_slug: str,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get market activity (trades, splits, merges).
        
        Args:
            market_slug: Market slug
            limit: Max results
        
        Returns:
            List of activity records
        """
        response = self._request(
            "GET",
            "/polymarket/activity",
            params={"market_slug": market_slug, "limit": limit}
        )
        return response.get("activities", [])
    
    def get_polymarket_wallet(
        self,
        eoa: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> Dict:
        """
        Get wallet positions and data.
        
        Args:
            eoa: EOA wallet address
            proxy: Proxy wallet address
        
        Returns:
            Wallet data including positions
        """
        params = {}
        if eoa:
            params["eoa"] = eoa
        elif proxy:
            params["proxy"] = proxy
        else:
            return {}
        
        response = self._request("GET", "/polymarket/wallet", params=params)
        return response
    
    def get_polymarket_wallet_pnl(
        self,
        eoa: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> Dict:
        """
        Get wallet profit and loss data for reconciliation.
        
        Args:
            eoa: EOA wallet address
            proxy: Proxy wallet address
        
        Returns:
            PnL data including realized/unrealized
        """
        params = {}
        if eoa:
            params["eoa"] = eoa
        elif proxy:
            params["proxy"] = proxy
        else:
            return {}
        
        response = self._request(
            "GET",
            "/polymarket/wallet/pnl",
            params=params
        )
        return response
    
    # ========================================================================
    # Kalshi Endpoints
    # ========================================================================
    
    def get_kalshi_markets(
        self,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get Kalshi markets."""
        params = {"limit": limit}
        if search:
            params["search"] = search
        
        response = self._request("GET", "/kalshi/markets", params=params)
        return response.get("data", [])
    
    def get_kalshi_orderbook_history(
        self,
        ticker: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1h",
    ) -> List[OrderbookSnapshot]:
        """Get historical orderbook for Kalshi market."""
        params = {"ticker": ticker, "interval": interval}
        
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        response = self._request(
            "GET",
            "/kalshi/orderbook-history",
            params=params
        )
        
        snapshots = []
        for item in response.get("data", []):
            try:
                snapshots.append(OrderbookSnapshot(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    market_id=ticker,
                    bids=[(b["price"], b["size"]) for b in item.get("bids", [])],
                    asks=[(a["price"], a["size"]) for a in item.get("asks", [])],
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse orderbook: {e}")
        
        return snapshots
    
    # ========================================================================
    # Cross-Platform Market Matching
    # ========================================================================
    
    def get_matched_sports_markets(
        self,
        sport: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get matched markets across Polymarket and Kalshi.
        
        Useful for cross-platform arbitrage.
        
        Args:
            sport: Filter by sport (e.g., "nfl", "nba")
            date: Filter by date (YYYY-MM-DD)
        
        Returns:
            List of matched market pairs
        """
        params = {}
        if sport:
            params["sport"] = sport
        if date:
            params["date"] = date
        
        response = self._request("GET", "/matching/sports", params=params)
        return response.get("data", [])
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def compute_execution_price(
        self,
        orderbook: OrderbookSnapshot,
        side: str,
        size: float,
    ) -> Tuple[float, float]:
        """
        Compute execution price and slippage for an order.
        
        Args:
            orderbook: Current orderbook snapshot
            side: 'buy' or 'sell'
            size: Order size in USD
        
        Returns:
            (execution_price, slippage_bps)
        """
        if side == 'buy':
            levels = orderbook.asks
        else:
            levels = orderbook.bids
        
        if not levels:
            return (0.5, 0)  # No orderbook, use mid
        
        remaining = size
        weighted_price = 0.0
        total_filled = 0.0
        
        for price, level_size in levels:
            fill_size = min(remaining, level_size)
            weighted_price += price * fill_size
            total_filled += fill_size
            remaining -= fill_size
            
            if remaining <= 0:
                break
        
        if total_filled == 0:
            return (levels[0][0], 0)
        
        avg_price = weighted_price / total_filled
        mid = orderbook.mid_price or levels[0][0]
        slippage_bps = abs(avg_price - mid) / mid * 10000
        
        return (avg_price, slippage_bps)


class DomeBacktestDataProvider:
    """
    Provides historical data from Dome API for backtesting.
    
    Integrates with our existing backtest infrastructure.
    """
    
    def __init__(self, client: DomeAPIClient):
        self.client = client
        self._orderbook_cache: Dict[str, List[OrderbookSnapshot]] = {}
        self._candle_cache: Dict[str, List[Candlestick]] = {}
    
    def load_orderbook_history(
        self,
        token_id: str,
        start_date: str,
        end_date: str,
    ) -> List[OrderbookSnapshot]:
        """Load and cache orderbook history."""
        cache_key = f"{token_id}_{start_date}_{end_date}"
        
        if cache_key not in self._orderbook_cache:
            self._orderbook_cache[cache_key] = self.client.get_polymarket_orderbook_history(
                token_id=token_id,
                start_time=datetime.fromisoformat(start_date),
                end_time=datetime.fromisoformat(end_date),
            )
        
        return self._orderbook_cache[cache_key]
    
    def get_spread_at_time(
        self,
        token_id: str,
        timestamp: datetime,
    ) -> Optional[float]:
        """Get spread at a specific time for cost modeling."""
        # Find closest orderbook snapshot
        history = self._orderbook_cache.get(token_id, [])
        
        closest = None
        min_diff = timedelta(days=1)
        
        for snapshot in history:
            diff = abs(snapshot.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = snapshot
        
        if closest:
            return closest.spread
        return None
    
    def get_candlestick_features(
        self,
        token_id: str,
        timestamp: datetime,
        lookback: int = 10,
    ) -> Dict[str, float]:
        """
        Get candlestick-based features for momentum strategy.
        
        Returns:
            Dict with features like trend, volatility, volume
        """
        candles = self._candle_cache.get(token_id, [])
        
        # Get candles before timestamp
        past_candles = [c for c in candles if c.timestamp < timestamp][-lookback:]
        
        if len(past_candles) < 2:
            return {}
        
        closes = [c.close for c in past_candles]
        volumes = [c.volume for c in past_candles]
        
        return {
            'price_change': (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0,
            'volatility': max(c.high - c.low for c in past_candles),
            'avg_volume': sum(volumes) / len(volumes),
            'trend': 1 if closes[-1] > closes[-2] else -1,
            'n_bullish': sum(1 for c in past_candles if c.is_bullish),
        }


# Convenience function
def create_dome_client() -> DomeAPIClient:
    """Create Dome API client from environment."""
    api_key = os.environ.get("DOME_API_KEY", "")
    if not api_key:
        logger.warning("DOME_API_KEY not set, using unauthenticated access")
    
    return DomeAPIClient(DomeConfig(api_key=api_key))
