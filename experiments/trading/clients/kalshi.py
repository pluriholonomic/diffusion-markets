"""
Kalshi REST API Client

Kalshi is a CFTC-regulated prediction market exchange.
API docs: https://trading-api.readme.io/reference/
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

from ..utils.models import (
    Platform, Side, Order, OrderType, OrderStatus, Fill, Market, Position
)


@dataclass
class KalshiConfig:
    """Configuration for Kalshi client."""
    email: str = ""
    password: str = ""
    api_key: str = ""
    base_url: str = "https://trading-api.kalshi.com/trade-api/v2"
    paper_trading: bool = True
    rate_limit_delay: float = 0.2


class KalshiClient:
    """
    Client for Kalshi Trading API.
    
    Supports:
    - Market data fetching
    - Order placement/cancellation
    - Position tracking
    - Paper trading mode
    """
    
    def __init__(self, config: Optional[KalshiConfig] = None):
        self.config = config or KalshiConfig()
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expiry: float = 0
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _ensure_authenticated(self):
        """Ensure we have a valid auth token."""
        if self._token and time.time() < self._token_expiry:
            return
        
        # Login to get token
        login_data = {
            "email": self.config.email,
            "password": self.config.password,
        }
        
        try:
            resp = self.session.post(
                f"{self.config.base_url}/login",
                json=login_data,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data.get("token", "")
            # Token valid for 24 hours, refresh after 23
            self._token_expiry = time.time() + 23 * 3600
        except Exception as e:
            self._token = None
            raise Exception(f"Kalshi login failed: {e}")
    
    def _get_headers(self, auth: bool = True) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    def _get(self, endpoint: str, params: Dict = None, auth: bool = False) -> Dict:
        """Make GET request."""
        self._rate_limit()
        if auth:
            self._ensure_authenticated()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            resp = self.session.get(
                url, 
                params=params, 
                headers=self._get_headers(auth),
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _post(self, endpoint: str, data: Dict, auth: bool = True) -> Dict:
        """Make POST request."""
        self._rate_limit()
        if auth:
            self._ensure_authenticated()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            resp = self.session.post(
                url,
                json=data,
                headers=self._get_headers(auth),
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _delete(self, endpoint: str, auth: bool = True) -> Dict:
        """Make DELETE request."""
        self._rate_limit()
        if auth:
            self._ensure_authenticated()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            resp = self.session.delete(
                url,
                headers=self._get_headers(auth),
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    # ===== Market Data =====
    
    def get_markets(
        self, 
        status: str = "open",
        limit: int = 100,
        cursor: str = None
    ) -> List[Market]:
        """Fetch available markets."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        
        result = self._get("/markets", params=params)
        
        markets = []
        for m in result.get("markets", []):
            try:
                # Kalshi prices are in cents (0-100)
                yes_price = m.get("yes_bid", 50) / 100.0
                no_price = 1 - yes_price
                
                market = Market(
                    market_id=m.get("ticker", ""),
                    platform=Platform.KALSHI,
                    question=m.get("title", ""),
                    category=m.get("category", m.get("event_ticker", "").split("-")[0]),
                    current_yes_price=yes_price,
                    current_no_price=no_price,
                    volume=float(m.get("volume", 0)),
                    liquidity=float(m.get("liquidity", 0)),
                    close_time=None,
                    metadata=m,
                )
                markets.append(market)
            except Exception:
                continue
        
        return markets
    
    def get_market(self, ticker: str) -> Optional[Market]:
        """Fetch single market by ticker."""
        result = self._get(f"/markets/{ticker}")
        
        if "error" in result or "market" not in result:
            return None
        
        m = result["market"]
        try:
            yes_price = m.get("yes_bid", 50) / 100.0
            
            return Market(
                market_id=m.get("ticker", ticker),
                platform=Platform.KALSHI,
                question=m.get("title", ""),
                category=m.get("category", ""),
                current_yes_price=yes_price,
                current_no_price=1 - yes_price,
                volume=float(m.get("volume", 0)),
                liquidity=float(m.get("liquidity", 0)),
                metadata=m,
            )
        except Exception:
            return None
    
    def get_orderbook(self, ticker: str) -> Dict[str, List]:
        """Fetch orderbook for a market."""
        result = self._get(f"/markets/{ticker}/orderbook")
        return {
            "yes_bids": result.get("orderbook", {}).get("yes", []),
            "no_bids": result.get("orderbook", {}).get("no", []),
        }
    
    # ===== Order Management =====
    
    def place_order(self, order: Order) -> Order:
        """Place an order (or simulate in paper trading mode)."""
        if self.config.paper_trading:
            return self._paper_trade_order(order)
        
        # Kalshi uses cents (0-100) for prices
        price_cents = int(order.price * 100)
        
        order_data = {
            "ticker": order.market_id,
            "action": "buy",
            "side": order.side.value,
            "type": order.order_type.value,
            "count": int(order.size / order.price),  # Number of contracts
            "yes_price" if order.side == Side.YES else "no_price": price_cents,
        }
        
        result = self._post("/portfolio/orders", order_data)
        
        if "error" in result:
            order.status = OrderStatus.REJECTED
            order.metadata["error"] = result["error"]
        else:
            order.status = OrderStatus.SUBMITTED
            order.platform_order_id = result.get("order", {}).get("order_id", "")
        
        order.updated_at = datetime.utcnow()
        return order
    
    def cancel_order(self, order: Order) -> Order:
        """Cancel an order."""
        if self.config.paper_trading:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            return order
        
        result = self._delete(f"/portfolio/orders/{order.platform_order_id}")
        
        if "error" not in result:
            order.status = OrderStatus.CANCELLED
        
        order.updated_at = datetime.utcnow()
        return order
    
    def get_order_status(self, order: Order) -> Order:
        """Check order status."""
        if self.config.paper_trading:
            return order
        
        result = self._get(f"/portfolio/orders/{order.platform_order_id}", auth=True)
        
        if "error" not in result and "order" in result:
            status_map = {
                "resting": OrderStatus.SUBMITTED,
                "pending": OrderStatus.PENDING,
                "executed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
            }
            order.status = status_map.get(
                result["order"].get("status", ""), 
                order.status
            )
        
        order.updated_at = datetime.utcnow()
        return order
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        result = self._get("/portfolio/orders", auth=True)
        
        orders = []
        for o in result.get("orders", []):
            try:
                orders.append(Order(
                    order_id=o.get("order_id", ""),
                    platform=Platform.KALSHI,
                    market_id=o.get("ticker", ""),
                    side=Side.YES if o.get("side") == "yes" else Side.NO,
                    order_type=OrderType.LIMIT,
                    size=float(o.get("remaining_count", 0)) * float(o.get("yes_price", 50)) / 100,
                    price=float(o.get("yes_price", 50)) / 100,
                    status=OrderStatus.SUBMITTED,
                    platform_order_id=o.get("order_id", ""),
                ))
            except Exception:
                continue
        
        return orders
    
    def _paper_trade_order(self, order: Order) -> Order:
        """Simulate order execution for paper trading."""
        order.status = OrderStatus.FILLED
        order.platform_order_id = f"paper_{order.order_id}"
        order.updated_at = datetime.utcnow()
        order.metadata["paper_trade"] = True
        return order
    
    # ===== Position Management =====
    
    def get_positions(self) -> List[Position]:
        """Fetch current positions."""
        if self.config.paper_trading:
            return []
        
        result = self._get("/portfolio/positions", auth=True)
        
        positions = []
        for p in result.get("market_positions", []):
            try:
                # Kalshi returns yes and no positions separately
                if p.get("position", 0) != 0:
                    side = Side.YES if p.get("position", 0) > 0 else Side.NO
                    size = abs(p.get("position", 0))
                    
                    positions.append(Position(
                        platform=Platform.KALSHI,
                        market_id=p.get("ticker", ""),
                        side=side,
                        size=size,
                        avg_entry_price=float(p.get("average_price", 50)) / 100,
                        current_price=float(p.get("market_price", 50)) / 100,
                    ))
            except Exception:
                continue
        
        return positions
    
    def get_balance(self) -> float:
        """Get account balance."""
        if self.config.paper_trading:
            return 0.0
        
        result = self._get("/portfolio/balance", auth=True)
        # Kalshi returns balance in cents
        return float(result.get("balance", 0)) / 100
    
    def get_fills(self, ticker: str = None, limit: int = 100) -> List[Fill]:
        """Get recent fills."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        
        result = self._get("/portfolio/fills", params=params, auth=True)
        
        fills = []
        for f in result.get("fills", []):
            try:
                fills.append(Fill(
                    fill_id=f.get("trade_id", ""),
                    order_id=f.get("order_id", ""),
                    platform=Platform.KALSHI,
                    market_id=f.get("ticker", ""),
                    side=Side.YES if f.get("side") == "yes" else Side.NO,
                    size=float(f.get("count", 0)),
                    price=float(f.get("yes_price", 50)) / 100,
                    fee=float(f.get("fee", 0)) / 100,
                    platform_fill_id=f.get("trade_id", ""),
                ))
            except Exception:
                continue
        
        return fills


# Factory function
def create_kalshi_client(paper_trading: bool = True) -> KalshiClient:
    """Create Kalshi client from environment variables."""
    config = KalshiConfig(
        email=os.environ.get("KALSHI_EMAIL", ""),
        password=os.environ.get("KALSHI_PASSWORD", ""),
        api_key=os.environ.get("KALSHI_API_KEY", ""),
        paper_trading=paper_trading,
    )
    return KalshiClient(config)
