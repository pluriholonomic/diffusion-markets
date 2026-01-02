"""
Polymarket CLOB API Client

Polymarket uses a hybrid AMM + orderbook (CLOB) system.
API docs: https://docs.polymarket.com/
"""

import os
import time
import json
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

from ..utils.models import (
    Platform, Side, Order, OrderType, OrderStatus, Fill, Market, Position
)


@dataclass
class PolymarketConfig:
    """Configuration for Polymarket client."""
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137  # Polygon mainnet
    paper_trading: bool = True
    rate_limit_delay: float = 0.2


class PolymarketClient:
    """
    Client for Polymarket CLOB API.
    
    Supports:
    - Market data fetching
    - Order placement/cancellation
    - Position tracking
    - Paper trading mode
    """
    
    def __init__(self, config: Optional[PolymarketConfig] = None):
        self.config = config or PolymarketConfig()
        self.session = requests.Session()
        self._last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Sign request for authenticated endpoints."""
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        
        signature = hmac.new(
            self.config.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "POLY_API_KEY": self.config.api_key,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_PASSPHRASE": self.config.api_passphrase,
        }
    
    def _get(self, endpoint: str, params: Dict = None, auth: bool = False) -> Dict:
        """Make GET request."""
        self._rate_limit()
        url = f"{self.config.base_url}{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        if auth:
            headers.update(self._sign_request("GET", endpoint))
        
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _post(self, endpoint: str, data: Dict, auth: bool = True) -> Dict:
        """Make POST request."""
        self._rate_limit()
        url = f"{self.config.base_url}{endpoint}"
        body = json.dumps(data)
        
        headers = {"Content-Type": "application/json"}
        if auth:
            headers.update(self._sign_request("POST", endpoint, body))
        
        try:
            resp = self.session.post(url, data=body, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    # ===== Market Data =====
    
    def get_markets(self, limit: int = 100, active_only: bool = True) -> List[Market]:
        """Fetch available markets."""
        params = {"limit": limit}
        if active_only:
            params["active"] = "true"
        
        # Use Gamma API for market data
        url = f"{self.config.gamma_url}/markets"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            return []
        
        markets = []
        for m in data:
            try:
                market = Market(
                    market_id=m.get("conditionId", m.get("id", "")),
                    platform=Platform.POLYMARKET,
                    question=m.get("question", ""),
                    category=m.get("category", "other"),
                    current_yes_price=float(m.get("outcomePrices", ["0.5", "0.5"])[0]),
                    current_no_price=float(m.get("outcomePrices", ["0.5", "0.5"])[1]),
                    volume=float(m.get("volume", 0)),
                    liquidity=float(m.get("liquidity", 0)),
                    close_time=None,  # Parse if available
                    metadata=m,
                )
                markets.append(market)
            except Exception:
                continue
        
        return markets
    
    def get_market(self, market_id: str) -> Optional[Market]:
        """Fetch single market by ID."""
        url = f"{self.config.gamma_url}/markets/{market_id}"
        try:
            resp = self.session.get(url, timeout=30)
            m = resp.json()
            
            return Market(
                market_id=m.get("conditionId", market_id),
                platform=Platform.POLYMARKET,
                question=m.get("question", ""),
                category=m.get("category", "other"),
                current_yes_price=float(m.get("outcomePrices", ["0.5", "0.5"])[0]),
                current_no_price=float(m.get("outcomePrices", ["0.5", "0.5"])[1]),
                volume=float(m.get("volume", 0)),
                liquidity=float(m.get("liquidity", 0)),
                metadata=m,
            )
        except Exception:
            return None
    
    def get_orderbook(self, token_id: str) -> Dict[str, List]:
        """Fetch orderbook for a token."""
        data = self._get(f"/book", params={"token_id": token_id})
        return {
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
        }
    
    # ===== Order Management =====
    
    def place_order(self, order: Order) -> Order:
        """Place an order (or simulate in paper trading mode)."""
        if self.config.paper_trading:
            return self._paper_trade_order(order)
        
        # Real order placement
        token_id = self._get_token_id(order.market_id, order.side)
        
        order_data = {
            "tokenID": token_id,
            "price": str(order.price),
            "size": str(order.size),
            "side": "BUY",  # We always buy the side we want
            "type": order.order_type.value.upper(),
        }
        
        result = self._post("/order", order_data)
        
        if "error" in result:
            order.status = OrderStatus.REJECTED
            order.metadata["error"] = result["error"]
        else:
            order.status = OrderStatus.SUBMITTED
            order.platform_order_id = result.get("orderID", "")
        
        order.updated_at = datetime.utcnow()
        return order
    
    def cancel_order(self, order: Order) -> Order:
        """Cancel an order."""
        if self.config.paper_trading:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            return order
        
        result = self._post(f"/order/{order.platform_order_id}/cancel", {})
        
        if "error" not in result:
            order.status = OrderStatus.CANCELLED
        
        order.updated_at = datetime.utcnow()
        return order
    
    def get_order_status(self, order: Order) -> Order:
        """Check order status."""
        if self.config.paper_trading:
            return order
        
        result = self._get(f"/order/{order.platform_order_id}", auth=True)
        
        if "error" not in result:
            status_map = {
                "OPEN": OrderStatus.SUBMITTED,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "EXPIRED": OrderStatus.EXPIRED,
            }
            order.status = status_map.get(result.get("status", ""), order.status)
        
        order.updated_at = datetime.utcnow()
        return order
    
    def _paper_trade_order(self, order: Order) -> Order:
        """Simulate order execution for paper trading."""
        # Simple simulation: assume immediate fill at limit price
        order.status = OrderStatus.FILLED
        order.platform_order_id = f"paper_{order.order_id}"
        order.updated_at = datetime.utcnow()
        order.metadata["paper_trade"] = True
        return order
    
    def _get_token_id(self, market_id: str, side: Side) -> str:
        """Get token ID for a market side."""
        # In Polymarket, each outcome has a separate token
        # This would need to be looked up from market data
        market = self.get_market(market_id)
        if market and "tokens" in market.metadata:
            tokens = market.metadata["tokens"]
            if side == Side.YES:
                return tokens[0].get("token_id", market_id)
            else:
                return tokens[1].get("token_id", market_id)
        return market_id
    
    # ===== Position Management =====
    
    def get_positions(self) -> List[Position]:
        """Fetch current positions."""
        if self.config.paper_trading:
            return []  # Paper trading tracks positions locally
        
        result = self._get("/positions", auth=True)
        
        positions = []
        for p in result.get("positions", []):
            try:
                positions.append(Position(
                    platform=Platform.POLYMARKET,
                    market_id=p.get("conditionId", ""),
                    side=Side.YES if p.get("side") == "YES" else Side.NO,
                    size=float(p.get("size", 0)),
                    avg_entry_price=float(p.get("avgPrice", 0)),
                    current_price=float(p.get("currentPrice", 0)),
                ))
            except Exception:
                continue
        
        return positions
    
    def get_balance(self) -> float:
        """Get USDC balance."""
        if self.config.paper_trading:
            return 0.0  # Paper trading tracks balance locally
        
        result = self._get("/balance", auth=True)
        return float(result.get("balance", 0))


# Factory function
def create_polymarket_client(paper_trading: bool = True) -> PolymarketClient:
    """Create Polymarket client from environment variables."""
    config = PolymarketConfig(
        api_key=os.environ.get("POLYMARKET_API_KEY", ""),
        api_secret=os.environ.get("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.environ.get("POLYMARKET_API_PASSPHRASE", ""),
        paper_trading=paper_trading,
    )
    return PolymarketClient(config)
