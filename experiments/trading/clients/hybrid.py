"""
Hybrid Market Clients

Fetches LIVE market data from APIs but simulates order execution.
Best of both worlds for realistic paper trading.
"""

import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from ..utils.models import (
    Platform, Side, Order, OrderType, OrderStatus, Fill, Market, Position
)

logger = logging.getLogger(__name__)


class HybridPolymarketClient:
    """
    Hybrid Polymarket client:
    - LIVE market data from Polymarket CLOB API (no auth required for reads)
    - SIMULATED order execution
    """
    
    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, rate_limit_delay: float = 0.2):
        self.rate_limit_delay = rate_limit_delay
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self._last_request_time = 0
        
    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def get_markets(self, limit: int = 100, active_only: bool = True) -> List[Market]:
        """Fetch LIVE markets from Polymarket API."""
        markets = []
        
        try:
            self._rate_limit()
            
            # Use gamma API for market metadata
            # Fetch more markets since many have 0 prices, sort by volume to get active ones
            # Note: Don't use active=true as it filters too aggressively
            params = {
                "limit": max(limit * 5, 500),  # Fetch more markets to cover positions
                "order": "volume",  # Sort by volume to get active markets first
                "ascending": "false",
            }
            
            response = requests.get(
                f"{self.GAMMA_URL}/markets",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                logger.warning(f"Unexpected Polymarket response format: {type(data)}")
                return []
            
            for item in data:
                try:
                    # Get current prices - gamma API uses outcomePrices (may be JSON string)
                    outcome_prices = item.get('outcomePrices', None)
                    yes_price = 0.5
                    no_price = 0.5
                    
                    if outcome_prices:
                        # May be a JSON string like '["0.5", "0.5"]'
                        if isinstance(outcome_prices, str):
                            import json
                            try:
                                outcome_prices = json.loads(outcome_prices)
                            except:
                                pass
                        
                        if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                            yes_price = float(outcome_prices[0])
                            no_price = float(outcome_prices[1])
                    
                    # Normalize prices
                    total = yes_price + no_price
                    if total > 0:
                        yes_price = yes_price / total
                        no_price = 1 - yes_price
                    
                    # Skip markets with exactly 0/1 prices (no trading activity)
                    if yes_price <= 0.0001 or yes_price >= 0.9999:
                        continue
                    
                    # Clamp
                    yes_price = max(0.01, min(0.99, yes_price))
                    no_price = 1 - yes_price
                    
                    # Prefer mid-range markets (0.15 - 0.85) that are more likely to move
                    # Extreme markets (0.01 or 0.99) rarely fluctuate
                    is_mid_range = 0.15 <= yes_price <= 0.85
                    
                    # Parse volume/liquidity (may be strings)
                    vol = item.get('volume', 0)
                    liq = item.get('liquidity', 0)
                    vol = float(vol) if vol else 0
                    liq = float(liq) if liq else 0
                    
                    market = Market(
                        market_id=item.get('conditionId', item.get('id', '')),
                        platform=Platform.POLYMARKET,
                        question=item.get('question', ''),
                        category=item.get('category', 'general'),
                        current_yes_price=yes_price,
                        current_no_price=no_price,
                        volume=vol,
                        liquidity=liq,
                        metadata={
                            'live': True,
                            'slug': item.get('slug'),
                            'end_date': item.get('endDate'),
                            'is_mid_range': is_mid_range,
                        }
                    )
                    markets.append(market)
                    
                except Exception as e:
                    logger.debug(f"Error parsing market: {e}")
                    continue
            
            # Sort to prioritize mid-range markets (more likely to have price movements)
            markets.sort(key=lambda m: (
                not m.metadata.get('is_mid_range', False),  # Mid-range first
                -m.volume  # Then by volume
            ))
            
            logger.info(f"Fetched {len(markets)} live markets from Polymarket")
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch Polymarket markets: {e}")
        except Exception as e:
            logger.warning(f"Error fetching Polymarket markets: {e}")
        
        return markets
    
    def get_orderbook(self, token_id: str) -> Dict[str, List]:
        """Fetch live orderbook for a token."""
        try:
            self._rate_limit()
            response = requests.get(
                f"{self.CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook: {e}")
            return {'bids': [], 'asks': []}
    
    def place_order(self, order: Order) -> Order:
        """SIMULATE order placement."""
        order.status = OrderStatus.FILLED
        order.platform_order_id = f"sim_pm_{order.order_id}"
        order.updated_at = datetime.utcnow()
        order.metadata['simulated'] = True
        order.metadata['execution_type'] = 'hybrid_simulated'
        
        self.orders[order.order_id] = order
        logger.info(f"[SIMULATED] Polymarket order filled: {order.order_id}")
        
        return order
    
    def cancel_order(self, order: Order) -> Order:
        """Cancel an order (simulated)."""
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        return order
    
    def get_positions(self) -> List[Position]:
        """Get simulated positions."""
        return list(self.positions.values())
    
    def get_balance(self) -> float:
        """Balance tracked externally in paper trading."""
        return 0.0


class HybridKalshiClient:
    """
    Hybrid Kalshi client:
    - LIVE market data from Kalshi public API
    - SIMULATED order execution
    """
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, rate_limit_delay: float = 0.3):
        self.rate_limit_delay = rate_limit_delay
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def get_markets(self, limit: int = 50, status: str = "open") -> List[Market]:
        """Fetch LIVE markets from Kalshi API."""
        markets = []
        
        try:
            self._rate_limit()
            
            params = {
                "limit": limit,
                "status": status,
            }
            
            response = requests.get(
                f"{self.BASE_URL}/markets",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('markets', []):
                try:
                    # Kalshi prices are in cents (0-100)
                    yes_bid = item.get('yes_bid', 0) or 0
                    last_price = item.get('last_price', 0) or 0
                    
                    # Prefer last_price, fall back to yes_bid
                    if last_price > 0:
                        yes_price = float(last_price) / 100
                    elif yes_bid > 0:
                        yes_price = float(yes_bid) / 100
                    else:
                        # Skip markets with no prices
                        continue
                    
                    no_price = 1 - yes_price
                    
                    # Skip markets with exactly 0/1 prices (no trading activity)
                    if yes_price <= 0.0001 or yes_price >= 0.9999:
                        continue
                    
                    # Parse volume
                    vol = item.get('volume', 0) or 0
                    oi = item.get('open_interest', 0) or 0
                    
                    market = Market(
                        market_id=item.get('ticker', ''),
                        platform=Platform.KALSHI,
                        question=item.get('title', ''),
                        category=item.get('category', 'general'),
                        current_yes_price=max(0.01, min(0.99, yes_price)),
                        current_no_price=max(0.01, min(0.99, no_price)),
                        volume=float(vol),
                        liquidity=float(oi),
                        metadata={
                            'live': True,
                            'event_ticker': item.get('event_ticker'),
                            'subtitle': item.get('subtitle'),
                            'close_time': item.get('close_time'),
                        }
                    )
                    markets.append(market)
                    
                except Exception as e:
                    logger.debug(f"Error parsing Kalshi market: {e}")
                    continue
            
            logger.info(f"Fetched {len(markets)} live markets from Kalshi")
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch Kalshi markets: {e}")
        except Exception as e:
            logger.warning(f"Error fetching Kalshi markets: {e}")
        
        return markets
    
    def get_orderbook(self, ticker: str) -> Dict[str, List]:
        """Fetch live orderbook."""
        try:
            self._rate_limit()
            response = requests.get(
                f"{self.BASE_URL}/markets/{ticker}/orderbook",
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get('orderbook', {'bids': [], 'asks': []})
        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi orderbook: {e}")
            return {'bids': [], 'asks': []}
    
    def place_order(self, order: Order) -> Order:
        """SIMULATE order placement."""
        order.status = OrderStatus.FILLED
        order.platform_order_id = f"sim_kalshi_{order.order_id}"
        order.updated_at = datetime.utcnow()
        order.metadata['simulated'] = True
        order.metadata['execution_type'] = 'hybrid_simulated'
        
        self.orders[order.order_id] = order
        logger.info(f"[SIMULATED] Kalshi order filled: {order.order_id}")
        
        return order
    
    def cancel_order(self, order: Order) -> Order:
        """Cancel an order (simulated)."""
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        return order
    
    def get_positions(self) -> List[Position]:
        """Get simulated positions."""
        return list(self.positions.values())
    
    def get_balance(self) -> float:
        """Balance tracked externally."""
        return 0.0


def create_hybrid_polymarket() -> HybridPolymarketClient:
    """Create hybrid Polymarket client."""
    return HybridPolymarketClient()


def create_hybrid_kalshi() -> HybridKalshiClient:
    """Create hybrid Kalshi client."""
    return HybridKalshiClient()
