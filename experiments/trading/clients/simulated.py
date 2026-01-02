"""
Simulated Market Client

Uses historical data to simulate live markets for paper trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..utils.models import (
    Platform, Side, Order, OrderType, OrderStatus, Fill, Market, Position
)


class SimulatedMarketClient:
    """
    Simulated client that uses historical data to provide market data.
    
    Useful for:
    - Paper trading without live API credentials
    - Backtesting strategies
    - Testing execution logic
    """
    
    def __init__(
        self,
        platform: Platform,
        data_path: str,
        sample_size: int = 50,
    ):
        self.platform = platform
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        
        # Load data
        self.data: Optional[pd.DataFrame] = None
        self._load_data()
        
        # State
        self.current_idx = 0
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
    
    def _load_data(self):
        """Load historical data."""
        if not self.data_path.exists():
            print(f"Warning: Data file not found: {self.data_path}")
            return
        
        df = pd.read_parquet(self.data_path)
        
        # Standardize columns
        if 'avg_price' in df.columns:
            df = df.rename(columns={'avg_price': 'price'})
        if 'y' in df.columns:
            df = df.rename(columns={'y': 'outcome'})
        
        df['price'] = df['price'].clip(0.01, 0.99)
        
        # Create market_id if not present
        if 'market_id' not in df.columns:
            if 'conditionId' in df.columns:
                df['market_id'] = df['conditionId']
            elif 'ticker' in df.columns:
                df['market_id'] = df['ticker']
            else:
                df['market_id'] = df.index.astype(str)
        
        if 'category' not in df.columns:
            df['category'] = 'general'
        
        if 'volume' not in df.columns:
            df['volume'] = 10000
        
        if 'liquidity' not in df.columns:
            df['liquidity'] = df['volume'] * 0.1
        
        self.data = df
    
    def get_markets(self, limit: int = None) -> List[Market]:
        """Get simulated markets from historical data."""
        if self.data is None or len(self.data) == 0:
            return []
        
        # Sample random markets
        n = min(limit or self.sample_size, len(self.data))
        sample = self.data.sample(n=n)
        
        markets = []
        for _, row in sample.iterrows():
            try:
                price = float(row['price'])
                
                market = Market(
                    market_id=str(row['market_id']),
                    platform=self.platform,
                    question=row.get('question', f"Market {row['market_id']}"),
                    category=str(row.get('category', 'general')),
                    current_yes_price=price,
                    current_no_price=1 - price,
                    volume=float(row.get('volume', 10000)),
                    liquidity=float(row.get('liquidity', 1000)),
                    outcome=int(row['outcome']) if pd.notna(row.get('outcome')) else None,
                    metadata={
                        'simulated': True,
                        'source_idx': row.name,
                    }
                )
                markets.append(market)
            except Exception as e:
                continue
        
        return markets
    
    def get_market(self, market_id: str) -> Optional[Market]:
        """Get a specific market."""
        if self.data is None:
            return None
        
        matches = self.data[self.data['market_id'] == market_id]
        if len(matches) == 0:
            return None
        
        row = matches.iloc[0]
        price = float(row['price'])
        
        return Market(
            market_id=str(row['market_id']),
            platform=self.platform,
            question=row.get('question', ''),
            category=str(row.get('category', 'general')),
            current_yes_price=price,
            current_no_price=1 - price,
            volume=float(row.get('volume', 10000)),
            liquidity=float(row.get('liquidity', 1000)),
            outcome=int(row['outcome']) if pd.notna(row.get('outcome')) else None,
            metadata={'simulated': True},
        )
    
    def place_order(self, order: Order) -> Order:
        """Simulate order placement and compute PnL."""
        # Get market outcome if available
        market = self.get_market(order.market_id)
        
        # Simulate fill (paper trading)
        order.status = OrderStatus.FILLED
        order.platform_order_id = f"sim_{order.order_id}"
        order.updated_at = datetime.utcnow()
        order.metadata['simulated'] = True
        
        # Compute PnL if we have outcome data
        outcome = market.outcome if market else None
        order.metadata['outcome'] = outcome
        
        if outcome is not None:
            # Compute PnL:
            # order.price is the price for the side being bet:
            # - YES bet: order.price = yes_price
            # - NO bet: order.price = no_price  
            #
            # When you bet at price p and win:
            # - You pay $p per contract to get $1 back
            # - Contracts bought = size / p
            # - Payout = size / p
            # - Profit = size/p - size = size * (1-p) / p
            #
            # When you lose, you lose your entire bet.
            price = order.price
            size = order.size
            
            if order.side.value == 'yes':
                if outcome == 1:
                    pnl = size * (1 - price) / price  # Win
                    order.metadata['result'] = 'win'
                else:
                    pnl = -size  # Lose entire bet
                    order.metadata['result'] = 'loss'
            else:  # NO
                if outcome == 0:
                    pnl = size * (1 - price) / price  # Win (same formula)
                    order.metadata['result'] = 'win'
                else:
                    pnl = -size  # Lose entire bet
                    order.metadata['result'] = 'loss'
            
            order.metadata['pnl'] = pnl
        else:
            order.metadata['pnl'] = 0  # Unknown outcome
            order.metadata['result'] = 'pending'
        
        self.orders[order.order_id] = order
        
        return order
    
    def cancel_order(self, order: Order) -> Order:
        """Cancel an order."""
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        return order
    
    def get_order_status(self, order: Order) -> Order:
        """Get order status."""
        return self.orders.get(order.order_id, order)
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())
    
    def get_balance(self) -> float:
        """Get simulated balance."""
        return 0.0  # Paper trading tracks balance externally
    
    def get_orderbook(self, market_id: str) -> Dict[str, List]:
        """Get simulated orderbook."""
        market = self.get_market(market_id)
        if not market:
            return {'bids': [], 'asks': []}
        
        price = market.current_yes_price
        
        # Simulate simple orderbook
        return {
            'bids': [{'price': price - 0.01, 'size': 1000}],
            'asks': [{'price': price + 0.01, 'size': 1000}],
        }


def create_simulated_polymarket(data_path: str = "data/polymarket/optimization_cache.parquet") -> SimulatedMarketClient:
    """Create simulated Polymarket client."""
    return SimulatedMarketClient(
        platform=Platform.POLYMARKET,
        data_path=data_path,
        sample_size=50,
    )


def create_simulated_kalshi(data_path: str = "data/kalshi/kalshi_backtest_clean.parquet") -> SimulatedMarketClient:
    """Create simulated Kalshi client."""
    return SimulatedMarketClient(
        platform=Platform.KALSHI,
        data_path=data_path,
        sample_size=50,
    )
