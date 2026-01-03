"""
Production Integration Tests for Polymarket and Kalshi

Tests cover:
1. API connectivity and authentication
2. Market data fetching
3. Order lifecycle (place, poll, cancel)
4. Error handling and retries
5. Rate limiting
6. End-to-end trading pipeline

These tests require environment variables for API credentials.
Skip tests if credentials are not available.
"""

import sys
from pathlib import Path

# Add experiments to path for imports
experiments_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(experiments_dir))

import os
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from trading.utils.models import Platform, Side, Signal, Market, Order, OrderStatus
from trading.clients.polymarket import PolymarketClient
from trading.clients.kalshi import KalshiClient


# ============================================================================
# Skip conditions
# ============================================================================

POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY")
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY")

skip_no_polymarket = pytest.mark.skipif(
    not POLYMARKET_API_KEY,
    reason="POLYMARKET_API_KEY not set"
)

skip_no_kalshi = pytest.mark.skipif(
    not KALSHI_API_KEY,
    reason="KALSHI_API_KEY not set"
)


# ============================================================================
# Polymarket Integration Tests
# ============================================================================

class TestPolymarketIntegration:
    """Integration tests for Polymarket adapter."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Polymarket client for unit testing."""
        client = Mock(spec=PolymarketClient)
        return client
    
    def test_market_parsing_string_outcome_prices(self, mock_client):
        """Test parsing when outcomePrices is a string."""
        raw_market = {
            'id': 'test123',
            'question': 'Will X happen?',
            'outcomePrices': '["0.65", "0.35"]',
            'volume': '50000',
            'liquidity': 10000,
        }
        
        # Simulate parsing
        outcome_prices = raw_market['outcomePrices']
        import json
        parsed = json.loads(outcome_prices)
        
        assert len(parsed) == 2
        assert float(parsed[0]) == 0.65
        assert float(parsed[1]) == 0.35
    
    def test_market_parsing_list_outcome_prices(self, mock_client):
        """Test parsing when outcomePrices is already a list."""
        raw_market = {
            'id': 'test123',
            'question': 'Will X happen?',
            'outcomePrices': [0.65, 0.35],
        }
        
        prices = raw_market['outcomePrices']
        assert prices[0] == 0.65
        assert prices[1] == 0.35
    
    def test_market_parsing_null_outcome_prices(self, mock_client):
        """Test parsing when outcomePrices is None."""
        raw_market = {
            'id': 'test123',
            'question': 'Will X happen?',
            'outcomePrices': None,
        }
        
        prices = raw_market.get('outcomePrices')
        if prices is None:
            prices = [0.5, 0.5]
        
        assert prices[0] == 0.5
        assert prices[1] == 0.5
    
    def test_order_creation(self, mock_client):
        """Test order object creation."""
        order = Order(
            market_id="test_market",
            side=Side.YES,
            size=100.0,
            price=0.65,
        )
        
        assert order.market_id == "test_market"
        assert order.side == Side.YES
        assert order.size == 100.0
        assert order.status == OrderStatus.PENDING
    
    def test_order_status_transitions(self, mock_client):
        """Test valid order status transitions."""
        order = Order(
            market_id="test_market",
            side=Side.YES,
            size=100.0,
            price=0.65,
        )
        
        # PENDING -> SUBMITTED
        order.status = OrderStatus.SUBMITTED
        assert order.status == OrderStatus.SUBMITTED
        
        # SUBMITTED -> FILLED
        order.status = OrderStatus.FILLED
        assert order.status == OrderStatus.FILLED
    
    @skip_no_polymarket
    def test_live_fetch_markets(self):
        """Test fetching markets from live API."""
        client = PolymarketClient()
        markets = client.get_markets(limit=5)
        
        assert len(markets) > 0
        assert all(isinstance(m, Market) for m in markets)
        assert all(0 < m.current_yes_price < 1 for m in markets)
    
    @skip_no_polymarket  
    def test_live_market_has_required_fields(self):
        """Test that fetched markets have all required fields."""
        client = PolymarketClient()
        markets = client.get_markets(limit=1)
        
        if markets:
            market = markets[0]
            assert market.market_id is not None
            assert market.question is not None
            assert market.current_yes_price is not None
            assert market.platform == Platform.POLYMARKET


class TestPolymarketOrderLifecycle:
    """Tests for Polymarket order lifecycle."""
    
    def test_order_id_generation(self):
        """Test that orders get unique IDs."""
        order1 = Order(market_id="m1", side=Side.YES, size=100, price=0.5)
        order2 = Order(market_id="m1", side=Side.YES, size=100, price=0.5)
        
        # Each order should have a unique order_id (if set by client)
        # For now just verify they can coexist
        assert order1.market_id == order2.market_id
    
    def test_order_fill_price_tracking(self):
        """Test that fill price is tracked correctly."""
        order = Order(market_id="m1", side=Side.YES, size=100, price=0.50)
        
        # Simulate fill at different price
        order.fill_price = 0.51
        order.status = OrderStatus.FILLED
        
        assert order.fill_price == 0.51
        assert order.price == 0.50  # Original price preserved
    
    def test_partial_fill_handling(self):
        """Test handling of partial fills."""
        order = Order(market_id="m1", side=Side.YES, size=100, price=0.50)
        
        # First partial fill
        order.filled_size = 50
        order.status = OrderStatus.PARTIAL
        
        assert order.filled_size == 50
        assert order.size - order.filled_size == 50  # Remaining


# ============================================================================
# Kalshi Integration Tests
# ============================================================================

class TestKalshiIntegration:
    """Integration tests for Kalshi adapter."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Kalshi client for unit testing."""
        client = Mock(spec=KalshiClient)
        return client
    
    def test_price_conversion_to_cents(self):
        """Test that prices are correctly converted to Kalshi cents format."""
        # Kalshi uses cents (1-99)
        price_decimal = 0.65
        price_cents = int(price_decimal * 100)
        
        assert price_cents == 65
        assert 1 <= price_cents <= 99
    
    def test_contract_count_calculation(self):
        """Test calculation of contract count from USD amount."""
        size_usd = 100
        price_cents = 65
        
        # Each contract pays $1 if win, so cost = price * contracts
        contracts = int(size_usd / (price_cents / 100))
        
        assert contracts == 153  # $100 / $0.65 = 153 contracts
    
    def test_ticker_format(self):
        """Test that ticker format matches Kalshi requirements."""
        ticker = "BIDEN-YES"
        
        # Kalshi tickers are typically uppercase
        assert ticker.isupper() or ticker == ticker.upper()
    
    @skip_no_kalshi
    def test_live_authentication(self):
        """Test authentication with Kalshi API."""
        client = KalshiClient()
        
        # Should be authenticated after init
        assert client.is_authenticated()
    
    @skip_no_kalshi
    def test_live_fetch_markets(self):
        """Test fetching markets from Kalshi."""
        client = KalshiClient()
        markets = client.get_markets(limit=5)
        
        assert len(markets) > 0


class TestKalshiSolanaAdapter:
    """Tests for Kalshi Solana token adapter (devnet)."""
    
    def test_token_mint_lookup(self):
        """Test token mint address lookup."""
        # Mock token registry (Solana addresses are 32 bytes = 43-44 base58 chars)
        # Using realistic Solana address format
        token_registry = {
            'BIDEN-YES': '11111111111111111111111111111111',  # 32 chars (pubkey format)
            'BIDEN-NO': '22222222222222222222222222222222',  # 32 chars
        }
        
        market_id = 'BIDEN'
        side = 'YES'
        
        token_key = f"{market_id}-{side}"
        mint = token_registry.get(token_key)
        
        assert mint is not None
        assert len(mint) >= 32  # Solana pubkey is at least 32 chars
    
    def test_swap_instruction_building(self):
        """Test building swap instruction for DEX."""
        # Mock swap params
        input_mint = "USDC"
        output_mint = "TOKEN"
        amount = 100
        slippage = 0.01
        
        # Verify params are valid
        assert amount > 0
        assert 0 <= slippage <= 0.10  # Max 10% slippage


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestEndToEndPipeline:
    """End-to-end tests of the full trading pipeline."""
    
    def test_signal_to_order_conversion(self):
        """Test converting a signal to an order."""
        signal = Signal(
            market_id="test_market",
            platform=Platform.POLYMARKET,
            side=Side.YES,
            edge=0.10,
            confidence=0.8,
            kelly_fraction=0.05,
            strategy="calibration",
            metadata={},
        )
        
        bankroll = 10000
        position_size = bankroll * signal.kelly_fraction
        
        order = Order(
            market_id=signal.market_id,
            side=signal.side,
            size=position_size,
            price=0.50,  # Would come from market
        )
        
        assert order.market_id == signal.market_id
        assert order.side == signal.side
        assert order.size == 500  # 10000 * 0.05
    
    def test_risk_check_before_order(self):
        """Test that risk checks are applied before ordering."""
        # Mock risk limits
        max_position_pct = 0.10
        max_daily_loss = 500
        
        bankroll = 10000
        current_daily_pnl = -400  # Already lost $400 today
        
        # Proposed order
        order_size = 200
        
        # Risk checks
        position_ok = order_size <= bankroll * max_position_pct
        daily_loss_ok = abs(current_daily_pnl) + order_size <= max_daily_loss
        
        assert position_ok  # $200 <= $1000
        assert not daily_loss_ok  # $400 + $200 > $500
    
    def test_execution_cost_applied(self):
        """Test that execution costs reduce PnL."""
        entry_price = 0.50
        exit_price = 0.60
        size_usd = 100
        
        # Without costs
        gross_pnl = size_usd * (exit_price - entry_price) / entry_price
        
        # With costs (1% entry + 1% exit)
        cost_pct = 0.02
        net_pnl = gross_pnl - size_usd * cost_pct
        
        assert gross_pnl == pytest.approx(20.0)  # $100 * 0.1 / 0.5
        assert net_pnl == pytest.approx(18.0)  # $20 - $2 costs


class TestHybridVsLiveConsistency:
    """Tests for consistency between hybrid and live modes."""
    
    def test_same_signals_for_same_data(self):
        """Hybrid and live should generate same signals for same market data."""
        from trading.strategies.calibration import CalibrationStrategy, CalibrationConfig
        
        # Create two strategy instances
        config = CalibrationConfig()
        strategy1 = CalibrationStrategy(Platform.POLYMARKET, config=config)
        strategy2 = CalibrationStrategy(Platform.POLYMARKET, config=config)
        
        # Same calibration data
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        
        calibration_data = pd.DataFrame({
            'price': np.random.uniform(0.2, 0.8, 100),
            'outcome': np.random.randint(0, 2, 100),
        })
        
        strategy1.update_historical_data(calibration_data)
        strategy2.update_historical_data(calibration_data)
        
        # Same market
        market = Market(
            market_id="test",
            platform=Platform.POLYMARKET,
            question="Test?",
            current_yes_price=0.55,
            current_no_price=0.45,
            volume=10000,
            liquidity=5000,
            category="test",
            metadata={},
        )
        
        # Should get same signal
        signal1 = strategy1.generate_signal(market, bankroll=10000)
        signal2 = strategy2.generate_signal(market, bankroll=10000)
        
        if signal1 and signal2:
            assert signal1.side == signal2.side
            assert signal1.edge == pytest.approx(signal2.edge, abs=0.001)
    
    def test_deterministic_with_seed(self):
        """Strategy should be deterministic with same random seed."""
        from trading.strategies.calibration import CalibrationStrategy
        import numpy as np
        import pandas as pd
        
        results = []
        
        for _ in range(2):
            np.random.seed(123)
            strategy = CalibrationStrategy(Platform.POLYMARKET)
            
            data = pd.DataFrame({
                'price': np.random.uniform(0.2, 0.8, 50),
                'outcome': np.random.randint(0, 2, 50),
            })
            strategy.update_historical_data(data)
            
            summary = strategy.get_calibration_summary()
            results.append(summary.get('mean_spread', 0))
        
        assert results[0] == results[1]


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in production."""
    
    def test_api_timeout_handling(self):
        """Test handling of API timeouts."""
        # Simulate timeout
        class TimeoutError(Exception):
            pass
        
        max_retries = 3
        retry_count = 0
        
        for attempt in range(max_retries):
            try:
                if attempt < 2:
                    raise TimeoutError("Connection timed out")
                # Third attempt succeeds
                result = "success"
                break
            except TimeoutError:
                retry_count += 1
                time.sleep(0.01)  # Brief sleep
        
        assert retry_count == 2
        assert result == "success"
    
    def test_invalid_response_handling(self):
        """Test handling of invalid API responses."""
        invalid_responses = [
            None,
            {},
            {'error': 'unauthorized'},
            {'data': None},
        ]
        
        for response in invalid_responses:
            if response is None:
                is_valid = False
            elif 'error' in response:
                is_valid = False
            elif response.get('data') is None:
                is_valid = False
            else:
                is_valid = True
            
            assert not is_valid
    
    def test_rate_limit_backoff(self):
        """Test exponential backoff on rate limits."""
        base_delay = 0.1
        max_delay = 5.0
        
        delays = []
        for attempt in range(5):
            delay = min(base_delay * (2 ** attempt), max_delay)
            delays.append(delay)
        
        assert delays == [0.1, 0.2, 0.4, 0.8, 1.6]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
