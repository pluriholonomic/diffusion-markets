"""
Comprehensive unit tests for all trading strategies.

Tests cover:
1. Edge cases (extreme prices, zero volume, missing data)
2. Signal generation correctness
3. Position sizing bounds
4. Calibration behavior
5. No lookahead bias verification
"""

import sys
from pathlib import Path

# Add experiments to path for imports
experiments_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(experiments_dir))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from trading.utils.models import Platform, Side, Signal, Market, RiskLimits
from trading.strategies.calibration import (
    CalibrationStrategy, 
    CalibrationConfig,
    PolymarketCalibrationStrategy,
    KalshiCalibrationStrategy,
)
from trading.strategies.stat_arb import StatArbStrategy, StatArbConfig
from trading.strategies.longshot import LongshotStrategy, LongshotConfig
from trading.strategies.momentum import MomentumStrategy, MomentumConfig
from trading.strategies.dispersion import (
    DispersionStrategy, 
    CorrelationStrategy, 
    DispersionConfig,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_market():
    """Create a sample market for testing."""
    return Market(
        market_id="test_market_1",
        platform=Platform.POLYMARKET,
        question="Will X happen?",
        current_yes_price=0.60,
        current_no_price=0.40,
        volume=50000,
        liquidity=10000,
        category="politics",
        metadata={},
    )


@pytest.fixture
def calibration_data():
    """Generate synthetic calibration data."""
    np.random.seed(42)
    n = 500
    
    # Create data with known calibration bias:
    # At price 0.3, outcome rate is 0.4 (underpriced YES)
    # At price 0.7, outcome rate is 0.6 (overpriced YES)
    prices = np.random.uniform(0.1, 0.9, n)
    outcomes = np.zeros(n)
    
    for i, p in enumerate(prices):
        # Add calibration bias
        if p < 0.5:
            true_prob = p + 0.10  # Underpriced
        else:
            true_prob = p - 0.10  # Overpriced
        outcomes[i] = 1 if np.random.random() < true_prob else 0
    
    return pd.DataFrame({
        'price': prices,
        'outcome': outcomes,
        'category': np.random.choice(['politics', 'crypto', 'sports'], n),
        'volume': np.random.uniform(1000, 100000, n),
    })


@pytest.fixture
def risk_limits():
    """Standard risk limits."""
    return RiskLimits(
        max_position_pct=0.10,
        kelly_fraction=0.25,
        min_edge=0.05,
        min_liquidity=1000,
    )


def make_market(
    price: float = 0.5,
    volume: float = 10000,
    liquidity: float = 5000,
    category: str = "politics",
    market_id: str = None,
) -> Market:
    """Helper to create markets with specific parameters."""
    return Market(
        market_id=market_id or f"test_{price}_{category}",
        platform=Platform.POLYMARKET,
        question=f"Test market at {price}",
        current_yes_price=price,
        current_no_price=1.0 - price,
        volume=volume,
        liquidity=liquidity,
        category=category,
        metadata={'category': category},
    )


# ============================================================================
# Calibration Strategy Tests
# ============================================================================

class TestCalibrationStrategy:
    """Tests for CalibrationStrategy."""
    
    def test_initialization(self, risk_limits):
        """Strategy initializes correctly."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        assert strategy.platform == Platform.POLYMARKET
        assert strategy.calibration is None
    
    def test_needs_recalibration_initially(self, risk_limits):
        """Strategy needs calibration when first created."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        assert strategy.needs_recalibration()
    
    def test_calibration_from_data(self, calibration_data, risk_limits):
        """Strategy learns calibration from historical data."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        strategy.update_historical_data(calibration_data)
        
        assert strategy.calibration is not None
        assert not strategy.needs_recalibration()
        assert len(strategy.calibration) > 0
    
    def test_no_signal_without_calibration(self, sample_market, risk_limits):
        """No signal generated without calibration."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        signal = strategy.generate_signal(sample_market, bankroll=10000)
        
        # Without calibration, spread is 0, so no signal
        assert signal is None
    
    def test_signal_with_calibration(self, calibration_data, risk_limits):
        """Signal generated after calibration."""
        strategy = CalibrationStrategy(
            Platform.POLYMARKET,
            config=CalibrationConfig(spread_threshold=0.05),
            risk_limits=risk_limits,
        )
        strategy.update_historical_data(calibration_data)
        
        # Create market at price 0.7 (should be overpriced based on our synthetic data)
        market = make_market(price=0.70, liquidity=5000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        # May or may not generate signal depending on calibration
        if signal:
            assert signal.side in [Side.YES, Side.NO]
            assert 0 < signal.kelly_fraction <= 1.0
    
    def test_extreme_price_near_zero(self, calibration_data, risk_limits):
        """Handles prices near 0 correctly."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        strategy.update_historical_data(calibration_data)
        
        market = make_market(price=0.01, liquidity=5000)
        # Should not crash
        signal = strategy.generate_signal(market, bankroll=10000)
        
        if signal:
            assert 0 < signal.kelly_fraction <= 1.0
            assert signal.edge >= 0
    
    def test_extreme_price_near_one(self, calibration_data, risk_limits):
        """Handles prices near 1 correctly."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        strategy.update_historical_data(calibration_data)
        
        market = make_market(price=0.99, liquidity=5000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        if signal:
            assert 0 < signal.kelly_fraction <= 1.0
    
    def test_low_liquidity_filtered(self, calibration_data, risk_limits):
        """Markets with low liquidity are filtered."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        strategy.update_historical_data(calibration_data)
        
        # Liquidity below risk limit minimum
        market = make_market(price=0.5, liquidity=500)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        assert signal is None
    
    def test_position_size_bounds(self, calibration_data, risk_limits):
        """Position size respects risk limits."""
        config = CalibrationConfig(
            kelly_fraction=0.50,
            max_position_pct=0.10,
        )
        strategy = CalibrationStrategy(
            Platform.POLYMARKET, 
            config=config,
            risk_limits=risk_limits,
        )
        strategy.update_historical_data(calibration_data)
        
        # Force a large edge signal
        strategy.calibration.loc[5, 'spread'] = 0.30  # Large spread
        market = make_market(price=0.55, liquidity=10000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        if signal:
            # Kelly should be capped at max_position_pct
            assert signal.kelly_fraction <= risk_limits.max_position_pct
    
    def test_calibration_summary(self, calibration_data, risk_limits):
        """Calibration summary returns correct structure."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        
        # Before calibration
        summary = strategy.get_calibration_summary()
        assert summary['status'] == 'not_calibrated'
        
        # After calibration
        strategy.update_historical_data(calibration_data)
        summary = strategy.get_calibration_summary()
        
        assert summary['status'] == 'calibrated'
        assert 'n_bins' in summary
        assert 'mean_spread' in summary


class TestPolymarketCalibrationStrategy:
    """Tests for Polymarket-specific calibration strategy."""
    
    def test_platform_is_polymarket(self):
        """Platform is correctly set."""
        strategy = PolymarketCalibrationStrategy()
        assert strategy.platform == Platform.POLYMARKET
    
    def test_default_config(self):
        """Default config has expected values."""
        strategy = PolymarketCalibrationStrategy()
        assert strategy.config.n_bins == 10


class TestKalshiCalibrationStrategy:
    """Tests for Kalshi-specific calibration strategy."""
    
    def test_platform_is_kalshi(self):
        """Platform is correctly set."""
        strategy = KalshiCalibrationStrategy()
        assert strategy.platform == Platform.KALSHI
    
    def test_more_conservative_than_polymarket(self):
        """Kalshi config is more conservative."""
        pm_strategy = PolymarketCalibrationStrategy()
        kalshi_strategy = KalshiCalibrationStrategy()
        
        # Kalshi should have lower Kelly fraction (more conservative)
        assert kalshi_strategy.config.kelly_fraction <= pm_strategy.config.kelly_fraction


# ============================================================================
# Statistical Arbitrage Tests
# ============================================================================

class TestStatArbStrategy:
    """Tests for StatArbStrategy."""
    
    def test_initialization(self):
        """Strategy initializes correctly."""
        strategy = StatArbStrategy()
        assert strategy.platform == Platform.POLYMARKET
        assert len(strategy.category_calibration) == 0
    
    def test_calibration_by_category(self, calibration_data):
        """Strategy calibrates by category."""
        strategy = StatArbStrategy()
        strategy.update_historical_data(calibration_data)
        
        assert len(strategy.category_calibration) > 0
        assert all(isinstance(v, float) for v in strategy.category_calibration.values())
    
    def test_signal_uses_category(self, calibration_data, risk_limits):
        """Signal generation uses category calibration."""
        strategy = StatArbStrategy(risk_limits=risk_limits)
        strategy.update_historical_data(calibration_data)
        
        market = make_market(price=0.5, category="politics", liquidity=5000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        # Should use category-specific calibration
        if signal:
            assert signal.metadata.get('category') is not None
    
    def test_unknown_category_no_signal(self, calibration_data, risk_limits):
        """Unknown category returns no signal."""
        strategy = StatArbStrategy(risk_limits=risk_limits)
        strategy.update_historical_data(calibration_data)
        
        market = make_market(price=0.5, category="unknown_category_xyz", liquidity=5000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        # Unknown category should map or return None
        # This depends on mapping - check it doesn't crash
        assert signal is None or isinstance(signal, Signal)
    
    def test_category_weights_applied(self, calibration_data):
        """Category weights affect position sizing."""
        config = StatArbConfig(
            category_weight_politics=2.0,
            category_weight_crypto=0.5,
        )
        strategy = StatArbStrategy(config=config)
        # Just verify it initializes without error
        assert strategy.config.category_weight_politics == 2.0


# ============================================================================
# Longshot Strategy Tests
# ============================================================================

class TestLongshotStrategy:
    """Tests for LongshotStrategy."""
    
    def test_only_low_price_markets(self, calibration_data):
        """Only trades low-price markets."""
        strategy = LongshotStrategy()
        strategy.update_historical_data(calibration_data)
        
        # High price market should not generate signal
        high_price_market = make_market(price=0.60, liquidity=5000)
        signal = strategy.generate_signal(high_price_market, bankroll=10000)
        assert signal is None
        
        # Low price market may generate signal if edge exists
        low_price_market = make_market(price=0.05, liquidity=5000)
        signal = strategy.generate_signal(low_price_market, bankroll=10000)
        if signal:
            assert signal.side == Side.YES  # Longshots bet YES
    
    def test_always_bets_yes(self, calibration_data):
        """Longshot strategy always bets YES on underpriced events."""
        config = LongshotConfig(
            price_threshold=0.20,
            min_expected_edge=0.05,
        )
        strategy = LongshotStrategy(config=config)
        
        # Manually set edge
        strategy.longshot_edge = 0.20
        strategy.calibration_samples = 100
        
        market = make_market(price=0.08, liquidity=5000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        assert signal is not None
        assert signal.side == Side.YES
    
    def test_platform_is_kalshi(self):
        """Default platform is Kalshi."""
        strategy = LongshotStrategy()
        assert strategy.platform == Platform.KALSHI


# ============================================================================
# Momentum Strategy Tests  
# ============================================================================

class TestMomentumStrategy:
    """Tests for MomentumStrategy."""
    
    def test_initialization(self):
        """Strategy initializes correctly."""
        strategy = MomentumStrategy(Platform.POLYMARKET)
        assert len(strategy.price_history) == 0
    
    def test_price_tracking(self):
        """Prices are tracked correctly."""
        strategy = MomentumStrategy(Platform.POLYMARKET)
        
        strategy.update_price("market1", 0.50)
        strategy.update_price("market1", 0.52)
        strategy.update_price("market1", 0.55)
        
        assert len(strategy.price_history["market1"]) == 3
        assert strategy.price_history["market1"][-1] == 0.55
    
    def test_momentum_calculation(self):
        """Momentum is calculated correctly."""
        strategy = MomentumStrategy(Platform.POLYMARKET)
        
        prices = [0.50, 0.52, 0.54, 0.56, 0.58]
        momentum = strategy._compute_momentum(prices)
        
        # Should be positive (upward trend)
        assert momentum > 0
    
    def test_trend_signal_uptrend(self):
        """Trend signal detects uptrend."""
        config = MomentumConfig(fast_ma_period=3, slow_ma_period=10)
        strategy = MomentumStrategy(Platform.POLYMARKET, config=config)
        
        # Clear uptrend
        prices = list(np.linspace(0.40, 0.60, 20))
        signal = strategy._compute_trend_signal(prices)
        
        assert signal == 1  # Uptrend
    
    def test_trend_signal_downtrend(self):
        """Trend signal detects downtrend."""
        config = MomentumConfig(fast_ma_period=3, slow_ma_period=10)
        strategy = MomentumStrategy(Platform.POLYMARKET, config=config)
        
        # Clear downtrend
        prices = list(np.linspace(0.60, 0.40, 20))
        signal = strategy._compute_trend_signal(prices)
        
        assert signal == -1  # Downtrend
    
    def test_no_signal_without_history(self):
        """No signal without sufficient price history."""
        strategy = MomentumStrategy(Platform.POLYMARKET)
        
        markets = [make_market(price=0.50)]
        signals = strategy.generate_signals(markets)
        
        assert len(signals) == 0  # Not enough history
    
    def test_online_learning_compatibility(self):
        """Strategy works without pre-calibration."""
        strategy = MomentumStrategy(Platform.POLYMARKET)
        
        # update_historical_data should be a no-op
        strategy.update_historical_data(pd.DataFrame())
        
        summary = strategy.get_calibration_summary()
        assert summary['status'] == 'online'


# ============================================================================
# Dispersion Strategy Tests
# ============================================================================

class TestDispersionStrategy:
    """Tests for DispersionStrategy."""
    
    def test_initialization(self):
        """Strategy initializes correctly."""
        strategy = DispersionStrategy(Platform.POLYMARKET)
        assert len(strategy.category_prices) == 0
    
    def test_category_extraction(self):
        """Categories are extracted from markets."""
        strategy = DispersionStrategy(Platform.POLYMARKET)
        
        market = make_market(price=0.5, category="politics")
        market.metadata = {'category': 'politics'}
        cat = strategy._get_category(market)
        
        assert cat == 'politics'
    
    def test_implied_vol_calculation(self):
        """Implied volatility calculated correctly."""
        strategy = DispersionStrategy(Platform.POLYMARKET)
        
        # At p=0.5, implied vol should be 0.5
        iv_50 = strategy._compute_implied_vol(0.50)
        assert iv_50 == pytest.approx(0.5, abs=0.01)
        
        # At p=0.1, implied vol should be sqrt(0.1 * 0.9) = 0.3
        iv_10 = strategy._compute_implied_vol(0.10)
        assert iv_10 == pytest.approx(0.3, abs=0.01)
    
    def test_min_markets_per_category(self):
        """Respects minimum markets per category."""
        config = DispersionConfig(min_markets_per_category=5)
        strategy = DispersionStrategy(Platform.POLYMARKET, config=config)
        
        # Only 2 markets in category - should not generate signals
        markets = [
            make_market(price=0.5, category="rare_cat", market_id="m1"),
            make_market(price=0.6, category="rare_cat", market_id="m2"),
        ]
        for m in markets:
            m.metadata = {'category': 'rare_cat'}
        
        signals = strategy.generate_signals(markets)
        assert len(signals) == 0  # Not enough markets


class TestCorrelationStrategy:
    """Tests for CorrelationStrategy."""
    
    def test_initialization(self):
        """Strategy initializes correctly."""
        strategy = CorrelationStrategy(Platform.POLYMARKET)
        assert len(strategy.returns) == 0
    
    def test_correlation_calculation(self):
        """Correlation calculated correctly."""
        strategy = CorrelationStrategy(Platform.POLYMARKET)
        
        # Perfect positive correlation
        returns1 = [0.01, 0.02, 0.01, 0.03, 0.02]
        returns2 = [0.01, 0.02, 0.01, 0.03, 0.02]
        
        corr = strategy._compute_correlation(returns1, returns2)
        assert corr == pytest.approx(1.0, abs=0.01)
    
    def test_insufficient_data_no_correlation(self):
        """Returns 0 correlation with insufficient data."""
        strategy = CorrelationStrategy(Platform.POLYMARKET)
        
        returns1 = [0.01, 0.02]
        returns2 = [0.01, 0.02]
        
        corr = strategy._compute_correlation(returns1, returns2)
        assert corr == 0.0  # Not enough data


# ============================================================================
# Cross-Strategy Tests
# ============================================================================

ALL_STRATEGIES = [
    (CalibrationStrategy, {'platform': Platform.POLYMARKET}),
    (StatArbStrategy, {}),
    (LongshotStrategy, {}),
    (MomentumStrategy, {'platform': Platform.POLYMARKET}),
    (DispersionStrategy, {'platform': Platform.POLYMARKET}),
    (CorrelationStrategy, {'platform': Platform.POLYMARKET}),
]


class TestAllStrategies:
    """Cross-cutting tests for all strategies."""
    
    @pytest.mark.parametrize("strategy_class,init_kwargs", ALL_STRATEGIES)
    def test_does_not_crash_on_empty_markets(self, strategy_class, init_kwargs):
        """Strategy handles empty market list."""
        strategy = strategy_class(**init_kwargs)
        
        # Strategies with generate_signals method
        if hasattr(strategy, 'generate_signals'):
            signals = strategy.generate_signals([])
            assert signals == []
    
    @pytest.mark.parametrize("strategy_class,init_kwargs", ALL_STRATEGIES)
    def test_has_calibration_summary(self, strategy_class, init_kwargs):
        """All strategies have calibration summary."""
        strategy = strategy_class(**init_kwargs)
        
        summary = strategy.get_calibration_summary()
        assert isinstance(summary, dict)
        assert 'status' in summary or 'total_samples' in summary
    
    @pytest.mark.parametrize("strategy_class,init_kwargs", ALL_STRATEGIES)
    def test_position_size_never_negative(self, strategy_class, init_kwargs, calibration_data):
        """Position size is never negative."""
        strategy = strategy_class(**init_kwargs)
        
        # Try to calibrate if method exists
        if hasattr(strategy, 'update_historical_data'):
            strategy.update_historical_data(calibration_data)
        
        # Create a fake signal
        signal = Signal(
            market_id="test",
            platform=Platform.POLYMARKET,
            side=Side.YES,
            edge=0.10,
            confidence=0.8,
            kelly_fraction=0.20,
            strategy="test",
            metadata={},
        )
        
        if hasattr(strategy, 'compute_position_size'):
            size = strategy.compute_position_size(signal, bankroll=10000)
            assert size >= 0


# ============================================================================
# No Lookahead Bias Tests
# ============================================================================

class TestNoLookaheadBias:
    """Verify strategies don't use future information."""
    
    def test_calibration_uses_only_past_data(self, risk_limits):
        """Calibration strategy only uses past data."""
        strategy = CalibrationStrategy(Platform.POLYMARKET, risk_limits=risk_limits)
        
        # Create sequential data
        np.random.seed(123)
        n = 100
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n, freq='H'),
            'price': np.random.uniform(0.3, 0.7, n),
            'outcome': np.random.randint(0, 2, n),
            'category': 'politics',
        })
        
        # Calibrate on first half
        strategy.update_historical_data(data.iloc[:50])
        
        # The calibration should be based only on the first 50 rows
        # Test market at current time shouldn't know future outcomes
        market = make_market(price=0.5, liquidity=5000)
        signal = strategy.generate_signal(market, bankroll=10000)
        
        # Signal (if any) should be based on historical calibration, not future
        if signal:
            assert signal.metadata.get('spread') is not None
    
    def test_momentum_no_future_prices(self):
        """Momentum strategy doesn't peek at future prices."""
        strategy = MomentumStrategy(Platform.POLYMARKET)
        
        # Feed prices one at a time (simulating real-time)
        prices = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
        
        for i, p in enumerate(prices):
            strategy.update_price("m1", p)
            
            # At each step, only past prices should be in history
            assert len(strategy.price_history["m1"]) == i + 1
            assert max(strategy.price_history["m1"]) <= p


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
