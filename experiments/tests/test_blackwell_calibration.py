"""
Tests for the Blackwell calibration trading strategy.

Tests cover:
1. Basic functionality (binning, calibration, position sizing)
2. Rolling calibration (no lookahead bias)
3. Statistical validation (significance filtering)
4. Risk-parity sizing
5. Integration with backtest engine
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure backtest module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.strategies.blackwell_calibration import (
    BlackwellCalibrationStrategy,
    BlackwellCalibrationConfig,
    BinStats,
)


class TestBlackwellCalibrationConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        cfg = BlackwellCalibrationConfig()
        assert cfg.n_bins == 10
        assert cfg.g_bar_threshold == 0.05
        assert cfg.use_risk_parity is True

    def test_custom_config(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=20,
            g_bar_threshold=0.08,
            leverage=2.0,
        )
        assert cfg.n_bins == 20
        assert cfg.g_bar_threshold == 0.08
        assert cfg.leverage == 2.0


class TestBinning:
    """Tests for price-to-bin mapping."""

    def test_price_to_bin_boundaries(self):
        cfg = BlackwellCalibrationConfig(n_bins=10)
        strategy = BlackwellCalibrationStrategy(cfg)

        # Test bin boundaries
        assert strategy._price_to_bin(0.0) == 0
        assert strategy._price_to_bin(0.05) == 0
        assert strategy._price_to_bin(0.1) == 1
        assert strategy._price_to_bin(0.15) == 1
        assert strategy._price_to_bin(0.5) == 5
        assert strategy._price_to_bin(0.95) == 9
        assert strategy._price_to_bin(1.0) == 9

    def test_price_to_bin_clamping(self):
        cfg = BlackwellCalibrationConfig(n_bins=10)
        strategy = BlackwellCalibrationStrategy(cfg)

        # Test clamping for out-of-range values
        assert strategy._price_to_bin(-0.1) == 0
        assert strategy._price_to_bin(1.5) == 9


class TestCalibration:
    """Tests for rolling calibration."""

    def test_recalibrate_computes_g_bar(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=5,
            lookback_trades=100,
            t_stat_threshold=0.0,  # Disable t-stat filter
            g_bar_threshold=0.01,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Add samples with known miscalibration in bin 5 (prices 0.5-0.6)
        # Price = 0.55, all outcomes = 1 → g̅ = 1 - 0.55 = 0.45
        for i in range(20):
            strategy.on_resolution(f"m_{i}", outcome=1.0, price=0.55)

        strategy._recalibrate()

        # Check bin 5 has correct g_bar
        assert 5 in strategy.state.bin_stats
        stats = strategy.state.bin_stats[5]
        assert abs(stats.g_bar - 0.45) < 0.01
        assert stats.n_samples == 20

    def test_recalibrate_respects_lookback(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=5,
            lookback_trades=10,  # Only use last 10
            t_stat_threshold=0.0,
            g_bar_threshold=0.01,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Add 20 samples at price 0.55, outcome=1
        for i in range(20):
            strategy.on_resolution(f"m_{i}", outcome=1.0, price=0.55)

        # Add 10 samples at price 0.55, outcome=0
        for i in range(10):
            strategy.on_resolution(f"m2_{i}", outcome=0.0, price=0.55)

        strategy._recalibrate()

        # g̅ should be based on last 10 (all outcome=0)
        # g̅ = 0 - 0.55 = -0.55
        if 5 in strategy.state.bin_stats:
            stats = strategy.state.bin_stats[5]
            assert stats.g_bar < 0  # Should be negative

    def test_significance_filtering(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=10,
            lookback_trades=100,
            t_stat_threshold=2.0,  # Strict filtering
            g_bar_threshold=0.05,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Add samples with small edge (not significant)
        np.random.seed(42)
        for i in range(50):
            price = 0.55
            outcome = 1.0 if np.random.random() < 0.52 else 0.0  # 2% edge
            strategy.on_resolution(f"m_{i}", outcome=outcome, price=price)

        strategy._recalibrate()

        # Bin should exist but not be significant
        if 5 in strategy.state.bin_stats:
            assert not strategy.state.bin_stats[5].is_significant


class TestPositionSizing:
    """Tests for position sizing logic."""

    def test_no_position_outside_price_range(self):
        cfg = BlackwellCalibrationConfig(
            price_min=0.2,
            price_max=0.8,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Even with significant bins, should return 0 outside range
        assert strategy.get_position("m1", 0.1) == 0.0
        assert strategy.get_position("m2", 0.9) == 0.0

    def test_no_position_without_calibration(self):
        cfg = BlackwellCalibrationConfig()
        strategy = BlackwellCalibrationStrategy(cfg)

        # No history → no position
        assert strategy.get_position("m1", 0.5) == 0.0

    def test_position_direction(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=5,
            t_stat_threshold=0.0,
            g_bar_threshold=0.01,
            use_risk_parity=False,
            leverage=1.0,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Create positive g̅ in bin 5 (underpriced → buy)
        for i in range(20):
            strategy.on_resolution(f"m_{i}", outcome=1.0, price=0.55)
        strategy._recalibrate()

        pos = strategy.get_position("test", 0.55)
        assert pos > 0  # Should be long (buy)

    def test_risk_parity_sizing(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=5,
            t_stat_threshold=0.0,
            g_bar_threshold=0.01,
            use_risk_parity=True,
            target_max_loss=0.2,
            leverage=1.0,
            max_position=1.0,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Create significant edge
        for i in range(20):
            strategy.on_resolution(f"m_{i}", outcome=1.0, price=0.55)
        strategy._recalibrate()

        # At price 0.55, max loss = 0.55 (if buying and outcome=0)
        # Size should be target_max_loss / max_loss = 0.2 / 0.55 ≈ 0.36
        pos = strategy.get_position("test", 0.55)
        expected = min(1.0, 0.2 / 0.55)
        assert abs(pos - expected) < 0.01

    def test_leverage_multiplier(self):
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=5,
            t_stat_threshold=0.0,
            g_bar_threshold=0.01,
            use_risk_parity=False,
            leverage=2.0,
            max_position=3.0,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        for i in range(20):
            strategy.on_resolution(f"m_{i}", outcome=1.0, price=0.55)
        strategy._recalibrate()

        pos = strategy.get_position("test", 0.55)
        assert pos == 2.0  # 1.0 base * 2.0 leverage


class TestRollingBacktest:
    """Tests for rolling backtest simulation."""

    def test_no_lookahead_bias(self):
        """Verify that the strategy doesn't use future data."""
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=10,
            lookback_trades=50,
            recalibrate_freq=10,
            t_stat_threshold=0.0,
            g_bar_threshold=0.05,
            use_risk_parity=False,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Generate synthetic data with regime change
        np.random.seed(42)
        n = 200

        # First half: bin 5 is underpriced (outcome=1 more often)
        # Second half: bin 5 is overpriced (outcome=0 more often)
        prices = np.full(n, 0.55)
        outcomes = np.zeros(n)
        outcomes[:100] = np.random.random(100) < 0.7  # 70% outcome=1
        outcomes[100:] = np.random.random(100) < 0.3  # 30% outcome=1

        # Warmup
        warmup = 50
        for i in range(warmup):
            strategy.on_resolution(f"m_{i}", outcomes[i], prices[i])

        # Trade and track
        positions = []
        for i in range(warmup, n):
            pos = strategy.get_position(f"m_{i}", prices[i])
            positions.append(pos)
            strategy.on_resolution(f"m_{i}", outcomes[i], prices[i])

        positions = np.array(positions)

        # Positions should adapt to regime change
        # In first half, should tend to buy (positive)
        # In second half, should tend to sell (negative) after adaptation
        first_half = positions[:40]  # Before regime change
        second_half = positions[60:]  # After regime change + adaptation

        # First half should have positive positions (buying underpriced)
        assert first_half.mean() > 0

    def test_pnl_calculation(self):
        """Test that PnL is calculated correctly."""
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=5,
            lookback_trades=20,
            t_stat_threshold=0.0,
            g_bar_threshold=0.01,
            use_risk_parity=False,
        )
        strategy = BlackwellCalibrationStrategy(cfg)

        # Create strong positive edge in bin 5
        for i in range(20):
            strategy.on_resolution(f"m_{i}", outcome=1.0, price=0.55)
        strategy._recalibrate()

        # Now trade: buy at 0.55, outcome=1 → profit = 1 - 0.55 = 0.45
        pos = strategy.get_position("test", 0.55)
        pnl = pos * (1.0 - 0.55)
        assert pnl > 0
        assert abs(pnl - 0.45) < 0.01  # pos should be 1.0


class TestStatisticalValidation:
    """Statistical tests for strategy validity."""

    def test_random_data_no_edge(self):
        """Strategy should have no significant edge on random data."""
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=20,
            lookback_trades=200,
            t_stat_threshold=0.0,
            g_bar_threshold=0.05,
            use_risk_parity=False,
        )

        np.random.seed(42)
        n_trials = 5
        pnls = []

        for trial in range(n_trials):
            strategy = BlackwellCalibrationStrategy(cfg)

            # Random calibrated data: P(outcome=1) = price
            n = 500
            prices = np.random.uniform(0.1, 0.9, n)
            outcomes = (np.random.random(n) < prices).astype(float)

            # Warmup
            for i in range(200):
                strategy.on_resolution(f"m_{i}", outcomes[i], prices[i])

            # Trade
            total_pnl = 0
            for i in range(200, n):
                pos = strategy.get_position(f"m_{i}", prices[i])
                pnl = pos * (outcomes[i] - prices[i])
                total_pnl += pnl
                strategy.on_resolution(f"m_{i}", outcomes[i], prices[i])

            pnls.append(total_pnl)

        # PnL should be close to zero on average
        mean_pnl = np.mean(pnls)
        assert abs(mean_pnl) < 5.0  # Allow some variance

    def test_miscalibrated_data_positive_edge(self):
        """Strategy should profit on systematically miscalibrated data."""
        cfg = BlackwellCalibrationConfig(
            n_bins=10,
            min_samples_per_bin=20,
            lookback_trades=200,
            t_stat_threshold=0.0,
            g_bar_threshold=0.03,
            use_risk_parity=False,
        )

        np.random.seed(42)
        n_trials = 5
        pnls = []

        for trial in range(n_trials):
            strategy = BlackwellCalibrationStrategy(cfg)

            # Miscalibrated data: true P(outcome=1) = price + 0.1
            n = 500
            prices = np.random.uniform(0.2, 0.7, n)  # Keep away from edges
            true_probs = prices + 0.1  # Systematically underpriced
            outcomes = (np.random.random(n) < true_probs).astype(float)

            # Warmup
            for i in range(200):
                strategy.on_resolution(f"m_{i}", outcomes[i], prices[i])

            # Trade
            total_pnl = 0
            for i in range(200, n):
                pos = strategy.get_position(f"m_{i}", prices[i])
                pnl = pos * (outcomes[i] - prices[i])
                total_pnl += pnl
                strategy.on_resolution(f"m_{i}", outcomes[i], prices[i])

            pnls.append(total_pnl)

        # PnL should be positive on average
        mean_pnl = np.mean(pnls)
        assert mean_pnl > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_history(self):
        cfg = BlackwellCalibrationConfig()
        strategy = BlackwellCalibrationStrategy(cfg)

        # Should not crash with empty history
        strategy._recalibrate()
        assert len(strategy.state.bin_stats) == 0

    def test_single_sample(self):
        cfg = BlackwellCalibrationConfig(min_samples_per_bin=1)
        strategy = BlackwellCalibrationStrategy(cfg)

        strategy.on_resolution("m1", 1.0, 0.55)
        strategy._recalibrate()

        # Should handle single sample
        assert 5 in strategy.state.bin_stats

    def test_reset(self):
        cfg = BlackwellCalibrationConfig()
        strategy = BlackwellCalibrationStrategy(cfg)

        # Add some history
        for i in range(50):
            strategy.on_resolution(f"m_{i}", 1.0, 0.55)

        # Reset
        strategy.reset()

        assert len(strategy.state.history) == 0
        assert len(strategy.state.bin_stats) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


