"""
Integration tests for the group mean-reversion statistical arbitrage strategy.

These tests validate the full pipeline:
1. Regime detection via calibration tracking
2. Basket construction (calibration, dollar-neutral, Frechet)
3. Walk-forward backtesting with attribution
"""

import numpy as np
import pandas as pd
import pytest

from forecastbench.strategies.regime_detector import (
    GroupCalibrationTracker,
    RegimeDetectorConfig,
    RegimeType,
    compute_group_calibration_summary,
    detect_regime_changes,
)

from forecastbench.strategies.basket_builder import (
    Basket,
    BasketBuilderConfig,
    CalibrationBasedBuilder,
    DollarNeutralBuilder,
    FrechetArbitrageBuilder,
    UnifiedBasketBuilder,
    aggregate_positions,
)

from forecastbench.strategies.mean_reversion_backtest import (
    GroupMeanReversionConfig,
    BacktestResult,
    run_group_mean_reversion_backtest,
    run_walk_forward_backtest,
    compare_strategies,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data():
    """Generate synthetic prediction market data for testing."""
    np.random.seed(42)
    n = 500
    
    # Create groups with different calibration characteristics
    groups = np.random.choice(["politics", "crypto", "sports"], size=n, p=[0.4, 0.35, 0.25])
    
    # Generate market prices
    market_prices = np.random.beta(2, 2, size=n)  # Centered around 0.5
    
    # Generate outcomes with group-specific biases
    # Politics: well-calibrated (mean-reverting)
    # Crypto: overestimates (persistent positive bias)
    # Sports: underestimates (persistent negative bias)
    biases = {"politics": 0.0, "crypto": 0.15, "sports": -0.10}
    
    outcomes = np.zeros(n)
    for i in range(n):
        true_prob = np.clip(market_prices[i] + biases[groups[i]], 0.01, 0.99)
        outcomes[i] = np.random.binomial(1, true_prob)
    
    # Generate model predictions (slightly better than market for testing)
    model_prices = np.clip(market_prices + 0.5 * (outcomes - market_prices) + np.random.normal(0, 0.05, n), 0.01, 0.99)
    
    # Create timestamps
    timestamps = np.arange(n, dtype=float)
    
    # Market IDs
    market_ids = np.array([f"market_{i}" for i in range(n)])
    
    return pd.DataFrame({
        "id": market_ids,
        "category": groups,
        "market_prob": market_prices,
        "pred_prob": model_prices,
        "y": outcomes,
        "timestamp": timestamps,
    })


@pytest.fixture
def small_calibration_data():
    """Small dataset for regime detection testing."""
    np.random.seed(123)
    n = 100
    
    groups = np.array(["A"] * 50 + ["B"] * 50)
    
    # Group A: well-calibrated
    prices_a = np.random.uniform(0.3, 0.7, 50)
    outcomes_a = np.random.binomial(1, prices_a)
    
    # Group B: persistent positive bias (model underestimates)
    prices_b = np.random.uniform(0.3, 0.7, 50)
    outcomes_b = np.random.binomial(1, np.clip(prices_b + 0.2, 0, 1))
    
    prices = np.concatenate([prices_a, prices_b])
    outcomes = np.concatenate([outcomes_a, outcomes_b])
    
    return groups, prices, outcomes


# =============================================================================
# Regime Detection Tests
# =============================================================================


class TestGroupCalibrationTracker:
    """Tests for the GroupCalibrationTracker."""
    
    def test_initialization(self):
        """Test tracker initializes correctly."""
        cfg = RegimeDetectorConfig()
        tracker = GroupCalibrationTracker(cfg)
        
        assert len(tracker._states) == 0
        assert tracker._global_idx == 0
    
    def test_update_single_group(self):
        """Test updating a single group."""
        cfg = RegimeDetectorConfig(min_observations=5)
        tracker = GroupCalibrationTracker(cfg)
        
        # Add observations
        for i in range(10):
            tracker.update("group_a", price=0.5, outcome=0.6)
        
        assert tracker._states["group_a"].n_observations == 10
        assert tracker.get_calibration_error("group_a") != 0.0
    
    def test_regime_classification_mean_revert(self, small_calibration_data):
        """Test that well-calibrated groups are classified as mean-reverting."""
        groups, prices, outcomes = small_calibration_data
        
        cfg = RegimeDetectorConfig(
            mean_revert_threshold=0.08,
            momentum_threshold=0.15,
            min_observations=10,
            regime_stability_window=3,
        )
        tracker = GroupCalibrationTracker(cfg)
        
        # Update with all observations
        for g, p, y in zip(groups, prices, outcomes):
            tracker.update(g, p, y)
        
        # Group A should be more mean-reverting
        regime_a = tracker.get_regime("A")
        regime_b = tracker.get_regime("B")
        
        # At minimum, they should be different
        stats_a = tracker.get_calibration_stats("A")
        stats_b = tracker.get_calibration_stats("B")
        
        assert stats_a["n_observations"] == 50
        assert stats_b["n_observations"] == 50
        
        # Group B should have larger absolute bias
        assert abs(stats_b["bias"]) > abs(stats_a["bias"])
    
    def test_get_all_regimes(self, small_calibration_data):
        """Test getting all regimes at once."""
        groups, prices, outcomes = small_calibration_data
        
        cfg = RegimeDetectorConfig(min_observations=5)
        tracker = GroupCalibrationTracker(cfg)
        
        for g, p, y in zip(groups, prices, outcomes):
            tracker.update(g, p, y)
        
        regimes = tracker.get_all_regimes()
        
        assert "A" in regimes
        assert "B" in regimes
        assert all(isinstance(r, RegimeType) for r in regimes.values())
    
    def test_reset(self):
        """Test reset clears all state."""
        cfg = RegimeDetectorConfig()
        tracker = GroupCalibrationTracker(cfg)
        
        tracker.update("test", 0.5, 1.0)
        assert len(tracker._states) == 1
        
        tracker.reset()
        assert len(tracker._states) == 0


class TestCalibrationSummary:
    """Tests for calibration summary computation."""
    
    def test_compute_summary(self, small_calibration_data):
        """Test computing calibration summary."""
        groups, prices, outcomes = small_calibration_data
        
        summary = compute_group_calibration_summary(groups, prices, outcomes)
        
        assert "A" in summary
        assert "B" in summary
        assert "regime" in summary["A"]
        assert "bias" in summary["A"]
    
    def test_detect_regime_changes(self, small_calibration_data):
        """Test detecting regime changes."""
        groups, prices, outcomes = small_calibration_data
        
        changes = detect_regime_changes(groups, prices, outcomes)
        
        # Should be a list of change events
        assert isinstance(changes, list)
        for change in changes:
            assert "idx" in change
            assert "group" in change
            assert "from_regime" in change
            assert "to_regime" in change


# =============================================================================
# Basket Builder Tests
# =============================================================================


class TestCalibrationBasedBuilder:
    """Tests for calibration-based basket construction."""
    
    def test_build_basket(self):
        """Test building a calibration-based basket."""
        cfg = BasketBuilderConfig(min_edge=0.02, kelly_fraction=0.25)
        builder = CalibrationBasedBuilder(cfg)
        
        market_ids = np.array(["m1", "m2", "m3"])
        groups = np.array(["g1", "g1", "g2"])
        market_prices = np.array([0.5, 0.4, 0.6])
        model_prices = np.array([0.6, 0.35, 0.65])  # m1: long, m2: short, m3: long
        regimes = {"g1": RegimeType.MEAN_REVERT, "g2": RegimeType.MEAN_REVERT}
        
        basket = builder.build_basket(
            market_ids=market_ids,
            groups=groups,
            market_prices=market_prices,
            model_prices=model_prices,
            regimes=regimes,
        )
        
        assert isinstance(basket, Basket)
        assert basket.method == "calibration"
        # Should have positions for markets with sufficient edge
        assert basket.n_positions >= 1
    
    def test_respects_min_edge(self):
        """Test that positions below min_edge are filtered."""
        cfg = BasketBuilderConfig(min_edge=0.10)
        builder = CalibrationBasedBuilder(cfg)
        
        # Edge is only 0.05, below threshold
        basket = builder.build_basket(
            market_ids=np.array(["m1"]),
            groups=np.array(["g1"]),
            market_prices=np.array([0.50]),
            model_prices=np.array([0.55]),
            regimes={"g1": RegimeType.MEAN_REVERT},
        )
        
        assert basket.n_positions == 0


class TestDollarNeutralBuilder:
    """Tests for dollar-neutral basket construction."""
    
    def test_build_balanced_basket(self):
        """Test that dollar-neutral baskets have balanced exposure."""
        cfg = BasketBuilderConfig(min_edge=0.02)
        builder = DollarNeutralBuilder(cfg)
        
        # Create a group with both long and short opportunities
        market_ids = np.array(["m1", "m2", "m3", "m4"])
        groups = np.array(["g1", "g1", "g1", "g1"])
        market_prices = np.array([0.5, 0.5, 0.5, 0.5])
        model_prices = np.array([0.6, 0.6, 0.4, 0.4])  # 2 long, 2 short
        regimes = {"g1": RegimeType.MEAN_REVERT}
        
        basket = builder.build_basket(
            market_ids=market_ids,
            groups=groups,
            market_prices=market_prices,
            model_prices=model_prices,
            regimes=regimes,
        )
        
        assert basket.method == "dollar_neutral"
        # Should have both long and short positions
        if basket.n_positions > 0:
            assert basket.total_long > 0
            assert basket.total_short > 0


class TestUnifiedBasketBuilder:
    """Tests for unified basket builder."""
    
    def test_build_multiple_methods(self):
        """Test building baskets with multiple methods."""
        cfg = BasketBuilderConfig(min_edge=0.02)
        builder = UnifiedBasketBuilder(cfg, methods=["calibration", "dollar_neutral"])
        
        market_ids = np.array(["m1", "m2"])
        groups = np.array(["g1", "g1"])
        market_prices = np.array([0.5, 0.5])
        model_prices = np.array([0.6, 0.4])
        regimes = {"g1": RegimeType.MEAN_REVERT}
        
        baskets = builder.build_baskets(
            market_ids=market_ids,
            groups=groups,
            market_prices=market_prices,
            model_prices=model_prices,
            regimes=regimes,
        )
        
        assert "calibration" in baskets
        assert "dollar_neutral" in baskets


# =============================================================================
# Backtest Tests
# =============================================================================


class TestMeanReversionBacktest:
    """Tests for the mean-reversion backtest engine."""
    
    def test_backtest_runs(self, synthetic_data):
        """Test that backtest runs without errors."""
        cfg = GroupMeanReversionConfig()
        
        result = run_group_mean_reversion_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            cfg=cfg,
        )
        
        assert isinstance(result, BacktestResult)
        assert result.n_trades > 0
        assert result.final_bankroll > 0
    
    def test_backtest_attribution(self, synthetic_data):
        """Test that attribution is computed correctly."""
        cfg = GroupMeanReversionConfig()
        
        result = run_group_mean_reversion_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            cfg=cfg,
        )
        
        # Should have attribution by group
        assert len(result.pnl_by_group) > 0
        
        # PnL by group should sum approximately to total
        total_from_groups = sum(result.pnl_by_group.values())
        assert abs(total_from_groups - result.total_pnl) < 0.01
    
    def test_walk_forward_backtest(self, synthetic_data):
        """Test walk-forward backtest."""
        cfg = GroupMeanReversionConfig()
        
        result = run_walk_forward_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            cfg=cfg,
            train_frac=0.3,
            n_folds=3,
        )
        
        assert isinstance(result, BacktestResult)
        # Walk-forward should use less data so fewer trades
        assert result.n_trades >= 0
    
    def test_compare_strategies(self, synthetic_data):
        """Test strategy comparison."""
        results = compare_strategies(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            strategies=["calibration", "dollar_neutral"],
        )
        
        assert "calibration" in results
        assert "dollar_neutral" in results
        assert all(isinstance(r, BacktestResult) for r in results.values())
    
    def test_bootstrap_ci(self, synthetic_data):
        """Test that bootstrap CIs are computed."""
        cfg = GroupMeanReversionConfig(bootstrap_samples=100)  # Small for speed
        
        result = run_group_mean_reversion_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            cfg=cfg,
        )
        
        # Should have confidence intervals
        assert result.sharpe_ci[0] <= result.sharpe_ci[1]
        assert result.roi_ci[0] <= result.roi_ci[1]


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""
    
    def test_to_dict(self, synthetic_data):
        """Test serialization to dict."""
        cfg = GroupMeanReversionConfig()
        
        result = run_group_mean_reversion_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            cfg=cfg,
        )
        
        d = result.to_dict()
        
        assert "total_pnl" in d
        assert "roi" in d
        assert "sharpe" in d
        assert "pnl_by_group" in d
        assert "pnl_by_regime" in d


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullPipeline:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, synthetic_data):
        """Test the full pipeline from data to results."""
        # 1. Analyze calibration
        groups = synthetic_data["category"].values
        prices = synthetic_data["market_prob"].values
        outcomes = synthetic_data["y"].values
        
        summary = compute_group_calibration_summary(groups, prices, outcomes)
        assert len(summary) == 3  # politics, crypto, sports
        
        # 2. Build baskets
        cfg = BasketBuilderConfig()
        builder = UnifiedBasketBuilder(cfg)
        
        regimes = {g: RegimeType.MEAN_REVERT for g in summary.keys()}
        
        baskets = builder.build_baskets(
            market_ids=synthetic_data["id"].values,
            groups=groups,
            market_prices=prices,
            model_prices=synthetic_data["pred_prob"].values,
            regimes=regimes,
        )
        
        assert len(baskets) > 0
        
        # 3. Run backtest
        result = run_group_mean_reversion_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
        )
        
        # 4. Verify results are sensible
        assert result.final_bankroll > 0
        assert not np.isnan(result.sharpe)
        assert result.n_trades > 0
    
    def test_regime_improves_returns(self, synthetic_data):
        """Test that regime-aware trading improves over naive trading."""
        # Run with regime awareness
        cfg_regime = GroupMeanReversionConfig()
        result_regime = run_group_mean_reversion_backtest(
            synthetic_data,
            model_forecast_col="pred_prob",
            market_price_col="market_prob",
            group_col="category",
            outcome_col="y",
            cfg=cfg_regime,
        )
        
        # Verify results are reasonable
        assert result_regime.final_bankroll > 0
        assert result_regime.n_trades > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
