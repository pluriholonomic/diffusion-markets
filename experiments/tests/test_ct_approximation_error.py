"""
Tests for C_t approximation error analysis.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.metrics.ct_approximation_error import (
    CtApproximationAnalyzer,
    CtApproximationMetrics,
    CtPrediction,
    compute_ct_error_from_samples,
)


class TestCtPrediction:
    """Tests for CtPrediction dataclass."""
    
    def test_basic_prediction(self):
        samples = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        pred = CtPrediction(
            market_id="test",
            samples=samples,
            mean_pred=0.5,
            std_pred=np.std(samples),
        )
        
        assert pred.market_id == "test"
        assert pred.mean_pred == 0.5
        
    def test_confidence_interval(self):
        samples = np.linspace(0.2, 0.8, 100)
        pred = CtPrediction(
            market_id="test",
            samples=samples,
            mean_pred=0.5,
            std_pred=np.std(samples),
        )
        
        lo, hi = pred.confidence_interval
        assert 0.2 < lo < 0.25
        assert 0.75 < hi < 0.8


class TestCtApproximationAnalyzer:
    """Tests for the main analyzer class."""
    
    def test_empty_analyzer(self):
        analyzer = CtApproximationAnalyzer()
        metrics = analyzer.compute_metrics()
        
        assert metrics.n_markets == 0
        
    def test_single_prediction(self):
        analyzer = CtApproximationAnalyzer()
        
        samples = np.array([0.4, 0.5, 0.6])
        analyzer.add_prediction("m1", samples, outcome=1.0, market_price=0.5)
        
        metrics = analyzer.compute_metrics()
        assert metrics.n_markets == 1
        
    def test_perfect_model(self):
        """Model that exactly matches outcomes."""
        analyzer = CtApproximationAnalyzer()
        
        # Model predicts exactly the outcome
        for i in range(100):
            outcome = float(i < 50)  # 50 ones, 50 zeros
            samples = np.full(10, outcome)  # Perfect prediction
            market_price = 0.5
            analyzer.add_prediction(f"m_{i}", samples, outcome, market_price)
        
        metrics = analyzer.compute_metrics()
        
        # Perfect model should have zero calibration error
        assert abs(metrics.model_calibration_error) < 0.01
        # Perfect model should have zero Brier score
        assert metrics.brier_score < 0.01
        
    def test_biased_model(self):
        """Model with systematic bias."""
        analyzer = CtApproximationAnalyzer()
        
        np.random.seed(42)
        for i in range(100):
            outcome = float(i < 30)  # 30% positive
            # Model always predicts 0.5 (biased for this distribution)
            samples = np.random.normal(0.5, 0.05, 10)
            samples = np.clip(samples, 0, 1)
            analyzer.add_prediction(f"m_{i}", samples, outcome, market_price=0.3)
        
        metrics = analyzer.compute_metrics()
        
        # Biased model should have negative calibration error
        # (predicts too high when true rate is 30%)
        assert metrics.model_calibration_error < 0
        
    def test_model_improvement_calculation(self):
        """Test model improvement vs market baseline."""
        analyzer = CtApproximationAnalyzer()
        
        np.random.seed(42)
        for i in range(100):
            outcome = float(i < 70)  # 70% positive
            market_price = 0.5  # Market is wrong
            # Model is better (predicts 0.7)
            samples = np.random.normal(0.7, 0.05, 10)
            samples = np.clip(samples, 0, 1)
            analyzer.add_prediction(f"m_{i}", samples, outcome, market_price)
        
        metrics = analyzer.compute_metrics()
        
        # Model should be better than market
        assert metrics.model_error < metrics.market_error
        assert metrics.model_improvement > 0
        
    def test_calibration_by_bin(self):
        """Test bin-wise calibration analysis."""
        analyzer = CtApproximationAnalyzer(n_bins=10)
        
        np.random.seed(42)
        # Create data with different calibration in different bins
        for i in range(200):
            if i < 100:
                # Bin 2-3 (prices 0.2-0.4): well calibrated
                price = np.random.uniform(0.2, 0.4)
                outcome = float(np.random.random() < price)
            else:
                # Bin 7-8 (prices 0.7-0.9): miscalibrated
                price = np.random.uniform(0.7, 0.9)
                outcome = 0.0  # Always wrong
            
            samples = np.random.normal(price, 0.05, 10)
            samples = np.clip(samples, 0, 1)
            analyzer.add_prediction(f"m_{i}", samples, outcome, market_price=price)
        
        metrics = analyzer.compute_metrics()
        
        # Should have bin-wise calibration data
        assert len(metrics.model_calibration_by_bin) > 0
        
    def test_reset(self):
        analyzer = CtApproximationAnalyzer()
        
        analyzer.add_prediction("m1", np.array([0.5]), 1.0, 0.5)
        assert len(analyzer.predictions) == 1
        
        analyzer.reset()
        assert len(analyzer.predictions) == 0


class TestMispredictionAnalysis:
    """Tests for misprediction analysis."""
    
    def test_overconfident_detection(self):
        analyzer = CtApproximationAnalyzer()
        
        # Model is overconfident: predicts 0.9 but outcome is 0
        for i in range(50):
            samples = np.random.normal(0.9, 0.05, 10)
            samples = np.clip(samples, 0, 1)
            analyzer.add_prediction(f"m_{i}", samples, outcome=0.0, market_price=0.5)
        
        # Model is underconfident: predicts 0.1 but outcome is 1
        for i in range(50):
            samples = np.random.normal(0.1, 0.05, 10)
            samples = np.clip(samples, 0, 1)
            analyzer.add_prediction(f"m_{i+50}", samples, outcome=1.0, market_price=0.5)
        
        analysis = analyzer.get_misprediction_analysis()
        
        # All predictions should be overconfident (wrong direction)
        assert analysis["overconfident_rate"] == 1.0
        
    def test_large_error_detection(self):
        analyzer = CtApproximationAnalyzer()
        
        # Small errors
        for i in range(80):
            samples = np.array([0.45, 0.5, 0.55])
            analyzer.add_prediction(f"m_{i}", samples, outcome=0.5, market_price=0.5)
        
        # Large errors
        for i in range(20):
            samples = np.array([0.9, 0.95, 1.0])
            analyzer.add_prediction(f"m_{i+80}", samples, outcome=0.0, market_price=0.5)
        
        analysis = analyzer.get_misprediction_analysis()
        
        # 20% should have large errors
        assert 0.15 < analysis["large_error_rate"] < 0.25


class TestTradingSignalQuality:
    """Tests for trading signal analysis."""
    
    def test_no_signals_when_aligned(self):
        analyzer = CtApproximationAnalyzer()
        
        # Model agrees with market
        for i in range(100):
            price = np.random.uniform(0.3, 0.7)
            samples = np.random.normal(price, 0.02, 10)  # Close to market
            samples = np.clip(samples, 0, 1)
            outcome = float(np.random.random() < price)
            analyzer.add_prediction(f"m_{i}", samples, outcome, price)
        
        signals = analyzer.get_trading_signal_quality()
        
        # Few signals when model ≈ market
        total_signals = signals["n_buy_signals"] + signals["n_sell_signals"]
        assert total_signals < 50  # Most should have no signal
        
    def test_signals_when_divergent(self):
        analyzer = CtApproximationAnalyzer()
        
        # Model consistently above market
        for i in range(100):
            market_price = 0.3
            samples = np.random.normal(0.5, 0.05, 10)  # Model says 0.5
            samples = np.clip(samples, 0, 1)
            outcome = float(np.random.random() < 0.5)  # True rate is 0.5
            analyzer.add_prediction(f"m_{i}", samples, outcome, market_price)
        
        signals = analyzer.get_trading_signal_quality()
        
        # Should have many buy signals (model > market)
        assert signals["n_buy_signals"] > 80
        # Win rate should be good since model is correct
        assert signals["buy_signal_win_rate"] > 0.4


class TestConvenienceFunction:
    """Tests for compute_ct_error_from_samples."""
    
    def test_basic_usage(self):
        n = 100
        n_samples = 10
        
        samples = np.random.rand(n, n_samples)
        outcomes = (np.random.rand(n) < 0.5).astype(float)
        market_prices = np.random.rand(n)
        
        metrics = compute_ct_error_from_samples(
            samples, outcomes, market_prices, n_bins=10
        )
        
        assert metrics.n_markets == n
        assert metrics.n_samples_per_market == n_samples
        
    def test_perfect_calibration(self):
        n = 100
        n_samples = 10
        
        # Perfect model: mean prediction = outcome
        outcomes = np.array([0.0] * 50 + [1.0] * 50)
        samples = np.zeros((n, n_samples))
        samples[:50] = np.random.normal(0.1, 0.02, (50, n_samples))
        samples[50:] = np.random.normal(0.9, 0.02, (50, n_samples))
        samples = np.clip(samples, 0, 1)
        
        market_prices = np.full(n, 0.5)  # Market is wrong
        
        metrics = compute_ct_error_from_samples(
            samples, outcomes, market_prices, n_bins=10
        )
        
        # Model should beat market
        assert metrics.model_error < metrics.market_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


