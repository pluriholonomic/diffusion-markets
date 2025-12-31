"""
C_t Approximation Error Analysis.

Measures how well the diffusion model's learned C_t approximates the true
constraint set, separate from market calibration error.

Key metrics:
1. Model Calibration Error: E[Y - p_model | conditioning]
2. Coverage Error: How often is outcome outside model's predicted range?
3. Sharpness: Variance of model predictions (confidence)
4. Brier Score: Proper scoring rule for probability predictions
5. Model-Market Divergence: |p_model - q_market|
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CtApproximationMetrics:
    """Metrics for C_t approximation error."""
    
    # Sample counts
    n_markets: int = 0
    n_samples_per_market: int = 0
    
    # Model calibration (by bin)
    model_calibration_error: float = 0.0  # E[Y - p_model]
    model_calibration_by_bin: Dict[int, float] = field(default_factory=dict)
    
    # Coverage
    coverage_rate: float = 0.0  # Fraction where Y in [min, max] of samples
    
    # Sharpness
    mean_sample_std: float = 0.0  # Average std of samples per market
    
    # Proper scoring
    brier_score: float = 0.0  # Mean (p_model - Y)²
    log_loss: float = 0.0  # Mean cross-entropy
    
    # Model-Market divergence
    mean_model_market_diff: float = 0.0  # E[|p_model - q_market|]
    model_market_correlation: float = 0.0  # corr(p_model, q_market)
    
    # Attribution (decomposition)
    total_error: float = 0.0  # Overall prediction error
    market_error: float = 0.0  # Error from market prices
    model_error: float = 0.0  # Error from model predictions
    model_improvement: float = 0.0  # How much model reduces error vs market


@dataclass
class CtPrediction:
    """Model prediction for a single market."""
    market_id: str
    samples: np.ndarray  # (n_samples,) probability predictions
    mean_pred: float
    std_pred: float
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval."""
        return (np.percentile(self.samples, 2.5), np.percentile(self.samples, 97.5))


class CtApproximationAnalyzer:
    """
    Analyzes approximation error of the diffusion model's C_t.
    
    Usage:
        analyzer = CtApproximationAnalyzer(n_bins=10)
        
        # Add predictions and outcomes
        for market_id, samples, outcome, market_price in data:
            analyzer.add_prediction(market_id, samples, outcome, market_price)
        
        # Get metrics
        metrics = analyzer.compute_metrics()
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._bin_edges = np.linspace(0, 1, n_bins + 1)
        
        # Storage
        self.predictions: List[CtPrediction] = []
        self.outcomes: List[float] = []
        self.market_prices: List[float] = []
        
    def reset(self) -> None:
        """Clear all stored data."""
        self.predictions = []
        self.outcomes = []
        self.market_prices = []
        
    def add_prediction(
        self,
        market_id: str,
        samples: np.ndarray,
        outcome: float,
        market_price: float,
    ) -> None:
        """
        Add a model prediction for analysis.
        
        Args:
            market_id: Market identifier
            samples: Array of sampled probabilities from diffusion model
            outcome: Actual outcome (0 or 1)
            market_price: Market's quoted probability
        """
        samples = np.asarray(samples).flatten()
        
        pred = CtPrediction(
            market_id=market_id,
            samples=samples,
            mean_pred=float(np.mean(samples)),
            std_pred=float(np.std(samples)),
        )
        
        self.predictions.append(pred)
        self.outcomes.append(float(outcome))
        self.market_prices.append(float(market_price))
        
    def compute_metrics(self) -> CtApproximationMetrics:
        """Compute all approximation error metrics."""
        if not self.predictions:
            return CtApproximationMetrics()
        
        n = len(self.predictions)
        outcomes = np.array(self.outcomes)
        market_prices = np.array(self.market_prices)
        model_preds = np.array([p.mean_pred for p in self.predictions])
        pred_stds = np.array([p.std_pred for p in self.predictions])
        
        metrics = CtApproximationMetrics(
            n_markets=n,
            n_samples_per_market=len(self.predictions[0].samples) if self.predictions else 0,
        )
        
        # 1. Model calibration error
        metrics.model_calibration_error = float(np.mean(outcomes - model_preds))
        
        # Calibration by bin
        bins = np.digitize(model_preds, self._bin_edges) - 1
        bins = np.clip(bins, 0, self.n_bins - 1)
        
        for b in range(self.n_bins):
            mask = bins == b
            if mask.sum() >= 5:
                metrics.model_calibration_by_bin[b] = float(np.mean(outcomes[mask] - model_preds[mask]))
        
        # 2. Coverage (outcome within sample range)
        coverage_count = 0
        for pred, outcome in zip(self.predictions, outcomes):
            min_sample = np.min(pred.samples)
            max_sample = np.max(pred.samples)
            # For binary outcomes, check if outcome is "reachable"
            # A probability p can lead to Y=1 with prob p, Y=0 with prob 1-p
            # So any p ∈ [0,1] can lead to either outcome
            # For continuous: check if outcome ∈ [min, max]
            if min_sample <= outcome <= max_sample or outcome in [0, 1]:
                coverage_count += 1
        metrics.coverage_rate = coverage_count / n
        
        # 3. Sharpness (model confidence)
        metrics.mean_sample_std = float(np.mean(pred_stds))
        
        # 4. Brier score
        metrics.brier_score = float(np.mean((model_preds - outcomes) ** 2))
        
        # 5. Log loss (clamp predictions to avoid log(0))
        eps = 1e-7
        model_preds_clipped = np.clip(model_preds, eps, 1 - eps)
        metrics.log_loss = float(-np.mean(
            outcomes * np.log(model_preds_clipped) + 
            (1 - outcomes) * np.log(1 - model_preds_clipped)
        ))
        
        # 6. Model-Market divergence
        metrics.mean_model_market_diff = float(np.mean(np.abs(model_preds - market_prices)))
        if np.std(model_preds) > 0 and np.std(market_prices) > 0:
            metrics.model_market_correlation = float(np.corrcoef(model_preds, market_prices)[0, 1])
        
        # 7. Error attribution
        metrics.market_error = float(np.mean((market_prices - outcomes) ** 2))
        metrics.model_error = float(np.mean((model_preds - outcomes) ** 2))
        metrics.total_error = metrics.model_error  # When using model
        
        # Improvement: how much better is model vs market?
        if metrics.market_error > 0:
            metrics.model_improvement = (metrics.market_error - metrics.model_error) / metrics.market_error
        
        return metrics
    
    def get_misprediction_analysis(self) -> Dict:
        """
        Detailed analysis of when and how the model mispredicts.
        
        Returns breakdown of mispredictions by:
        - Direction (overconfident vs underconfident)
        - Magnitude
        - Market price bin
        """
        if not self.predictions:
            return {}
        
        outcomes = np.array(self.outcomes)
        market_prices = np.array(self.market_prices)
        model_preds = np.array([p.mean_pred for p in self.predictions])
        
        # Residuals
        model_residuals = outcomes - model_preds  # Positive = underestimated
        market_residuals = outcomes - market_prices
        
        # Categorize mispredictions
        overconfident = (model_preds > 0.5) & (outcomes == 0) | (model_preds < 0.5) & (outcomes == 1)
        underconfident = ~overconfident
        
        # By magnitude
        large_errors = np.abs(model_residuals) > 0.3
        
        analysis = {
            "n_total": len(outcomes),
            
            # Direction
            "n_overconfident": int(overconfident.sum()),
            "n_underconfident": int(underconfident.sum()),
            "overconfident_rate": float(overconfident.mean()),
            
            # Magnitude
            "mean_absolute_error": float(np.mean(np.abs(model_residuals))),
            "n_large_errors": int(large_errors.sum()),
            "large_error_rate": float(large_errors.mean()),
            
            # Comparison to market
            "model_beats_market_rate": float(np.mean(np.abs(model_residuals) < np.abs(market_residuals))),
            "mean_model_error": float(np.mean(model_residuals)),
            "mean_market_error": float(np.mean(market_residuals)),
            
            # By market price bin
            "by_market_bin": {},
        }
        
        # Analysis by market price bin
        bins = np.digitize(market_prices, self._bin_edges) - 1
        bins = np.clip(bins, 0, self.n_bins - 1)
        
        for b in range(self.n_bins):
            mask = bins == b
            if mask.sum() >= 5:
                analysis["by_market_bin"][b] = {
                    "n": int(mask.sum()),
                    "model_mae": float(np.mean(np.abs(model_residuals[mask]))),
                    "market_mae": float(np.mean(np.abs(market_residuals[mask]))),
                    "model_beats_market": float(np.mean(np.abs(model_residuals[mask]) < np.abs(market_residuals[mask]))),
                    "mean_model_pred": float(np.mean(model_preds[mask])),
                    "mean_market_price": float(np.mean(market_prices[mask])),
                    "mean_outcome": float(np.mean(outcomes[mask])),
                }
        
        return analysis
    
    def get_trading_signal_quality(self) -> Dict:
        """
        Analyze quality of trading signals from model vs market disagreement.
        
        When model and market disagree, who is right?
        """
        if not self.predictions:
            return {}
        
        outcomes = np.array(self.outcomes)
        market_prices = np.array(self.market_prices)
        model_preds = np.array([p.mean_pred for p in self.predictions])
        
        # Model says "buy" when model_pred > market_price (model thinks underpriced)
        # Model says "sell" when model_pred < market_price (model thinks overpriced)
        
        divergence = model_preds - market_prices
        
        # Threshold for signal
        signal_threshold = 0.05
        
        buy_signal = divergence > signal_threshold
        sell_signal = divergence < -signal_threshold
        no_signal = ~buy_signal & ~sell_signal
        
        # PnL if we follow model's signal
        # Buy: profit if outcome > market_price
        # Sell: profit if outcome < market_price
        
        buy_pnl = outcomes[buy_signal] - market_prices[buy_signal] if buy_signal.any() else np.array([])
        sell_pnl = market_prices[sell_signal] - outcomes[sell_signal] if sell_signal.any() else np.array([])
        
        return {
            "signal_threshold": signal_threshold,
            "n_buy_signals": int(buy_signal.sum()),
            "n_sell_signals": int(sell_signal.sum()),
            "n_no_signal": int(no_signal.sum()),
            
            "buy_signal_win_rate": float(np.mean(buy_pnl > 0)) if len(buy_pnl) > 0 else 0,
            "sell_signal_win_rate": float(np.mean(sell_pnl > 0)) if len(sell_pnl) > 0 else 0,
            
            "buy_signal_mean_pnl": float(np.mean(buy_pnl)) if len(buy_pnl) > 0 else 0,
            "sell_signal_mean_pnl": float(np.mean(sell_pnl)) if len(sell_pnl) > 0 else 0,
            
            "total_signal_pnl": float(np.sum(buy_pnl) + np.sum(sell_pnl)),
            "mean_divergence_when_signal": float(np.mean(np.abs(divergence[buy_signal | sell_signal]))) if (buy_signal | sell_signal).any() else 0,
        }


def compute_ct_error_from_samples(
    samples: np.ndarray,
    outcomes: np.ndarray,
    market_prices: np.ndarray,
    n_bins: int = 10,
) -> CtApproximationMetrics:
    """
    Convenience function to compute C_t approximation error.
    
    Args:
        samples: (n_markets, n_samples) array of model predictions
        outcomes: (n_markets,) array of actual outcomes
        market_prices: (n_markets,) array of market prices
        n_bins: Number of bins for calibration analysis
        
    Returns:
        CtApproximationMetrics
    """
    analyzer = CtApproximationAnalyzer(n_bins=n_bins)
    
    for i in range(len(outcomes)):
        analyzer.add_prediction(
            market_id=f"market_{i}",
            samples=samples[i] if samples.ndim > 1 else samples,
            outcome=outcomes[i],
            market_price=market_prices[i],
        )
    
    return analyzer.compute_metrics()


