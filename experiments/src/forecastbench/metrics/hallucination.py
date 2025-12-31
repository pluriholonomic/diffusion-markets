"""
Hallucination and calibration metrics based on Kalai framework.

Implements metrics for:
- Hallucination rate (wrong predictions with high confidence)
- Abstention-aware calibration
- Distance-based confidence scoring
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HallucinationMetrics:
    """Container for hallucination metrics."""
    
    # Core metrics
    hallucination_rate: float
    abstention_rate: float
    error_rate_when_acting: float
    
    # Confidence-stratified
    high_conf_error_rate: float
    low_conf_error_rate: float
    
    # Theoretical bounds
    kalai_bound: float
    
    # Sample sizes
    n_total: int
    n_acted: int
    n_abstained: int


def compute_hallucination_rate(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    confidences: np.ndarray,
    high_conf_threshold: float = 0.75,
) -> float:
    """
    Compute hallucination rate: fraction of high-confidence wrong predictions.
    
    Args:
        predictions: Predicted probabilities (n,)
        outcomes: Actual outcomes 0/1 (n,)
        confidences: Confidence scores (n,)
        high_conf_threshold: Threshold for "high confidence"
        
    Returns:
        Hallucination rate
    """
    high_conf_mask = confidences >= high_conf_threshold
    
    if high_conf_mask.sum() == 0:
        return 0.0
    
    # Determine prediction correctness
    # A prediction is "correct" if it agrees with outcome direction
    pred_yes = predictions > 0.5
    outcome_yes = outcomes > 0.5
    correct = pred_yes == outcome_yes
    
    # Hallucination = high confidence AND wrong
    hallucinations = high_conf_mask & ~correct
    
    return hallucinations.sum() / high_conf_mask.sum()


def compute_abstention_aware_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    distances: np.ndarray,
    abstention_threshold: float = 0.05,
    high_conf_threshold: float = 0.75,
) -> HallucinationMetrics:
    """
    Compute comprehensive hallucination and abstention metrics.
    
    Uses projection distance d(q, C_t) as confidence measure.
    Higher distance = more confident (market is mispriced).
    
    Args:
        predictions: Model predictions (n,)
        outcomes: Actual outcomes (n,)
        distances: Projection distances to C_t (n,)
        abstention_threshold: Minimum distance to act
        high_conf_threshold: Distance threshold for "high confidence"
        
    Returns:
        HallucinationMetrics dataclass
    """
    n_total = len(predictions)
    
    # Determine which predictions we "act on" vs abstain
    acted_mask = distances >= abstention_threshold
    n_acted = acted_mask.sum()
    n_abstained = n_total - n_acted
    
    # Compute correctness
    pred_yes = predictions > 0.5
    outcome_yes = outcomes > 0.5
    correct = pred_yes == outcome_yes
    
    # Error rate when acting
    if n_acted > 0:
        error_rate_when_acting = (~correct[acted_mask]).sum() / n_acted
    else:
        error_rate_when_acting = 0.0
    
    # Abstention rate
    abstention_rate = n_abstained / n_total if n_total > 0 else 0.0
    
    # High/low confidence error rates
    high_conf_mask = distances >= high_conf_threshold
    low_conf_mask = (distances >= abstention_threshold) & (distances < high_conf_threshold)
    
    if high_conf_mask.sum() > 0:
        high_conf_error_rate = (~correct[high_conf_mask]).sum() / high_conf_mask.sum()
    else:
        high_conf_error_rate = 0.0
    
    if low_conf_mask.sum() > 0:
        low_conf_error_rate = (~correct[low_conf_mask]).sum() / low_conf_mask.sum()
    else:
        low_conf_error_rate = 0.0
    
    # Hallucination rate: high confidence errors
    hallucination_rate = compute_hallucination_rate(
        predictions, outcomes, distances, high_conf_threshold
    )
    
    # Kalai bound: hallucination ≤ abstention_rate if error_rate ≤ 0.5
    kalai_bound = min(abstention_rate, 0.5)
    
    return HallucinationMetrics(
        hallucination_rate=hallucination_rate,
        abstention_rate=abstention_rate,
        error_rate_when_acting=error_rate_when_acting,
        high_conf_error_rate=high_conf_error_rate,
        low_conf_error_rate=low_conf_error_rate,
        kalai_bound=kalai_bound,
        n_total=n_total,
        n_acted=n_acted,
        n_abstained=n_abstained,
    )


def validate_kalai_bound(
    metrics: HallucinationMetrics,
) -> Dict[str, bool]:
    """
    Validate whether the Kalai bound holds.
    
    The key theorem states that if:
    - Forecaster abstains when confidence is low
    - Error rate when acting is ≤ 0.5
    
    Then: hallucination_rate ≤ abstention_rate
    
    Args:
        metrics: Computed hallucination metrics
        
    Returns:
        Dictionary with validation results
    """
    return {
        "bound_holds": metrics.hallucination_rate <= metrics.kalai_bound,
        "error_rate_acceptable": metrics.error_rate_when_acting <= 0.5,
        "theoretical_valid": (
            metrics.error_rate_when_acting <= 0.5 and 
            metrics.hallucination_rate <= metrics.abstention_rate
        ),
    }


class HallucinationTracker:
    """
    Online tracker for hallucination metrics during backtesting.
    """
    
    def __init__(
        self,
        abstention_threshold: float = 0.05,
        high_conf_threshold: float = 0.75,
    ):
        self.abstention_threshold = abstention_threshold
        self.high_conf_threshold = high_conf_threshold
        
        self.predictions: List[float] = []
        self.outcomes: List[float] = []
        self.distances: List[float] = []
    
    def update(
        self,
        prediction: float,
        outcome: float,
        distance: float,
    ):
        """Add a single observation."""
        self.predictions.append(prediction)
        self.outcomes.append(outcome)
        self.distances.append(distance)
    
    def update_batch(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        distances: np.ndarray,
    ):
        """Add a batch of observations."""
        self.predictions.extend(predictions.tolist())
        self.outcomes.extend(outcomes.tolist())
        self.distances.extend(distances.tolist())
    
    def compute_metrics(self) -> HallucinationMetrics:
        """Compute current metrics."""
        return compute_abstention_aware_metrics(
            np.array(self.predictions),
            np.array(self.outcomes),
            np.array(self.distances),
            self.abstention_threshold,
            self.high_conf_threshold,
        )
    
    def validate(self) -> Dict[str, bool]:
        """Validate Kalai bounds on current data."""
        metrics = self.compute_metrics()
        return validate_kalai_bound(metrics)
    
    def reset(self):
        """Clear all observations."""
        self.predictions.clear()
        self.outcomes.clear()
        self.distances.clear()


def compute_calibration_by_distance(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    distances: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute calibration curve stratified by distance.
    
    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        distances: Projection distances
        n_bins: Number of distance bins
        
    Returns:
        Dictionary with bin edges, mean predictions, mean outcomes, calibration errors
    """
    # Bin by distance
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(distances, percentiles)
    bin_edges[-1] += 1e-6  # Include max
    
    bin_preds = []
    bin_outcomes = []
    bin_errors = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_pred = predictions[mask].mean()
            mean_outcome = outcomes[mask].mean()
            bin_preds.append(mean_pred)
            bin_outcomes.append(mean_outcome)
            bin_errors.append(abs(mean_pred - mean_outcome))
            bin_counts.append(mask.sum())
        else:
            bin_preds.append(np.nan)
            bin_outcomes.append(np.nan)
            bin_errors.append(np.nan)
            bin_counts.append(0)
    
    return {
        "bin_edges": bin_edges,
        "mean_predictions": np.array(bin_preds),
        "mean_outcomes": np.array(bin_outcomes),
        "calibration_errors": np.array(bin_errors),
        "counts": np.array(bin_counts),
    }



