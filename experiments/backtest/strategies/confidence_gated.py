"""
Confidence-gated abstention strategy.

This strategy implements the Kalai-Blackwell framework where:
- Forecaster abstains if confidence is below threshold
- Only trades when projection distance d(q, C_t) exceeds minimum

This provides theoretical guarantees on hallucination rate.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConfidenceGatedConfig:
    """Configuration for confidence-gated strategy."""
    
    # Minimum d(q, C_t) to trade (abstain if below)
    min_distance_threshold: float = 0.05
    
    # Maximum probability confidence to trade (abstain if too extreme)
    max_confidence_threshold: float = 0.95
    
    # Position sizing method
    sizing_method: str = "proportional"  # "fixed", "proportional", "kelly"
    
    # Base position size
    base_position: float = 1.0
    
    # Maximum position per trade
    max_position: float = 10.0
    
    # Minimum samples for valid C_t
    min_ct_samples: int = 32


@dataclass
class TradeDecision:
    """Result of a trade decision."""
    
    should_trade: bool
    abstention_reason: Optional[str] = None
    position_size: float = 0.0
    direction: Optional[np.ndarray] = None
    confidence_score: float = 0.0
    distance_to_ct: float = 0.0


class ConfidenceGatedStrategy:
    """
    A strategy that abstains when confidence is low.
    
    Implements the Kalai-Blackwell guarantee:
    - Hallucination rate bounded by abstention rate
    - Only acts when projection distance exceeds threshold
    """
    
    def __init__(self, config: ConfidenceGatedConfig):
        self.config = config
        self.trade_history: List[Dict] = []
        self.abstention_counts = {
            "low_distance": 0,
            "high_confidence": 0,
            "insufficient_samples": 0,
            "traded": 0,
        }
    
    def decide(
        self,
        q: np.ndarray,
        ct_samples: np.ndarray,
        project_fn: callable,
    ) -> TradeDecision:
        """
        Decide whether to trade based on confidence gating.
        
        Args:
            q: Current market prices (k,)
            ct_samples: Samples from C_t (n_samples, k)
            project_fn: Function to project q onto conv(ct_samples)
            
        Returns:
            TradeDecision with trade recommendation
        """
        # Check sample sufficiency
        n_samples = ct_samples.shape[0]
        if n_samples < self.config.min_ct_samples:
            self.abstention_counts["insufficient_samples"] += 1
            return TradeDecision(
                should_trade=False,
                abstention_reason=f"Insufficient C_t samples: {n_samples} < {self.config.min_ct_samples}",
                confidence_score=0.0,
            )
        
        # Compute projection
        proj_result = project_fn(x=q, samples=ct_samples)
        distance = proj_result["distance"]
        direction = proj_result["direction"]
        projected = proj_result["projected"]
        
        # Check distance threshold (low distance = don't trade)
        if distance < self.config.min_distance_threshold:
            self.abstention_counts["low_distance"] += 1
            return TradeDecision(
                should_trade=False,
                abstention_reason=f"Distance too low: {distance:.4f} < {self.config.min_distance_threshold}",
                distance_to_ct=distance,
                confidence_score=distance,
            )
        
        # Check confidence (extreme prices = don't trade)
        max_prob = np.max(np.abs(q - 0.5)) + 0.5  # Transform to [0.5, 1] scale
        if max_prob > self.config.max_confidence_threshold:
            self.abstention_counts["high_confidence"] += 1
            return TradeDecision(
                should_trade=False,
                abstention_reason=f"Price too extreme: {max_prob:.4f} > {self.config.max_confidence_threshold}",
                distance_to_ct=distance,
                confidence_score=1.0 - max_prob,
            )
        
        # Compute position size
        if self.config.sizing_method == "fixed":
            position_size = self.config.base_position
        elif self.config.sizing_method == "proportional":
            # Size proportional to distance
            position_size = self.config.base_position * distance
        elif self.config.sizing_method == "kelly":
            # Kelly sizing based on distance as edge estimate
            edge = distance  # Use distance as edge proxy
            odds = 1.0  # Even money for simplicity
            kelly_frac = edge / odds if odds > 0 else 0
            position_size = self.config.base_position * kelly_frac
        else:
            position_size = self.config.base_position
        
        # Clip position
        position_size = min(position_size, self.config.max_position)
        
        self.abstention_counts["traded"] += 1
        
        return TradeDecision(
            should_trade=True,
            position_size=position_size,
            direction=direction,
            confidence_score=distance,
            distance_to_ct=distance,
        )
    
    def get_abstention_rate(self) -> float:
        """Get the abstention rate (fraction of decisions that didn't trade)."""
        total = sum(self.abstention_counts.values())
        if total == 0:
            return 0.0
        return 1.0 - (self.abstention_counts["traded"] / total)
    
    def get_abstention_breakdown(self) -> Dict[str, float]:
        """Get breakdown of abstention reasons."""
        total = sum(self.abstention_counts.values())
        if total == 0:
            return {k: 0.0 for k in self.abstention_counts}
        return {k: v / total for k, v in self.abstention_counts.items()}
    
    def reset_counts(self):
        """Reset abstention counts."""
        for key in self.abstention_counts:
            self.abstention_counts[key] = 0


def compute_hallucination_bound(
    abstention_rate: float,
    base_error_rate: float = 0.5,
) -> float:
    """
    Compute the Kalai-Blackwell hallucination bound.
    
    The key insight: if we abstain at rate α, then our
    hallucination rate is bounded by α (assuming the base
    forecaster has error rate ≤ 0.5 when it chooses to act).
    
    Args:
        abstention_rate: Fraction of decisions where we abstained
        base_error_rate: Error rate of base forecaster (default 0.5)
        
    Returns:
        Upper bound on hallucination rate
    """
    # Theorem: If forecaster abstains at rate α and has error ≤ 0.5
    # when acting, then hallucination rate ≤ α
    return min(abstention_rate, base_error_rate)



