"""
Correct Blackwell Approachability Implementation.

Based on Section 4.2 of the paper:
- g_t(i) := (Y_t - q_t) * h^i(X_t, q_t)  -- payoff vector
- C_ε := [-ε, ε]^M                        -- constraint set
- AppErr_T := d_∞(g̅_T, C_ε)              -- approachability error

No-arbitrage ⟺ g̅_T ∈ C_ε for all test functions h^i
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple
import numpy as np


# Type alias for test functions
# h(context, price) -> float
TestFunction = Callable[[Dict, float], float]


@dataclass
class TestFunctionFamily:
    """A family of test functions for Blackwell approachability."""
    
    functions: List[TestFunction] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.functions)
    
    def add(self, name: str, fn: TestFunction) -> None:
        """Add a test function to the family."""
        self.names.append(name)
        self.functions.append(fn)
    
    def evaluate(self, context: Dict, price: float) -> np.ndarray:
        """Evaluate all test functions on (context, price)."""
        return np.array([h(context, price) for h in self.functions])
    
    @classmethod
    def default_family(cls) -> "TestFunctionFamily":
        """Create a default test function family."""
        family = cls()
        
        # Unconditional (overall calibration)
        family.add("unconditional", lambda ctx, q: 1.0)
        
        # Price bins (binned calibration)
        family.add("bin_0.0-0.2", lambda ctx, q: 1.0 if q < 0.2 else 0.0)
        family.add("bin_0.2-0.4", lambda ctx, q: 1.0 if 0.2 <= q < 0.4 else 0.0)
        family.add("bin_0.4-0.6", lambda ctx, q: 1.0 if 0.4 <= q < 0.6 else 0.0)
        family.add("bin_0.6-0.8", lambda ctx, q: 1.0 if 0.6 <= q < 0.8 else 0.0)
        family.add("bin_0.8-1.0", lambda ctx, q: 1.0 if q >= 0.8 else 0.0)
        
        # Group indicators (if available in context)
        family.add("is_crypto", lambda ctx, q: float(ctx.get("is_crypto", 0)))
        family.add("is_sports", lambda ctx, q: float(ctx.get("is_sports", 0)))
        family.add("is_politics", lambda ctx, q: float(ctx.get("is_politics", 0)))
        
        # Confidence-weighted (model confidence if available)
        family.add("high_confidence", 
                   lambda ctx, q: 1.0 if ctx.get("model_confidence", 0) > 0.8 else 0.0)
        family.add("low_confidence",
                   lambda ctx, q: 1.0 if ctx.get("model_confidence", 0) < 0.3 else 0.0)
        
        return family


@dataclass
class ApproachabilityState:
    """Tracks cumulative payoff vectors for approachability."""
    
    n_tests: int
    epsilon: float = 0.05  # No-arbitrage tolerance
    
    # Cumulative payoff vector
    g_cumsum: np.ndarray = field(default=None)
    t: int = 0
    
    # History for analysis
    g_history: List[np.ndarray] = field(default_factory=list)
    violation_history: List[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        if self.g_cumsum is None:
            self.g_cumsum = np.zeros(self.n_tests)
    
    def update(self, outcome: float, price: float, h_values: np.ndarray) -> Dict:
        """
        Update with new observation.
        
        Args:
            outcome: Y_t ∈ {0, 1} (or [0,1] for soft outcomes)
            price: q_t ∈ [0, 1]
            h_values: h^i(X_t, q_t) for all test functions
        
        Returns:
            Dict with current approachability metrics
        """
        # Payoff vector: g_t(i) = (Y_t - q_t) * h^i(X_t, q_t)
        g_t = (outcome - price) * h_values
        
        self.g_cumsum += g_t
        self.t += 1
        
        # Track per-test counts for proper averaging
        if not hasattr(self, 'h_counts'):
            self.h_counts = np.zeros(self.n_tests)
        self.h_counts += (h_values != 0).astype(float)
        
        # CORRECT: Per-test average (only count when test is active)
        # g̅(i) = E[g_t(i) | h^i != 0] = sum(g_t(i)) / count(h^i != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            g_bar = np.where(self.h_counts > 0, self.g_cumsum / self.h_counts, 0)
        
        # Check violations
        violations = np.abs(g_bar) > self.epsilon
        
        # Store history
        self.g_history.append(g_t.copy())
        self.violation_history.append(violations.copy())
        
        # Approachability error: d_∞(g̅_T, C_ε) = max_i |g̅_i| - ε
        app_err = max(0, np.max(np.abs(g_bar)) - self.epsilon)
        
        return {
            "g_t": g_t,
            "g_bar": g_bar,
            "app_err": app_err,
            "violations": violations,
            "n_violations": violations.sum(),
            "worst_violation_idx": np.argmax(np.abs(g_bar)) if app_err > 0 else -1,
        }
    
    def get_metrics(self) -> Dict:
        """Get current approachability metrics."""
        if self.t == 0:
            return {"app_err": 0.0, "g_bar": np.zeros(self.n_tests)}
        
        g_bar = self.g_cumsum / self.t
        app_err = max(0, np.max(np.abs(g_bar)) - self.epsilon)
        violations = np.abs(g_bar) > self.epsilon
        
        return {
            "t": self.t,
            "g_bar": g_bar,
            "app_err": app_err,
            "violations": violations,
            "n_violations": violations.sum(),
        }
    
    def get_arbitrage_direction(self) -> Tuple[int, float, int]:
        """
        Get the direction to trade to reduce approachability error.
        
        Returns:
            (direction, magnitude, test_idx)
            - direction: +1 (buy/overweight) or -1 (sell/underweight)
            - magnitude: suggested trade size (0 if no arbitrage)
            - test_idx: which test function is violated most
        """
        if self.t == 0:
            return (0, 0.0, -1)
        
        g_bar = self.g_cumsum / self.t
        
        # Find the most violated constraint
        abs_violations = np.abs(g_bar) - self.epsilon
        most_violated_idx = np.argmax(abs_violations)
        most_violated_val = abs_violations[most_violated_idx]
        
        if most_violated_val <= 0:
            return (0, 0.0, -1)  # No arbitrage
        
        # Direction to project back:
        # If g_bar[i] > ε, we've been over-predicting on test i → need to reduce
        # If g_bar[i] < -ε, we've been under-predicting → need to increase
        direction = -1 if g_bar[most_violated_idx] > 0 else 1
        magnitude = abs(g_bar[most_violated_idx]) - self.epsilon
        
        return (direction, magnitude, most_violated_idx)


class BlackwellApproachabilityTracker:
    """
    Full Blackwell approachability tracker for backtesting.
    
    Usage:
        tracker = BlackwellApproachabilityTracker()
        
        for outcome, price, context in data:
            result = tracker.update(outcome, price, context)
            if result["n_violations"] > 0:
                # Arbitrage exists
                direction, magnitude, test_idx = tracker.get_trade_signal()
    """
    
    def __init__(
        self,
        test_family: Optional[TestFunctionFamily] = None,
        epsilon: float = 0.05,
    ):
        self.test_family = test_family or TestFunctionFamily.default_family()
        self.state = ApproachabilityState(
            n_tests=len(self.test_family),
            epsilon=epsilon,
        )
    
    def update(self, outcome: float, price: float, context: Dict) -> Dict:
        """Update with new observation."""
        h_values = self.test_family.evaluate(context, price)
        result = self.state.update(outcome, price, h_values)
        result["test_names"] = self.test_family.names
        return result
    
    def get_trade_signal(self) -> Tuple[int, float, str]:
        """
        Get trade signal based on current violations.
        
        Returns:
            (direction, magnitude, test_name)
        """
        direction, magnitude, idx = self.state.get_arbitrage_direction()
        test_name = self.test_family.names[idx] if idx >= 0 else ""
        return (direction, magnitude, test_name)
    
    def get_summary(self) -> Dict:
        """Get summary of approachability over the entire run."""
        metrics = self.state.get_metrics()
        
        # Per-test violation rates
        if self.state.t > 0:
            violation_rates = np.mean(self.state.violation_history, axis=0)
        else:
            violation_rates = np.zeros(len(self.test_family))
        
        return {
            "total_observations": self.state.t,
            "final_app_err": metrics["app_err"],
            "final_g_bar": metrics["g_bar"].tolist(),
            "final_n_violations": int(metrics["n_violations"]),
            "test_names": self.test_family.names,
            "violation_rates": violation_rates.tolist(),
        }


def run_approachability_test(
    outcomes: np.ndarray,
    prices: np.ndarray,
    contexts: List[Dict],
    epsilon: float = 0.05,
    test_family: Optional[TestFunctionFamily] = None,
) -> Dict:
    """
    Run full approachability test on a dataset.
    
    Args:
        outcomes: Array of Y_t ∈ {0, 1}
        prices: Array of q_t ∈ [0, 1]
        contexts: List of context dicts for each observation
        epsilon: No-arbitrage tolerance
        test_family: Optional custom test function family
    
    Returns:
        Dict with approachability results
    """
    tracker = BlackwellApproachabilityTracker(
        test_family=test_family,
        epsilon=epsilon,
    )
    
    app_err_history = []
    
    for outcome, price, context in zip(outcomes, prices, contexts):
        result = tracker.update(outcome, price, context)
        app_err_history.append(result["app_err"])
    
    summary = tracker.get_summary()
    summary["app_err_history"] = app_err_history
    
    # Compute approachability rate (should decay like 1/√T)
    if len(app_err_history) > 100:
        t_vals = np.arange(1, len(app_err_history) + 1)
        # Fit log(app_err) ~ -α log(t)
        log_t = np.log(t_vals[10:])  # Skip first few
        log_err = np.log(np.maximum(app_err_history[10:], 1e-10))
        
        # Linear regression
        valid = np.isfinite(log_err)
        if valid.sum() > 10:
            coef = np.polyfit(log_t[valid], log_err[valid], 1)
            decay_rate = -coef[0]  # Should be ~0.5 for 1/√T
            summary["decay_rate"] = float(decay_rate)
    
    return summary

