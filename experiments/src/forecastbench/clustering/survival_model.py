"""
Survival Models for Market Lifetime Estimation.

These models estimate P(market survives > t | features) which is used to
weight correlation estimates in survival-weighted clustering algorithms.

Key insight: Markets near resolution provide less reliable correlation
information because:
1. They will soon disappear, so learned correlations can't be exploited
2. Price dynamics near 0 or 1 are dominated by resolution mechanics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class SurvivalObservation:
    """
    A single survival observation for training.
    
    Attributes:
        market_id: Unique identifier
        duration: Time from birth to death (or censoring)
        event: Whether death was observed (True) or censored (False)
        features: Covariates for the survival model
    """
    market_id: str
    duration: float
    event: bool  # True if resolved, False if censored
    features: Dict[str, float] = field(default_factory=dict)


class SurvivalModel(ABC):
    """
    Abstract base class for survival models.
    
    Survival models estimate the probability that a market will survive
    (not resolve) beyond a given time horizon.
    """
    
    @abstractmethod
    def fit(self, observations: List[SurvivalObservation]) -> None:
        """Fit the survival model to historical data."""
        pass
    
    @abstractmethod
    def survival_probability(
        self,
        time_since_birth: float,
        horizon: float,
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimate probability of survival beyond horizon.
        
        Args:
            time_since_birth: Current age of the market
            horizon: Time horizon to survive beyond
            features: Optional covariates
            
        Returns:
            P(T > time_since_birth + horizon | T > time_since_birth, features)
        """
        pass
    
    def survival_weight(
        self,
        time_since_birth: float,
        horizon: float = 1.0,
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute weight for correlation estimation.
        
        Markets with higher survival probability get higher weight.
        """
        return self.survival_probability(time_since_birth, horizon, features)
    
    def joint_survival_weight(
        self,
        market1_age: float,
        market2_age: float,
        horizon: float = 1.0,
        features1: Optional[Dict[str, float]] = None,
        features2: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute joint survival weight for a pair of markets.
        
        Assumes independence of survival times (can be relaxed in subclasses).
        """
        s1 = self.survival_probability(market1_age, horizon, features1)
        s2 = self.survival_probability(market2_age, horizon, features2)
        return s1 * s2


class ExponentialSurvival(SurvivalModel):
    """
    Simple exponential survival model with constant hazard rate.
    
    S(t) = exp(-lambda * t)
    
    Optionally supports category-specific hazard rates.
    """
    
    def __init__(
        self,
        default_hazard: float = 0.01,
        category_hazards: Optional[Dict[str, float]] = None,
    ):
        self.default_hazard = default_hazard
        self.category_hazards = category_hazards or {}
        self._fitted = False
    
    def fit(self, observations: List[SurvivalObservation]) -> None:
        """
        Fit hazard rates using maximum likelihood.
        
        For exponential distribution: lambda_hat = n_events / sum(durations)
        """
        if not observations:
            return
        
        # Group by category
        category_data: Dict[str, Tuple[int, float]] = {}
        total_events = 0
        total_duration = 0.0
        
        for obs in observations:
            category = obs.features.get("category", "default")
            events, duration = category_data.get(category, (0, 0.0))
            
            if obs.event:
                events += 1
                total_events += 1
            
            duration += obs.duration
            total_duration += obs.duration
            category_data[category] = (events, duration)
        
        # Estimate hazard rates
        if total_duration > 0:
            self.default_hazard = max(total_events / total_duration, 1e-6)
        
        for category, (events, duration) in category_data.items():
            if duration > 0 and events > 0:
                self.category_hazards[category] = events / duration
        
        self._fitted = True
    
    def _get_hazard(self, features: Optional[Dict[str, float]] = None) -> float:
        """Get hazard rate for given features."""
        if features is None:
            return self.default_hazard
        
        category = features.get("category", "default")
        return self.category_hazards.get(category, self.default_hazard)
    
    def survival_probability(
        self,
        time_since_birth: float,
        horizon: float,
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        For exponential: P(T > t + h | T > t) = exp(-lambda * h)
        
        The memoryless property means survival doesn't depend on current age.
        """
        hazard = self._get_hazard(features)
        return np.exp(-hazard * horizon)


class KaplanMeierSurvival(SurvivalModel):
    """
    Non-parametric Kaplan-Meier survival estimator.
    
    Estimates S(t) empirically from observed death times,
    properly handling censored observations.
    """
    
    def __init__(self, time_grid_resolution: float = 1.0):
        self.time_grid_resolution = time_grid_resolution
        self.survival_curve: np.ndarray = np.array([1.0])
        self.time_grid: np.ndarray = np.array([0.0])
        self._fitted = False
    
    def fit(self, observations: List[SurvivalObservation]) -> None:
        """
        Fit Kaplan-Meier survival curve.
        
        S(t) = prod_{t_i <= t} (1 - d_i / n_i)
        
        where d_i is number of deaths at time t_i and n_i is number at risk.
        """
        if not observations:
            return
        
        # Collect event times and censoring indicators
        times = np.array([obs.duration for obs in observations])
        events = np.array([obs.event for obs in observations])
        
        # Sort by time
        order = np.argsort(times)
        times = times[order]
        events = events[order]
        
        # Build Kaplan-Meier curve
        unique_times = np.unique(times[events])  # Only event times matter
        
        if len(unique_times) == 0:
            # No events observed
            self.time_grid = np.array([0.0, times.max()])
            self.survival_curve = np.array([1.0, 1.0])
            self._fitted = True
            return
        
        n = len(times)
        survival = 1.0
        survival_values = [1.0]
        time_values = [0.0]
        
        at_risk = n
        for t in unique_times:
            # Number of events at time t
            d = np.sum((times == t) & events)
            # Number censored before t (no longer at risk)
            censored = np.sum((times < t) & ~events)
            at_risk = n - np.sum(times < t)
            
            if at_risk > 0:
                survival *= (1 - d / at_risk)
            
            time_values.append(t)
            survival_values.append(survival)
        
        # Extend to max observed time
        max_time = times.max()
        if time_values[-1] < max_time:
            time_values.append(max_time)
            survival_values.append(survival)
        
        self.time_grid = np.array(time_values)
        self.survival_curve = np.array(survival_values)
        self._fitted = True
    
    def _get_survival_at_time(self, t: float) -> float:
        """Get S(t) from the fitted curve."""
        if not self._fitted or len(self.time_grid) == 0:
            return 1.0
        
        if t <= 0:
            return 1.0
        if t >= self.time_grid[-1]:
            return self.survival_curve[-1]
        
        # Find the interval containing t
        idx = np.searchsorted(self.time_grid, t, side='right') - 1
        return self.survival_curve[max(0, idx)]
    
    def survival_probability(
        self,
        time_since_birth: float,
        horizon: float,
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Conditional survival: P(T > t + h | T > t) = S(t + h) / S(t)
        """
        s_current = self._get_survival_at_time(time_since_birth)
        s_future = self._get_survival_at_time(time_since_birth + horizon)
        
        if s_current <= 0:
            return 0.0
        
        return min(s_future / s_current, 1.0)


class CoxSurvival(SurvivalModel):
    """
    Cox Proportional Hazards survival model.
    
    h(t | x) = h_0(t) * exp(beta' * x)
    
    Allows survival to depend on market features like:
    - Price proximity to 0 or 1 (high certainty -> near resolution)
    - Category
    - Volume/liquidity
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        regularization: float = 0.01,
    ):
        self.feature_names = feature_names or ["price_extremity", "category_encoded"]
        self.regularization = regularization
        self.coefficients: np.ndarray = np.zeros(len(self.feature_names))
        self.baseline_survival: KaplanMeierSurvival = KaplanMeierSurvival()
        self._fitted = False
    
    def _extract_features(self, features: Optional[Dict[str, float]]) -> np.ndarray:
        """Extract feature vector from feature dict."""
        if features is None:
            return np.zeros(len(self.feature_names))
        
        x = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if name == "price_extremity":
                # Measure how close price is to 0 or 1
                price = features.get("last_price", 0.5)
                x[i] = 2 * abs(price - 0.5)  # 0 at p=0.5, 1 at p=0 or p=1
            elif name in features:
                x[i] = features[name]
        
        return x
    
    def fit(self, observations: List[SurvivalObservation]) -> None:
        """
        Fit Cox model using partial likelihood.
        
        For simplicity, we use coordinate descent with a fixed number of iterations.
        In production, use lifelines or scikit-survival.
        """
        if not observations:
            return
        
        n = len(observations)
        p = len(self.feature_names)
        
        # Extract data
        X = np.array([self._extract_features(obs.features) for obs in observations])
        times = np.array([obs.duration for obs in observations])
        events = np.array([obs.event for obs in observations], dtype=np.float64)
        
        # Sort by time (descending for risk set computation)
        order = np.argsort(-times)
        X = X[order]
        times = times[order]
        events = events[order]
        
        # Initialize coefficients
        beta = np.zeros(p)
        
        # Simple gradient descent on partial log-likelihood
        lr = 0.01
        for iteration in range(100):
            # Compute risk set weights
            eta = X @ beta
            exp_eta = np.exp(eta - eta.max())  # Numerically stable
            
            # Cumulative sums for risk sets
            risk_sum = np.cumsum(exp_eta)
            
            # Gradient of partial log-likelihood
            grad = np.zeros(p)
            for i in range(n):
                if events[i]:
                    grad += X[i] - np.sum(X[:i+1] * exp_eta[:i+1, None], axis=0) / risk_sum[i]
            
            # Add regularization
            grad -= self.regularization * beta
            
            # Update
            beta += lr * grad
        
        self.coefficients = beta
        
        # Fit baseline survival
        self.baseline_survival.fit(observations)
        self._fitted = True
    
    def survival_probability(
        self,
        time_since_birth: float,
        horizon: float,
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Cox conditional survival:
        
        S(t + h | t, x) = [S_0(t + h) / S_0(t)]^exp(beta' * x)
        """
        x = self._extract_features(features)
        risk_score = np.exp(np.dot(self.coefficients, x))
        
        # Get baseline conditional survival
        s_baseline = self.baseline_survival.survival_probability(
            time_since_birth, horizon, None
        )
        
        # Apply proportional hazards adjustment
        # Higher risk score -> lower survival
        return np.power(s_baseline, risk_score)


class AdaptiveSurvival(SurvivalModel):
    """
    Online adaptive survival model that updates with each observation.
    
    Maintains exponential moving averages of death rates per category,
    updated as new resolutions are observed.
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.1,
        default_hazard: float = 0.01,
    ):
        self.ema_alpha = ema_alpha
        self.default_hazard = default_hazard
        
        # Per-category statistics
        self._category_stats: Dict[str, Dict[str, float]] = {}
        self._global_stats = {"total_time": 0.0, "total_events": 0}
    
    def fit(self, observations: List[SurvivalObservation]) -> None:
        """Initialize from batch data."""
        for obs in observations:
            self.update(obs)
    
    def update(self, observation: SurvivalObservation) -> None:
        """Update model with a single observation."""
        category = observation.features.get("category", "default")
        
        if category not in self._category_stats:
            self._category_stats[category] = {
                "ema_hazard": self.default_hazard,
                "total_time": 0.0,
                "total_events": 0,
            }
        
        stats = self._category_stats[category]
        stats["total_time"] += observation.duration
        
        if observation.event:
            stats["total_events"] += 1
            self._global_stats["total_events"] += 1
            
            # Update EMA of hazard rate
            instantaneous_hazard = 1.0 / max(observation.duration, 1e-6)
            stats["ema_hazard"] = (
                (1 - self.ema_alpha) * stats["ema_hazard"]
                + self.ema_alpha * instantaneous_hazard
            )
        
        self._global_stats["total_time"] += observation.duration
    
    def survival_probability(
        self,
        time_since_birth: float,
        horizon: float,
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Use EMA hazard rate for exponential survival."""
        category = "default"
        if features:
            category = features.get("category", "default")
        
        stats = self._category_stats.get(category)
        if stats is None:
            hazard = self.default_hazard
        else:
            hazard = stats["ema_hazard"]
        
        return np.exp(-hazard * horizon)
    
    def get_hazard_rates(self) -> Dict[str, float]:
        """Get current hazard rate estimates per category."""
        return {
            cat: stats["ema_hazard"]
            for cat, stats in self._category_stats.items()
        }
