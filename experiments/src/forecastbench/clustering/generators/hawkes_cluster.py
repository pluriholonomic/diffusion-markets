"""
Hawkes Process Cluster Generator.

Generates event streams from a multivariate Hawkes process with
block structure in the excitation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class HawkesEvent:
    """A single event in the Hawkes process."""
    timestamp: float
    market_id: str
    cluster_id: int
    magnitude: float = 1.0


@dataclass
class HawkesClusterConfig:
    """Configuration for Hawkes cluster generator."""
    
    # Cluster structure
    n_clusters: int = 5
    markets_per_cluster: int = 10
    
    # Hawkes parameters
    base_rate: float = 0.1  # Background intensity
    within_alpha: float = 0.5  # Excitation within cluster
    between_alpha: float = 0.05  # Excitation between clusters
    decay_rate: float = 1.0  # Kernel decay
    
    # Death model
    death_rate: float = 0.01
    death_price_effect: float = 1.0  # How much price extremity affects death
    min_lifetime: float = 10.0
    
    # Simulation
    max_time: float = 100.0
    
    # Random seed
    seed: int = 42


class HawkesClusterGenerator:
    """
    Generate event streams from a multivariate Hawkes process.
    
    The excitation matrix has block structure:
    - High excitation within clusters
    - Low excitation between clusters
    
    Uses Ogata's thinning algorithm for simulation.
    
    Example:
        gen = HawkesClusterGenerator(HawkesClusterConfig())
        events, deaths, labels, alpha = gen.generate()
    """
    
    def __init__(self, config: Optional[HawkesClusterConfig] = None):
        self.cfg = config or HawkesClusterConfig()
    
    def generate(self) -> Tuple[List[HawkesEvent], List[Tuple[float, str]], np.ndarray, np.ndarray]:
        """
        Generate synthetic Hawkes process data.
        
        Returns:
            events: List of HawkesEvent
            death_events: List of (timestamp, market_id) tuples
            labels: (n_markets,) cluster labels
            alpha: (n_markets, n_markets) true excitation matrix
        """
        rng = np.random.default_rng(self.cfg.seed)
        
        n_clusters = self.cfg.n_clusters
        n_per_cluster = self.cfg.markets_per_cluster
        n_markets = n_clusters * n_per_cluster
        
        # Create labels
        labels = np.repeat(np.arange(n_clusters), n_per_cluster)
        
        # Build excitation matrix
        alpha = self._build_excitation_matrix(n_clusters, n_per_cluster)
        
        # Simulate Hawkes process
        events = self._simulate_hawkes(rng, n_markets, labels, alpha)
        
        # Generate deaths
        death_events = self._generate_deaths(rng, n_markets, events)
        
        return events, death_events, labels, alpha
    
    def _build_excitation_matrix(
        self,
        n_clusters: int,
        n_per_cluster: int,
    ) -> np.ndarray:
        """Build block excitation matrix."""
        n = n_clusters * n_per_cluster
        alpha = np.ones((n, n)) * self.cfg.between_alpha
        
        # Set within-cluster excitation
        for c in range(n_clusters):
            start = c * n_per_cluster
            end = (c + 1) * n_per_cluster
            alpha[start:end, start:end] = self.cfg.within_alpha
        
        return alpha
    
    def _simulate_hawkes(
        self,
        rng: np.random.Generator,
        n_markets: int,
        labels: np.ndarray,
        alpha: np.ndarray,
    ) -> List[HawkesEvent]:
        """
        Simulate Hawkes process using Ogata's thinning algorithm.
        """
        events: List[HawkesEvent] = []
        
        # Track event times per market
        market_events: Dict[int, List[float]] = {i: [] for i in range(n_markets)}
        
        t = 0.0
        max_t = self.cfg.max_time
        
        while t < max_t:
            # Compute upper bound on total intensity
            lambda_bar = self._compute_intensity_bound(t, market_events, alpha)
            
            if lambda_bar < 1e-10:
                lambda_bar = n_markets * self.cfg.base_rate
            
            # Sample next candidate time
            dt = rng.exponential(1 / lambda_bar)
            t += dt
            
            if t >= max_t:
                break
            
            # Compute true intensity at time t
            intensities = np.array([
                self._compute_intensity(i, t, market_events, alpha)
                for i in range(n_markets)
            ])
            
            total_intensity = intensities.sum()
            
            # Accept/reject
            if rng.random() < total_intensity / lambda_bar:
                # Choose market
                probs = intensities / total_intensity
                market_idx = rng.choice(n_markets, p=probs)
                
                # Record event
                event = HawkesEvent(
                    timestamp=t,
                    market_id=f"market_{market_idx}",
                    cluster_id=int(labels[market_idx]),
                    magnitude=1.0 + rng.exponential(0.5),
                )
                events.append(event)
                market_events[market_idx].append(t)
        
        return events
    
    def _compute_intensity(
        self,
        market_idx: int,
        t: float,
        market_events: Dict[int, List[float]],
        alpha: np.ndarray,
    ) -> float:
        """Compute intensity for a single market at time t."""
        intensity = self.cfg.base_rate
        
        for j, event_times in market_events.items():
            for event_t in event_times:
                if event_t < t:
                    dt = t - event_t
                    intensity += alpha[market_idx, j] * self._kernel(dt)
        
        return intensity
    
    def _compute_intensity_bound(
        self,
        t: float,
        market_events: Dict[int, List[float]],
        alpha: np.ndarray,
    ) -> float:
        """Compute upper bound on total intensity."""
        n_markets = len(market_events)
        
        # Base intensity
        bound = n_markets * self.cfg.base_rate
        
        # Add excitation from recent events
        for j, event_times in market_events.items():
            for event_t in event_times:
                if event_t < t:
                    dt = t - event_t
                    bound += alpha[:, j].sum() * self._kernel(dt)
        
        return bound
    
    def _kernel(self, dt: float) -> float:
        """Exponential kernel."""
        if dt <= 0:
            return 0.0
        return self.cfg.decay_rate * np.exp(-self.cfg.decay_rate * dt)
    
    def _generate_deaths(
        self,
        rng: np.random.Generator,
        n_markets: int,
        events: List[HawkesEvent],
    ) -> List[Tuple[float, str]]:
        """Generate death events."""
        death_events = []
        
        # Count events per market to estimate "activity"
        event_counts: Dict[int, int] = {}
        for event in events:
            idx = int(event.market_id.split("_")[1])
            event_counts[idx] = event_counts.get(idx, 0) + 1
        
        for i in range(n_markets):
            # Death time based on activity
            activity = event_counts.get(i, 1)
            adjusted_rate = self.cfg.death_rate * (1 + self.cfg.death_price_effect * activity / 10)
            
            death_time = rng.exponential(1 / adjusted_rate)
            death_time = max(death_time, self.cfg.min_lifetime)
            
            if death_time < self.cfg.max_time:
                death_events.append((death_time, f"market_{i}"))
        
        death_events.sort(key=lambda x: x[0])
        return death_events
    
    def convert_to_prices(
        self,
        events: List[HawkesEvent],
        n_markets: int,
        n_timesteps: int = 100,
    ) -> np.ndarray:
        """
        Convert event stream to price time series.
        
        Prices move based on event arrivals.
        """
        max_t = self.cfg.max_time
        dt = max_t / n_timesteps
        
        prices = np.ones((n_timesteps, n_markets)) * 0.5
        
        # Initialize
        rng = np.random.default_rng(self.cfg.seed + 1)
        prices[0] = 0.5 + rng.standard_normal(n_markets) * 0.05
        prices[0] = np.clip(prices[0], 0.1, 0.9)
        
        for t_idx in range(1, n_timesteps):
            t_start = (t_idx - 1) * dt
            t_end = t_idx * dt
            
            # Events in this interval
            for event in events:
                if t_start <= event.timestamp < t_end:
                    idx = int(event.market_id.split("_")[1])
                    # Move price
                    direction = 1 if rng.random() > 0.5 else -1
                    prices[t_idx, idx] += direction * 0.02 * event.magnitude
            
            # Mean reversion
            prices[t_idx] = prices[t_idx-1] + 0.1 * (0.5 - prices[t_idx-1])
            prices[t_idx] = np.clip(prices[t_idx], 0.01, 0.99)
        
        return prices
