"""
Birth-Death Hawkes Process (BDHP) for Prediction Market Clustering.

Extends the multivariate Hawkes process with explicit death (resolution)
intensities. Markets that are mutually exciting (high alpha_ij) are
clustered together.

Key innovations:
1. Death intensity models market resolution probability
2. Mutual excitation reveals correlation structure
3. Handles irregular event times naturally
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from forecastbench.clustering.base import (
    OnlineClusteringBase,
    MarketState,
    ClusterState,
    ClusterAssignment,
)


class EventType(str, Enum):
    """Types of events in the Hawkes process."""
    PRICE_UP = "price_up"
    PRICE_DOWN = "price_down"
    BIRTH = "birth"
    DEATH = "death"


@dataclass
class HawkesEvent:
    """A single event in the Hawkes process."""
    timestamp: float
    market_id: str
    event_type: EventType
    magnitude: float = 1.0  # Size of price move


@dataclass
class BDHPConfig:
    """Configuration for Birth-Death Hawkes Process."""
    
    # Kernel parameters
    kernel_type: str = "exponential"  # or "power_law"
    decay_rate: float = 0.1  # Beta in exp(-beta * t)
    
    # Excitation parameters
    base_intensity: float = 0.1  # mu
    self_excitation: float = 0.3  # alpha_ii
    cross_excitation_init: float = 0.1  # alpha_ij initial
    
    # Death intensity parameters
    death_base_rate: float = 0.01
    death_price_sensitivity: float = 2.0  # Higher = more death near 0/1
    
    # Learning parameters
    learning_rate: float = 0.01
    excitation_decay: float = 0.99  # Decay old excitations
    
    # Clustering parameters
    n_clusters: Optional[int] = None
    excitation_threshold: float = 0.1  # Min excitation for edge
    
    # Event detection
    price_change_threshold: float = 0.02  # Min price change for event
    
    # Update frequency
    recluster_every: int = 20


class BirthDeathHawkes(OnlineClusteringBase):
    """
    Birth-Death Hawkes Process for market clustering.
    
    Models market price movements as a multivariate point process where:
    - Each market has a baseline intensity for price moves
    - Markets can excite each other (correlated moves)
    - Death (resolution) has its own intensity depending on price
    
    The excitation matrix alpha reveals cluster structure:
    markets with high mutual excitation are in the same cluster.
    
    Example:
        bdhp = BirthDeathHawkes(BDHPConfig(decay_rate=0.1))
        
        # Process events
        for event in event_stream:
            bdhp.add_event(event)
        
        # Get clusters based on excitation structure
        clusters = bdhp.get_clusters()
        
        # Get excitation between markets
        alpha_ij = bdhp.get_excitation("market_A", "market_B")
    """
    
    def __init__(self, config: Optional[BDHPConfig] = None):
        super().__init__(config.__dict__ if config else None)
        self.cfg = config or BDHPConfig()
        
        # Market indexing
        self._market_index: Dict[str, int] = {}
        self._index_to_market: Dict[int, str] = {}
        self._next_index: int = 0
        
        # Excitation matrix: alpha[i, j] = how much j excites i
        self._alpha: np.ndarray = np.array([[]])
        
        # Base intensities per market
        self._mu: np.ndarray = np.array([])
        
        # Event history for kernel computation
        self._event_history: List[HawkesEvent] = []
        self._max_history_length: int = 10000
        
        # Intensity state (for online updates)
        self._intensity: np.ndarray = np.array([])  # Current intensity per market
        
        # Previous prices for event detection
        self._prev_prices: Dict[str, float] = {}
        
        self._update_count: int = 0
    
    def _ensure_capacity(self, min_size: int) -> None:
        """Ensure arrays have capacity for at least min_size markets."""
        current_size = len(self._mu)
        if current_size >= min_size:
            return
        
        new_size = max(min_size, current_size * 2, 16)
        
        # Resize 1D arrays
        new_mu = np.ones(new_size) * self.cfg.base_intensity
        new_intensity = np.ones(new_size) * self.cfg.base_intensity
        
        if current_size > 0:
            new_mu[:current_size] = self._mu
            new_intensity[:current_size] = self._intensity
        
        self._mu = new_mu
        self._intensity = new_intensity
        
        # Resize excitation matrix
        new_alpha = np.eye(new_size) * self.cfg.self_excitation
        # Off-diagonal initialized to cross_excitation_init
        new_alpha += (1 - np.eye(new_size)) * self.cfg.cross_excitation_init
        
        if current_size > 0:
            new_alpha[:current_size, :current_size] = self._alpha
        
        self._alpha = new_alpha
    
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Initialize Hawkes state for new market."""
        idx = self._next_index
        self._next_index += 1
        
        self._market_index[market_id] = idx
        self._index_to_market[idx] = market_id
        
        self._ensure_capacity(idx + 1)
        
        self._prev_prices[market_id] = state.last_price
        
        # Record birth event
        event = HawkesEvent(
            timestamp=self._current_time,
            market_id=market_id,
            event_type=EventType.BIRTH,
        )
        self._add_event_to_history(event)
    
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """Record death event and update model."""
        # Record death event
        event = HawkesEvent(
            timestamp=self._current_time,
            market_id=market_id,
            event_type=EventType.DEATH,
        )
        self._add_event_to_history(event)
        
        # Clean up
        self._prev_prices.pop(market_id, None)
    
    def _add_event_to_history(self, event: HawkesEvent) -> None:
        """Add event to history, maintaining max length."""
        self._event_history.append(event)
        
        # Trim old events
        if len(self._event_history) > self._max_history_length:
            self._event_history = self._event_history[-self._max_history_length:]
    
    def _kernel(self, dt: float) -> float:
        """Compute kernel value at time lag dt."""
        if dt <= 0:
            return 0.0
        
        if self.cfg.kernel_type == "exponential":
            return self.cfg.decay_rate * np.exp(-self.cfg.decay_rate * dt)
        elif self.cfg.kernel_type == "power_law":
            return 1.0 / (1 + dt) ** 2
        else:
            return np.exp(-self.cfg.decay_rate * dt)
    
    def _compute_intensity(
        self,
        market_id: str,
        timestamp: float,
    ) -> float:
        """Compute current intensity for a market."""
        if market_id not in self._market_index:
            return self.cfg.base_intensity
        
        idx = self._market_index[market_id]
        
        # Base intensity
        intensity = self._mu[idx]
        
        # Add excitation from past events
        for event in self._event_history:
            if event.event_type in (EventType.PRICE_UP, EventType.PRICE_DOWN):
                if event.market_id not in self._market_index:
                    continue
                
                j = self._market_index[event.market_id]
                dt = timestamp - event.timestamp
                
                if dt > 0:
                    intensity += self._alpha[idx, j] * self._kernel(dt) * event.magnitude
        
        return max(intensity, 1e-10)
    
    def _death_intensity(self, market_id: str) -> float:
        """
        Compute death intensity for a market.
        
        Higher when price is near 0 or 1 (high certainty -> near resolution).
        """
        if market_id not in self._markets:
            return self.cfg.death_base_rate
        
        price = self._markets[market_id].last_price
        
        # Distance from 0.5 (neutral)
        distance_from_center = 2 * abs(price - 0.5)  # 0 at p=0.5, 1 at p=0/1
        
        # Death intensity increases near extremes
        death_intensity = self.cfg.death_base_rate * (
            1 + self.cfg.death_price_sensitivity * distance_from_center ** 2
        )
        
        return death_intensity
    
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """
        Update with new price observations.
        
        Detects price movement events and updates excitation matrix.
        """
        self._current_time = timestamp
        self._update_count += 1
        
        # Detect events and update
        events = []
        
        for market_id, price in prices.items():
            if market_id not in self._markets:
                continue
            if not self._markets[market_id].is_active:
                continue
            
            self._markets[market_id].add_price(timestamp, price)
            
            # Detect price movement event
            prev_price = self._prev_prices.get(market_id, price)
            price_change = price - prev_price
            
            if abs(price_change) >= self.cfg.price_change_threshold:
                event_type = EventType.PRICE_UP if price_change > 0 else EventType.PRICE_DOWN
                event = HawkesEvent(
                    timestamp=timestamp,
                    market_id=market_id,
                    event_type=event_type,
                    magnitude=abs(price_change) / self.cfg.price_change_threshold,
                )
                events.append(event)
                self._add_event_to_history(event)
            
            self._prev_prices[market_id] = price
        
        # Update excitation matrix based on observed events
        if events:
            self._update_excitation(events, timestamp)
        
        # Decay old excitations
        self._alpha *= self.cfg.excitation_decay
        
        # Restore self-excitation minimum
        for i in range(len(self._mu)):
            self._alpha[i, i] = max(self._alpha[i, i], self.cfg.self_excitation * 0.5)
        
        # Periodically recluster
        if self._update_count % self.cfg.recluster_every == 0:
            self._update_clusters()
    
    def _update_excitation(
        self,
        new_events: List[HawkesEvent],
        timestamp: float,
    ) -> None:
        """
        Update excitation matrix based on co-occurring events.
        
        If events in markets i and j occur close together, increase alpha_ij.
        """
        # Get recent events (within kernel support)
        lookback = 5.0 / self.cfg.decay_rate  # ~5 decay times
        recent_events = [
            e for e in self._event_history
            if timestamp - e.timestamp < lookback
            and e.event_type in (EventType.PRICE_UP, EventType.PRICE_DOWN)
        ]
        
        # For each new event, check for co-occurring events
        for event in new_events:
            if event.market_id not in self._market_index:
                continue
            
            i = self._market_index[event.market_id]
            
            for other_event in recent_events:
                if other_event.market_id not in self._market_index:
                    continue
                if other_event.market_id == event.market_id:
                    continue
                
                j = self._market_index[other_event.market_id]
                dt = abs(event.timestamp - other_event.timestamp)
                
                if dt < 1e-10:
                    dt = 0.01  # Simultaneous events
                
                # Update excitation based on kernel
                kernel_val = self._kernel(dt)
                
                # Same direction = positive excitation
                same_direction = event.event_type == other_event.event_type
                sign = 1.0 if same_direction else -0.5
                
                update = self.cfg.learning_rate * sign * kernel_val * event.magnitude
                
                # Symmetric update
                self._alpha[i, j] += update
                self._alpha[j, i] += update
                
                # Keep positive
                self._alpha[i, j] = max(self._alpha[i, j], 0)
                self._alpha[j, i] = max(self._alpha[j, i], 0)
    
    def get_excitation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Get the excitation matrix for active markets.
        
        Returns:
            (market_ids, alpha) where alpha[i,j] = excitation from j to i
        """
        active_ids = []
        active_indices = []
        
        for market_id, idx in self._market_index.items():
            if market_id not in self._markets:
                continue
            if not self._markets[market_id].is_active:
                continue
            
            active_ids.append(market_id)
            active_indices.append(idx)
        
        if not active_indices:
            return [], np.array([[]])
        
        alpha = self._alpha[np.ix_(active_indices, active_indices)]
        return active_ids, alpha
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Convert excitation matrix to correlation-like measure.
        
        Uses symmetrized and normalized excitation.
        """
        market_ids, alpha = self.get_excitation_matrix()
        n = len(market_ids)
        
        if n == 0:
            return [], np.array([[]])
        
        # Symmetrize
        alpha_sym = (alpha + alpha.T) / 2
        
        # Normalize to [-1, 1] range
        diag = np.diag(alpha_sym)
        diag = np.maximum(diag, 1e-10)
        d = np.sqrt(diag)
        corr = alpha_sym / np.outer(d, d)
        
        # Clip and fix diagonal
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)
        
        return market_ids, corr
    
    def get_excitation(self, market_id_1: str, market_id_2: str) -> float:
        """Get excitation between two specific markets."""
        if market_id_1 not in self._market_index or market_id_2 not in self._market_index:
            return 0.0
        
        i = self._market_index[market_id_1]
        j = self._market_index[market_id_2]
        
        return (self._alpha[i, j] + self._alpha[j, i]) / 2
    
    def _update_clusters(self) -> None:
        """Update clusters based on excitation structure."""
        market_ids, alpha = self.get_excitation_matrix()
        n = len(market_ids)
        
        if n < 2:
            for mid in market_ids:
                if 0 not in self._clusters:
                    self._clusters[0] = ClusterState(cluster_id=0)
                self._clusters[0].add_member(mid)
                self._assignments[mid] = ClusterAssignment(
                    market_id=mid, cluster_id=0, confidence=1.0
                )
            return
        
        # Symmetrize excitation
        alpha_sym = (alpha + alpha.T) / 2
        
        # Convert to distance (high excitation = low distance)
        max_alpha = alpha_sym.max()
        if max_alpha > 0:
            dist = 1 - alpha_sym / max_alpha
        else:
            dist = np.ones((n, n))
        
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, 2)
        
        # Hierarchical clustering
        try:
            dist_condensed = squareform(dist, checks=False)
            Z = linkage(dist_condensed, method='average')
        except Exception:
            self._assign_all_to_one_cluster(market_ids)
            return
        
        # Cut tree
        if self.cfg.n_clusters is not None:
            labels = fcluster(Z, self.cfg.n_clusters, criterion='maxclust')
        else:
            # Use excitation threshold
            threshold = 1 - self.cfg.excitation_threshold / (max_alpha + 1e-10)
            labels = fcluster(Z, threshold, criterion='distance')
        
        # Clear old clusters
        for cluster in self._clusters.values():
            cluster.member_ids.clear()
        
        # Assign to clusters
        for i, (mid, label) in enumerate(zip(market_ids, labels)):
            cluster_id = int(label) - 1
            
            if cluster_id not in self._clusters:
                self._clusters[cluster_id] = ClusterState(cluster_id=cluster_id)
            
            self._clusters[cluster_id].add_member(mid)
            self._assignments[mid] = ClusterAssignment(
                market_id=mid,
                cluster_id=cluster_id,
                confidence=1.0,
            )
    
    def _assign_all_to_one_cluster(self, market_ids: List[str]) -> None:
        """Fallback: assign all to cluster 0."""
        if 0 not in self._clusters:
            self._clusters[0] = ClusterState(cluster_id=0)
        
        self._clusters[0].member_ids.clear()
        
        for mid in market_ids:
            self._clusters[0].add_member(mid)
            self._assignments[mid] = ClusterAssignment(
                market_id=mid, cluster_id=0, confidence=1.0
            )
    
    def add_event(self, event: HawkesEvent) -> None:
        """Manually add an event to the process."""
        self._current_time = max(self._current_time, event.timestamp)
        self._add_event_to_history(event)
        
        if event.event_type in (EventType.PRICE_UP, EventType.PRICE_DOWN):
            self._update_excitation([event], event.timestamp)
    
    def get_event_history(
        self,
        market_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
    ) -> List[HawkesEvent]:
        """Get filtered event history."""
        events = self._event_history
        
        if market_id is not None:
            events = [e for e in events if e.market_id == market_id]
        
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        
        return events
    
    def predict_next_event_intensity(self, market_id: str) -> float:
        """Predict intensity of next event for a market."""
        return self._compute_intensity(market_id, self._current_time)
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self._market_index.clear()
        self._index_to_market.clear()
        self._next_index = 0
        self._alpha = np.array([[]])
        self._mu = np.array([])
        self._event_history.clear()
        self._intensity = np.array([])
        self._prev_prices.clear()
        self._update_count = 0
