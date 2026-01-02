"""
Base classes for online clustering algorithms.

All clustering algorithms inherit from OnlineClusteringBase and implement
the core interface for handling market birth, death, and price updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np


class MarketStatus(str, Enum):
    """Status of a market in the clustering system."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUSPENDED = "suspended"


@dataclass
class MarketState:
    """
    State information for a single market.
    
    Attributes:
        market_id: Unique identifier for the market
        status: Current status (active, resolved, suspended)
        birth_time: Time when market was first observed
        death_time: Time when market resolved (None if still active)
        category: Category/topic of the market
        last_price: Most recent price observation
        price_history: List of (timestamp, price) tuples
        features: Additional features for clustering
    """
    market_id: str
    status: MarketStatus = MarketStatus.ACTIVE
    birth_time: float = 0.0
    death_time: Optional[float] = None
    category: str = ""
    last_price: float = 0.5
    price_history: List[Tuple[float, float]] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        return self.status == MarketStatus.ACTIVE
    
    @property
    def lifetime(self) -> Optional[float]:
        if self.death_time is not None:
            return self.death_time - self.birth_time
        return None
    
    def add_price(self, timestamp: float, price: float) -> None:
        """Add a price observation."""
        self.price_history.append((timestamp, price))
        self.last_price = price
    
    def get_returns(self, log_returns: bool = False) -> np.ndarray:
        """Compute returns from price history."""
        if len(self.price_history) < 2:
            return np.array([])
        
        prices = np.array([p for _, p in self.price_history])
        prices = np.clip(prices, 0.01, 0.99)  # Avoid log(0)
        
        if log_returns:
            return np.diff(np.log(prices))
        else:
            return np.diff(prices) / prices[:-1]


@dataclass
class ClusterAssignment:
    """
    Cluster assignment for a market.
    
    Attributes:
        market_id: The market being assigned
        cluster_id: ID of the assigned cluster
        confidence: Confidence in the assignment [0, 1]
        soft_assignments: Optional dict of cluster_id -> probability
    """
    market_id: str
    cluster_id: int
    confidence: float = 1.0
    soft_assignments: Optional[Dict[int, float]] = None
    
    def get_soft_assignment(self, cluster_id: int) -> float:
        """Get probability of belonging to a specific cluster."""
        if self.soft_assignments is None:
            return 1.0 if cluster_id == self.cluster_id else 0.0
        return self.soft_assignments.get(cluster_id, 0.0)


@dataclass
class ClusterState:
    """
    State of a single cluster.
    
    Attributes:
        cluster_id: Unique identifier
        member_ids: Set of market IDs in this cluster
        centroid: Cluster centroid (if applicable)
        sufficient_stats: Sufficient statistics for online updates
    """
    cluster_id: int
    member_ids: Set[str] = field(default_factory=set)
    centroid: Optional[np.ndarray] = None
    sufficient_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.member_ids)
    
    def add_member(self, market_id: str) -> None:
        self.member_ids.add(market_id)
    
    def remove_member(self, market_id: str) -> None:
        self.member_ids.discard(market_id)


class OnlineClusteringBase(ABC):
    """
    Abstract base class for online clustering algorithms.
    
    All prediction market clustering algorithms must implement this interface
    to handle the unique challenges of market birth, death, and streaming updates.
    
    Usage:
        clustering = MyClusteringAlgorithm(config)
        
        # Add a new market
        clustering.add_market("market_123", features={"category": "politics"})
        
        # Update with price observations
        clustering.update(timestamp=1.0, prices={"market_123": 0.65})
        
        # Market resolves
        clustering.remove_market("market_123", timestamp=2.0)
        
        # Get current clusters
        clusters = clustering.get_clusters()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._markets: Dict[str, MarketState] = {}
        self._clusters: Dict[int, ClusterState] = {}
        self._assignments: Dict[str, ClusterAssignment] = {}
        self._current_time: float = 0.0
        self._next_cluster_id: int = 0
    
    @property
    def n_active_markets(self) -> int:
        """Number of currently active markets."""
        return sum(1 for m in self._markets.values() if m.is_active)
    
    @property
    def n_clusters(self) -> int:
        """Number of non-empty clusters."""
        return sum(1 for c in self._clusters.values() if c.size > 0)
    
    @property
    def active_market_ids(self) -> List[str]:
        """List of active market IDs."""
        return [m.market_id for m in self._markets.values() if m.is_active]
    
    def add_market(
        self,
        market_id: str,
        timestamp: Optional[float] = None,
        initial_price: float = 0.5,
        category: str = "",
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Add a new market to the clustering system.
        
        Args:
            market_id: Unique identifier for the market
            timestamp: Time of market creation
            initial_price: Initial price of the market
            category: Category/topic of the market
            features: Additional features for clustering
        """
        if timestamp is not None:
            self._current_time = max(self._current_time, timestamp)
        
        state = MarketState(
            market_id=market_id,
            status=MarketStatus.ACTIVE,
            birth_time=self._current_time,
            category=category,
            last_price=initial_price,
            features=features or {},
        )
        state.add_price(self._current_time, initial_price)
        self._markets[market_id] = state
        
        # Subclass-specific initialization
        self._on_market_added(market_id, state)
    
    def remove_market(
        self,
        market_id: str,
        timestamp: Optional[float] = None,
        outcome: Optional[int] = None,
    ) -> None:
        """
        Remove a market (market resolved/died).
        
        Args:
            market_id: ID of the market to remove
            timestamp: Time of resolution
            outcome: Resolution outcome (0 or 1)
        """
        if market_id not in self._markets:
            return
        
        if timestamp is not None:
            self._current_time = max(self._current_time, timestamp)
        
        state = self._markets[market_id]
        state.status = MarketStatus.RESOLVED
        state.death_time = self._current_time
        
        # Remove from cluster
        if market_id in self._assignments:
            cluster_id = self._assignments[market_id].cluster_id
            if cluster_id in self._clusters:
                self._clusters[cluster_id].remove_member(market_id)
            del self._assignments[market_id]
        
        # Subclass-specific cleanup
        self._on_market_removed(market_id, state, outcome)
    
    @abstractmethod
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """
        Update clustering with new price observations.
        
        Args:
            timestamp: Current timestamp
            prices: Dict mapping market_id -> current price
        """
        pass
    
    @abstractmethod
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Hook called when a new market is added."""
        pass
    
    @abstractmethod
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """Hook called when a market is removed."""
        pass
    
    def get_clusters(self) -> Dict[int, List[str]]:
        """
        Get current cluster assignments.
        
        Returns:
            Dict mapping cluster_id -> list of market_ids
        """
        result = {}
        for cluster_id, cluster in self._clusters.items():
            active_members = [
                m for m in cluster.member_ids
                if m in self._markets and self._markets[m].is_active
            ]
            if active_members:
                result[cluster_id] = active_members
        return result
    
    def get_assignment(self, market_id: str) -> Optional[ClusterAssignment]:
        """Get cluster assignment for a market."""
        return self._assignments.get(market_id)
    
    def get_cluster_for_market(self, market_id: str) -> Optional[int]:
        """Get cluster ID for a market."""
        assignment = self._assignments.get(market_id)
        return assignment.cluster_id if assignment else None
    
    def get_markets_in_cluster(self, cluster_id: int) -> List[str]:
        """Get all active markets in a cluster."""
        if cluster_id not in self._clusters:
            return []
        return [
            m for m in self._clusters[cluster_id].member_ids
            if m in self._markets and self._markets[m].is_active
        ]
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Get estimated correlation matrix for active markets.
        
        Returns:
            (market_ids, correlation_matrix) tuple
        """
        # Default implementation - subclasses should override
        active_ids = self.active_market_ids
        n = len(active_ids)
        corr = np.eye(n)
        return active_ids, corr
    
    def fit_predict(
        self,
        price_data: np.ndarray,
        death_events: List[Tuple[int, str]],
        market_ids: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Batch fitting for evaluation purposes.
        
        Args:
            price_data: (T, n_markets) array of prices
            death_events: List of (timestep, market_id) death events
            market_ids: Optional list of market IDs
            
        Returns:
            Array of cluster labels for each market
        """
        T, n_markets = price_data.shape
        if market_ids is None:
            market_ids = [f"market_{i}" for i in range(n_markets)]
        
        # Initialize all markets
        for i, mid in enumerate(market_ids):
            self.add_market(mid, timestamp=0.0, initial_price=price_data[0, i])
        
        # Create death event lookup
        death_lookup = {}
        for t, mid in death_events:
            death_lookup.setdefault(t, []).append(mid)
        
        # Process time series
        for t in range(1, T):
            prices = {
                mid: price_data[t, i]
                for i, mid in enumerate(market_ids)
                if mid in self._markets and self._markets[mid].is_active
            }
            self.update(timestamp=float(t), prices=prices)
            
            # Handle deaths at this timestep
            for mid in death_lookup.get(t, []):
                self.remove_market(mid, timestamp=float(t))
        
        # Extract final labels
        labels = np.zeros(n_markets, dtype=np.int32)
        for i, mid in enumerate(market_ids):
            assignment = self._assignments.get(mid)
            if assignment:
                labels[i] = assignment.cluster_id
            else:
                labels[i] = -1  # Unassigned/dead
        
        return labels
    
    def reset(self) -> None:
        """Reset all state."""
        self._markets.clear()
        self._clusters.clear()
        self._assignments.clear()
        self._current_time = 0.0
        self._next_cluster_id = 0
    
    def snapshot(self) -> Dict[str, Any]:
        """Get serializable snapshot of current state."""
        return {
            "n_active_markets": self.n_active_markets,
            "n_clusters": self.n_clusters,
            "clusters": {
                cid: list(c.member_ids)
                for cid, c in self._clusters.items()
                if c.size > 0
            },
            "current_time": self._current_time,
        }
