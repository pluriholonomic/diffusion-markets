"""
Online Low-Rank Covariance with Missing Data (OLRCM).

Core idea: Maintain a low-rank factorization of the covariance matrix,
treating resolved markets as missing data. Factor loadings reveal
the latent correlation structure and define clusters.

Advantages:
- Efficient O(k * |active|) updates per observation
- Factor loadings reveal latent correlation structure
- Natural handling of missing data (dead markets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from forecastbench.clustering.base import (
    OnlineClusteringBase,
    MarketState,
    ClusterState,
    ClusterAssignment,
)


@dataclass
class OLRCMConfig:
    """Configuration for OLRCM algorithm."""
    
    # Factor model parameters
    n_factors: int = 10  # Number of latent factors
    learning_rate: float = 0.01  # SGD learning rate
    momentum: float = 0.9  # Momentum for SGD
    
    # Regularization
    l2_reg: float = 0.001  # L2 regularization on loadings
    
    # Clustering parameters
    n_clusters: Optional[int] = None
    distance_threshold: float = 0.5
    min_cluster_size: int = 2
    
    # Update parameters
    recluster_every: int = 10
    min_observations: int = 5
    
    # Initialization
    init_scale: float = 0.1


class OLRCM(OnlineClusteringBase):
    """
    Online Low-Rank Covariance with Missing Data.
    
    Maintains a low-rank factorization:
        Cov ≈ V^T V + sigma^2 I
    
    where V is a (k × d) matrix of factor loadings.
    
    When a market dies (resolves), its column in V is retained but
    the market is excluded from updates. This handles the missing
    data pattern naturally.
    
    Clustering is performed on the factor loadings: markets with
    similar loadings are in the same cluster.
    
    Example:
        olrcm = OLRCM(OLRCMConfig(n_factors=5))
        
        for t, prices in enumerate(price_stream):
            olrcm.update(timestamp=t, prices=prices)
            
            if t % 100 == 0:
                clusters = olrcm.get_clusters()
                loadings = olrcm.get_factor_loadings()
    """
    
    def __init__(self, config: Optional[OLRCMConfig] = None):
        super().__init__(config.__dict__ if config else None)
        self.cfg = config or OLRCMConfig()
        
        # Factor model state
        self._market_index: Dict[str, int] = {}
        self._index_to_market: Dict[int, str] = {}
        self._next_index: int = 0
        
        # Factor loadings V: (n_factors, max_markets)
        self._V: np.ndarray = np.array([[]])
        # Idiosyncratic variance estimates
        self._sigma2: np.ndarray = np.array([])
        
        # Momentum for SGD
        self._V_momentum: np.ndarray = np.array([[]])
        
        # Mean estimates for centering
        self._means: np.ndarray = np.array([])
        self._n_obs: np.ndarray = np.array([])
        
        # Previous values for return computation
        self._prev_values: Dict[str, float] = {}
        
        self._update_count: int = 0
    
    def _ensure_capacity(self, min_size: int) -> None:
        """Ensure arrays have capacity for at least min_size markets."""
        k = self.cfg.n_factors
        current_size = self._V.shape[1] if self._V.size > 0 else 0
        
        if current_size >= min_size:
            return
        
        new_size = max(min_size, current_size * 2, 16)
        
        # Resize factor loadings
        new_V = np.random.randn(k, new_size) * self.cfg.init_scale
        new_V_momentum = np.zeros((k, new_size))
        
        if current_size > 0:
            new_V[:, :current_size] = self._V
            new_V_momentum[:, :current_size] = self._V_momentum
        
        self._V = new_V
        self._V_momentum = new_V_momentum
        
        # Resize other arrays
        new_sigma2 = np.ones(new_size) * 0.1
        new_means = np.zeros(new_size)
        new_n_obs = np.zeros(new_size)
        
        if current_size > 0:
            new_sigma2[:current_size] = self._sigma2
            new_means[:current_size] = self._means
            new_n_obs[:current_size] = self._n_obs
        
        self._sigma2 = new_sigma2
        self._means = new_means
        self._n_obs = new_n_obs
    
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Initialize factor loadings for new market."""
        idx = self._next_index
        self._next_index += 1
        
        self._market_index[market_id] = idx
        self._index_to_market[idx] = market_id
        
        self._ensure_capacity(idx + 1)
        
        # Initialize loadings randomly
        self._V[:, idx] = np.random.randn(self.cfg.n_factors) * self.cfg.init_scale
        self._sigma2[idx] = 0.1
        self._means[idx] = state.last_price
        self._n_obs[idx] = 0
        self._prev_values[market_id] = state.last_price
    
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """
        Handle market death.
        
        The loadings are retained but the market is excluded from
        future updates. This allows correlation with dead markets
        to be analyzed retrospectively.
        """
        # Remove from prev_values tracking
        self._prev_values.pop(market_id, None)
    
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """
        Update factor model with new observations.
        
        Uses stochastic gradient descent to update factor loadings
        to minimize reconstruction error.
        """
        self._current_time = timestamp
        self._update_count += 1
        
        # Update price histories
        for market_id, price in prices.items():
            if market_id in self._markets:
                self._markets[market_id].add_price(timestamp, price)
        
        # Compute returns for active markets
        active_indices = []
        returns = []
        
        for market_id, price in prices.items():
            if market_id not in self._markets:
                continue
            if not self._markets[market_id].is_active:
                continue
            
            idx = self._market_index[market_id]
            
            # Compute return
            prev = self._prev_values.get(market_id, price)
            prev = np.clip(prev, 0.01, 0.99)
            price_clipped = np.clip(price, 0.01, 0.99)
            ret = np.log(price_clipped) - np.log(prev)
            
            # Update mean estimate (EMA)
            alpha = 0.01
            self._means[idx] = (1 - alpha) * self._means[idx] + alpha * ret
            self._n_obs[idx] += 1
            
            active_indices.append(idx)
            returns.append(ret - self._means[idx])  # Center
            self._prev_values[market_id] = price
        
        if len(active_indices) < 2:
            return
        
        active_indices = np.array(active_indices)
        returns = np.array(returns)
        
        # Get current loadings for active markets
        V_active = self._V[:, active_indices]  # (k, n_active)
        
        # Predicted covariance contribution: V^T @ V
        # For reconstruction: r ≈ V^T @ f where f is latent factor
        # We estimate f as: f = (V @ V^T + lambda*I)^-1 @ V @ r
        # For efficiency, use: f ≈ V @ r / (||V||^2 + lambda)
        
        k = self.cfg.n_factors
        n_active = len(active_indices)
        
        # Estimate latent factors
        # f = V_active @ returns gives k-dimensional factor estimate
        f = V_active @ returns  # (k,)
        
        # Reconstruct returns
        reconstructed = V_active.T @ f  # (n_active,)
        
        # Compute residuals
        residuals = returns - reconstructed  # (n_active,)
        
        # Gradient for loadings: d/dV ||r - V^T f||^2
        # = -2 * (r - V^T f) @ f^T
        grad = -2 * np.outer(f, residuals)  # (k, n_active)
        
        # Add L2 regularization
        grad += 2 * self.cfg.l2_reg * V_active
        
        # Momentum update
        self._V_momentum[:, active_indices] = (
            self.cfg.momentum * self._V_momentum[:, active_indices]
            - self.cfg.learning_rate * grad
        )
        
        self._V[:, active_indices] += self._V_momentum[:, active_indices]
        
        # Update idiosyncratic variance
        alpha_var = 0.01
        for i, idx in enumerate(active_indices):
            self._sigma2[idx] = (
                (1 - alpha_var) * self._sigma2[idx]
                + alpha_var * residuals[i] ** 2
            )
        
        # Periodically recluster
        if self._update_count % self.cfg.recluster_every == 0:
            self._update_clusters()
    
    def get_factor_loadings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get factor loadings for active markets.
        
        Returns:
            (market_ids, loadings) where loadings is (n_factors, n_markets)
        """
        active_ids = []
        active_indices = []
        
        for market_id, idx in self._market_index.items():
            if market_id not in self._markets:
                continue
            if not self._markets[market_id].is_active:
                continue
            if self._n_obs[idx] < self.cfg.min_observations:
                continue
            
            active_ids.append(market_id)
            active_indices.append(idx)
        
        if not active_indices:
            return [], np.array([[]])
        
        loadings = self._V[:, active_indices]
        return active_ids, loadings
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Get correlation matrix implied by factor model.
        
        Cov = V^T V + diag(sigma^2)
        Corr = D^{-1/2} Cov D^{-1/2} where D = diag(Cov)
        """
        market_ids, loadings = self.get_factor_loadings()
        n = len(market_ids)
        
        if n == 0:
            return [], np.array([[]])
        
        # Covariance from factor model
        cov = loadings.T @ loadings  # (n, n)
        
        # Add idiosyncratic variance
        indices = [self._market_index[mid] for mid in market_ids]
        for i, idx in enumerate(indices):
            cov[i, i] += self._sigma2[idx]
        
        # Convert to correlation
        d = np.sqrt(np.diag(cov))
        d = np.maximum(d, 1e-10)
        corr = cov / np.outer(d, d)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)
        
        return market_ids, corr
    
    def _update_clusters(self) -> None:
        """Update clusters based on factor loading similarity."""
        market_ids, loadings = self.get_factor_loadings()
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
        
        # Cluster on loadings using cosine distance
        # Normalize loadings
        norms = np.linalg.norm(loadings, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        loadings_normalized = loadings / norms  # (k, n)
        
        # Compute pairwise distances
        # Cosine distance = 1 - cosine similarity
        cosine_sim = loadings_normalized.T @ loadings_normalized
        dist_matrix = 1 - cosine_sim
        dist_matrix = np.clip(dist_matrix, 0, 2)
        np.fill_diagonal(dist_matrix, 0)
        
        # Hierarchical clustering
        try:
            dist_condensed = pdist(loadings_normalized.T, metric='cosine')
            Z = linkage(dist_condensed, method='average')
        except Exception:
            self._assign_all_to_one_cluster(market_ids)
            return
        
        # Cut tree
        if self.cfg.n_clusters is not None:
            labels = fcluster(Z, self.cfg.n_clusters, criterion='maxclust')
        else:
            labels = fcluster(Z, self.cfg.distance_threshold, criterion='distance')
        
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
    
    def get_factor_exposures(self, market_id: str) -> Optional[np.ndarray]:
        """Get factor exposures for a specific market."""
        if market_id not in self._market_index:
            return None
        
        idx = self._market_index[market_id]
        return self._V[:, idx].copy()
    
    def predict_correlation(
        self,
        market_id_1: str,
        market_id_2: str,
    ) -> float:
        """Predict correlation between two markets using factor model."""
        if market_id_1 not in self._market_index or market_id_2 not in self._market_index:
            return 0.0
        
        idx1 = self._market_index[market_id_1]
        idx2 = self._market_index[market_id_2]
        
        v1 = self._V[:, idx1]
        v2 = self._V[:, idx2]
        
        # Covariance from loadings
        cov = np.dot(v1, v2)
        
        # Variances
        var1 = np.dot(v1, v1) + self._sigma2[idx1]
        var2 = np.dot(v2, v2) + self._sigma2[idx2]
        
        if var1 <= 0 or var2 <= 0:
            return 0.0
        
        return cov / np.sqrt(var1 * var2)
    
    def explained_variance_ratio(self) -> np.ndarray:
        """
        Compute fraction of variance explained by each factor.
        
        Returns array of shape (n_factors,) with explained variance ratios.
        """
        market_ids, loadings = self.get_factor_loadings()
        if len(market_ids) == 0:
            return np.zeros(self.cfg.n_factors)
        
        # Variance explained by each factor
        factor_vars = np.sum(loadings ** 2, axis=1)  # (k,)
        
        # Total variance
        indices = [self._market_index[mid] for mid in market_ids]
        total_var = np.sum(factor_vars) + np.sum(self._sigma2[indices])
        
        if total_var <= 0:
            return np.zeros(self.cfg.n_factors)
        
        return factor_vars / total_var
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self._market_index.clear()
        self._index_to_market.clear()
        self._next_index = 0
        self._V = np.array([[]])
        self._sigma2 = np.array([])
        self._V_momentum = np.array([[]])
        self._means = np.array([])
        self._n_obs = np.array([])
        self._prev_values.clear()
        self._update_count = 0
