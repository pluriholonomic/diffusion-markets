"""
Survival-Weighted Online Correlation Clustering (SWOCC).

Core idea: Weight correlation estimates by expected remaining lifetime
using survival analysis. Markets near resolution contribute less to
correlation estimates because:
1. They will soon disappear
2. Their price dynamics are dominated by resolution mechanics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from forecastbench.clustering.base import (
    OnlineClusteringBase,
    MarketState,
    ClusterState,
    ClusterAssignment,
)
from forecastbench.clustering.survival_model import (
    SurvivalModel,
    ExponentialSurvival,
    AdaptiveSurvival,
    SurvivalObservation,
)


@dataclass
class SWOCCConfig:
    """Configuration for SWOCC algorithm."""
    
    # EMA parameters
    ema_alpha: float = 0.1  # Decay factor for correlation updates
    
    # Survival weighting
    use_survival_weights: bool = True
    survival_horizon: float = 7.0  # Days to look ahead for survival
    
    # Clustering parameters
    n_clusters: Optional[int] = None  # If None, use threshold-based
    distance_threshold: float = 0.5  # For threshold-based clustering
    min_cluster_size: int = 2
    
    # Update frequency
    recluster_every: int = 10  # Recluster every N updates
    min_observations: int = 5  # Min observations before including in clustering
    
    # Correlation estimation
    use_returns: bool = True  # Use returns vs raw prices
    shrinkage: float = 0.1  # Shrinkage toward identity


class SWOCC(OnlineClusteringBase):
    """
    Survival-Weighted Online Correlation Clustering.
    
    Maintains an online estimate of the correlation matrix between active
    markets, weighted by survival probabilities. Clusters are extracted
    using hierarchical clustering on the correlation-based distance matrix.
    
    Key features:
    1. Survival weighting: Correlations involving markets near resolution
       are down-weighted
    2. Online updates: Correlation matrix updated incrementally
    3. Adaptive: Clusters updated as markets appear/disappear
    
    Example:
        swocc = SWOCC()
        
        # Add markets
        swocc.add_market("A", category="politics")
        swocc.add_market("B", category="politics")
        swocc.add_market("C", category="sports")
        
        # Update with prices
        for t, prices in enumerate(price_stream):
            swocc.update(timestamp=t, prices=prices)
        
        # Get clusters
        clusters = swocc.get_clusters()
    """
    
    def __init__(
        self,
        config: Optional[SWOCCConfig] = None,
        survival_model: Optional[SurvivalModel] = None,
    ):
        super().__init__(config.__dict__ if config else None)
        self.cfg = config or SWOCCConfig()
        
        # Survival model for weighting
        self.survival_model = survival_model or AdaptiveSurvival()
        
        # Correlation estimation state
        self._market_index: Dict[str, int] = {}  # market_id -> matrix index
        self._index_to_market: Dict[int, str] = {}
        self._next_index: int = 0
        
        # Sufficient statistics for online covariance
        self._n_obs: np.ndarray = np.array([])  # (n,) observation counts
        self._sum_x: np.ndarray = np.array([])  # (n,) sum of values
        self._sum_xx: np.ndarray = np.array([])  # (n,) sum of squared values
        self._sum_xy: np.ndarray = np.array([[]])  # (n, n) sum of products
        self._sum_weights: np.ndarray = np.array([[]])  # (n, n) sum of weights
        
        # Caches
        self._correlation_cache: Optional[np.ndarray] = None
        self._update_count: int = 0
        
    def _ensure_capacity(self, min_size: int) -> None:
        """Ensure arrays have capacity for at least min_size markets."""
        current_size = len(self._n_obs)
        if current_size >= min_size:
            return
        
        new_size = max(min_size, current_size * 2, 16)
        
        # Resize 1D arrays
        new_n_obs = np.zeros(new_size)
        new_sum_x = np.zeros(new_size)
        new_sum_xx = np.zeros(new_size)
        
        if current_size > 0:
            new_n_obs[:current_size] = self._n_obs
            new_sum_x[:current_size] = self._sum_x
            new_sum_xx[:current_size] = self._sum_xx
        
        self._n_obs = new_n_obs
        self._sum_x = new_sum_x
        self._sum_xx = new_sum_xx
        
        # Resize 2D arrays
        new_sum_xy = np.zeros((new_size, new_size))
        new_sum_weights = np.zeros((new_size, new_size))
        
        if current_size > 0:
            new_sum_xy[:current_size, :current_size] = self._sum_xy
            new_sum_weights[:current_size, :current_size] = self._sum_weights
        
        self._sum_xy = new_sum_xy
        self._sum_weights = new_sum_weights
    
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Initialize correlation tracking for new market."""
        idx = self._next_index
        self._next_index += 1
        
        self._market_index[market_id] = idx
        self._index_to_market[idx] = market_id
        
        self._ensure_capacity(idx + 1)
        
        # Invalidate cache
        self._correlation_cache = None
    
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """Update survival model and handle market death."""
        # Update survival model with this observation
        obs = SurvivalObservation(
            market_id=market_id,
            duration=state.lifetime or 0.0,
            event=True,  # Market resolved
            features={"category": state.category, **state.features},
        )
        
        if isinstance(self.survival_model, AdaptiveSurvival):
            self.survival_model.update(obs)
        
        # Invalidate cache
        self._correlation_cache = None
    
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """
        Update correlation estimates with new price observations.
        
        Args:
            timestamp: Current timestamp
            prices: Dict mapping market_id -> current price
        """
        self._current_time = timestamp
        self._update_count += 1
        
        # Update price histories
        for market_id, price in prices.items():
            if market_id in self._markets:
                self._markets[market_id].add_price(timestamp, price)
        
        # Compute values to use (returns or prices)
        active_ids = []
        values = []
        survival_weights = []
        
        for market_id, price in prices.items():
            if market_id not in self._markets:
                continue
            
            state = self._markets[market_id]
            if not state.is_active:
                continue
            
            # Get value for correlation
            if self.cfg.use_returns and len(state.price_history) >= 2:
                prev_price = state.price_history[-2][1]
                prev_price = np.clip(prev_price, 0.01, 0.99)
                price = np.clip(price, 0.01, 0.99)
                val = np.log(price) - np.log(prev_price)
            else:
                val = price
            
            # Get survival weight
            if self.cfg.use_survival_weights:
                age = timestamp - state.birth_time
                features = {"category": state.category, **state.features}
                weight = self.survival_model.survival_weight(
                    age, self.cfg.survival_horizon, features
                )
            else:
                weight = 1.0
            
            active_ids.append(market_id)
            values.append(val)
            survival_weights.append(weight)
        
        if len(active_ids) < 2:
            return
        
        # Update sufficient statistics
        values = np.array(values)
        survival_weights = np.array(survival_weights)
        
        for i, mid_i in enumerate(active_ids):
            idx_i = self._market_index[mid_i]
            w_i = survival_weights[i]
            x_i = values[i]
            
            # Update marginal statistics (EMA-style)
            alpha = self.cfg.ema_alpha
            self._n_obs[idx_i] += 1
            self._sum_x[idx_i] = (1 - alpha) * self._sum_x[idx_i] + alpha * x_i
            self._sum_xx[idx_i] = (1 - alpha) * self._sum_xx[idx_i] + alpha * x_i * x_i
            
            # Update pairwise statistics
            for j, mid_j in enumerate(active_ids):
                if j <= i:
                    continue
                
                idx_j = self._market_index[mid_j]
                w_j = survival_weights[j]
                x_j = values[j]
                
                # Joint weight
                w_ij = w_i * w_j
                
                # Update weighted sum of products
                self._sum_xy[idx_i, idx_j] = (
                    (1 - alpha * w_ij) * self._sum_xy[idx_i, idx_j]
                    + alpha * w_ij * x_i * x_j
                )
                self._sum_xy[idx_j, idx_i] = self._sum_xy[idx_i, idx_j]
                
                self._sum_weights[idx_i, idx_j] = (
                    (1 - alpha) * self._sum_weights[idx_i, idx_j]
                    + alpha * w_ij
                )
                self._sum_weights[idx_j, idx_i] = self._sum_weights[idx_i, idx_j]
        
        # Invalidate cache
        self._correlation_cache = None
        
        # Periodically recluster
        if self._update_count % self.cfg.recluster_every == 0:
            self._update_clusters()
    
    def _compute_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Compute correlation matrix from sufficient statistics.
        
        Returns:
            (market_ids, correlation_matrix) for active markets with enough data
        """
        # Get active markets with enough observations
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
        
        n = len(active_ids)
        if n == 0:
            return [], np.array([[]])
        
        # Build correlation matrix
        corr = np.eye(n)
        
        for i in range(n):
            idx_i = active_indices[i]
            var_i = self._sum_xx[idx_i] - self._sum_x[idx_i] ** 2
            
            for j in range(i + 1, n):
                idx_j = active_indices[j]
                var_j = self._sum_xx[idx_j] - self._sum_x[idx_j] ** 2
                
                # Covariance
                cov_ij = self._sum_xy[idx_i, idx_j] - self._sum_x[idx_i] * self._sum_x[idx_j]
                
                # Correlation
                if var_i > 1e-10 and var_j > 1e-10:
                    rho = cov_ij / np.sqrt(var_i * var_j)
                    rho = np.clip(rho, -1.0, 1.0)
                else:
                    rho = 0.0
                
                corr[i, j] = rho
                corr[j, i] = rho
        
        # Apply shrinkage toward identity
        if self.cfg.shrinkage > 0:
            corr = (1 - self.cfg.shrinkage) * corr + self.cfg.shrinkage * np.eye(n)
        
        return active_ids, corr
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Get current correlation matrix estimate."""
        if self._correlation_cache is not None:
            return self._correlation_cache
        
        result = self._compute_correlation_matrix()
        self._correlation_cache = result
        return result
    
    def _update_clusters(self) -> None:
        """Update cluster assignments based on current correlations."""
        market_ids, corr = self.get_correlation_matrix()
        n = len(market_ids)
        
        if n < 2:
            # Single market or none - assign to cluster 0
            for mid in market_ids:
                cluster_id = 0
                if cluster_id not in self._clusters:
                    self._clusters[cluster_id] = ClusterState(cluster_id=cluster_id)
                
                self._clusters[cluster_id].add_member(mid)
                self._assignments[mid] = ClusterAssignment(
                    market_id=mid,
                    cluster_id=cluster_id,
                    confidence=1.0,
                )
            return
        
        # Convert correlation to distance
        # Using 1 - |corr| so highly correlated markets are close
        dist_matrix = 1 - np.abs(corr)
        np.fill_diagonal(dist_matrix, 0)
        
        # Ensure symmetry and valid distance
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        # Convert to condensed form for scipy
        dist_condensed = squareform(dist_matrix, checks=False)
        
        # Hierarchical clustering
        try:
            Z = linkage(dist_condensed, method='average')
        except Exception:
            # Fallback: all in one cluster
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
            cluster_id = int(label) - 1  # fcluster uses 1-based labels
            
            if cluster_id not in self._clusters:
                self._clusters[cluster_id] = ClusterState(cluster_id=cluster_id)
            
            self._clusters[cluster_id].add_member(mid)
            self._assignments[mid] = ClusterAssignment(
                market_id=mid,
                cluster_id=cluster_id,
                confidence=1.0,
            )
    
    def _assign_all_to_one_cluster(self, market_ids: List[str]) -> None:
        """Fallback: assign all markets to cluster 0."""
        cluster_id = 0
        if cluster_id not in self._clusters:
            self._clusters[cluster_id] = ClusterState(cluster_id=cluster_id)
        
        self._clusters[cluster_id].member_ids.clear()
        
        for mid in market_ids:
            self._clusters[cluster_id].add_member(mid)
            self._assignments[mid] = ClusterAssignment(
                market_id=mid,
                cluster_id=cluster_id,
                confidence=1.0,
            )
    
    def get_intra_cluster_correlation(self, cluster_id: int) -> float:
        """Get average correlation within a cluster."""
        members = self.get_markets_in_cluster(cluster_id)
        if len(members) < 2:
            return 1.0
        
        market_ids, corr = self.get_correlation_matrix()
        id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
        
        total_corr = 0.0
        count = 0
        
        for i, mid_i in enumerate(members):
            if mid_i not in id_to_idx:
                continue
            idx_i = id_to_idx[mid_i]
            
            for j, mid_j in enumerate(members):
                if j <= i or mid_j not in id_to_idx:
                    continue
                idx_j = id_to_idx[mid_j]
                
                total_corr += corr[idx_i, idx_j]
                count += 1
        
        return total_corr / count if count > 0 else 1.0
    
    def get_inter_cluster_correlation(
        self,
        cluster_id_1: int,
        cluster_id_2: int,
    ) -> float:
        """Get average correlation between two clusters."""
        members_1 = self.get_markets_in_cluster(cluster_id_1)
        members_2 = self.get_markets_in_cluster(cluster_id_2)
        
        if not members_1 or not members_2:
            return 0.0
        
        market_ids, corr = self.get_correlation_matrix()
        id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
        
        total_corr = 0.0
        count = 0
        
        for mid_i in members_1:
            if mid_i not in id_to_idx:
                continue
            idx_i = id_to_idx[mid_i]
            
            for mid_j in members_2:
                if mid_j not in id_to_idx:
                    continue
                idx_j = id_to_idx[mid_j]
                
                total_corr += corr[idx_i, idx_j]
                count += 1
        
        return total_corr / count if count > 0 else 0.0
    
    def fit(
        self,
        observations: List[SurvivalObservation],
    ) -> None:
        """
        Fit survival model from historical data.
        
        Call before streaming updates to initialize survival weights.
        """
        self.survival_model.fit(observations)
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self._market_index.clear()
        self._index_to_market.clear()
        self._next_index = 0
        self._n_obs = np.array([])
        self._sum_x = np.array([])
        self._sum_xx = np.array([])
        self._sum_xy = np.array([[]])
        self._sum_weights = np.array([[]])
        self._correlation_cache = None
        self._update_count = 0
