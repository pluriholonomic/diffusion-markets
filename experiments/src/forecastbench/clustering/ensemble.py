"""
Ensemble Clustering Methods.

Combines multiple clustering algorithms to produce more robust
cluster assignments and correlation estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class EnsembleConfig:
    """Configuration for ensemble clustering."""
    
    # Aggregation method
    aggregation: str = "average"  # "average", "vote", "stack"
    
    # Clustering from aggregated similarity
    n_clusters: Optional[int] = None
    distance_threshold: float = 0.5
    
    # Update frequency
    recluster_every: int = 10


class EnsembleClustering(OnlineClusteringBase):
    """
    Ensemble clustering combining multiple algorithms.
    
    Aggregates correlation estimates from multiple base algorithms
    to produce more robust cluster assignments.
    
    Methods:
    - average: Average correlation matrices
    - vote: Majority voting on cluster assignments
    - stack: Learn optimal weighting of algorithms
    
    Example:
        from forecastbench.clustering import SWOCC, OLRCM
        
        base_algorithms = [
            SWOCC(),
            OLRCM(),
        ]
        
        ensemble = EnsembleClustering(base_algorithms)
        
        for t, prices in enumerate(price_stream):
            ensemble.update(timestamp=t, prices=prices)
        
        clusters = ensemble.get_clusters()
    """
    
    def __init__(
        self,
        base_algorithms: List[OnlineClusteringBase],
        config: Optional[EnsembleConfig] = None,
    ):
        super().__init__(config.__dict__ if config else None)
        self.cfg = config or EnsembleConfig()
        self.base_algorithms = base_algorithms
        
        self._update_count = 0
    
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Add market to all base algorithms."""
        for algo in self.base_algorithms:
            algo.add_market(
                market_id=market_id,
                timestamp=state.birth_time,
                initial_price=state.last_price,
                category=state.category,
                features=state.features,
            )
    
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """Remove market from all base algorithms."""
        for algo in self.base_algorithms:
            algo.remove_market(market_id, timestamp=state.death_time, outcome=outcome)
    
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """Update all base algorithms."""
        self._current_time = timestamp
        self._update_count += 1
        
        # Update price histories
        for market_id, price in prices.items():
            if market_id in self._markets:
                self._markets[market_id].add_price(timestamp, price)
        
        # Update all base algorithms
        for algo in self.base_algorithms:
            algo.update(timestamp=timestamp, prices=prices)
        
        # Periodically update ensemble clusters
        if self._update_count % self.cfg.recluster_every == 0:
            self._update_clusters()
    
    def _aggregate_correlations(self) -> Tuple[List[str], np.ndarray]:
        """
        Aggregate correlation matrices from base algorithms.
        """
        all_market_ids = set()
        correlations = []
        
        for algo in self.base_algorithms:
            market_ids, corr = algo.get_correlation_matrix()
            if len(market_ids) > 0:
                all_market_ids.update(market_ids)
                correlations.append((market_ids, corr))
        
        if not correlations:
            return [], np.array([[]])
        
        # Get common market IDs
        common_ids = list(all_market_ids)
        n = len(common_ids)
        
        if n == 0:
            return [], np.array([[]])
        
        id_to_idx = {mid: i for i, mid in enumerate(common_ids)}
        
        # Aggregate correlations
        if self.cfg.aggregation == "average":
            agg_corr = np.zeros((n, n))
            counts = np.zeros((n, n))
            
            for market_ids, corr in correlations:
                for i, mid_i in enumerate(market_ids):
                    for j, mid_j in enumerate(market_ids):
                        idx_i = id_to_idx.get(mid_i)
                        idx_j = id_to_idx.get(mid_j)
                        
                        if idx_i is not None and idx_j is not None:
                            agg_corr[idx_i, idx_j] += corr[i, j]
                            counts[idx_i, idx_j] += 1
            
            # Average
            counts = np.maximum(counts, 1)
            agg_corr = agg_corr / counts
            
            # Ensure valid correlation matrix
            np.fill_diagonal(agg_corr, 1.0)
            agg_corr = np.clip(agg_corr, -1, 1)
            
        elif self.cfg.aggregation == "vote":
            # Use binary co-clustering indicator
            agg_corr = np.zeros((n, n))
            
            for algo in self.base_algorithms:
                clusters = algo.get_clusters()
                for cluster_id, members in clusters.items():
                    for mid_i in members:
                        for mid_j in members:
                            idx_i = id_to_idx.get(mid_i)
                            idx_j = id_to_idx.get(mid_j)
                            
                            if idx_i is not None and idx_j is not None:
                                agg_corr[idx_i, idx_j] += 1
            
            # Normalize
            agg_corr = agg_corr / len(self.base_algorithms)
            np.fill_diagonal(agg_corr, 1.0)
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.cfg.aggregation}")
        
        return common_ids, agg_corr
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Get aggregated correlation matrix."""
        return self._aggregate_correlations()
    
    def _update_clusters(self) -> None:
        """Update clusters based on aggregated correlations."""
        market_ids, corr = self._aggregate_correlations()
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
        
        # Convert to distance
        dist = 1 - np.abs(corr)
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
            
            # Compute confidence from agreement across algorithms
            agreements = 0
            for algo in self.base_algorithms:
                algo_cluster = algo.get_cluster_for_market(mid)
                if algo_cluster is not None:
                    # Check if same markets are in same cluster
                    algo_members = set(algo.get_markets_in_cluster(algo_cluster))
                    our_members = set(self._clusters[cluster_id].member_ids)
                    if len(algo_members & our_members) > 1:
                        agreements += 1
            
            confidence = agreements / len(self.base_algorithms) if self.base_algorithms else 1.0
            
            self._assignments[mid] = ClusterAssignment(
                market_id=mid,
                cluster_id=cluster_id,
                confidence=confidence,
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
    
    def get_algorithm_agreement(self) -> float:
        """
        Compute agreement score across base algorithms.
        
        Returns value in [0, 1] where 1 = perfect agreement.
        """
        if len(self.base_algorithms) < 2:
            return 1.0
        
        agreements = []
        
        for i, algo1 in enumerate(self.base_algorithms):
            for algo2 in self.base_algorithms[i+1:]:
                # Compare cluster assignments
                common_markets = set(algo1.active_market_ids) & set(algo2.active_market_ids)
                
                if len(common_markets) < 2:
                    continue
                
                # Count pair agreements
                same = 0
                total = 0
                
                markets = list(common_markets)
                for j, mid1 in enumerate(markets):
                    c1_1 = algo1.get_cluster_for_market(mid1)
                    c2_1 = algo2.get_cluster_for_market(mid1)
                    
                    for mid2 in markets[j+1:]:
                        c1_2 = algo1.get_cluster_for_market(mid2)
                        c2_2 = algo2.get_cluster_for_market(mid2)
                        
                        if c1_1 is not None and c1_2 is not None and c2_1 is not None and c2_2 is not None:
                            # Same cluster in algo1 iff same cluster in algo2
                            if (c1_1 == c1_2) == (c2_1 == c2_2):
                                same += 1
                            total += 1
                
                if total > 0:
                    agreements.append(same / total)
        
        return np.mean(agreements) if agreements else 1.0
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        for algo in self.base_algorithms:
            algo.reset()
        self._update_count = 0
