"""
Streaming Dirichlet Process Mixture Model (SDPM) for Market Clustering.

Uses Bayesian nonparametrics where clusters can grow/shrink as markets
appear and disappear. The Chinese Restaurant Process (CRP) prior naturally
handles varying cluster counts.

Key advantages:
- No need to specify number of clusters
- Principled uncertainty quantification
- Natural handling of market birth/death
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from scipy.special import logsumexp

from forecastbench.clustering.base import (
    OnlineClusteringBase,
    MarketState,
    ClusterState,
    ClusterAssignment,
)


@dataclass
class DPClusterStats:
    """Sufficient statistics for a cluster in the DP mixture."""
    
    cluster_id: int
    member_ids: Set[str] = field(default_factory=set)
    
    # Sufficient statistics for Gaussian likelihood
    n: int = 0
    sum_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sum_xx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Cached parameters
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    precision: float = 1.0
    
    @property
    def size(self) -> int:
        return len(self.member_ids)
    
    def add_observation(self, x: np.ndarray) -> None:
        """Add observation to sufficient statistics."""
        if self.sum_x.size == 0:
            d = len(x)
            self.sum_x = np.zeros(d)
            self.sum_xx = np.zeros(d)
            self.mean = np.zeros(d)
        
        self.n += 1
        self.sum_x += x
        self.sum_xx += x * x
        self._update_params()
    
    def remove_observation(self, x: np.ndarray) -> None:
        """Remove observation from sufficient statistics."""
        if self.n <= 0:
            return
        
        self.n -= 1
        self.sum_x -= x
        self.sum_xx -= x * x
        self._update_params()
    
    def _update_params(self) -> None:
        """Update cached parameters."""
        if self.n > 0:
            self.mean = self.sum_x / self.n
            variance = self.sum_xx / self.n - self.mean ** 2
            variance = np.maximum(variance, 1e-6)
            self.precision = 1.0 / np.mean(variance)
        else:
            self.mean = np.zeros_like(self.mean)
            self.precision = 1.0
    
    def log_predictive(self, x: np.ndarray, prior_precision: float = 0.1) -> float:
        """
        Compute log predictive probability of x given cluster.
        
        Uses Normal-Normal conjugacy with known precision.
        """
        if self.n == 0:
            # Prior: N(0, 1/prior_precision)
            return -0.5 * prior_precision * np.sum(x ** 2)
        
        # Posterior predictive
        d = len(x)
        diff = x - self.mean
        
        # Precision weighted by number of observations
        effective_precision = self.precision * self.n / (self.n + 1)
        
        log_prob = -0.5 * effective_precision * np.sum(diff ** 2)
        log_prob -= 0.5 * d * np.log(2 * np.pi / effective_precision)
        
        return log_prob


@dataclass
class SDPMConfig:
    """Configuration for Streaming DP Mixture."""
    
    # DP concentration parameter
    # Higher alpha = more clusters
    concentration: float = 1.0
    
    # Prior parameters
    prior_mean: float = 0.0
    prior_precision: float = 0.1
    
    # Feature extraction
    feature_dim: int = 10  # Dimension of market features
    use_returns: bool = True
    return_window: int = 20  # Window for computing return features
    
    # Online Gibbs sampling
    n_gibbs_sweeps: int = 1  # Sweeps per update
    temperature: float = 1.0  # Annealing temperature
    
    # Cluster management
    min_cluster_size: int = 1
    merge_threshold: float = 0.9  # Merge if similarity > threshold
    
    # Update frequency
    reassign_every: int = 10


class StreamingDPClustering(OnlineClusteringBase):
    """
    Streaming Dirichlet Process Mixture Model.
    
    Uses the Chinese Restaurant Process (CRP) metaphor:
    - Existing clusters are "tables" with customers (markets)
    - New markets choose tables proportional to occupancy
    - Or start a new table with probability proportional to alpha
    
    When a market dies, it simply leaves its table. Empty tables
    are removed automatically.
    
    Example:
        sdpm = StreamingDPClustering(SDPMConfig(concentration=1.0))
        
        # Markets arrive
        sdpm.add_market("A", features={"return_mean": 0.01})
        sdpm.add_market("B", features={"return_mean": 0.02})
        
        # Update with observations
        sdpm.update(timestamp=1.0, prices={"A": 0.5, "B": 0.6})
        
        # Market dies
        sdpm.remove_market("A")
        
        # Get clusters
        clusters = sdpm.get_clusters()
    """
    
    def __init__(self, config: Optional[SDPMConfig] = None):
        super().__init__(config.__dict__ if config else None)
        self.cfg = config or SDPMConfig()
        
        # Cluster statistics (indexed by cluster_id)
        self._cluster_stats: Dict[int, DPClusterStats] = {}
        
        # Market features
        self._market_features: Dict[str, np.ndarray] = {}
        self._feature_history: Dict[str, List[np.ndarray]] = {}
        
        # For feature extraction
        self._prev_prices: Dict[str, float] = {}
        self._return_buffer: Dict[str, List[float]] = {}
        
        self._update_count: int = 0
    
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Assign new market to a cluster using CRP."""
        self._prev_prices[market_id] = state.last_price
        self._return_buffer[market_id] = []
        
        # Initialize with zero features
        features = np.zeros(self.cfg.feature_dim)
        self._market_features[market_id] = features
        self._feature_history[market_id] = [features]
        
        # Assign using CRP
        cluster_id = self._crp_assign(market_id, features)
        
        # Update assignments
        self._assignments[market_id] = ClusterAssignment(
            market_id=market_id,
            cluster_id=cluster_id,
            confidence=1.0,
        )
    
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """Remove market from its cluster."""
        if market_id not in self._assignments:
            return
        
        cluster_id = self._assignments[market_id].cluster_id
        
        # Remove from cluster stats
        if cluster_id in self._cluster_stats:
            stats = self._cluster_stats[cluster_id]
            stats.member_ids.discard(market_id)
            
            # Remove feature contribution
            if market_id in self._market_features:
                features = self._market_features[market_id]
                stats.remove_observation(features)
            
            # Delete empty cluster
            if stats.size == 0:
                del self._cluster_stats[cluster_id]
                if cluster_id in self._clusters:
                    del self._clusters[cluster_id]
        
        # Clean up market data
        self._market_features.pop(market_id, None)
        self._feature_history.pop(market_id, None)
        self._prev_prices.pop(market_id, None)
        self._return_buffer.pop(market_id, None)
    
    def _crp_assign(
        self,
        market_id: str,
        features: np.ndarray,
    ) -> int:
        """
        Assign market to cluster using Chinese Restaurant Process.
        
        P(cluster k) ∝ n_k * likelihood(x | cluster k)
        P(new cluster) ∝ alpha * prior(x)
        """
        n_total = sum(s.size for s in self._cluster_stats.values())
        
        # Compute log probabilities for each existing cluster
        log_probs = []
        cluster_ids = []
        
        for cid, stats in self._cluster_stats.items():
            if stats.size == 0:
                continue
            
            # CRP prior: log(n_k)
            log_prior = np.log(max(stats.size, 1e-10))
            
            # Likelihood
            log_lik = stats.log_predictive(features, self.cfg.prior_precision)
            
            # Clip to avoid extreme values
            log_lik = np.clip(log_lik, -100, 100)
            
            log_probs.append((log_prior + log_lik) / self.cfg.temperature)
            cluster_ids.append(cid)
        
        # New cluster probability
        log_new = np.log(max(self.cfg.concentration, 1e-10))
        feature_sq = np.sum((features - self.cfg.prior_mean) ** 2)
        feature_sq = np.clip(feature_sq, 0, 1000)  # Prevent extreme values
        log_new += -0.5 * self.cfg.prior_precision * feature_sq
        log_new /= self.cfg.temperature
        
        log_probs.append(log_new)
        cluster_ids.append(-1)  # Sentinel for new cluster
        
        # Normalize and sample
        log_probs = np.array(log_probs)
        
        # Handle NaN/Inf
        log_probs = np.nan_to_num(log_probs, nan=-100, posinf=100, neginf=-100)
        
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # Ensure valid probability distribution
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()
        
        chosen_idx = np.random.choice(len(probs), p=probs)
        chosen_cluster = cluster_ids[chosen_idx]
        
        if chosen_cluster == -1:
            # Create new cluster
            new_id = self._next_cluster_id
            self._next_cluster_id += 1
            
            self._cluster_stats[new_id] = DPClusterStats(cluster_id=new_id)
            self._clusters[new_id] = ClusterState(cluster_id=new_id)
            
            chosen_cluster = new_id
        
        # Add to cluster
        self._cluster_stats[chosen_cluster].member_ids.add(market_id)
        self._cluster_stats[chosen_cluster].add_observation(features)
        self._clusters[chosen_cluster].add_member(market_id)
        
        return chosen_cluster
    
    def _extract_features(self, market_id: str) -> np.ndarray:
        """Extract feature vector for a market."""
        returns = self._return_buffer.get(market_id, [])
        
        if len(returns) < 2:
            return np.zeros(self.cfg.feature_dim)
        
        returns = np.array(returns[-self.cfg.return_window:])
        
        # Feature vector: moments and autocorrelations
        features = []
        
        # Mean
        features.append(np.mean(returns))
        
        # Std
        features.append(np.std(returns))
        
        # Skewness
        if np.std(returns) > 1e-10:
            features.append(np.mean((returns - np.mean(returns)) ** 3) / np.std(returns) ** 3)
        else:
            features.append(0.0)
        
        # Kurtosis
        if np.std(returns) > 1e-10:
            features.append(np.mean((returns - np.mean(returns)) ** 4) / np.std(returns) ** 4)
        else:
            features.append(3.0)
        
        # Autocorrelations
        for lag in [1, 2, 3]:
            if len(returns) > lag:
                ac = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                features.append(ac if not np.isnan(ac) else 0.0)
            else:
                features.append(0.0)
        
        # Pad or truncate to feature_dim
        features = np.array(features)
        if len(features) < self.cfg.feature_dim:
            features = np.pad(features, (0, self.cfg.feature_dim - len(features)))
        else:
            features = features[:self.cfg.feature_dim]
        
        return features
    
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """Update with new price observations."""
        self._current_time = timestamp
        self._update_count += 1
        
        # Update price histories and extract features
        for market_id, price in prices.items():
            if market_id not in self._markets:
                continue
            if not self._markets[market_id].is_active:
                continue
            
            self._markets[market_id].add_price(timestamp, price)
            
            # Compute return
            prev_price = self._prev_prices.get(market_id, price)
            prev_price = np.clip(prev_price, 0.01, 0.99)
            price_clipped = np.clip(price, 0.01, 0.99)
            
            if self.cfg.use_returns:
                ret = np.log(price_clipped) - np.log(prev_price)
            else:
                ret = price_clipped - prev_price
            
            self._return_buffer.setdefault(market_id, []).append(ret)
            
            # Keep buffer bounded
            if len(self._return_buffer[market_id]) > self.cfg.return_window * 2:
                self._return_buffer[market_id] = self._return_buffer[market_id][-self.cfg.return_window:]
            
            self._prev_prices[market_id] = price
            
            # Update features
            new_features = self._extract_features(market_id)
            old_features = self._market_features.get(market_id, new_features)
            
            # Update cluster stats
            if market_id in self._assignments:
                cluster_id = self._assignments[market_id].cluster_id
                if cluster_id in self._cluster_stats:
                    self._cluster_stats[cluster_id].remove_observation(old_features)
                    self._cluster_stats[cluster_id].add_observation(new_features)
            
            self._market_features[market_id] = new_features
            self._feature_history.setdefault(market_id, []).append(new_features)
        
        # Periodically reassign using Gibbs sampling
        if self._update_count % self.cfg.reassign_every == 0:
            self._gibbs_sweep()
    
    def _gibbs_sweep(self) -> None:
        """
        Perform Gibbs sampling sweep over all markets.
        
        For each market, remove it from its cluster and reassign
        using the CRP posterior.
        """
        active_markets = [
            mid for mid in self._markets
            if self._markets[mid].is_active and mid in self._market_features
        ]
        
        # Random order
        np.random.shuffle(active_markets)
        
        for _ in range(self.cfg.n_gibbs_sweeps):
            for market_id in active_markets:
                if market_id not in self._assignments:
                    continue
                
                old_cluster = self._assignments[market_id].cluster_id
                features = self._market_features[market_id]
                
                # Remove from current cluster
                if old_cluster in self._cluster_stats:
                    stats = self._cluster_stats[old_cluster]
                    stats.member_ids.discard(market_id)
                    stats.remove_observation(features)
                    
                    if old_cluster in self._clusters:
                        self._clusters[old_cluster].remove_member(market_id)
                    
                    # Remove empty cluster
                    if stats.size == 0:
                        del self._cluster_stats[old_cluster]
                        if old_cluster in self._clusters:
                            del self._clusters[old_cluster]
                
                # Reassign
                new_cluster = self._crp_assign(market_id, features)
                
                self._assignments[market_id] = ClusterAssignment(
                    market_id=market_id,
                    cluster_id=new_cluster,
                    confidence=1.0,
                )
    
    def get_cluster_parameters(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get estimated parameters for a cluster."""
        if cluster_id not in self._cluster_stats:
            return None
        
        stats = self._cluster_stats[cluster_id]
        
        return {
            "mean": stats.mean.tolist(),
            "precision": stats.precision,
            "n_observations": stats.n,
            "size": stats.size,
        }
    
    def get_assignment_probabilities(
        self,
        market_id: str,
    ) -> Dict[int, float]:
        """Get soft assignment probabilities for a market."""
        if market_id not in self._market_features:
            return {}
        
        features = self._market_features[market_id]
        
        # Compute probabilities for all clusters
        log_probs = {}
        
        for cid, stats in self._cluster_stats.items():
            if stats.size == 0:
                continue
            
            log_prior = np.log(stats.size)
            log_lik = stats.log_predictive(features, self.cfg.prior_precision)
            log_probs[cid] = log_prior + log_lik
        
        # Add new cluster
        log_probs[-1] = np.log(self.cfg.concentration) - 0.5 * self.cfg.prior_precision * np.sum(
            (features - self.cfg.prior_mean) ** 2
        )
        
        # Normalize
        log_total = logsumexp(list(log_probs.values()))
        probs = {cid: np.exp(lp - log_total) for cid, lp in log_probs.items()}
        
        # Remove new cluster from output
        probs.pop(-1, None)
        
        return probs
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Estimate correlation based on cluster co-membership probability.
        
        Markets in the same cluster have correlation 1, different clusters 0.
        Soft assignments give intermediate values.
        """
        active_ids = self.active_market_ids
        n = len(active_ids)
        
        if n == 0:
            return [], np.array([[]])
        
        # Build correlation from soft assignments
        corr = np.eye(n)
        
        for i, mid_i in enumerate(active_ids):
            probs_i = self.get_assignment_probabilities(mid_i)
            
            for j in range(i + 1, n):
                mid_j = active_ids[j]
                probs_j = self.get_assignment_probabilities(mid_j)
                
                # Correlation = probability of being in same cluster
                same_cluster_prob = sum(
                    probs_i.get(cid, 0) * probs_j.get(cid, 0)
                    for cid in set(probs_i.keys()) | set(probs_j.keys())
                )
                
                corr[i, j] = same_cluster_prob
                corr[j, i] = same_cluster_prob
        
        return active_ids, corr
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self._cluster_stats.clear()
        self._market_features.clear()
        self._feature_history.clear()
        self._prev_prices.clear()
        self._return_buffer.clear()
        self._update_count = 0
