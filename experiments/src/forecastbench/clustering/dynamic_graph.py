"""
Dynamic Graph Attention Network (DGAT) for Market Correlation Learning.

Maintains a graph where nodes are active markets and edges represent
correlations. Uses attention mechanisms to learn edge weights that
predict future co-movements.

Note: This is a simplified numpy-based implementation. For production
use with large datasets, consider PyTorch Geometric or DGL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
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
class DGATConfig:
    """Configuration for Dynamic Graph Attention Network."""
    
    # Node features
    node_feature_dim: int = 16
    hidden_dim: int = 32
    n_attention_heads: int = 4
    
    # Edge features
    edge_feature_dim: int = 8
    
    # Learning
    learning_rate: float = 0.01
    momentum: float = 0.9
    l2_reg: float = 0.001
    
    # Attention
    attention_dropout: float = 0.1
    leaky_relu_slope: float = 0.2
    
    # Graph construction
    k_neighbors: int = 10  # Connect to k most similar nodes
    min_edge_weight: float = 0.01
    
    # Clustering
    n_clusters: Optional[int] = None
    distance_threshold: float = 0.5
    
    # Update
    recluster_every: int = 10
    feature_window: int = 20


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def leaky_relu(x: np.ndarray, slope: float = 0.2) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, slope * x)


class DynamicGraphAttention(OnlineClusteringBase):
    """
    Dynamic Graph Attention Network for market clustering.
    
    Maintains a dynamic graph where:
    - Nodes = active markets with feature vectors
    - Edges = potential correlations with learned weights
    - Attention = mechanism to weight neighbor information
    
    When a market resolves, its node and all incident edges are removed.
    New markets are added as new nodes.
    
    Clustering is performed on the learned node embeddings or
    attention-weighted adjacency matrix.
    
    Example:
        dgat = DynamicGraphAttention(DGATConfig(hidden_dim=32))
        
        for t, prices in enumerate(price_stream):
            dgat.update(timestamp=t, prices=prices)
        
        # Get clusters based on learned embeddings
        clusters = dgat.get_clusters()
        
        # Get attention weights between markets
        attn = dgat.get_attention_matrix()
    """
    
    def __init__(self, config: Optional[DGATConfig] = None):
        super().__init__(config.__dict__ if config else None)
        self.cfg = config or DGATConfig()
        
        # Node indexing
        self._market_index: Dict[str, int] = {}
        self._index_to_market: Dict[int, str] = {}
        self._next_index: int = 0
        
        # Node features and embeddings
        self._node_features: np.ndarray = np.array([[]])  # (n, node_feature_dim)
        self._node_embeddings: np.ndarray = np.array([[]])  # (n, hidden_dim)
        
        # Adjacency and attention
        self._adjacency: np.ndarray = np.array([[]])  # (n, n) binary
        self._attention_weights: np.ndarray = np.array([[]])  # (n, n, n_heads)
        
        # Model parameters
        self._W_node: np.ndarray = np.random.randn(
            self.cfg.node_feature_dim, self.cfg.hidden_dim
        ) * 0.1
        self._W_attn: np.ndarray = np.random.randn(
            self.cfg.n_attention_heads, 2 * self.cfg.hidden_dim
        ) * 0.1
        
        # Momentum for updates
        self._W_node_momentum: np.ndarray = np.zeros_like(self._W_node)
        self._W_attn_momentum: np.ndarray = np.zeros_like(self._W_attn)
        
        # Feature extraction
        self._prev_prices: Dict[str, float] = {}
        self._return_buffer: Dict[str, List[float]] = {}
        
        self._update_count: int = 0
    
    def _ensure_capacity(self, min_size: int) -> None:
        """Ensure arrays have capacity for at least min_size nodes."""
        current_size = self._node_features.shape[0] if self._node_features.size > 0 else 0
        
        if current_size >= min_size:
            return
        
        new_size = max(min_size, current_size * 2, 16)
        
        # Node features
        new_features = np.zeros((new_size, self.cfg.node_feature_dim))
        new_embeddings = np.zeros((new_size, self.cfg.hidden_dim))
        
        if current_size > 0:
            new_features[:current_size] = self._node_features
            new_embeddings[:current_size] = self._node_embeddings
        
        self._node_features = new_features
        self._node_embeddings = new_embeddings
        
        # Adjacency and attention
        new_adj = np.zeros((new_size, new_size))
        new_attn = np.zeros((new_size, new_size, self.cfg.n_attention_heads))
        
        if current_size > 0:
            new_adj[:current_size, :current_size] = self._adjacency
            new_attn[:current_size, :current_size] = self._attention_weights
        
        self._adjacency = new_adj
        self._attention_weights = new_attn
    
    def _on_market_added(self, market_id: str, state: MarketState) -> None:
        """Add node to graph."""
        idx = self._next_index
        self._next_index += 1
        
        self._market_index[market_id] = idx
        self._index_to_market[idx] = market_id
        
        self._ensure_capacity(idx + 1)
        
        # Initialize node features
        self._node_features[idx] = np.random.randn(self.cfg.node_feature_dim) * 0.01
        
        # Connect to existing nodes (initially sparse)
        active_indices = self._get_active_indices()
        for other_idx in active_indices:
            if other_idx != idx:
                # Initial connection based on random similarity
                self._adjacency[idx, other_idx] = 1
                self._adjacency[other_idx, idx] = 1
        
        self._prev_prices[market_id] = state.last_price
        self._return_buffer[market_id] = []
    
    def _on_market_removed(
        self,
        market_id: str,
        state: MarketState,
        outcome: Optional[int],
    ) -> None:
        """Remove node and incident edges."""
        if market_id not in self._market_index:
            return
        
        idx = self._market_index[market_id]
        
        # Remove edges
        self._adjacency[idx, :] = 0
        self._adjacency[:, idx] = 0
        self._attention_weights[idx, :, :] = 0
        self._attention_weights[:, idx, :] = 0
        
        # Clean up
        self._prev_prices.pop(market_id, None)
        self._return_buffer.pop(market_id, None)
    
    def _get_active_indices(self) -> List[int]:
        """Get indices of active markets."""
        return [
            self._market_index[mid]
            for mid in self._markets
            if self._markets[mid].is_active and mid in self._market_index
        ]
    
    def _extract_node_features(self, market_id: str) -> np.ndarray:
        """Extract feature vector for a market node."""
        returns = self._return_buffer.get(market_id, [])
        state = self._markets.get(market_id)
        
        features = np.zeros(self.cfg.node_feature_dim)
        
        if state is None:
            return features
        
        # Price-based features
        features[0] = state.last_price - 0.5  # Centered price
        features[1] = 2 * abs(state.last_price - 0.5)  # Extremity
        
        if len(returns) >= 2:
            returns_arr = np.array(returns[-self.cfg.feature_window:])
            
            # Return statistics
            features[2] = np.mean(returns_arr)
            features[3] = np.std(returns_arr)
            
            # Recent momentum
            if len(returns_arr) >= 5:
                features[4] = np.mean(returns_arr[-5:])
            
            # Autocorrelation
            if len(returns_arr) >= 3:
                ac = np.corrcoef(returns_arr[:-1], returns_arr[1:])[0, 1]
                features[5] = ac if not np.isnan(ac) else 0
        
        # Age-based features
        age = self._current_time - state.birth_time
        features[6] = np.log1p(age)
        
        return features
    
    def _compute_attention(self, active_indices: List[int]) -> np.ndarray:
        """
        Compute attention weights between active nodes.
        
        Uses scaled dot-product attention with multiple heads.
        """
        n = len(active_indices)
        if n < 2:
            return np.zeros((n, n, self.cfg.n_attention_heads))
        
        # Get embeddings for active nodes
        embeddings = self._node_embeddings[active_indices]  # (n, hidden_dim)
        
        # Compute attention for each head
        attention = np.zeros((n, n, self.cfg.n_attention_heads))
        
        for h in range(self.cfg.n_attention_heads):
            # Attention weights for this head
            w = self._W_attn[h]  # (2 * hidden_dim,)
            
            # Compute attention scores
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Concatenate embeddings
                    concat = np.concatenate([embeddings[i], embeddings[j]])
                    
                    # Compute score
                    score = np.dot(w, concat)
                    score = leaky_relu(score, self.cfg.leaky_relu_slope)
                    
                    attention[i, j, h] = score
            
            # Softmax over neighbors
            for i in range(n):
                mask = np.arange(n) != i
                if np.any(mask):
                    attention[i, mask, h] = softmax(attention[i, mask, h])
        
        return attention
    
    def _graph_attention_layer(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
        attention: np.ndarray,
    ) -> np.ndarray:
        """
        Apply graph attention layer.
        
        h'_i = sigma(sum_j alpha_ij * W * h_j)
        """
        n = features.shape[0]
        hidden_dim = self.cfg.hidden_dim
        n_heads = self.cfg.n_attention_heads
        
        # Transform features
        transformed = features @ self._W_node  # (n, hidden_dim)
        
        # Aggregate using attention
        output = np.zeros((n, hidden_dim))
        
        for i in range(n):
            for h in range(n_heads):
                for j in range(n):
                    if adjacency[i, j] > 0:
                        output[i] += attention[i, j, h] * transformed[j] / n_heads
        
        # Activation
        output = leaky_relu(output, self.cfg.leaky_relu_slope)
        
        return output
    
    def update(
        self,
        timestamp: float,
        prices: Dict[str, float],
    ) -> None:
        """Update graph with new observations."""
        self._current_time = timestamp
        self._update_count += 1
        
        # Update features
        for market_id, price in prices.items():
            if market_id not in self._markets:
                continue
            if not self._markets[market_id].is_active:
                continue
            
            self._markets[market_id].add_price(timestamp, price)
            
            # Compute return
            prev = self._prev_prices.get(market_id, price)
            prev = np.clip(prev, 0.01, 0.99)
            price_clipped = np.clip(price, 0.01, 0.99)
            ret = np.log(price_clipped) - np.log(prev)
            
            self._return_buffer.setdefault(market_id, []).append(ret)
            if len(self._return_buffer[market_id]) > self.cfg.feature_window * 2:
                self._return_buffer[market_id] = self._return_buffer[market_id][-self.cfg.feature_window:]
            
            self._prev_prices[market_id] = price
            
            # Update node features
            idx = self._market_index[market_id]
            self._node_features[idx] = self._extract_node_features(market_id)
        
        # Forward pass and update embeddings
        active_indices = self._get_active_indices()
        if len(active_indices) >= 2:
            self._forward_pass(active_indices)
            self._update_adjacency(active_indices)
        
        # Periodically recluster
        if self._update_count % self.cfg.recluster_every == 0:
            self._update_clusters()
    
    def _forward_pass(self, active_indices: List[int]) -> None:
        """Run forward pass of graph attention network."""
        n = len(active_indices)
        
        # Get active node features
        features = self._node_features[active_indices]
        adj = self._adjacency[np.ix_(active_indices, active_indices)]
        
        # Ensure connectivity
        adj = np.maximum(adj, np.eye(n) * 0.1)
        
        # Compute attention
        attention = self._compute_attention(active_indices)
        
        # Store attention weights
        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                self._attention_weights[idx_i, idx_j] = attention[i, j]
        
        # Apply graph attention layer
        embeddings = self._graph_attention_layer(features, adj, attention)
        
        # Update embeddings
        for i, idx in enumerate(active_indices):
            alpha = 0.1  # EMA update
            self._node_embeddings[idx] = (
                (1 - alpha) * self._node_embeddings[idx]
                + alpha * embeddings[i]
            )
    
    def _update_adjacency(self, active_indices: List[int]) -> None:
        """Update adjacency based on embedding similarity."""
        n = len(active_indices)
        if n < 2:
            return
        
        embeddings = self._node_embeddings[active_indices]
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings_normed = embeddings / norms
        
        # Compute cosine similarity
        similarity = embeddings_normed @ embeddings_normed.T
        
        # Update adjacency (keep k nearest neighbors)
        for i, idx_i in enumerate(active_indices):
            # Get k nearest neighbors
            sims = similarity[i].copy()
            sims[i] = -np.inf  # Exclude self
            
            top_k = np.argsort(sims)[-self.cfg.k_neighbors:]
            
            for j, idx_j in enumerate(active_indices):
                if j in top_k and sims[j] > self.cfg.min_edge_weight:
                    self._adjacency[idx_i, idx_j] = 1
                else:
                    self._adjacency[idx_i, idx_j] = 0
    
    def _update_clusters(self) -> None:
        """Update clusters based on node embeddings."""
        active_indices = self._get_active_indices()
        n = len(active_indices)
        
        if n < 2:
            for idx in active_indices:
                mid = self._index_to_market[idx]
                if 0 not in self._clusters:
                    self._clusters[0] = ClusterState(cluster_id=0)
                self._clusters[0].add_member(mid)
                self._assignments[mid] = ClusterAssignment(
                    market_id=mid, cluster_id=0, confidence=1.0
                )
            return
        
        # Get embeddings
        embeddings = self._node_embeddings[active_indices]
        
        # Cluster on embeddings
        try:
            dist = pdist(embeddings, metric='cosine')
            Z = linkage(dist, method='average')
        except Exception:
            self._assign_all_to_one_cluster(active_indices)
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
        for i, (idx, label) in enumerate(zip(active_indices, labels)):
            mid = self._index_to_market[idx]
            cluster_id = int(label) - 1
            
            if cluster_id not in self._clusters:
                self._clusters[cluster_id] = ClusterState(cluster_id=cluster_id)
            
            self._clusters[cluster_id].add_member(mid)
            self._assignments[mid] = ClusterAssignment(
                market_id=mid,
                cluster_id=cluster_id,
                confidence=1.0,
            )
    
    def _assign_all_to_one_cluster(self, active_indices: List[int]) -> None:
        """Fallback: assign all to cluster 0."""
        if 0 not in self._clusters:
            self._clusters[0] = ClusterState(cluster_id=0)
        
        self._clusters[0].member_ids.clear()
        
        for idx in active_indices:
            mid = self._index_to_market[idx]
            self._clusters[0].add_member(mid)
            self._assignments[mid] = ClusterAssignment(
                market_id=mid, cluster_id=0, confidence=1.0
            )
    
    def get_attention_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Get attention-weighted adjacency matrix.
        
        Returns average attention across heads.
        """
        active_indices = self._get_active_indices()
        n = len(active_indices)
        
        if n == 0:
            return [], np.array([[]])
        
        market_ids = [self._index_to_market[idx] for idx in active_indices]
        
        # Average attention across heads
        attn = self._attention_weights[np.ix_(active_indices, active_indices)]
        attn_avg = np.mean(attn, axis=2)
        
        return market_ids, attn_avg
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Get correlation based on embedding similarity."""
        active_indices = self._get_active_indices()
        n = len(active_indices)
        
        if n == 0:
            return [], np.array([[]])
        
        market_ids = [self._index_to_market[idx] for idx in active_indices]
        embeddings = self._node_embeddings[active_indices]
        
        # Cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings_normed = embeddings / norms
        
        corr = embeddings_normed @ embeddings_normed.T
        np.fill_diagonal(corr, 1.0)
        
        return market_ids, corr
    
    def get_node_embedding(self, market_id: str) -> Optional[np.ndarray]:
        """Get learned embedding for a market."""
        if market_id not in self._market_index:
            return None
        
        idx = self._market_index[market_id]
        return self._node_embeddings[idx].copy()
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self._market_index.clear()
        self._index_to_market.clear()
        self._next_index = 0
        self._node_features = np.array([[]])
        self._node_embeddings = np.array([[]])
        self._adjacency = np.array([[]])
        self._attention_weights = np.array([[]])
        self._prev_prices.clear()
        self._return_buffer.clear()
        self._update_count = 0


# Convenience import
from scipy.spatial.distance import pdist
