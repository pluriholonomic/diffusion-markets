"""
Block Correlation Generator.

Generates markets with known block-diagonal correlation structure
and controlled death times for testing clustering algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BlockCorrelationConfig:
    """Configuration for block correlation generator."""
    
    # Cluster structure
    n_clusters: int = 5
    markets_per_cluster: int = 10
    
    # Correlation structure
    intra_cluster_corr: float = 0.7
    inter_cluster_corr: float = 0.1
    
    # Time series parameters
    n_timesteps: int = 500
    volatility: float = 0.02
    mean_reversion: float = 0.1
    
    # Death model
    death_rate: float = 0.01  # Hazard rate
    death_correlation: bool = False  # Whether deaths are correlated within clusters
    min_lifetime: int = 50  # Minimum lifetime before death possible
    
    # Random seed
    seed: int = 42


class BlockCorrelationGenerator:
    """
    Generate synthetic market data with block-diagonal correlation.
    
    Creates price series where markets within the same cluster are
    highly correlated, while markets in different clusters have
    low correlation.
    
    Example:
        gen = BlockCorrelationGenerator(BlockCorrelationConfig(n_clusters=5))
        prices, deaths, labels, corr = gen.generate()
        
        # prices: (T, n_markets) price array
        # deaths: List of (timestep, market_id) tuples
        # labels: (n_markets,) cluster labels
        # corr: (n_markets, n_markets) true correlation matrix
    """
    
    def __init__(self, config: Optional[BlockCorrelationConfig] = None):
        self.cfg = config or BlockCorrelationConfig()
    
    def generate(self) -> Tuple[np.ndarray, List[Tuple[int, str]], np.ndarray, np.ndarray]:
        """
        Generate synthetic dataset.
        
        Returns:
            prices: (T, n_markets) price array
            death_events: List of (timestep, market_id) tuples
            labels: (n_markets,) ground truth cluster labels
            correlation: (n_markets, n_markets) true correlation matrix
        """
        rng = np.random.default_rng(self.cfg.seed)
        
        n_clusters = self.cfg.n_clusters
        n_per_cluster = self.cfg.markets_per_cluster
        n_markets = n_clusters * n_per_cluster
        T = self.cfg.n_timesteps
        
        # Create ground truth labels
        labels = np.repeat(np.arange(n_clusters), n_per_cluster)
        
        # Build correlation matrix
        corr = self._build_correlation_matrix(n_clusters, n_per_cluster)
        
        # Generate correlated returns using Cholesky decomposition
        L = np.linalg.cholesky(corr)
        
        # Generate innovations
        innovations = rng.standard_normal((T, n_markets))
        correlated_innovations = innovations @ L.T
        
        # Scale by volatility
        returns = self.cfg.volatility * correlated_innovations
        
        # Generate prices with mean reversion
        prices = np.zeros((T, n_markets))
        prices[0] = 0.5 + rng.standard_normal(n_markets) * 0.1
        prices[0] = np.clip(prices[0], 0.1, 0.9)
        
        for t in range(1, T):
            # Mean reversion toward 0.5
            drift = self.cfg.mean_reversion * (0.5 - prices[t-1])
            prices[t] = prices[t-1] + drift + returns[t]
            prices[t] = np.clip(prices[t], 0.01, 0.99)
        
        # Generate death events
        death_events = self._generate_deaths(rng, n_markets, T, labels)
        
        # Apply deaths to prices (set to NaN after death)
        market_ids = [f"market_{i}" for i in range(n_markets)]
        death_times = {mid: T for mid in market_ids}
        
        for t, mid in death_events:
            idx = int(mid.split("_")[1])
            death_times[mid] = t
            prices[t:, idx] = np.nan
        
        return prices, death_events, labels, corr
    
    def _build_correlation_matrix(
        self,
        n_clusters: int,
        n_per_cluster: int,
    ) -> np.ndarray:
        """Build block-diagonal correlation matrix."""
        n = n_clusters * n_per_cluster
        corr = np.ones((n, n)) * self.cfg.inter_cluster_corr
        
        # Set intra-cluster correlations
        for c in range(n_clusters):
            start = c * n_per_cluster
            end = (c + 1) * n_per_cluster
            corr[start:end, start:end] = self.cfg.intra_cluster_corr
        
        # Diagonal is 1
        np.fill_diagonal(corr, 1.0)
        
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 0:
            # Add small diagonal to make positive definite
            corr += (-eigvals.min() + 0.01) * np.eye(n)
            # Renormalize to correlation matrix
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)
        
        return corr
    
    def _generate_deaths(
        self,
        rng: np.random.Generator,
        n_markets: int,
        T: int,
        labels: np.ndarray,
    ) -> List[Tuple[int, str]]:
        """Generate death events."""
        death_events = []
        
        if self.cfg.death_correlation:
            # Correlated deaths within clusters
            n_clusters = self.cfg.n_clusters
            
            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_indices = np.where(cluster_mask)[0]
                
                # Sample a common death time component
                common_death_rate = self.cfg.death_rate * 0.7
                cluster_death_time = rng.exponential(1 / common_death_rate)
                
                for idx in cluster_indices:
                    # Individual variation
                    individual_offset = rng.exponential(1 / (self.cfg.death_rate * 0.3))
                    death_time = int(cluster_death_time + individual_offset)
                    death_time = max(death_time, self.cfg.min_lifetime)
                    
                    if death_time < T:
                        death_events.append((death_time, f"market_{idx}"))
        else:
            # Independent deaths
            for i in range(n_markets):
                death_time = rng.exponential(1 / self.cfg.death_rate)
                death_time = max(int(death_time), self.cfg.min_lifetime)
                
                if death_time < T:
                    death_events.append((death_time, f"market_{i}"))
        
        # Sort by time
        death_events.sort(key=lambda x: x[0])
        
        return death_events
    
    def get_market_ids(self) -> List[str]:
        """Get market IDs."""
        n = self.cfg.n_clusters * self.cfg.markets_per_cluster
        return [f"market_{i}" for i in range(n)]
