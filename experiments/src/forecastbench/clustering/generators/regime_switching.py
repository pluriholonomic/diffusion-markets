"""
Regime-Switching Correlation Generator.

Generates data where the correlation structure switches between
different regimes according to a Markov process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class RegimeSwitchingConfig:
    """Configuration for regime-switching generator."""
    
    # Markets
    n_markets: int = 20
    
    # Regimes
    n_regimes: int = 3
    regime_persistence: float = 0.95  # P(stay in same regime)
    
    # Correlation structures per regime
    # If None, will generate random block structures
    correlation_matrices: Optional[List[np.ndarray]] = None
    
    # Time series
    n_timesteps: int = 500
    volatility: float = 0.02
    
    # Death model
    death_rate: float = 0.01
    min_lifetime: int = 50
    
    # Random seed
    seed: int = 42


class RegimeSwitchingGenerator:
    """
    Generate data with regime-switching correlation structure.
    
    Tests whether clustering algorithms can:
    1. Detect regime changes
    2. Adapt to new correlation structures
    3. Handle spurious regime detection
    
    Example:
        gen = RegimeSwitchingGenerator(RegimeSwitchingConfig(n_regimes=3))
        prices, deaths, regimes, corr_matrices = gen.generate()
    """
    
    def __init__(self, config: Optional[RegimeSwitchingConfig] = None):
        self.cfg = config or RegimeSwitchingConfig()
    
    def generate(self) -> Tuple[np.ndarray, List[Tuple[int, str]], np.ndarray, List[np.ndarray]]:
        """
        Generate synthetic regime-switching data.
        
        Returns:
            prices: (T, n_markets) price array
            death_events: List of (timestep, market_id) tuples
            regimes: (T,) regime indicator at each timestep
            correlation_matrices: List of correlation matrices per regime
        """
        rng = np.random.default_rng(self.cfg.seed)
        
        n_markets = self.cfg.n_markets
        n_regimes = self.cfg.n_regimes
        T = self.cfg.n_timesteps
        
        # Generate or use provided correlation matrices
        if self.cfg.correlation_matrices is not None:
            corr_matrices = self.cfg.correlation_matrices
        else:
            corr_matrices = self._generate_regime_correlations(rng, n_markets, n_regimes)
        
        # Build transition matrix
        trans_mat = self._build_transition_matrix(n_regimes)
        
        # Simulate regime sequence
        regimes = self._simulate_regimes(rng, T, trans_mat)
        
        # Generate returns under each regime
        prices = self._generate_prices(rng, regimes, corr_matrices)
        
        # Generate deaths
        death_events = self._generate_deaths(rng, n_markets, T)
        
        # Apply deaths
        for t, mid in death_events:
            idx = int(mid.split("_")[1])
            prices[t:, idx] = np.nan
        
        return prices, death_events, regimes, corr_matrices
    
    def _generate_regime_correlations(
        self,
        rng: np.random.Generator,
        n_markets: int,
        n_regimes: int,
    ) -> List[np.ndarray]:
        """Generate different correlation structures for each regime."""
        matrices = []
        
        for r in range(n_regimes):
            # Random block structure
            n_clusters = 2 + r  # Different number of clusters per regime
            cluster_size = n_markets // n_clusters
            
            corr = np.eye(n_markets)
            
            for c in range(n_clusters):
                start = c * cluster_size
                end = min((c + 1) * cluster_size, n_markets)
                
                # Random intra-cluster correlation
                intra_corr = 0.4 + 0.4 * rng.random()
                corr[start:end, start:end] = intra_corr
            
            np.fill_diagonal(corr, 1.0)
            
            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(corr)
            if eigvals.min() < 0:
                corr += (-eigvals.min() + 0.01) * np.eye(n_markets)
                d = np.sqrt(np.diag(corr))
                corr = corr / np.outer(d, d)
            
            matrices.append(corr)
        
        return matrices
    
    def _build_transition_matrix(self, n_regimes: int) -> np.ndarray:
        """Build Markov transition matrix."""
        p = self.cfg.regime_persistence
        q = (1 - p) / (n_regimes - 1) if n_regimes > 1 else 0
        
        trans = np.full((n_regimes, n_regimes), q)
        np.fill_diagonal(trans, p)
        
        return trans
    
    def _simulate_regimes(
        self,
        rng: np.random.Generator,
        T: int,
        trans_mat: np.ndarray,
    ) -> np.ndarray:
        """Simulate regime sequence from Markov chain."""
        n_regimes = trans_mat.shape[0]
        regimes = np.zeros(T, dtype=np.int32)
        
        # Start in regime 0
        regimes[0] = 0
        
        for t in range(1, T):
            current = regimes[t-1]
            probs = trans_mat[current]
            regimes[t] = rng.choice(n_regimes, p=probs)
        
        return regimes
    
    def _generate_prices(
        self,
        rng: np.random.Generator,
        regimes: np.ndarray,
        corr_matrices: List[np.ndarray],
    ) -> np.ndarray:
        """Generate prices under regime-switching correlation."""
        T = len(regimes)
        n_markets = corr_matrices[0].shape[0]
        
        # Precompute Cholesky factors
        L_matrices = []
        for corr in corr_matrices:
            try:
                L = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                # Fallback to diagonal
                L = np.eye(n_markets)
            L_matrices.append(L)
        
        prices = np.zeros((T, n_markets))
        prices[0] = 0.5 + rng.standard_normal(n_markets) * 0.05
        prices[0] = np.clip(prices[0], 0.1, 0.9)
        
        for t in range(1, T):
            regime = regimes[t]
            L = L_matrices[regime]
            
            # Correlated innovations
            z = rng.standard_normal(n_markets)
            innovations = L @ z * self.cfg.volatility
            
            # Mean reversion
            drift = 0.1 * (0.5 - prices[t-1])
            
            prices[t] = prices[t-1] + drift + innovations
            prices[t] = np.clip(prices[t], 0.01, 0.99)
        
        return prices
    
    def _generate_deaths(
        self,
        rng: np.random.Generator,
        n_markets: int,
        T: int,
    ) -> List[Tuple[int, str]]:
        """Generate death events."""
        death_events = []
        
        for i in range(n_markets):
            death_time = rng.exponential(1 / self.cfg.death_rate)
            death_time = max(int(death_time), self.cfg.min_lifetime)
            
            if death_time < T:
                death_events.append((death_time, f"market_{i}"))
        
        death_events.sort(key=lambda x: x[0])
        return death_events
    
    def get_regime_change_times(self, regimes: np.ndarray) -> List[int]:
        """Get timesteps where regime changes occur."""
        changes = []
        for t in range(1, len(regimes)):
            if regimes[t] != regimes[t-1]:
                changes.append(t)
        return changes
    
    def get_cluster_labels_for_regime(
        self,
        regime: int,
        corr_matrices: List[np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Extract cluster labels from correlation matrix.
        
        Uses simple thresholding on correlation.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        
        corr = corr_matrices[regime]
        n = corr.shape[0]
        
        dist = 1 - np.abs(corr)
        np.fill_diagonal(dist, 0)
        
        dist_condensed = squareform(dist, checks=False)
        Z = linkage(dist_condensed, method='average')
        labels = fcluster(Z, threshold, criterion='distance')
        
        return labels - 1  # 0-indexed
