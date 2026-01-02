"""
Factor Model Generator.

Generates returns from a latent factor model where factors define clusters.
Markets in the same cluster share factor loadings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class FactorModelConfig:
    """Configuration for factor model generator."""
    
    # Factor structure
    n_factors: int = 5
    markets_per_factor: int = 10
    
    # Factor parameters
    factor_vol: float = 0.03
    idio_vol: float = 0.01
    loading_dispersion: float = 0.3  # Variation in loadings within cluster
    
    # Time series
    n_timesteps: int = 500
    
    # Death model
    resolution_threshold: float = 0.3  # Cumulative return that triggers death
    min_lifetime: int = 20
    
    # Random seed
    seed: int = 42


class FactorModelGenerator:
    """
    Generate returns from a latent factor model.
    
    r_i(t) = sum_k beta_ik * f_k(t) + epsilon_i(t)
    
    Markets in the same cluster have similar loadings on a dominant factor.
    
    Example:
        gen = FactorModelGenerator(FactorModelConfig(n_factors=5))
        prices, deaths, labels, loadings = gen.generate()
    """
    
    def __init__(self, config: Optional[FactorModelConfig] = None):
        self.cfg = config or FactorModelConfig()
    
    def generate(self) -> Tuple[np.ndarray, List[Tuple[int, str]], np.ndarray, np.ndarray]:
        """
        Generate synthetic factor model data.
        
        Returns:
            prices: (T, n_markets) price array
            death_events: List of (timestep, market_id) tuples
            labels: (n_markets,) cluster labels (based on dominant factor)
            loadings: (n_factors, n_markets) true factor loadings
        """
        rng = np.random.default_rng(self.cfg.seed)
        
        n_factors = self.cfg.n_factors
        n_per_factor = self.cfg.markets_per_factor
        n_markets = n_factors * n_per_factor
        T = self.cfg.n_timesteps
        
        # Create labels (cluster = dominant factor)
        labels = np.repeat(np.arange(n_factors), n_per_factor)
        
        # Generate factor loadings
        loadings = self._generate_loadings(rng, n_factors, n_per_factor)
        
        # Generate factor returns
        factor_returns = rng.standard_normal((T, n_factors)) * self.cfg.factor_vol
        
        # Generate idiosyncratic returns
        idio_returns = rng.standard_normal((T, n_markets)) * self.cfg.idio_vol
        
        # Combine: r = F @ loadings + epsilon
        returns = factor_returns @ loadings + idio_returns
        
        # Generate prices from returns
        prices = np.zeros((T, n_markets))
        prices[0] = 0.5
        
        cumulative_returns = np.zeros(n_markets)
        death_events = []
        death_times = np.full(n_markets, T)
        
        for t in range(1, T):
            cumulative_returns += returns[t]
            
            # Check for deaths
            for i in range(n_markets):
                if t >= self.cfg.min_lifetime and death_times[i] == T:
                    if abs(cumulative_returns[i]) > self.cfg.resolution_threshold:
                        death_times[i] = t
                        death_events.append((t, f"market_{i}"))
            
            # Update prices (clip to valid range)
            prices[t] = prices[t-1] + returns[t]
            prices[t] = np.clip(prices[t], 0.01, 0.99)
            
            # Set dead markets to NaN
            for i in range(n_markets):
                if death_times[i] <= t:
                    prices[t, i] = np.nan
        
        death_events.sort(key=lambda x: x[0])
        
        return prices, death_events, labels, loadings
    
    def _generate_loadings(
        self,
        rng: np.random.Generator,
        n_factors: int,
        n_per_factor: int,
    ) -> np.ndarray:
        """
        Generate factor loadings with cluster structure.
        
        Each cluster has high loading on its dominant factor.
        """
        n_markets = n_factors * n_per_factor
        loadings = np.zeros((n_factors, n_markets))
        
        for k in range(n_factors):
            start = k * n_per_factor
            end = (k + 1) * n_per_factor
            
            # High loading on dominant factor
            loadings[k, start:end] = 1.0 + rng.standard_normal(n_per_factor) * self.cfg.loading_dispersion
            
            # Small loading on other factors
            for other_k in range(n_factors):
                if other_k != k:
                    loadings[other_k, start:end] = rng.standard_normal(n_per_factor) * 0.1
        
        return loadings
    
    def get_covariance_matrix(self, loadings: np.ndarray) -> np.ndarray:
        """
        Compute true covariance matrix from factor loadings.
        
        Cov = loadings.T @ loadings + idio_vol^2 * I
        """
        n_markets = loadings.shape[1]
        cov = loadings.T @ loadings * self.cfg.factor_vol ** 2
        cov += self.cfg.idio_vol ** 2 * np.eye(n_markets)
        return cov
    
    def get_correlation_matrix(self, loadings: np.ndarray) -> np.ndarray:
        """Compute true correlation matrix."""
        cov = self.get_covariance_matrix(loadings)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        return corr
