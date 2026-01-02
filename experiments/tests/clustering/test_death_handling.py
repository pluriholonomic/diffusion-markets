"""
Tests for death handling across all clustering algorithms.
"""

import pytest
import numpy as np
from sklearn.metrics import adjusted_rand_score

from forecastbench.clustering.generators import BlockCorrelationGenerator, BlockCorrelationConfig


class TestDeathHandling:
    """Test robustness to market death events across algorithms."""
    
    def test_graceful_death_removal(self, algorithm):
        """Algorithm should not crash when market dies."""
        # Add some markets
        for i in range(5):
            algorithm.add_market(f"m{i}", timestamp=0.0, initial_price=0.5)
        
        # Run some updates
        for t in range(10):
            prices = {f"m{i}": 0.5 + np.random.randn() * 0.01 for i in range(5)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        # Kill some markets
        algorithm.remove_market("m0", timestamp=10.0)
        algorithm.remove_market("m2", timestamp=10.0)
        
        # Continue updates with remaining markets
        for t in range(10, 20):
            prices = {f"m{i}": 0.5 + np.random.randn() * 0.01 for i in [1, 3, 4]}
            algorithm.update(timestamp=float(t), prices=prices)
        
        # Should complete without error
        clusters = algorithm.get_clusters()
        assert isinstance(clusters, dict)
    
    def test_progressive_deaths(self, algorithm):
        """Handle markets dying one by one."""
        n_markets = 10
        
        for i in range(n_markets):
            algorithm.add_market(f"m{i}", timestamp=0.0)
        
        # Kill one market at each timestep
        for t in range(n_markets):
            active = [f"m{i}" for i in range(t, n_markets)]
            if active:
                prices = {mid: 0.5 for mid in active}
                algorithm.update(timestamp=float(t), prices=prices)
            
            if t < n_markets:
                algorithm.remove_market(f"m{t}", timestamp=float(t))
        
        # All markets dead
        assert algorithm.n_active_markets == 0
    
    def test_correlated_deaths(self, algorithm):
        """Test with correlated death patterns."""
        config = BlockCorrelationConfig(
            n_clusters=3,
            markets_per_cluster=5,
            death_correlation=True,
            death_rate=0.05,
            n_timesteps=100,
        )
        gen = BlockCorrelationGenerator(config)
        prices, deaths, labels, _ = gen.generate()
        market_ids = gen.get_market_ids()
        
        algorithm.reset()
        pred_labels = algorithm.fit_predict(prices, deaths, market_ids)
        
        # Should complete without error
        assert len(pred_labels) == len(labels)
    
    def test_death_at_cluster_boundary(self, algorithm):
        """Test when deaths occur at cluster boundaries."""
        # Create two clear clusters
        algorithm.add_market("cluster1_a", timestamp=0.0)
        algorithm.add_market("cluster1_b", timestamp=0.0)
        algorithm.add_market("cluster2_a", timestamp=0.0)
        algorithm.add_market("cluster2_b", timestamp=0.0)
        
        # Run updates with clear cluster structure
        for t in range(20):
            # Cluster 1 moves together
            p1 = 0.5 + 0.1 * np.sin(t / 5)
            # Cluster 2 moves opposite
            p2 = 0.5 - 0.1 * np.sin(t / 5)
            
            prices = {
                "cluster1_a": p1 + np.random.randn() * 0.01,
                "cluster1_b": p1 + np.random.randn() * 0.01,
                "cluster2_a": p2 + np.random.randn() * 0.01,
                "cluster2_b": p2 + np.random.randn() * 0.01,
            }
            algorithm.update(timestamp=float(t), prices=prices)
        
        # Kill one from each cluster
        algorithm.remove_market("cluster1_a")
        algorithm.remove_market("cluster2_a")
        
        # Remaining should still be clusterable
        clusters = algorithm.get_clusters()
        assert len(algorithm.active_market_ids) == 2


class TestDeathTimingEffects:
    """Test effects of death timing on clustering."""
    
    def test_early_death(self, algorithm):
        """Markets that die early should not corrupt clustering."""
        # Long-lived markets
        for i in range(5):
            algorithm.add_market(f"long_{i}", timestamp=0.0)
        
        # Short-lived market
        algorithm.add_market("short", timestamp=0.0)
        
        # Short dies immediately
        algorithm.update(timestamp=0.0, prices={"short": 0.5, **{f"long_{i}": 0.5 for i in range(5)}})
        algorithm.remove_market("short", timestamp=1.0)
        
        # Continue with long-lived
        for t in range(1, 50):
            prices = {f"long_{i}": 0.5 + np.random.randn() * 0.01 for i in range(5)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        clusters = algorithm.get_clusters()
        assert algorithm.n_active_markets == 5
    
    def test_simultaneous_deaths(self, algorithm):
        """Handle multiple simultaneous deaths."""
        for i in range(10):
            algorithm.add_market(f"m{i}", timestamp=0.0)
        
        for t in range(10):
            prices = {f"m{i}": 0.5 for i in range(10)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        # Kill half simultaneously
        for i in range(5):
            algorithm.remove_market(f"m{i}", timestamp=10.0)
        
        assert algorithm.n_active_markets == 5
        
        # Continue
        for t in range(10, 20):
            prices = {f"m{i}": 0.5 for i in range(5, 10)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        clusters = algorithm.get_clusters()
        assert isinstance(clusters, dict)
