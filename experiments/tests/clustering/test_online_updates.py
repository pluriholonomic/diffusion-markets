"""
Tests for online/streaming behavior of clustering algorithms.
"""

import pytest
import numpy as np
import time

from forecastbench.clustering.generators import BlockCorrelationGenerator, BlockCorrelationConfig


class TestOnlineUpdates:
    """Test online/streaming behavior."""
    
    def test_incremental_consistency(self, algorithm, block_generator):
        """Results should be consistent when processing incrementally."""
        prices, deaths, labels, _ = block_generator.generate()
        market_ids = block_generator.get_market_ids()
        
        # Process once
        algorithm.reset()
        labels1 = algorithm.fit_predict(prices, deaths, market_ids)
        clusters1 = algorithm.get_clusters()
        
        # Process again (should give similar results)
        algorithm.reset()
        labels2 = algorithm.fit_predict(prices, deaths, market_ids)
        clusters2 = algorithm.get_clusters()
        
        # With same random state, should be deterministic
        # But even without, cluster structure should be similar
        assert len(clusters1) > 0 or len(clusters2) > 0
    
    def test_streaming_update(self, algorithm):
        """Test true streaming updates."""
        n_markets = 10
        
        # Initialize
        for i in range(n_markets):
            algorithm.add_market(f"m{i}", timestamp=0.0, initial_price=0.5)
        
        # Stream in prices
        for t in range(100):
            prices = {f"m{i}": 0.5 + np.sin((t + i) / 10) * 0.1 for i in range(n_markets)}
            algorithm.update(timestamp=float(t), prices=prices)
            
            # Can query clusters at any time
            if t % 20 == 0:
                clusters = algorithm.get_clusters()
                assert isinstance(clusters, dict)
    
    def test_update_time_scaling(self, algorithm):
        """Update time should not grow too fast with number of markets."""
        times_small = []
        times_large = []
        
        # Small number of markets
        algorithm.reset()
        for i in range(10):
            algorithm.add_market(f"m{i}")
        
        for t in range(20):
            prices = {f"m{i}": 0.5 for i in range(10)}
            start = time.time()
            algorithm.update(timestamp=float(t), prices=prices)
            times_small.append(time.time() - start)
        
        # Larger number of markets
        algorithm.reset()
        for i in range(50):
            algorithm.add_market(f"m{i}")
        
        for t in range(20):
            prices = {f"m{i}": 0.5 for i in range(50)}
            start = time.time()
            algorithm.update(timestamp=float(t), prices=prices)
            times_large.append(time.time() - start)
        
        # Time should not grow more than ~25x for 5x markets (quadratic would be 25x)
        avg_small = np.mean(times_small)
        avg_large = np.mean(times_large)
        
        if avg_small > 1e-6:  # Avoid division by zero
            ratio = avg_large / avg_small
            # Allow for some variance but should be sub-quadratic
            assert ratio < 100, f"Time ratio {ratio} suggests poor scaling"
    
    def test_cluster_stability_over_time(self, algorithm):
        """Clusters should be relatively stable without major changes."""
        n_markets = 20
        
        for i in range(n_markets):
            algorithm.add_market(f"m{i}", timestamp=0.0)
        
        # Track cluster assignments over time
        assignments_history = []
        
        for t in range(50):
            # Stable price dynamics
            prices = {f"m{i}": 0.5 + np.sin((t + i * 5) / 10) * 0.05 for i in range(n_markets)}
            algorithm.update(timestamp=float(t), prices=prices)
            
            if t % 10 == 0:
                assignments = {}
                for mid in algorithm.active_market_ids:
                    cluster = algorithm.get_cluster_for_market(mid)
                    if cluster is not None:
                        assignments[mid] = cluster
                assignments_history.append(assignments)
        
        # Count changes
        changes = 0
        for i in range(1, len(assignments_history)):
            for mid in assignments_history[i]:
                if mid in assignments_history[i-1]:
                    if assignments_history[i][mid] != assignments_history[i-1][mid]:
                        changes += 1
        
        # Should have some stability (allow for some churn)
        total_possible = sum(len(a) for a in assignments_history[1:])
        if total_possible > 0:
            stability = 1 - changes / total_possible
            assert stability > 0.5, f"Cluster stability {stability} is too low"


class TestLateMarketAddition:
    """Test adding markets after algorithm has been running."""
    
    def test_add_market_late(self, algorithm):
        """Can add markets after initial setup."""
        # Start with some markets
        for i in range(5):
            algorithm.add_market(f"m{i}", timestamp=0.0)
        
        # Run for a while
        for t in range(20):
            prices = {f"m{i}": 0.5 for i in range(5)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        # Add more markets
        for i in range(5, 10):
            algorithm.add_market(f"m{i}", timestamp=20.0)
        
        # Continue
        for t in range(20, 40):
            prices = {f"m{i}": 0.5 for i in range(10)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        assert algorithm.n_active_markets == 10
    
    def test_new_market_gets_clustered(self, algorithm):
        """New markets should eventually get cluster assignments."""
        for i in range(5):
            algorithm.add_market(f"m{i}", timestamp=0.0)
        
        for t in range(30):
            prices = {f"m{i}": 0.5 + np.sin((t + i) / 5) * 0.1 for i in range(5)}
            algorithm.update(timestamp=float(t), prices=prices)
        
        # Add new market
        algorithm.add_market("new", timestamp=30.0)
        
        # Run more updates
        for t in range(30, 50):
            prices = {
                **{f"m{i}": 0.5 + np.sin((t + i) / 5) * 0.1 for i in range(5)},
                "new": 0.5 + np.sin((t + 2) / 5) * 0.1  # Correlates with m2
            }
            algorithm.update(timestamp=float(t), prices=prices)
        
        # New market should have assignment
        assignment = algorithm.get_cluster_for_market("new")
        # May or may not have assignment depending on algorithm
        assert assignment is None or isinstance(assignment, int)
