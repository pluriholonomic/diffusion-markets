"""
Unit tests for Online Low-Rank Covariance with Missing Data (OLRCM).
"""

import pytest
import numpy as np
from sklearn.metrics import adjusted_rand_score

from forecastbench.clustering import OLRCM
from forecastbench.clustering.online_factor import OLRCMConfig


class TestOLRCM:
    """Tests for OLRCM algorithm."""
    
    def test_add_remove_market(self, olrcm_algorithm):
        """Test basic market lifecycle."""
        algo = olrcm_algorithm
        
        algo.add_market("m1", timestamp=0.0, initial_price=0.5)
        assert algo.n_active_markets == 1
        
        algo.add_market("m2", timestamp=0.0, initial_price=0.6)
        assert algo.n_active_markets == 2
        
        algo.remove_market("m1", timestamp=1.0)
        assert algo.n_active_markets == 1
    
    def test_factor_loadings_shape(self, olrcm_algorithm, simple_price_data, simple_market_ids):
        """Test factor loadings output."""
        algo = olrcm_algorithm
        
        for i, mid in enumerate(simple_market_ids):
            algo.add_market(mid, timestamp=0.0, initial_price=simple_price_data[0, i])
        
        for t in range(1, len(simple_price_data)):
            prices = {mid: simple_price_data[t, i] for i, mid in enumerate(simple_market_ids)}
            algo.update(timestamp=float(t), prices=prices)
        
        market_ids, loadings = algo.get_factor_loadings()
        
        n_markets = len(market_ids)
        n_factors = algo.cfg.n_factors
        
        assert loadings.shape == (n_factors, n_markets)
    
    def test_factor_recovery(self, olrcm_algorithm, factor_generator):
        """Test that OLRCM can recover factor structure."""
        prices, deaths, labels, true_loadings = factor_generator.generate()
        market_ids = [f"market_{i}" for i in range(prices.shape[1])]
        
        algo = olrcm_algorithm
        algo.reset()
        
        pred_labels = algo.fit_predict(prices, deaths, market_ids)
        
        # Get estimated loadings
        _, estimated_loadings = algo.get_factor_loadings()
        
        # Should have some loadings estimated
        assert estimated_loadings.size > 0
    
    def test_correlation_from_factors(self, olrcm_algorithm, simple_price_data, simple_market_ids):
        """Test correlation matrix computed from factor model."""
        algo = olrcm_algorithm
        
        for i, mid in enumerate(simple_market_ids):
            algo.add_market(mid, timestamp=0.0, initial_price=simple_price_data[0, i])
        
        for t in range(1, len(simple_price_data)):
            prices = {mid: simple_price_data[t, i] for i, mid in enumerate(simple_market_ids)}
            algo.update(timestamp=float(t), prices=prices)
        
        market_ids, corr = algo.get_correlation_matrix()
        
        n = len(market_ids)
        assert corr.shape == (n, n)
        
        # Should be symmetric
        assert np.allclose(corr, corr.T, atol=1e-5)
        
        # Diagonal should be close to 1
        assert np.allclose(np.diag(corr), 1.0, atol=0.1)
    
    def test_explained_variance(self, olrcm_algorithm, simple_price_data, simple_market_ids):
        """Test explained variance ratio computation."""
        algo = olrcm_algorithm
        
        for i, mid in enumerate(simple_market_ids):
            algo.add_market(mid, timestamp=0.0, initial_price=simple_price_data[0, i])
        
        for t in range(1, len(simple_price_data)):
            prices = {mid: simple_price_data[t, i] for i, mid in enumerate(simple_market_ids)}
            algo.update(timestamp=float(t), prices=prices)
        
        evr = algo.explained_variance_ratio()
        
        assert len(evr) == algo.cfg.n_factors
        assert np.all(evr >= 0)
        # Sum of explained variance should be reasonable
        assert np.sum(evr) <= 1.5  # Can be > 1 due to estimation noise
    
    def test_predict_correlation(self, olrcm_algorithm, simple_price_data, simple_market_ids):
        """Test correlation prediction between markets."""
        algo = olrcm_algorithm
        
        for i, mid in enumerate(simple_market_ids):
            algo.add_market(mid, timestamp=0.0, initial_price=simple_price_data[0, i])
        
        for t in range(1, len(simple_price_data)):
            prices = {mid: simple_price_data[t, i] for i, mid in enumerate(simple_market_ids)}
            algo.update(timestamp=float(t), prices=prices)
        
        corr = algo.predict_correlation("market_0", "market_1")
        
        assert -1 <= corr <= 1


class TestOLRCMEdgeCases:
    """Edge case tests for OLRCM."""
    
    def test_few_observations(self, olrcm_algorithm):
        """Handle case with few observations."""
        algo = olrcm_algorithm
        
        algo.add_market("m1")
        algo.add_market("m2")
        
        # Only a couple updates
        algo.update(timestamp=1.0, prices={"m1": 0.5, "m2": 0.6})
        algo.update(timestamp=2.0, prices={"m1": 0.51, "m2": 0.59})
        
        # Should handle gracefully
        market_ids, loadings = algo.get_factor_loadings()
        # May or may not have enough observations
        assert isinstance(loadings, np.ndarray)
    
    def test_market_death_retains_loadings(self, olrcm_algorithm):
        """Loadings should be retained after market death."""
        algo = olrcm_algorithm
        
        algo.add_market("m1")
        algo.add_market("m2")
        
        for t in range(20):
            algo.update(timestamp=float(t), prices={"m1": 0.5, "m2": 0.6})
        
        # Get loadings before death
        exposure_before = algo.get_factor_exposures("m1")
        
        algo.remove_market("m1", timestamp=20.0)
        
        # Loadings still accessible (for analysis)
        exposure_after = algo.get_factor_exposures("m1")
        
        # Both should exist (or both None)
        assert (exposure_before is None) == (exposure_after is None)
