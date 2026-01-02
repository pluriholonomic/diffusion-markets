"""
Pytest fixtures for clustering tests.
"""

import pytest
import numpy as np
from typing import Dict, Any

# Import generators
from forecastbench.clustering.generators import (
    BlockCorrelationGenerator,
    BlockCorrelationConfig,
    FactorModelGenerator,
    FactorModelConfig,
    HawkesClusterGenerator,
    HawkesClusterConfig,
    RegimeSwitchingGenerator,
    RegimeSwitchingConfig,
)

# Import algorithms
from forecastbench.clustering import (
    SWOCC,
    OLRCM,
    BirthDeathHawkes,
    StreamingDPClustering,
)
from forecastbench.clustering.survival_weighted import SWOCCConfig
from forecastbench.clustering.online_factor import OLRCMConfig
from forecastbench.clustering.birth_death_hawkes import BDHPConfig
from forecastbench.clustering.streaming_dp import SDPMConfig
from forecastbench.clustering.dynamic_graph import DynamicGraphAttention, DGATConfig


@pytest.fixture
def block_generator():
    """Block correlation data generator."""
    config = BlockCorrelationConfig(
        n_clusters=3,
        markets_per_cluster=10,
        intra_cluster_corr=0.7,
        inter_cluster_corr=0.1,
        n_timesteps=200,
        death_rate=0.005,
        seed=42,
    )
    return BlockCorrelationGenerator(config)


@pytest.fixture
def factor_generator():
    """Factor model data generator."""
    config = FactorModelConfig(
        n_factors=3,
        markets_per_factor=10,
        n_timesteps=200,
        seed=42,
    )
    return FactorModelGenerator(config)


@pytest.fixture
def hawkes_generator():
    """Hawkes process data generator."""
    config = HawkesClusterConfig(
        n_clusters=3,
        markets_per_cluster=10,
        max_time=50.0,
        seed=42,
    )
    return HawkesClusterGenerator(config)


@pytest.fixture
def regime_generator():
    """Regime-switching data generator."""
    config = RegimeSwitchingConfig(
        n_markets=30,
        n_regimes=2,
        n_timesteps=200,
        seed=42,
    )
    return RegimeSwitchingGenerator(config)


@pytest.fixture
def swocc_algorithm():
    """SWOCC algorithm instance."""
    config = SWOCCConfig(
        ema_alpha=0.1,
        recluster_every=10,
        min_observations=5,
    )
    return SWOCC(config)


@pytest.fixture
def olrcm_algorithm():
    """OLRCM algorithm instance."""
    config = OLRCMConfig(
        n_factors=5,
        recluster_every=10,
    )
    return OLRCM(config)


@pytest.fixture
def bdhp_algorithm():
    """Birth-Death Hawkes algorithm instance."""
    config = BDHPConfig(
        decay_rate=0.1,
        recluster_every=10,
    )
    return BirthDeathHawkes(config)


@pytest.fixture
def sdpm_algorithm():
    """Streaming DP algorithm instance."""
    config = SDPMConfig(
        concentration=1.0,
        reassign_every=10,
    )
    return StreamingDPClustering(config)


@pytest.fixture
def dgat_algorithm():
    """Dynamic graph attention algorithm instance."""
    config = DGATConfig(
        hidden_dim=16,
        recluster_every=10,
    )
    return DynamicGraphAttention(config)


@pytest.fixture(params=[
    "swocc_algorithm",
    "olrcm_algorithm",
    "bdhp_algorithm",
    "sdpm_algorithm",
    "dgat_algorithm",
])
def algorithm(request):
    """Parametrized fixture for all algorithms."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def simple_price_data():
    """Simple price data for basic tests."""
    np.random.seed(42)
    T, n = 100, 10
    prices = 0.5 + np.cumsum(np.random.randn(T, n) * 0.01, axis=0)
    prices = np.clip(prices, 0.01, 0.99)
    return prices


@pytest.fixture
def simple_market_ids():
    """Market IDs for simple data."""
    return [f"market_{i}" for i in range(10)]
