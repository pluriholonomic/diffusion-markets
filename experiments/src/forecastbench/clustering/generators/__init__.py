"""
Synthetic Data Generators for Clustering Algorithm Testing.

These generators create datasets with known ground-truth cluster structure
to enable rigorous unit testing of clustering algorithms.
"""

from forecastbench.clustering.generators.block_correlation import (
    BlockCorrelationGenerator,
    BlockCorrelationConfig,
)
from forecastbench.clustering.generators.hawkes_cluster import (
    HawkesClusterGenerator,
    HawkesClusterConfig,
    HawkesEvent,
)
from forecastbench.clustering.generators.factor_model import (
    FactorModelGenerator,
    FactorModelConfig,
)
from forecastbench.clustering.generators.regime_switching import (
    RegimeSwitchingGenerator,
    RegimeSwitchingConfig,
)

__all__ = [
    "BlockCorrelationGenerator",
    "BlockCorrelationConfig",
    "HawkesClusterGenerator",
    "HawkesClusterConfig",
    "HawkesEvent",
    "FactorModelGenerator",
    "FactorModelConfig",
    "RegimeSwitchingGenerator",
    "RegimeSwitchingConfig",
]
