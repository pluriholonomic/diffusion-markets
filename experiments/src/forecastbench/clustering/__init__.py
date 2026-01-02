"""
Online Adaptive Clustering for Prediction Markets.

This module provides clustering algorithms that handle:
- Market "death" (resolution) events
- Online/streaming updates
- Survival-weighted correlation estimation
- Dynamic cluster membership

Algorithms:
- SWOCC: Survival-Weighted Online Correlation Clustering
- OLRCM: Online Low-Rank Covariance with Missing Data
- BirthDeathHawkes: Birth-Death Hawkes Process
- StreamingDPClustering: Streaming Dirichlet Process Mixture
- DynamicGraphAttention: Dynamic Graph Attention Network
- EnsembleClustering: Ensemble of multiple algorithms

Example:
    from forecastbench.clustering import SWOCC, ClusteringEvaluator
    
    # Create algorithm
    algo = SWOCC()
    
    # Add markets
    algo.add_market("market_1", timestamp=0.0, initial_price=0.5)
    algo.add_market("market_2", timestamp=0.0, initial_price=0.6)
    
    # Update with prices
    algo.update(timestamp=1.0, prices={"market_1": 0.55, "market_2": 0.58})
    
    # Get clusters
    clusters = algo.get_clusters()
    
    # Market dies
    algo.remove_market("market_1", timestamp=2.0, outcome=1)
"""

from forecastbench.clustering.base import (
    OnlineClusteringBase,
    ClusterAssignment,
    ClusterState,
    MarketState,
    MarketStatus,
)
from forecastbench.clustering.survival_model import (
    SurvivalModel,
    SurvivalObservation,
    KaplanMeierSurvival,
    CoxSurvival,
    ExponentialSurvival,
    AdaptiveSurvival,
)
from forecastbench.clustering.survival_weighted import SWOCC, SWOCCConfig
from forecastbench.clustering.online_factor import OLRCM, OLRCMConfig
from forecastbench.clustering.birth_death_hawkes import BirthDeathHawkes, BDHPConfig
from forecastbench.clustering.streaming_dp import StreamingDPClustering, SDPMConfig
from forecastbench.clustering.dynamic_graph import DynamicGraphAttention, DGATConfig
from forecastbench.clustering.ensemble import EnsembleClustering, EnsembleConfig
from forecastbench.clustering.evaluation import ClusteringEvaluator, ClusteringMetrics

__all__ = [
    # Base classes
    "OnlineClusteringBase",
    "ClusterAssignment",
    "ClusterState",
    "MarketState",
    "MarketStatus",
    # Survival models
    "SurvivalModel",
    "SurvivalObservation",
    "KaplanMeierSurvival",
    "CoxSurvival",
    "ExponentialSurvival",
    "AdaptiveSurvival",
    # Clustering algorithms
    "SWOCC",
    "SWOCCConfig",
    "OLRCM",
    "OLRCMConfig",
    "BirthDeathHawkes",
    "BDHPConfig",
    "StreamingDPClustering",
    "SDPMConfig",
    "DynamicGraphAttention",
    "DGATConfig",
    "EnsembleClustering",
    "EnsembleConfig",
    # Evaluation
    "ClusteringEvaluator",
    "ClusteringMetrics",
]
