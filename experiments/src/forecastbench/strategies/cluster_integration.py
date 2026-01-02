"""
Integration of Adaptive Clustering with Trading Strategies.

This module provides adapters to use the clustering algorithms
with existing trading strategies in the forecastbench framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from forecastbench.clustering import (
    SWOCC,
    OLRCM,
    OnlineClusteringBase,
    SWOCCConfig,
    OLRCMConfig,
)


@dataclass
class ClusterTradingSignal:
    """Trading signal derived from clustering."""
    
    market_id: str
    cluster_id: int
    
    # Cluster-level signals
    cluster_calibration_error: float  # Mean calibration error in cluster
    cluster_spread_zscore: float  # Market's spread vs cluster mean
    
    # Confidence
    cluster_confidence: float
    n_cluster_members: int
    
    # Suggested trade
    direction: int  # +1 buy YES, -1 buy NO, 0 no trade
    position_weight: float  # Suggested position weight [0, 1]


class ClusteringAdapter:
    """
    Adapter to use clustering with existing trading infrastructure.
    
    Wraps clustering algorithms and provides methods compatible with
    the existing portfolio_strategies.py interface.
    
    Example:
        adapter = ClusteringAdapter(algorithm="SWOCC")
        
        # Fit on training data
        adapter.fit(train_df)
        
        # Get clusters for trading
        clusters = adapter.get_market_clusters(test_df)
        
        # Get trading signals
        signals = adapter.get_trading_signals(test_df)
    """
    
    def __init__(
        self,
        algorithm: str = "SWOCC",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.algorithm_name = algorithm
        self._config = config or {}
        self._algorithm: Optional[OnlineClusteringBase] = None
        self._fitted = False
    
    def _create_algorithm(self) -> OnlineClusteringBase:
        """Create clustering algorithm instance."""
        if self.algorithm_name == "SWOCC":
            cfg = SWOCCConfig(
                ema_alpha=self._config.get("ema_alpha", 0.1),
                use_survival_weights=self._config.get("use_survival_weights", True),
                recluster_every=self._config.get("recluster_every", 10),
            )
            return SWOCC(cfg)
        
        elif self.algorithm_name == "OLRCM":
            cfg = OLRCMConfig(
                n_factors=self._config.get("n_factors", 10),
                recluster_every=self._config.get("recluster_every", 10),
            )
            return OLRCM(cfg)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def fit(
        self,
        df: pd.DataFrame,
        price_col: str = "avg_price",
        market_id_col: str = "market_id",
        outcome_col: str = "y",
        timestamp_col: Optional[str] = None,
    ) -> "ClusteringAdapter":
        """
        Fit clustering on training data.
        
        Args:
            df: Training DataFrame
            price_col: Column with prices
            market_id_col: Column with market IDs
            outcome_col: Column with outcomes
            timestamp_col: Optional column with timestamps
            
        Returns:
            self for chaining
        """
        self._algorithm = self._create_algorithm()
        
        # Get unique markets
        if market_id_col not in df.columns:
            df = df.copy()
            df[market_id_col] = df.index.astype(str)
        
        market_ids = df[market_id_col].unique()
        
        # Initialize all markets
        for mid in market_ids:
            market_data = df[df[market_id_col] == mid]
            if len(market_data) == 0:
                continue
            
            initial_price = market_data[price_col].iloc[0]
            initial_price = np.clip(initial_price, 0.01, 0.99)
            
            self._algorithm.add_market(
                market_id=str(mid),
                timestamp=0.0,
                initial_price=initial_price,
            )
        
        # If we have time series data, process it
        if timestamp_col and timestamp_col in df.columns:
            df_sorted = df.sort_values(timestamp_col)
            
            for t, (_, row) in enumerate(df_sorted.iterrows()):
                mid = str(row[market_id_col])
                price = np.clip(row[price_col], 0.01, 0.99)
                
                self._algorithm.update(
                    timestamp=float(t),
                    prices={mid: price},
                )
        else:
            # Single update with all prices
            prices = {}
            for mid in market_ids:
                market_data = df[df[market_id_col] == mid]
                if len(market_data) > 0:
                    price = market_data[price_col].iloc[-1]
                    prices[str(mid)] = np.clip(price, 0.01, 0.99)
            
            self._algorithm.update(timestamp=1.0, prices=prices)
        
        self._fitted = True
        return self
    
    def get_market_clusters(
        self,
        df: Optional[pd.DataFrame] = None,
        market_id_col: str = "market_id",
    ) -> Dict[int, List[str]]:
        """
        Get current cluster assignments.
        
        Compatible with find_market_clusters() from portfolio_strategies.py.
        
        Returns:
            Dict mapping cluster_id -> list of market_ids
        """
        if not self._fitted or self._algorithm is None:
            raise RuntimeError("Must call fit() first")
        
        return self._algorithm.get_clusters()
    
    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Get estimated correlation matrix.
        
        Returns:
            (market_ids, correlation_matrix) tuple
        """
        if not self._fitted or self._algorithm is None:
            raise RuntimeError("Must call fit() first")
        
        return self._algorithm.get_correlation_matrix()
    
    def get_trading_signals(
        self,
        df: pd.DataFrame,
        price_col: str = "avg_price",
        market_id_col: str = "market_id",
        calibration_threshold: float = 0.05,
    ) -> List[ClusterTradingSignal]:
        """
        Generate trading signals based on clustering.
        
        Args:
            df: DataFrame with market data
            price_col: Column with prices
            market_id_col: Column with market IDs
            calibration_threshold: Minimum calibration error to trade
            
        Returns:
            List of ClusterTradingSignal
        """
        if not self._fitted or self._algorithm is None:
            raise RuntimeError("Must call fit() first")
        
        signals = []
        clusters = self._algorithm.get_clusters()
        
        if market_id_col not in df.columns:
            df = df.copy()
            df[market_id_col] = df.index.astype(str)
        
        # Get cluster statistics
        cluster_stats = {}
        for cluster_id, members in clusters.items():
            member_prices = []
            for mid in members:
                market_data = df[df[market_id_col].astype(str) == mid]
                if len(market_data) > 0:
                    price = market_data[price_col].iloc[-1]
                    member_prices.append(price)
            
            if member_prices:
                cluster_stats[cluster_id] = {
                    "mean_price": np.mean(member_prices),
                    "std_price": np.std(member_prices),
                    "n_members": len(members),
                }
        
        # Generate signals
        for idx, row in df.iterrows():
            mid = str(row[market_id_col])
            price = row[price_col]
            
            cluster_id = self._algorithm.get_cluster_for_market(mid)
            if cluster_id is None:
                continue
            
            if cluster_id not in cluster_stats:
                continue
            
            stats = cluster_stats[cluster_id]
            
            # Compute spread from cluster mean
            spread = price - stats["mean_price"]
            zscore = spread / (stats["std_price"] + 1e-6)
            
            # Trade direction based on mean reversion
            if abs(zscore) > 1.0:
                direction = -1 if zscore > 0 else 1
                position_weight = min(abs(zscore) / 3.0, 1.0)
            else:
                direction = 0
                position_weight = 0.0
            
            assignment = self._algorithm.get_assignment(mid)
            confidence = assignment.confidence if assignment else 0.5
            
            signals.append(ClusterTradingSignal(
                market_id=mid,
                cluster_id=cluster_id,
                cluster_calibration_error=abs(spread),
                cluster_spread_zscore=zscore,
                cluster_confidence=confidence,
                n_cluster_members=stats["n_members"],
                direction=direction,
                position_weight=position_weight,
            ))
        
        return signals
    
    def update_with_resolution(
        self,
        market_id: str,
        outcome: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update clustering when a market resolves.
        
        Args:
            market_id: ID of resolved market
            outcome: Resolution outcome (0 or 1)
            timestamp: Optional resolution timestamp
        """
        if self._algorithm is not None:
            self._algorithm.remove_market(
                market_id=market_id,
                timestamp=timestamp,
                outcome=outcome,
            )


def find_market_clusters_adaptive(
    df: pd.DataFrame,
    algorithm: str = "SWOCC",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2,
    **kwargs,
) -> Dict[int, List[int]]:
    """
    Drop-in replacement for find_market_clusters() using adaptive clustering.
    
    Compatible with portfolio_strategies.py interface.
    
    Args:
        df: DataFrame with market data
        algorithm: Clustering algorithm ("SWOCC" or "OLRCM")
        n_clusters: Target number of clusters (optional)
        min_cluster_size: Minimum cluster size
        **kwargs: Additional config options
        
    Returns:
        Dict mapping cluster_id -> list of DataFrame indices
    """
    adapter = ClusteringAdapter(algorithm=algorithm, config=kwargs)
    
    # Determine price column
    price_col = "avg_price" if "avg_price" in df.columns else "price"
    if price_col not in df.columns:
        price_col = "first_price" if "first_price" in df.columns else df.columns[0]
    
    adapter.fit(df, price_col=price_col)
    
    clusters = adapter.get_market_clusters()
    
    # Convert market_ids back to indices
    # Create mapping from market_id to index
    if "market_id" in df.columns:
        id_col = "market_id"
    elif "id" in df.columns:
        id_col = "id"
    else:
        # Use index as ID
        id_to_idx = {str(i): i for i in range(len(df))}
        
        result = {}
        for cid, members in clusters.items():
            indices = [id_to_idx[m] for m in members if m in id_to_idx]
            if len(indices) >= min_cluster_size:
                result[cid] = indices
        
        return result
    
    id_to_idx = {str(v): i for i, v in enumerate(df[id_col])}
    
    result = {}
    for cid, members in clusters.items():
        indices = [id_to_idx[m] for m in members if m in id_to_idx]
        if len(indices) >= min_cluster_size:
            result[cid] = indices
    
    return result
