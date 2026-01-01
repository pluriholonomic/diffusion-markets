"""
Regime Detection for Prediction Markets

This module provides tools to detect whether a market/category is in a
trending (momentum) regime or mean-reverting regime.

Key Methods:
1. Hurst Exponent - H > 0.5 = trending, H < 0.5 = mean-reverting
2. Autocorrelation Analysis - positive AC = trending, negative AC = mean-reverting
3. Variance Ratio Test - tests random walk hypothesis
4. Combined Regime Score

Usage:
    detector = RegimeDetector()
    regime = detector.detect_regime(price_series)
    # regime.type in ['trending', 'mean_reverting', 'random_walk']
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class RegimeType(Enum):
    TRENDING = 'trending'
    MEAN_REVERTING = 'mean_reverting'
    RANDOM_WALK = 'random_walk'
    UNKNOWN = 'unknown'


@dataclass
class RegimeResult:
    """Result of regime detection"""
    regime_type: RegimeType
    hurst_exponent: float
    autocorrelation: float
    variance_ratio: float
    confidence: float  # 0-1, how confident we are in the regime call
    details: Dict[str, Any]


def compute_hurst_exponent(
    series: np.ndarray,
    max_lag: int = None,
    method: str = 'rs'
) -> float:
    """
    Compute Hurst exponent using R/S (Rescaled Range) analysis.
    
    H > 0.5: Trending/persistent (momentum works)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting (mean reversion works)
    
    Args:
        series: Price or return series
        max_lag: Maximum lag to consider
        method: 'rs' for R/S analysis, 'dfa' for DFA (simplified)
    
    Returns:
        Hurst exponent (0 to 1)
    """
    n = len(series)
    if n < 20:
        return 0.5  # Not enough data
    
    if max_lag is None:
        max_lag = min(n // 4, 100)
    
    # R/S Analysis
    lags = range(10, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        # Divide series into chunks of size 'lag'
        n_chunks = n // lag
        if n_chunks < 1:
            continue
            
        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i * lag:(i + 1) * lag]
            
            # Mean-adjusted cumulative deviation
            mean_chunk = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_chunk)
            
            # Range
            R = np.max(cumdev) - np.min(cumdev)
            
            # Standard deviation
            S = np.std(chunk, ddof=1) if len(chunk) > 1 else 1e-8
            
            if S > 1e-8:
                rs_chunk.append(R / S)
        
        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))
    
    if len(rs_values) < 3:
        return 0.5
    
    # Linear regression in log-log space: log(R/S) = H * log(n) + c
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    
    # Simple OLS
    n_points = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_x2 = np.sum(log_lags ** 2)
    
    denominator = n_points * sum_x2 - sum_x ** 2
    if abs(denominator) < 1e-10:
        return 0.5
    
    H = (n_points * sum_xy - sum_x * sum_y) / denominator
    
    # Clamp to valid range
    return np.clip(H, 0.0, 1.0)


def compute_autocorrelation(
    series: np.ndarray,
    lags: List[int] = None
) -> Dict[int, float]:
    """
    Compute autocorrelation at various lags.
    
    Positive autocorrelation → trending
    Negative autocorrelation → mean-reverting
    Zero autocorrelation → random walk
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    if var < 1e-10:
        return {lag: 0.0 for lag in lags}
    
    result = {}
    for lag in lags:
        if lag >= n:
            result[lag] = 0.0
            continue
        
        # Autocorrelation at lag k
        cov = np.mean((series[lag:] - mean) * (series[:-lag] - mean))
        result[lag] = cov / var
    
    return result


def compute_variance_ratio(
    series: np.ndarray,
    k: int = 2
) -> float:
    """
    Variance Ratio Test (Lo and MacKinlay)
    
    Under random walk: VR(k) = 1
    VR(k) > 1: Trending
    VR(k) < 1: Mean-reverting
    
    Args:
        series: Price or return series
        k: Holding period for variance ratio
    
    Returns:
        Variance ratio
    """
    n = len(series)
    if n < k * 2:
        return 1.0
    
    # Returns
    returns = np.diff(series)
    if len(returns) < k:
        return 1.0
    
    # Variance of 1-period returns
    var_1 = np.var(returns, ddof=1)
    if var_1 < 1e-10:
        return 1.0
    
    # Variance of k-period returns
    k_returns = series[k:] - series[:-k]
    var_k = np.var(k_returns, ddof=1)
    
    # Variance ratio
    vr = var_k / (k * var_1)
    
    return vr


class RegimeDetector:
    """
    Detects market regime (trending vs mean-reverting) using multiple methods.
    """
    
    def __init__(
        self,
        hurst_threshold_trending: float = 0.55,
        hurst_threshold_reverting: float = 0.45,
        ac_threshold: float = 0.1,
        vr_threshold_trending: float = 1.1,
        vr_threshold_reverting: float = 0.9,
    ):
        self.hurst_threshold_trending = hurst_threshold_trending
        self.hurst_threshold_reverting = hurst_threshold_reverting
        self.ac_threshold = ac_threshold
        self.vr_threshold_trending = vr_threshold_trending
        self.vr_threshold_reverting = vr_threshold_reverting
    
    def detect_regime(
        self,
        series: np.ndarray,
        return_details: bool = True
    ) -> RegimeResult:
        """
        Detect regime for a given price/return series.
        
        Returns:
            RegimeResult with regime type and confidence
        """
        if len(series) < 20:
            return RegimeResult(
                regime_type=RegimeType.UNKNOWN,
                hurst_exponent=0.5,
                autocorrelation=0.0,
                variance_ratio=1.0,
                confidence=0.0,
                details={'error': 'Insufficient data'}
            )
        
        # Compute all metrics
        hurst = compute_hurst_exponent(series)
        ac_dict = compute_autocorrelation(series, lags=[1, 2, 5])
        ac_1 = ac_dict.get(1, 0.0)
        vr = compute_variance_ratio(series, k=2)
        
        # Vote on regime
        votes = {'trending': 0, 'mean_reverting': 0, 'random_walk': 0}
        
        # Hurst vote
        if hurst > self.hurst_threshold_trending:
            votes['trending'] += 2
        elif hurst < self.hurst_threshold_reverting:
            votes['mean_reverting'] += 2
        else:
            votes['random_walk'] += 1
        
        # Autocorrelation vote
        if ac_1 > self.ac_threshold:
            votes['trending'] += 1
        elif ac_1 < -self.ac_threshold:
            votes['mean_reverting'] += 1
        else:
            votes['random_walk'] += 1
        
        # Variance ratio vote
        if vr > self.vr_threshold_trending:
            votes['trending'] += 1
        elif vr < self.vr_threshold_reverting:
            votes['mean_reverting'] += 1
        else:
            votes['random_walk'] += 1
        
        # Determine regime
        max_votes = max(votes.values())
        total_votes = sum(votes.values())
        
        if votes['trending'] == max_votes:
            regime_type = RegimeType.TRENDING
        elif votes['mean_reverting'] == max_votes:
            regime_type = RegimeType.MEAN_REVERTING
        else:
            regime_type = RegimeType.RANDOM_WALK
        
        confidence = max_votes / total_votes if total_votes > 0 else 0.5
        
        details = {
            'votes': votes,
            'autocorrelation_all': ac_dict,
            'hurst_raw': hurst,
        }
        
        return RegimeResult(
            regime_type=regime_type,
            hurst_exponent=hurst,
            autocorrelation=ac_1,
            variance_ratio=vr,
            confidence=confidence,
            details=details
        )
    
    def detect_regime_rolling(
        self,
        series: np.ndarray,
        window: int = 50,
        step: int = 10
    ) -> List[Tuple[int, RegimeResult]]:
        """
        Detect regime over rolling windows.
        
        Returns:
            List of (end_index, RegimeResult) pairs
        """
        results = []
        n = len(series)
        
        for end in range(window, n + 1, step):
            start = end - window
            window_series = series[start:end]
            result = self.detect_regime(window_series)
            results.append((end, result))
        
        return results
    
    def get_strategy_weights(
        self,
        regime: RegimeResult
    ) -> Dict[str, float]:
        """
        Get recommended strategy weights based on regime.
        
        Returns:
            Dict with weights for 'momentum' and 'mean_reversion'
        """
        if regime.regime_type == RegimeType.TRENDING:
            # Favor momentum
            momentum_weight = 0.7 + 0.3 * regime.confidence
            mean_revert_weight = 1.0 - momentum_weight
        elif regime.regime_type == RegimeType.MEAN_REVERTING:
            # Favor mean reversion
            mean_revert_weight = 0.7 + 0.3 * regime.confidence
            momentum_weight = 1.0 - mean_revert_weight
        else:
            # Random walk - reduce both or equal weight
            momentum_weight = 0.3
            mean_revert_weight = 0.3
        
        return {
            'momentum': momentum_weight,
            'mean_reversion': mean_revert_weight,
            'regime_type': regime.regime_type.value,
            'confidence': regime.confidence,
        }


def detect_category_regimes(
    df: pd.DataFrame,
    price_col: str = 'avg_price',
    category_col: str = 'category',
    min_samples: int = 50,
) -> Dict[str, RegimeResult]:
    """
    Detect regime for each category in the dataset.
    
    Returns:
        Dict mapping category name to RegimeResult
    """
    if category_col not in df.columns:
        return {}
    
    detector = RegimeDetector()
    results = {}
    
    for category, cat_data in df.groupby(category_col):
        if len(cat_data) < min_samples:
            continue
        
        # Get price series (sorted by time if available)
        if 'timestamp' in cat_data.columns:
            cat_data = cat_data.sort_values('timestamp')
        
        prices = cat_data[price_col].values
        regime = detector.detect_regime(prices)
        results[category] = regime
    
    return results


def detect_market_regimes(
    df: pd.DataFrame,
    price_col: str = 'avg_price',
    market_col: str = 'conditionId',
    min_samples: int = 20,
) -> Dict[str, RegimeResult]:
    """
    Detect regime for each individual market.
    """
    if market_col not in df.columns:
        return {}
    
    detector = RegimeDetector()
    results = {}
    
    for market_id, market_data in df.groupby(market_col):
        if len(market_data) < min_samples:
            continue
        
        if 'timestamp' in market_data.columns:
            market_data = market_data.sort_values('timestamp')
        
        prices = market_data[price_col].values
        regime = detector.detect_regime(prices)
        results[market_id] = regime
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Regime Detection')
    parser.add_argument('--data', type=str, default='optimization_cache.parquet')
    parser.add_argument('--by', type=str, default='category', 
                        choices=['category', 'market', 'overall'])
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    price_col = 'avg_price' if 'avg_price' in df.columns else 'price'
    
    if args.by == 'overall':
        prices = df[price_col].values
        detector = RegimeDetector()
        result = detector.detect_regime(prices)
        
        print(f"\nOverall Regime Detection:")
        print(f"  Regime: {result.regime_type.value}")
        print(f"  Hurst: {result.hurst_exponent:.3f}")
        print(f"  AC(1): {result.autocorrelation:.3f}")
        print(f"  VR(2): {result.variance_ratio:.3f}")
        print(f"  Confidence: {result.confidence:.2f}")
        
    elif args.by == 'category':
        results = detect_category_regimes(df, price_col)
        
        print(f"\nCategory Regime Detection ({len(results)} categories):")
        print(f"{'Category':<30} {'Regime':<15} {'Hurst':>8} {'AC(1)':>8} {'Conf':>8}")
        print("-" * 75)
        
        for cat, result in sorted(results.items(), key=lambda x: -x[1].confidence):
            print(f"{str(cat)[:30]:<30} {result.regime_type.value:<15} "
                  f"{result.hurst_exponent:>8.3f} {result.autocorrelation:>8.3f} "
                  f"{result.confidence:>8.2f}")
        
        # Summary
        trending = sum(1 for r in results.values() if r.regime_type == RegimeType.TRENDING)
        reverting = sum(1 for r in results.values() if r.regime_type == RegimeType.MEAN_REVERTING)
        random = sum(1 for r in results.values() if r.regime_type == RegimeType.RANDOM_WALK)
        
        print(f"\nSummary: Trending={trending}, Mean-Reverting={reverting}, Random Walk={random}")
        
    else:  # market
        results = detect_market_regimes(df, price_col)
        
        print(f"\nMarket Regime Detection ({len(results)} markets):")
        
        # Summary stats
        trending = sum(1 for r in results.values() if r.regime_type == RegimeType.TRENDING)
        reverting = sum(1 for r in results.values() if r.regime_type == RegimeType.MEAN_REVERTING)
        random = sum(1 for r in results.values() if r.regime_type == RegimeType.RANDOM_WALK)
        
        print(f"Summary: Trending={trending}, Mean-Reverting={reverting}, Random Walk={random}")
        
        avg_hurst = np.mean([r.hurst_exponent for r in results.values()])
        print(f"Average Hurst: {avg_hurst:.3f}")
