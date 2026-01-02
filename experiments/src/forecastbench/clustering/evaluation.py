"""
Evaluation Framework for Clustering Algorithms.

Provides metrics and utilities for comparing clustering algorithms
on prediction market data, including:
- Cluster quality metrics
- Correlation prediction accuracy
- Trading performance
- Resolution robustness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import spearmanr, pearsonr
import warnings

from forecastbench.clustering.base import OnlineClusteringBase


@dataclass
class ClusteringMetrics:
    """Container for clustering evaluation metrics."""
    
    # Cluster quality
    n_clusters: int = 0
    avg_cluster_size: float = 0.0
    cluster_size_std: float = 0.0
    
    # Agreement with ground truth (if available)
    ari: float = 0.0  # Adjusted Rand Index
    nmi: float = 0.0  # Normalized Mutual Information
    
    # Internal validity
    silhouette: float = 0.0
    intra_cluster_corr: float = 0.0
    inter_cluster_corr: float = 0.0
    correlation_ratio: float = 0.0  # intra / inter
    
    # Stability
    cluster_stability: float = 0.0  # 1 - fraction of reassignments
    
    # Correlation prediction
    correlation_mae: float = 0.0
    correlation_rmse: float = 0.0
    correlation_spearman: float = 0.0
    
    # Computational
    update_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "avg_cluster_size": self.avg_cluster_size,
            "cluster_size_std": self.cluster_size_std,
            "ari": self.ari,
            "nmi": self.nmi,
            "silhouette": self.silhouette,
            "intra_cluster_corr": self.intra_cluster_corr,
            "inter_cluster_corr": self.inter_cluster_corr,
            "correlation_ratio": self.correlation_ratio,
            "cluster_stability": self.cluster_stability,
            "correlation_mae": self.correlation_mae,
            "correlation_rmse": self.correlation_rmse,
            "correlation_spearman": self.correlation_spearman,
            "update_time_ms": self.update_time_ms,
        }


class ClusteringEvaluator:
    """
    Evaluator for comparing clustering algorithms.
    
    Provides methods for:
    1. Running algorithms on datasets
    2. Computing evaluation metrics
    3. Statistical comparison between algorithms
    
    Example:
        evaluator = ClusteringEvaluator()
        
        # Run on synthetic data
        metrics = evaluator.evaluate_on_synthetic(
            algorithm=SWOCC(),
            generator=BlockCorrelationGenerator(),
        )
        
        # Compare algorithms
        comparison = evaluator.compare_algorithms(
            algorithms={"SWOCC": SWOCC(), "OLRCM": OLRCM()},
            price_data=df,
        )
    """
    
    def __init__(self):
        self._metrics_history: List[Dict[str, Any]] = []
    
    def evaluate_clustering(
        self,
        algorithm: OnlineClusteringBase,
        price_data: np.ndarray,
        death_events: List[Tuple[int, str]],
        market_ids: Optional[List[str]] = None,
        true_labels: Optional[np.ndarray] = None,
        true_correlation: Optional[np.ndarray] = None,
    ) -> ClusteringMetrics:
        """
        Evaluate a clustering algorithm on a dataset.
        
        Args:
            algorithm: Clustering algorithm to evaluate
            price_data: (T, n_markets) array of prices
            death_events: List of (timestep, market_id) death events
            market_ids: Optional list of market IDs
            true_labels: Optional ground truth cluster labels
            true_correlation: Optional ground truth correlation matrix
            
        Returns:
            ClusteringMetrics with evaluation results
        """
        import time
        
        T, n_markets = price_data.shape
        if market_ids is None:
            market_ids = [f"market_{i}" for i in range(n_markets)]
        
        # Reset algorithm
        algorithm.reset()
        
        # Initialize markets
        for i, mid in enumerate(market_ids):
            algorithm.add_market(mid, timestamp=0.0, initial_price=price_data[0, i])
        
        # Create death lookup
        death_lookup = {}
        for t, mid in death_events:
            death_lookup.setdefault(t, []).append(mid)
        
        # Track assignments for stability
        prev_assignments: Dict[str, int] = {}
        reassignments = 0
        total_checks = 0
        
        # Process time series
        update_times = []
        for t in range(1, T):
            prices = {
                mid: price_data[t, i]
                for i, mid in enumerate(market_ids)
                if mid in algorithm._markets and algorithm._markets[mid].is_active
            }
            
            start = time.time()
            algorithm.update(timestamp=float(t), prices=prices)
            update_times.append(time.time() - start)
            
            # Handle deaths
            for mid in death_lookup.get(t, []):
                algorithm.remove_market(mid, timestamp=float(t))
            
            # Track stability
            for mid in prices:
                assignment = algorithm.get_cluster_for_market(mid)
                if assignment is not None:
                    if mid in prev_assignments and prev_assignments[mid] != assignment:
                        reassignments += 1
                    prev_assignments[mid] = assignment
                    total_checks += 1
        
        # Compute metrics
        metrics = ClusteringMetrics()
        
        # Cluster sizes
        clusters = algorithm.get_clusters()
        if clusters:
            sizes = [len(members) for members in clusters.values()]
            metrics.n_clusters = len(clusters)
            metrics.avg_cluster_size = np.mean(sizes)
            metrics.cluster_size_std = np.std(sizes)
        
        # Stability
        if total_checks > 0:
            metrics.cluster_stability = 1 - reassignments / total_checks
        
        # Update time
        metrics.update_time_ms = np.mean(update_times) * 1000 if update_times else 0
        
        # Get predicted labels and correlation
        pred_labels = np.zeros(n_markets, dtype=np.int32)
        alive_mask = np.zeros(n_markets, dtype=bool)
        
        for i, mid in enumerate(market_ids):
            assignment = algorithm.get_cluster_for_market(mid)
            if assignment is not None:
                pred_labels[i] = assignment
                alive_mask[i] = True
            else:
                pred_labels[i] = -1
        
        # Agreement with ground truth
        if true_labels is not None:
            valid = (pred_labels >= 0) & (true_labels >= 0)
            if np.sum(valid) > 1:
                metrics.ari = adjusted_rand_score(true_labels[valid], pred_labels[valid])
                metrics.nmi = normalized_mutual_info_score(true_labels[valid], pred_labels[valid])
        
        # Correlation-based metrics
        market_ids_corr, pred_corr = algorithm.get_correlation_matrix()
        
        if len(market_ids_corr) >= 2:
            # Intra/inter cluster correlation
            metrics.intra_cluster_corr, metrics.inter_cluster_corr = self._compute_intra_inter_corr(
                algorithm, market_ids_corr, pred_corr
            )
            if metrics.inter_cluster_corr > 1e-6:
                metrics.correlation_ratio = metrics.intra_cluster_corr / metrics.inter_cluster_corr
            
            # Silhouette score
            try:
                labels_for_silhouette = [
                    algorithm.get_cluster_for_market(mid) or 0
                    for mid in market_ids_corr
                ]
                if len(set(labels_for_silhouette)) > 1:
                    # Use correlation-based distance
                    dist = 1 - np.abs(pred_corr)
                    np.fill_diagonal(dist, 0)
                    metrics.silhouette = silhouette_score(dist, labels_for_silhouette, metric='precomputed')
            except Exception:
                pass
        
        # Correlation prediction accuracy
        if true_correlation is not None and len(market_ids_corr) >= 2:
            metrics.correlation_mae, metrics.correlation_rmse, metrics.correlation_spearman = \
                self._compute_correlation_accuracy(
                    market_ids_corr, market_ids, pred_corr, true_correlation
                )
        
        return metrics
    
    def _compute_intra_inter_corr(
        self,
        algorithm: OnlineClusteringBase,
        market_ids: List[str],
        corr_matrix: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute average intra and inter cluster correlations."""
        n = len(market_ids)
        if n < 2:
            return 1.0, 0.0
        
        id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
        
        intra_sum = 0.0
        intra_count = 0
        inter_sum = 0.0
        inter_count = 0
        
        clusters = algorithm.get_clusters()
        
        for cluster_id, members in clusters.items():
            members_in_corr = [m for m in members if m in id_to_idx]
            
            # Intra-cluster
            for i, m1 in enumerate(members_in_corr):
                for m2 in members_in_corr[i+1:]:
                    idx1, idx2 = id_to_idx[m1], id_to_idx[m2]
                    intra_sum += abs(corr_matrix[idx1, idx2])
                    intra_count += 1
            
            # Inter-cluster
            for other_id, other_members in clusters.items():
                if other_id <= cluster_id:
                    continue
                
                other_in_corr = [m for m in other_members if m in id_to_idx]
                
                for m1 in members_in_corr:
                    for m2 in other_in_corr:
                        idx1, idx2 = id_to_idx[m1], id_to_idx[m2]
                        inter_sum += abs(corr_matrix[idx1, idx2])
                        inter_count += 1
        
        intra = intra_sum / intra_count if intra_count > 0 else 1.0
        inter = inter_sum / inter_count if inter_count > 0 else 0.0
        
        return intra, inter
    
    def _compute_correlation_accuracy(
        self,
        pred_market_ids: List[str],
        all_market_ids: List[str],
        pred_corr: np.ndarray,
        true_corr: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute correlation prediction accuracy."""
        # Build mapping
        all_id_to_idx = {mid: i for i, mid in enumerate(all_market_ids)}
        pred_id_to_idx = {mid: i for i, mid in enumerate(pred_market_ids)}
        
        # Collect pairs
        true_vals = []
        pred_vals = []
        
        for i, m1 in enumerate(pred_market_ids):
            for j, m2 in enumerate(pred_market_ids[i+1:], i+1):
                if m1 not in all_id_to_idx or m2 not in all_id_to_idx:
                    continue
                
                true_i, true_j = all_id_to_idx[m1], all_id_to_idx[m2]
                
                if true_i < true_corr.shape[0] and true_j < true_corr.shape[1]:
                    true_vals.append(true_corr[true_i, true_j])
                    pred_vals.append(pred_corr[i, j])
        
        if len(true_vals) < 2:
            return 0.0, 0.0, 0.0
        
        true_vals = np.array(true_vals)
        pred_vals = np.array(pred_vals)
        
        mae = np.mean(np.abs(true_vals - pred_vals))
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman, _ = spearmanr(true_vals, pred_vals)
            if np.isnan(spearman):
                spearman = 0.0
        
        return mae, rmse, spearman
    
    def compare_algorithms(
        self,
        algorithms: Dict[str, OnlineClusteringBase],
        price_data: np.ndarray,
        death_events: List[Tuple[int, str]],
        market_ids: Optional[List[str]] = None,
        true_labels: Optional[np.ndarray] = None,
        true_correlation: Optional[np.ndarray] = None,
        n_runs: int = 1,
    ) -> Dict[str, ClusteringMetrics]:
        """
        Compare multiple algorithms on the same dataset.
        
        Args:
            algorithms: Dict mapping name -> algorithm instance
            price_data: Price data array
            death_events: Death events
            market_ids: Market IDs
            true_labels: Ground truth labels
            true_correlation: Ground truth correlation
            n_runs: Number of runs for averaging
            
        Returns:
            Dict mapping algorithm name -> averaged metrics
        """
        results = {}
        
        for name, algo in algorithms.items():
            run_metrics = []
            
            for _ in range(n_runs):
                metrics = self.evaluate_clustering(
                    algorithm=algo,
                    price_data=price_data,
                    death_events=death_events,
                    market_ids=market_ids,
                    true_labels=true_labels,
                    true_correlation=true_correlation,
                )
                run_metrics.append(metrics)
            
            # Average metrics
            if n_runs == 1:
                results[name] = run_metrics[0]
            else:
                results[name] = self._average_metrics(run_metrics)
        
        return results
    
    def _average_metrics(self, metrics_list: List[ClusteringMetrics]) -> ClusteringMetrics:
        """Average multiple ClusteringMetrics."""
        avg = ClusteringMetrics()
        n = len(metrics_list)
        
        for field_name in avg.__dataclass_fields__:
            values = [getattr(m, field_name) for m in metrics_list]
            setattr(avg, field_name, np.mean(values))
        
        return avg
    
    def statistical_comparison(
        self,
        results: Dict[str, List[float]],
        baseline: str = "Baseline",
        confidence: float = 0.95,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Statistical comparison of algorithm performance.
        
        Args:
            results: Dict mapping algorithm name -> list of metric values
            baseline: Name of baseline algorithm
            confidence: Confidence level for intervals
            
        Returns:
            Dict with statistical comparison results
        """
        from scipy import stats
        
        comparison = {}
        
        baseline_vals = np.array(results.get(baseline, []))
        
        for name, vals in results.items():
            vals = np.array(vals)
            n = len(vals)
            
            if n == 0:
                continue
            
            # Point estimates
            mean = np.mean(vals)
            std = np.std(vals)
            
            # Bootstrap CI
            ci_low, ci_high = self._bootstrap_ci(vals, confidence)
            
            comp = {
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
            }
            
            # Compare to baseline
            if name != baseline and len(baseline_vals) > 0:
                # Wilcoxon signed-rank test (if paired)
                if len(vals) == len(baseline_vals):
                    try:
                        stat, pval = stats.wilcoxon(vals, baseline_vals)
                        comp["vs_baseline_pval"] = pval
                    except Exception:
                        comp["vs_baseline_pval"] = 1.0
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((std ** 2 + np.std(baseline_vals) ** 2) / 2)
                if pooled_std > 0:
                    comp["cohens_d"] = (mean - np.mean(baseline_vals)) / pooled_std
                else:
                    comp["cohens_d"] = 0.0
            
            comparison[name] = comp
        
        return comparison
    
    def _bootstrap_ci(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(values)
        if n < 2:
            return float(values[0]) if n == 1 else (0.0, 0.0)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return ci_low, ci_high


def evaluate_resolution_robustness(
    algorithm: OnlineClusteringBase,
    price_data: np.ndarray,
    death_times: np.ndarray,
    market_ids: List[str],
    true_correlation: np.ndarray,
    ttr_bins: List[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate clustering accuracy as function of time-to-resolution.
    
    Args:
        algorithm: Clustering algorithm
        price_data: (T, n_markets) price array
        death_times: Array of death times per market
        market_ids: Market IDs
        true_correlation: Ground truth correlation
        ttr_bins: List of (low, high) TTR bins
        
    Returns:
        Dict mapping bin_label -> metrics
    """
    if ttr_bins is None:
        ttr_bins = [(0, 1), (1, 7), (7, 30), (30, 90), (90, float('inf'))]
    
    bin_labels = [f"{low}-{high}d" for low, high in ttr_bins]
    
    T, n_markets = price_data.shape
    evaluator = ClusteringEvaluator()
    
    results = {}
    
    for (low, high), label in zip(ttr_bins, bin_labels):
        # Find markets in this TTR bin at final time
        mask = (death_times - T >= low) & (death_times - T < high)
        
        if np.sum(mask) < 2:
            continue
        
        # Subset data
        subset_indices = np.where(mask)[0]
        subset_ids = [market_ids[i] for i in subset_indices]
        subset_prices = price_data[:, subset_indices]
        subset_corr = true_correlation[np.ix_(subset_indices, subset_indices)]
        
        # Create death events for subset
        death_events = []
        for i, idx in enumerate(subset_indices):
            if death_times[idx] < T:
                death_events.append((int(death_times[idx]), subset_ids[i]))
        
        # Evaluate
        algorithm.reset()
        metrics = evaluator.evaluate_clustering(
            algorithm=algorithm,
            price_data=subset_prices,
            death_events=death_events,
            market_ids=subset_ids,
            true_correlation=subset_corr,
        )
        
        results[label] = metrics.to_dict()
    
    return results
