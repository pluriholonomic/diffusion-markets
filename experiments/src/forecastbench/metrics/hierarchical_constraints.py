"""
Hierarchical Constraint Sets for H4: Arbitrage Bounds via d(q, C_t).

This module implements the hierarchical constraint framework:
- Inner constraints: Multicalibration (E[Y-p | group, bin(p)] = 0)
- Outer constraints: Fréchet cross-market bounds (P(A∧B) ≤ min(P(A), P(B)))

The distance d(q, C_t) to this hierarchical constraint set provides
an upper bound on extractable statistical arbitrage.

Key insight: The constraint set forms a ladder:
  C_outer ⊃ C_inner  (Fréchet is weaker than multicalibration)
  
Violations at each level correspond to different arbitrage opportunities:
- Fréchet violations → static arbitrage across related markets
- Multicalibration violations → statistical arbitrage via group-conditional miscalibration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Multicalibration Tracker
# =============================================================================

@dataclass
class MulticalibrationTracker:
    """
    Track multicalibration constraints for arbitrary group crossings.
    
    Multicalibration requires:
        E[Y - p | group, bin(p)] = 0 for all (group, bin) pairs
    
    This generalizes simple calibration (bin-only) to group-conditional calibration,
    which is the inner constraint in our hierarchical framework.
    
    Groups can be arbitrary categorical features (topic, volume bucket, etc.)
    or crossings thereof (topic × volume_q5).
    
    Attributes:
        group_cols: List of column names to cross for group definition
        n_bins: Number of prediction bins
        
    Usage:
        tracker = MulticalibrationTracker(group_cols=["topic", "volume_q5"], n_bins=10)
        for batch in data:
            tracker.update(p=batch["pred"], y=batch["y"], df=batch["df"])
        metrics = tracker.get_metrics()
    """
    
    group_cols: List[str] = field(default_factory=lambda: ["topic"])
    n_bins: int = 10
    
    # Internal state
    _group_bin_sums: Dict[Tuple, np.ndarray] = field(default_factory=dict)
    _group_bin_counts: Dict[Tuple, np.ndarray] = field(default_factory=dict)
    _step: int = 0
    _history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self._group_bin_sums = {}
        self._group_bin_counts = {}
        self._step = 0
        self._history = []
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._group_bin_sums = {}
        self._group_bin_counts = {}
        self._step = 0
        self._history = []
    
    def _get_group_key(self, df: pd.DataFrame, idx: int) -> Tuple:
        """Extract group key for a single row."""
        key_parts = []
        for col in self.group_cols:
            if col in df.columns:
                val = df.iloc[idx][col]
                key_parts.append(str(val) if pd.notna(val) else "__NA__")
            else:
                key_parts.append("__MISSING__")
        return tuple(key_parts)
    
    def _get_bin_idx(self, p: float) -> int:
        """Get bin index for a prediction."""
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        idx = np.searchsorted(bin_edges[1:], p, side='right')
        return min(idx, self.n_bins - 1)
    
    def update(
        self,
        p: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        *,
        log_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Update constraint tracking with new batch.
        
        Args:
            p: Predictions (N,)
            y: Outcomes (N,)
            df: DataFrame with group columns (must have same length as p, y)
            log_every: Log metrics every N steps
            
        Returns:
            Current constraint violation metrics
        """
        p = np.asarray(p, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        n = len(p)
        
        if len(df) != n:
            raise ValueError(f"df length {len(df)} != p length {n}")
        
        # Update cumulative sums for each (group, bin) pair
        for i in range(n):
            group_key = self._get_group_key(df, i)
            bin_idx = self._get_bin_idx(p[i])
            residual = y[i] - p[i]
            
            if group_key not in self._group_bin_sums:
                self._group_bin_sums[group_key] = np.zeros(self.n_bins, dtype=np.float64)
                self._group_bin_counts[group_key] = np.zeros(self.n_bins, dtype=np.float64)
            
            self._group_bin_sums[group_key][bin_idx] += residual
            self._group_bin_counts[group_key][bin_idx] += 1
        
        self._step += n
        
        # Compute current violations
        metrics = self._compute_violations()
        
        if self._step % log_every == 0:
            self._history.append({
                "step": self._step,
                **metrics,
            })
        
        return metrics
    
    def _compute_violations(self) -> Dict[str, Any]:
        """Compute current constraint violations across all (group, bin) pairs."""
        all_violations = []
        group_max_violations = {}
        
        for group_key, sums in self._group_bin_sums.items():
            counts = self._group_bin_counts[group_key]
            
            group_violations = []
            for b in range(self.n_bins):
                if counts[b] > 0:
                    violation = abs(sums[b] / counts[b])
                    all_violations.append(violation)
                    group_violations.append(violation)
            
            if group_violations:
                group_max_violations[group_key] = max(group_violations)
        
        if not all_violations:
            return {
                "max_violation": 0.0,
                "mean_violation": 0.0,
                "worst_group": None,
                "worst_group_violation": 0.0,
                "n_groups": 0,
                "n_active_constraints": 0,
            }
        
        worst_group = max(group_max_violations.keys(), key=lambda k: group_max_violations[k])
        
        return {
            "max_violation": float(np.max(all_violations)),
            "mean_violation": float(np.mean(all_violations)),
            "worst_group": worst_group,
            "worst_group_violation": float(group_max_violations[worst_group]),
            "n_groups": len(group_max_violations),
            "n_active_constraints": len(all_violations),
        }
    
    def get_violation_matrix(self) -> Tuple[np.ndarray, List[Tuple], List[int]]:
        """
        Get (n_groups, n_bins) matrix of current violations.
        
        Returns:
            (matrix, group_keys, bin_indices) where matrix[i,j] is the violation
            for group_keys[i] in bin j.
        """
        group_keys = sorted(self._group_bin_sums.keys())
        n_groups = len(group_keys)
        
        matrix = np.zeros((n_groups, self.n_bins), dtype=np.float64)
        
        for i, group_key in enumerate(group_keys):
            sums = self._group_bin_sums[group_key]
            counts = self._group_bin_counts[group_key]
            for b in range(self.n_bins):
                if counts[b] > 0:
                    matrix[i, b] = abs(sums[b] / counts[b])
        
        return matrix, group_keys, list(range(self.n_bins))
    
    def get_history(self) -> List[Dict]:
        """Get history of logged metrics."""
        return self._history
    
    def compute_distance_to_C(self) -> float:
        """
        Compute distance to multicalibration constraint set.
        
        Distance = max over all (group, bin) of |E[Y-p | group, bin]|
        """
        metrics = self._compute_violations()
        return metrics["max_violation"]


# =============================================================================
# Fréchet Constraint Tracker
# =============================================================================

@dataclass
class FrechetConstraintTracker:
    """
    Track Fréchet cross-market constraints for bundled markets.
    
    For markets A, B, and A∧B (conjunction), Fréchet bounds require:
        max(0, P(A) + P(B) - 1) ≤ P(A∧B) ≤ min(P(A), P(B))
    
    This is the outer constraint in our hierarchical framework.
    
    We use category-based bundling to group related markets together,
    then check if the joint predictions satisfy Fréchet constraints.
    
    Attributes:
        bundle_col: Column name for bundling (e.g., "category")
        bundle_size: Number of markets per bundle (default 3 for A, B, A∧B triples)
        constraint_type: Type of constraint ("frechet", "implication", "mutual_exclusion")
    """
    
    bundle_col: str = "category"
    bundle_size: int = 3
    constraint_type: str = "frechet"
    
    # Internal state
    _bundle_violations: List[np.ndarray] = field(default_factory=list)
    _n_bundles_seen: int = 0
    _history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self._bundle_violations = []
        self._n_bundles_seen = 0
        self._history = []
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._bundle_violations = []
        self._n_bundles_seen = 0
        self._history = []
    
    def compute_bundle_violations(
        self,
        p: np.ndarray,
        *,
        include_box: bool = False,
    ) -> np.ndarray:
        """
        Compute Fréchet violations for a single bundle.
        
        Args:
            p: Predictions for bundle (bundle_size,)
            include_box: Whether to include box constraints
            
        Returns:
            Violation vector where positive values indicate constraint violation
        """
        p = np.asarray(p, dtype=np.float64).flatten()
        
        if self.constraint_type == "frechet" and len(p) >= 3:
            # Interpret as (p_a, p_b, p_ab) triple
            p_a, p_b, p_ab = p[0], p[1], p[2]
            
            violations = []
            # Upper bound: p_ab <= min(p_a, p_b)
            violations.append(max(0, p_ab - min(p_a, p_b)))
            # Lower bound: p_ab >= max(0, p_a + p_b - 1)
            violations.append(max(0, max(0, p_a + p_b - 1) - p_ab))
            
            if include_box:
                for pi in [p_a, p_b, p_ab]:
                    violations.append(max(0, -pi))  # pi >= 0
                    violations.append(max(0, pi - 1))  # pi <= 1
            
            return np.array(violations, dtype=np.float64)
        
        elif self.constraint_type == "implication" and len(p) >= 2:
            # A => B means p_a <= p_b
            p_a, p_b = p[0], p[1]
            violations = [max(0, p_a - p_b)]
            return np.array(violations, dtype=np.float64)
        
        elif self.constraint_type == "mutual_exclusion":
            # Sum <= 1
            violations = [max(0, np.sum(p) - 1)]
            return np.array(violations, dtype=np.float64)
        
        else:
            # No constraints for unknown type
            return np.zeros(0, dtype=np.float64)
    
    def update_from_bundles(
        self,
        bundle_idx: np.ndarray,
        mask: np.ndarray,
        p: np.ndarray,
        *,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Update with pre-computed bundles.
        
        Args:
            bundle_idx: (n_bundles, bundle_size) array of row indices
            mask: (n_bundles, bundle_size) bool array (True = valid)
            p: (N,) array of predictions
            log_every: Log metrics every N bundles
            
        Returns:
            Current constraint violation metrics
        """
        n_bundles = bundle_idx.shape[0]
        
        for i in range(n_bundles):
            valid_mask = mask[i]
            if valid_mask.sum() < self.bundle_size:
                continue  # Skip incomplete bundles
            
            idx = bundle_idx[i]
            bundle_p = p[idx[valid_mask]]
            
            violations = self.compute_bundle_violations(bundle_p)
            if len(violations) > 0:
                self._bundle_violations.append(violations)
                self._n_bundles_seen += 1
        
        metrics = self._compute_metrics()
        
        if self._n_bundles_seen % log_every == 0 and self._n_bundles_seen > 0:
            self._history.append({
                "n_bundles": self._n_bundles_seen,
                **metrics,
            })
        
        return metrics
    
    def update(
        self,
        df: pd.DataFrame,
        p: np.ndarray,
        *,
        seed: int = 0,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Update by creating bundles from DataFrame.
        
        Args:
            df: DataFrame with bundle_col column
            p: (N,) array of predictions
            seed: RNG seed for bundle creation
            log_every: Log metrics every N bundles
            
        Returns:
            Current constraint violation metrics
        """
        from forecastbench.data.bundles import make_group_bundles
        
        df_reset = df.reset_index(drop=True)
        p = np.asarray(p, dtype=np.float64).flatten()
        
        if self.bundle_col not in df_reset.columns:
            return {
                "max_violation": 0.0,
                "mean_violation": 0.0,
                "frac_violated": 0.0,
                "n_bundles": 0,
                "error": f"Missing bundle column: {self.bundle_col}",
            }
        
        bundle_idx, mask = make_group_bundles(
            df_reset,
            group_col=self.bundle_col,
            bundle_size=self.bundle_size,
            seed=seed,
            drop_last=True,  # Only complete bundles for Fréchet
        )
        
        return self.update_from_bundles(bundle_idx, mask, p, log_every=log_every)
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute current constraint metrics."""
        if not self._bundle_violations:
            return {
                "max_violation": 0.0,
                "mean_violation": 0.0,
                "frac_violated": 0.0,
                "n_bundles": 0,
            }
        
        all_violations = np.concatenate(self._bundle_violations)
        
        return {
            "max_violation": float(np.max(all_violations)),
            "mean_violation": float(np.mean(all_violations)),
            "frac_violated": float(np.mean(all_violations > 0.01)),
            "n_bundles": self._n_bundles_seen,
        }
    
    def get_history(self) -> List[Dict]:
        """Get history of logged metrics."""
        return self._history
    
    def compute_distance_to_C(self) -> float:
        """
        Compute distance to Fréchet constraint set.
        
        Distance = max over all bundles of max constraint violation
        """
        metrics = self._compute_metrics()
        return metrics["max_violation"]


# =============================================================================
# Hierarchical Constraint Set
# =============================================================================

@dataclass
class HierarchicalConstraintSet:
    """
    Hierarchical constraint set combining multicalibration (inner) + Fréchet (outer).
    
    The constraint set C_t is defined as:
        C_t = C_multicalib ∩ C_frechet
    
    Distance to C_t is:
        d(q, C_t) = max(d_multicalib(q), d_frechet(q))
    
    This distance provides an upper bound on extractable statistical arbitrage.
    
    Attributes:
        multicalib_tracker: Multicalibration constraint tracker
        frechet_tracker: Fréchet constraint tracker
    """
    
    multicalib_tracker: MulticalibrationTracker
    frechet_tracker: FrechetConstraintTracker
    
    @classmethod
    def create(
        cls,
        *,
        group_cols: List[str] = None,
        n_bins: int = 10,
        bundle_col: str = "category",
        bundle_size: int = 3,
        constraint_type: str = "frechet",
    ) -> "HierarchicalConstraintSet":
        """
        Factory method to create hierarchical constraint set.
        
        Args:
            group_cols: Columns for multicalibration groups
            n_bins: Number of prediction bins
            bundle_col: Column for Fréchet bundling
            bundle_size: Markets per bundle
            constraint_type: Type of Fréchet constraint
            
        Returns:
            Configured HierarchicalConstraintSet
        """
        if group_cols is None:
            group_cols = ["topic", "volume_q5"]
        
        multicalib = MulticalibrationTracker(
            group_cols=group_cols,
            n_bins=n_bins,
        )
        
        frechet = FrechetConstraintTracker(
            bundle_col=bundle_col,
            bundle_size=bundle_size,
            constraint_type=constraint_type,
        )
        
        return cls(
            multicalib_tracker=multicalib,
            frechet_tracker=frechet,
        )
    
    def reset(self) -> None:
        """Reset both trackers."""
        self.multicalib_tracker.reset()
        self.frechet_tracker.reset()
    
    def update(
        self,
        p: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        *,
        seed: int = 0,
        log_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Update both constraint trackers.
        
        Args:
            p: Predictions (N,)
            y: Outcomes (N,)
            df: DataFrame with group and bundle columns
            seed: RNG seed for bundling
            log_every: Log frequency
            
        Returns:
            Combined constraint metrics
        """
        multicalib_metrics = self.multicalib_tracker.update(
            p=p, y=y, df=df, log_every=log_every
        )
        
        frechet_metrics = self.frechet_tracker.update(
            df=df, p=p, seed=seed, log_every=log_every
        )
        
        # Combined distance is max of both
        d_multicalib = multicalib_metrics["max_violation"]
        d_frechet = frechet_metrics["max_violation"]
        
        return {
            "distance_to_C": float(max(d_multicalib, d_frechet)),
            "d_multicalib": float(d_multicalib),
            "d_frechet": float(d_frechet),
            "multicalib": multicalib_metrics,
            "frechet": frechet_metrics,
        }
    
    def compute_distance_to_C(self) -> float:
        """
        Compute distance to hierarchical constraint set C_t.
        
        Returns:
            max(d_multicalib, d_frechet)
        """
        return max(
            self.multicalib_tracker.compute_distance_to_C(),
            self.frechet_tracker.compute_distance_to_C(),
        )


# =============================================================================
# Arbitrage Bound Computation
# =============================================================================

def compute_arbitrage_bound_hierarchical(
    q_market: np.ndarray,
    p_model: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    *,
    group_cols: List[str] = None,
    n_bins: int = 10,
    bundle_col: str = "category",
    bundle_size: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute statistical arbitrage bounds using hierarchical constraint distance.
    
    KEY INSIGHT: The distance d(q, C_t) from market prices q to the hierarchical
    constraint set C_t (multicalibration ∩ Fréchet) provides an upper bound on
    extractable statistical arbitrage.
    
    We measure:
    1. Distance to C_t for market prices vs model predictions
    2. Realized PnL from betting on model vs market
    3. Correlation between distance and profit
    
    Args:
        q_market: Market prices (N,)
        p_model: Model predictions (N,)
        y: Realized outcomes (N,)
        df: DataFrame with group and bundle columns
        group_cols: Columns for multicalibration
        n_bins: Prediction bins
        bundle_col: Column for Fréchet bundling
        bundle_size: Markets per bundle
        seed: RNG seed
        
    Returns:
        Dict with arbitrage bounds and analysis
    """
    if group_cols is None:
        group_cols = ["topic", "volume_q5"]
    
    q_market = np.asarray(q_market, dtype=np.float64).flatten()
    p_model = np.asarray(p_model, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = len(y)
    
    # Create constraint trackers for market and model
    market_constraints = HierarchicalConstraintSet.create(
        group_cols=group_cols,
        n_bins=n_bins,
        bundle_col=bundle_col,
        bundle_size=bundle_size,
    )
    
    model_constraints = HierarchicalConstraintSet.create(
        group_cols=group_cols,
        n_bins=n_bins,
        bundle_col=bundle_col,
        bundle_size=bundle_size,
    )
    
    # Update both trackers
    market_metrics = market_constraints.update(
        p=q_market, y=y, df=df, seed=seed
    )
    
    model_metrics = model_constraints.update(
        p=p_model, y=y, df=df, seed=seed
    )
    
    # Compute realized PnL
    position = p_model - q_market  # Long if model > market
    pnl = position * (y - q_market)  # Profit = position * (outcome - price)
    
    # Correlation between distance and profit
    market_distance = market_metrics["distance_to_C"]
    model_distance = model_metrics["distance_to_C"]
    
    # Arbitrage capture rate
    total_arbitrage = float(np.sum(np.abs(y - q_market)))
    captured_arbitrage = float(np.sum(pnl[pnl > 0]))
    arbitrage_capture_rate = captured_arbitrage / max(total_arbitrage, 1e-10)
    
    # Sharpe ratio
    if np.std(pnl) > 1e-10:
        sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(252))
    else:
        sharpe = 0.0
    
    return {
        "n": n,
        "market_distance_to_C": float(market_distance),
        "model_distance_to_C": float(model_distance),
        "distance_reduction": float(1 - model_distance / max(market_distance, 1e-10)),
        "market_metrics": market_metrics,
        "model_metrics": model_metrics,
        "mean_pnl": float(np.mean(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "sharpe_ratio": sharpe,
        "arbitrage_capture_rate": arbitrage_capture_rate,
        "interpretation": (
            f"Market distance to C_t: {market_distance:.4f} "
            f"(d_multicalib={market_metrics['d_multicalib']:.4f}, "
            f"d_frechet={market_metrics['d_frechet']:.4f}). "
            f"Model reduces to {model_distance:.4f} "
            f"({100*(1-model_distance/max(market_distance,1e-10)):.1f}% reduction). "
            f"Sharpe: {sharpe:.2f}"
        ),
    }


def compare_arbitrage_detection_hierarchical(
    q_market: np.ndarray,
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    *,
    group_cols: List[str] = None,
    n_bins: int = 10,
    bundle_col: str = "category",
    bundle_size: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compare AR-only vs AR+Diffusion for statistical arbitrage using hierarchical constraints.
    
    This directly tests H3 and H4:
    - H3: Does diffusion improve learning of C_t?
    - H4: Does d(q, C_t) bound extractable profit?
    
    Args:
        q_market: Market prices
        p_ar: AR-only predictions
        p_hybrid: AR+Diffusion predictions
        y: Realized outcomes
        df: DataFrame with group and bundle columns
        group_cols: Columns for multicalibration
        n_bins: Prediction bins
        bundle_col: Column for Fréchet bundling
        bundle_size: Markets per bundle
        seed: RNG seed
        
    Returns:
        Comparative arbitrage detection results
    """
    ar_arb = compute_arbitrage_bound_hierarchical(
        q_market=q_market,
        p_model=p_ar,
        y=y,
        df=df,
        group_cols=group_cols,
        n_bins=n_bins,
        bundle_col=bundle_col,
        bundle_size=bundle_size,
        seed=seed,
    )
    
    hybrid_arb = compute_arbitrage_bound_hierarchical(
        q_market=q_market,
        p_model=p_hybrid,
        y=y,
        df=df,
        group_cols=group_cols,
        n_bins=n_bins,
        bundle_col=bundle_col,
        bundle_size=bundle_size,
        seed=seed,
    )
    
    return {
        "ar": ar_arb,
        "hybrid": hybrid_arb,
        "improvement": {
            "distance_reduction_gain": (
                hybrid_arb["distance_reduction"] - ar_arb["distance_reduction"]
            ),
            "pnl_improvement": hybrid_arb["mean_pnl"] - ar_arb["mean_pnl"],
            "pnl_improvement_pct": (
                (hybrid_arb["mean_pnl"] - ar_arb["mean_pnl"]) / 
                max(abs(ar_arb["mean_pnl"]), 1e-10)
            ),
            "capture_rate_improvement": (
                hybrid_arb["arbitrage_capture_rate"] - ar_arb["arbitrage_capture_rate"]
            ),
            "sharpe_improvement": hybrid_arb["sharpe_ratio"] - ar_arb["sharpe_ratio"],
            "multicalib_improvement": (
                ar_arb["model_metrics"]["d_multicalib"] - 
                hybrid_arb["model_metrics"]["d_multicalib"]
            ),
            "frechet_improvement": (
                ar_arb["model_metrics"]["d_frechet"] - 
                hybrid_arb["model_metrics"]["d_frechet"]
            ),
        },
        "summary": (
            f"AR captures {ar_arb['arbitrage_capture_rate']*100:.1f}% of arbitrage, "
            f"Hybrid captures {hybrid_arb['arbitrage_capture_rate']*100:.1f}% "
            f"(+{(hybrid_arb['arbitrage_capture_rate']-ar_arb['arbitrage_capture_rate'])*100:.1f}pp). "
            f"Sharpe: AR={ar_arb['sharpe_ratio']:.2f}, Hybrid={hybrid_arb['sharpe_ratio']:.2f}"
        ),
        "h3_supported": (
            hybrid_arb["model_distance_to_C"] < ar_arb["model_distance_to_C"]
        ),
        "h4_supported": (
            hybrid_arb["arbitrage_capture_rate"] > ar_arb["arbitrage_capture_rate"]
        ),
    }



