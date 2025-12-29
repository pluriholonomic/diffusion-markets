"""
C_t Sample Approximation Validation.

Validates that the MC samples used to represent C_t are sufficient
to accurately approximate the learned constraint set.

Background
----------
The Blackwell constraint set C_t is approximated by sampling from a trained
diffusion model. The key question is: how many samples are needed?

Theory:
- The projection d(q, C_t) uses conv(samples) to approximate C_t
- As n_samples → ∞, conv(samples) → C_t (by law of large numbers)
- But we need finite n_samples for tractable computation

The sufficiency of n_samples depends on:
1. Dimension k (number of markets in the bundle)
2. Geometry of C_t (smooth vs. complex boundary)
3. Required accuracy for downstream decisions

Key Tests
---------
1. **Sample convergence**: Does projection distance stabilize as n increases?
   - Measure CV (coefficient of variation) across trials
   - CV < 0.15 indicates stable projection estimates
   - Typical: k=10 markets needs ~64-256 samples

2. **Hull coverage**: Are new samples close to the sample hull?
   - If samples are sufficient, new samples should be ~inside the hull
   - Uses leave-one-out cross-validation

3. **Hull stability**: Does hull size/radius stabilize?
   - Second moment (radius) should not grow significantly with more samples
   - Growing radius suggests samples haven't captured full spread

Empirical Guidelines
--------------------
- k=5-10 markets: 64 samples often sufficient
- k=10-20 markets: 128-256 samples recommended
- k>20 markets: 256-512 samples, or use projection sub-sampling

These are rough heuristics; the validator provides quantitative checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CtValidationResult:
    """Results from C_t validation tests."""

    n_samples_used: int
    n_markets: int

    # Convergence test
    converged: bool
    converged_at: Optional[int]
    convergence_cv: float  # Coefficient of variation at n_samples_used

    # Coverage test
    coverage_frac_inside: float  # Fraction of test samples inside hull
    coverage_frac_close: float  # Fraction within 5%
    coverage_p95_dist: float  # 95th percentile distance

    # Hull stability
    hull_radius2: float
    hull_radius2_stable: bool

    # Recommendation
    recommendation: str
    sufficient: bool

    def to_dict(self) -> Dict:
        return {
            "n_samples_used": self.n_samples_used,
            "n_markets": self.n_markets,
            "converged": self.converged,
            "converged_at": self.converged_at,
            "convergence_cv": self.convergence_cv,
            "coverage_frac_inside": self.coverage_frac_inside,
            "coverage_frac_close": self.coverage_frac_close,
            "coverage_p95_dist": self.coverage_p95_dist,
            "hull_radius2": self.hull_radius2,
            "hull_radius2_stable": self.hull_radius2_stable,
            "recommendation": self.recommendation,
            "sufficient": self.sufficient,
        }


@dataclass
class CtValidationConfig:
    """Configuration for C_t validation."""

    # Sample counts to test for convergence
    sample_counts: Tuple[int, ...] = (16, 32, 64, 128, 256)

    # Number of trials for variance estimation
    n_trials: int = 5

    # Coverage test parameters
    n_test_samples: int = 256  # Test samples for coverage check
    coverage_threshold: float = 0.90  # Fraction that should be within tolerance
    distance_tolerance: float = 0.05  # 5% distance is "close"

    # Convergence threshold
    # Note: CV of ~0.15 typically achievable with 256+ samples for k=10 markets
    # CV of ~0.10 may require 512+ samples depending on distribution complexity
    cv_threshold: float = 0.15  # Coefficient of variation for "converged"

    # Stability threshold
    radius_cv_threshold: float = 0.15  # Radius CV for "stable"


class CtValidator:
    """
    Validates that MC samples sufficiently approximate C_t.

    The core question: Is n=64 enough samples to accurately represent
    the convex hull of the diffusion model's output distribution?

    We check:
    1. Convergence: Does d(q, C_t) stabilize as n increases?
    2. Coverage: Are new samples mostly inside the estimated hull?
    3. Stability: Does hull size stabilize?
    """

    def __init__(self, cfg: CtValidationConfig = CtValidationConfig()):
        self.cfg = cfg
        self._history: List[CtValidationResult] = []

    def validate_sample_convergence(
        self,
        sample_fn: Callable[[int, int], np.ndarray],
        q: np.ndarray,
        project_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict]],
    ) -> Dict:
        """
        Check if projection distance converges as sample count increases.

        Args:
            sample_fn: Function(n_samples, seed) -> (n_samples, k) samples
            q: (k,) market prices to project
            project_fn: Function(x, samples) -> (proj, feats)

        Returns:
            Convergence analysis
        """
        results = {n: [] for n in self.cfg.sample_counts}

        for trial in range(self.cfg.n_trials):
            seed = trial * 1000

            for n in self.cfg.sample_counts:
                samples = sample_fn(n, seed)
                _, feats = project_fn(q, samples)
                results[n].append(feats.get("dist_l2", 0.0))

        # Compute statistics at each sample count
        summary = {}
        for n in self.cfg.sample_counts:
            dists = results[n]
            mean_dist = float(np.mean(dists))
            std_dist = float(np.std(dists))
            cv = std_dist / (mean_dist + 1e-12)
            summary[n] = {
                "mean_dist": mean_dist,
                "std_dist": std_dist,
                "cv": cv,
                "converged": cv < self.cfg.cv_threshold,
            }

        # Find first sample count where converged
        converged_at = None
        for n in sorted(self.cfg.sample_counts):
            if summary[n]["converged"]:
                converged_at = n
                break

        return {
            "summary": summary,
            "converged_at": converged_at,
            "cv_at_max": summary[max(self.cfg.sample_counts)]["cv"],
        }

    def validate_hull_coverage(
        self,
        sample_fn: Callable[[int, int], np.ndarray],
        n_hull: int,
        project_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict]],
        seed: int = 0,
    ) -> Dict:
        """
        Check if test samples are close to the estimated hull.

        If n_hull samples are sufficient, most new samples from the
        same distribution should be inside or very close to conv(n_hull).

        Args:
            sample_fn: Function(n_samples, seed) -> (n_samples, k) samples
            n_hull: Number of samples used for hull
            project_fn: Function(x, samples) -> (proj, feats)
            seed: Random seed

        Returns:
            Coverage analysis
        """
        # Build hull from n_hull samples
        hull_samples = sample_fn(n_hull, seed)

        # Sample test points
        test_samples = sample_fn(self.cfg.n_test_samples, seed + 999999)

        # For each test point, compute distance to hull
        distances = []
        for i in range(len(test_samples)):
            _, feats = project_fn(test_samples[i], hull_samples)
            distances.append(feats.get("dist_l2", 0.0))

        distances = np.array(distances)

        return {
            "mean_dist": float(np.mean(distances)),
            "std_dist": float(np.std(distances)),
            "max_dist": float(np.max(distances)),
            "p50_dist": float(np.percentile(distances, 50)),
            "p95_dist": float(np.percentile(distances, 95)),
            "p99_dist": float(np.percentile(distances, 99)),
            "frac_inside": float(np.mean(distances < 0.01)),  # Within 1%
            "frac_close": float(np.mean(distances < self.cfg.distance_tolerance)),
        }

    def validate_hull_stability(
        self,
        sample_fn: Callable[[int, int], np.ndarray],
        summarize_fn: Callable[[np.ndarray], Dict],
    ) -> Dict:
        """
        Check if hull properties stabilize as samples increase.

        Args:
            sample_fn: Function(n_samples, seed) -> (n_samples, k) samples
            summarize_fn: Function(samples) -> dict with "radius2", etc.

        Returns:
            Stability analysis
        """
        metrics = []

        for n in self.cfg.sample_counts:
            trial_radii = []
            for trial in range(self.cfg.n_trials):
                samples = sample_fn(n, trial * 1000)
                summary = summarize_fn(samples)
                trial_radii.append(summary.get("radius2", 0.0))

            metrics.append({
                "n": n,
                "mean_radius2": float(np.mean(trial_radii)),
                "std_radius2": float(np.std(trial_radii)),
                "cv": float(np.std(trial_radii) / (np.mean(trial_radii) + 1e-12)),
            })

        # Check convergence of radius
        radii = [m["mean_radius2"] for m in metrics]
        final_cv = metrics[-1]["cv"]

        # Check monotonicity (radius should increase and stabilize)
        diffs = np.diff(radii)
        mostly_increasing = np.mean(diffs >= -1e-6) > 0.7

        return {
            "metrics": metrics,
            "final_radius2": radii[-1],
            "final_cv": final_cv,
            "stable": final_cv < self.cfg.radius_cv_threshold,
            "mostly_increasing": mostly_increasing,
        }

    def run_full_validation(
        self,
        sample_fn: Callable[[int, int], np.ndarray],
        q: np.ndarray,
        project_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict]],
        summarize_fn: Callable[[np.ndarray], Dict],
        n_samples_used: int = 64,
        seed: int = 0,
    ) -> CtValidationResult:
        """
        Run all validation tests and produce a recommendation.

        Args:
            sample_fn: Function(n_samples, seed) -> (n_samples, k) samples
            q: (k,) market prices to project
            project_fn: Function(x, samples) -> (proj, feats)
            summarize_fn: Function(samples) -> dict with "radius2", etc.
            n_samples_used: The sample count being validated
            seed: Random seed

        Returns:
            CtValidationResult with recommendation
        """
        k = len(q)

        # Run tests
        convergence = self.validate_sample_convergence(sample_fn, q, project_fn)
        coverage = self.validate_hull_coverage(sample_fn, n_samples_used, project_fn, seed)
        stability = self.validate_hull_stability(sample_fn, summarize_fn)

        # Get CV at the sample count being used
        cv_at_n = convergence["summary"].get(n_samples_used, {}).get("cv", 1.0)

        # Determine sufficiency
        converged = convergence["converged_at"] is not None and convergence["converged_at"] <= n_samples_used
        good_coverage = coverage["frac_close"] >= self.cfg.coverage_threshold
        stable = stability["stable"]

        sufficient = converged and good_coverage and stable

        # Generate recommendation
        if sufficient:
            recommendation = f"{n_samples_used} samples is sufficient for k={k} markets"
        else:
            issues = []
            if not converged:
                if convergence["converged_at"]:
                    issues.append(f"need {convergence['converged_at']} samples for convergence")
                else:
                    issues.append(f"projection does not converge even at {max(self.cfg.sample_counts)} samples")
            if not good_coverage:
                issues.append(f"only {coverage['frac_close']:.1%} of test samples within tolerance")
            if not stable:
                issues.append(f"hull radius not stable (CV={stability['final_cv']:.2f})")

            if convergence["converged_at"] and convergence["converged_at"] > n_samples_used:
                recommendation = f"Increase to {convergence['converged_at']} samples: {'; '.join(issues)}"
            else:
                recommendation = f"Sample quality issues: {'; '.join(issues)}"

        result = CtValidationResult(
            n_samples_used=n_samples_used,
            n_markets=k,
            converged=converged,
            converged_at=convergence["converged_at"],
            convergence_cv=cv_at_n,
            coverage_frac_inside=coverage["frac_inside"],
            coverage_frac_close=coverage["frac_close"],
            coverage_p95_dist=coverage["p95_dist"],
            hull_radius2=stability["final_radius2"],
            hull_radius2_stable=stable,
            recommendation=recommendation,
            sufficient=sufficient,
        )

        self._history.append(result)
        return result

    def get_history(self) -> List[CtValidationResult]:
        """Get all validation results."""
        return self._history

    def summary_stats(self) -> Dict:
        """Get summary statistics across all validations."""
        if not self._history:
            return {"n_validations": 0}

        sufficient_count = sum(1 for r in self._history if r.sufficient)
        mean_coverage = float(np.mean([r.coverage_frac_close for r in self._history]))
        mean_cv = float(np.mean([r.convergence_cv for r in self._history]))

        return {
            "n_validations": len(self._history),
            "sufficient_rate": sufficient_count / len(self._history),
            "mean_coverage_frac": mean_coverage,
            "mean_convergence_cv": mean_cv,
        }


def quick_validate_ct(
    samples: np.ndarray,
    q: np.ndarray,
    n_test: int = 100,
    seed: int = 0,
) -> Dict:
    """
    Quick validation of C_t samples without needing a model.

    Uses bootstrap resampling to estimate coverage.

    Args:
        samples: (mc, k) existing MC samples
        q: (k,) market prices
        n_test: Number of bootstrap samples to test

    Returns:
        Quick validation metrics
    """
    from forecastbench.utils.convex_hull_projection import (
        ct_projection_features,
        summarize_ct_samples,
    )

    rng = np.random.default_rng(seed)
    mc, k = samples.shape

    # 1. Project q to samples
    _, feats = ct_projection_features(x=q, samples=samples)
    dist_to_ct = feats["dist_l2"]

    # 2. Hull summary
    hull_summary = summarize_ct_samples(samples)

    # 3. Bootstrap coverage: how often is a held-out sample inside the hull?
    #    Use leave-one-out cross-validation
    loo_distances = []
    for i in range(min(mc, n_test)):
        # Leave out sample i
        held_out = samples[i]
        remaining = np.delete(samples, i, axis=0)

        if len(remaining) < 2:
            continue

        _, loo_feats = ct_projection_features(x=held_out, samples=remaining)
        loo_distances.append(loo_feats["dist_l2"])

    loo_distances = np.array(loo_distances)

    # 4. Variance estimation: split samples and compare projections
    half = mc // 2
    _, feats_a = ct_projection_features(x=q, samples=samples[:half])
    _, feats_b = ct_projection_features(x=q, samples=samples[half:])
    split_diff = abs(feats_a["dist_l2"] - feats_b["dist_l2"])

    return {
        "n_samples": mc,
        "n_markets": k,
        "dist_to_ct": dist_to_ct,
        "hull_radius2": hull_summary["radius2"],
        "loo_mean_dist": float(np.mean(loo_distances)) if len(loo_distances) > 0 else 0.0,
        "loo_p95_dist": float(np.percentile(loo_distances, 95)) if len(loo_distances) > 0 else 0.0,
        "loo_frac_inside": float(np.mean(loo_distances < 0.01)) if len(loo_distances) > 0 else 0.0,
        "split_half_diff": split_diff,
        "stability_ok": split_diff < 0.05,
        "coverage_ok": float(np.mean(loo_distances < 0.05)) > 0.9 if len(loo_distances) > 0 else False,
    }

