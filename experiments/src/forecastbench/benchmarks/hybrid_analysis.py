"""
Hybrid Analysis: Per-sample correction classification for AR+Diffusion models.

This module analyzes how the diffusion refinement affects individual predictions:
- Does diffusion correct AR errors?
- Does diffusion corrupt AR successes?
- Is there correlation between AR error magnitude and diffusion correction?

These analyses directly test whether diffusion is learning to repair
high-degree errors that AR cannot capture (the spectral cliff hypothesis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Per-Sample Correction Classification
# =============================================================================

@dataclass
class CorrectionClassification:
    """
    Classification of per-sample corrections.
    
    Categories:
    - ar_wrong_diff_corrects: AR was wrong (error > threshold), diffusion corrected
    - ar_right_diff_corrupts: AR was right (error < threshold), diffusion corrupted
    - both_improve: Both AR and hybrid get closer to truth
    - both_degrade: Both AR and hybrid get further from truth
    - diff_helps: Hybrid error < AR error (regardless of threshold)
    - diff_hurts: Hybrid error > AR error (regardless of threshold)
    """
    
    ar_wrong_diff_corrects: np.ndarray
    ar_right_diff_corrupts: np.ndarray
    both_improve: np.ndarray
    both_degrade: np.ndarray
    diff_helps: np.ndarray
    diff_hurts: np.ndarray
    
    @property
    def n_samples(self) -> int:
        return len(self.diff_helps)
    
    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        n = self.n_samples
        if n == 0:
            return {"error": "No samples"}
        
        return {
            "n_samples": n,
            "ar_wrong_diff_corrects_rate": float(np.mean(self.ar_wrong_diff_corrects)),
            "ar_right_diff_corrupts_rate": float(np.mean(self.ar_right_diff_corrupts)),
            "both_improve_rate": float(np.mean(self.both_improve)),
            "both_degrade_rate": float(np.mean(self.both_degrade)),
            "diff_helps_rate": float(np.mean(self.diff_helps)),
            "diff_hurts_rate": float(np.mean(self.diff_hurts)),
            "net_help_rate": float(np.mean(self.diff_helps) - np.mean(self.diff_hurts)),
            "correction_ratio": float(
                np.sum(self.ar_wrong_diff_corrects) / 
                max(np.sum(self.ar_right_diff_corrupts), 1)
            ),
        }


def classify_corrections(
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
    *,
    error_threshold: float = 0.5,
) -> CorrectionClassification:
    """
    Classify each sample based on how diffusion affected the prediction.
    
    Args:
        p_ar: AR-only predictions (N,)
        p_hybrid: AR+Diffusion predictions (N,)
        y: True outcomes (N,) - binary 0/1
        error_threshold: Threshold for "wrong" prediction (|p - y| > threshold)
        
    Returns:
        CorrectionClassification with per-sample boolean arrays
    """
    p_ar = np.asarray(p_ar, dtype=np.float64).flatten()
    p_hybrid = np.asarray(p_hybrid, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Compute errors
    ar_error = np.abs(p_ar - y)
    hybrid_error = np.abs(p_hybrid - y)
    
    # Classification
    ar_wrong = ar_error > error_threshold
    ar_right = ~ar_wrong
    
    # Diffusion helps/hurts
    diff_helps = hybrid_error < ar_error
    diff_hurts = hybrid_error > ar_error
    
    # Combined classifications
    ar_wrong_diff_corrects = ar_wrong & diff_helps
    ar_right_diff_corrupts = ar_right & diff_hurts
    
    # Both improve: both get closer to y than 0.5 baseline
    baseline_error = np.abs(0.5 - y)
    both_improve = (ar_error < baseline_error) & (hybrid_error < baseline_error)
    both_degrade = (ar_error > baseline_error) & (hybrid_error > baseline_error)
    
    return CorrectionClassification(
        ar_wrong_diff_corrects=ar_wrong_diff_corrects,
        ar_right_diff_corrupts=ar_right_diff_corrupts,
        both_improve=both_improve,
        both_degrade=both_degrade,
        diff_helps=diff_helps,
        diff_hurts=diff_hurts,
    )


# =============================================================================
# Error Correlation Analysis
# =============================================================================

def compute_error_correlations(
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze correlation between AR errors and diffusion corrections.
    
    Key insight: If diffusion is learning to correct high-degree errors,
    we expect high correlation between:
    - |p_ar - y| (AR error magnitude)
    - |p_ar - p_hybrid| (diffusion correction magnitude)
    
    Args:
        p_ar: AR-only predictions
        p_hybrid: AR+Diffusion predictions
        y: True outcomes
        
    Returns:
        Correlation analysis results
    """
    p_ar = np.asarray(p_ar, dtype=np.float64).flatten()
    p_hybrid = np.asarray(p_hybrid, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    ar_error = np.abs(p_ar - y)
    hybrid_error = np.abs(p_hybrid - y)
    correction_magnitude = np.abs(p_ar - p_hybrid)
    error_reduction = ar_error - hybrid_error  # Positive = diffusion helped
    
    # Correlation: does diffusion correction magnitude correlate with AR error?
    mask = np.isfinite(ar_error) & np.isfinite(correction_magnitude)
    if mask.sum() < 10:
        return {"error": "Not enough valid samples"}
    
    corr_error_correction = float(np.corrcoef(
        ar_error[mask], correction_magnitude[mask]
    )[0, 1])
    
    # Correlation: does correction magnitude predict improvement?
    corr_correction_improvement = float(np.corrcoef(
        correction_magnitude[mask], error_reduction[mask]
    )[0, 1])
    
    # Stratified analysis: how does diffusion do on high vs low AR error samples?
    ar_error_median = float(np.median(ar_error))
    high_error_mask = ar_error > ar_error_median
    low_error_mask = ~high_error_mask
    
    high_error_improvement = float(np.mean(error_reduction[high_error_mask]))
    low_error_improvement = float(np.mean(error_reduction[low_error_mask]))
    
    return {
        "corr_ar_error_vs_correction_magnitude": corr_error_correction,
        "corr_correction_magnitude_vs_improvement": corr_correction_improvement,
        "ar_error_median": ar_error_median,
        "high_error_improvement": high_error_improvement,
        "low_error_improvement": low_error_improvement,
        "targets_high_error": high_error_improvement > low_error_improvement,
        "mean_ar_error": float(np.mean(ar_error)),
        "mean_hybrid_error": float(np.mean(hybrid_error)),
        "mean_correction_magnitude": float(np.mean(correction_magnitude)),
        "mean_error_reduction": float(np.mean(error_reduction)),
        "interpretation": (
            f"Correlation between AR error and correction magnitude: {corr_error_correction:.3f}. "
            f"High-error samples improve by {high_error_improvement:.4f}, "
            f"low-error by {low_error_improvement:.4f}. "
            f"{'Diffusion targets high-error samples.' if high_error_improvement > low_error_improvement else 'Diffusion does not specifically target high-error samples.'}"
        ),
    }


# =============================================================================
# Spectral Analysis (Proxy for Degree Structure)
# =============================================================================

def estimate_error_complexity(
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
    features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Estimate the "complexity" of errors using feature-based proxies.
    
    For parity-like problems, high-degree errors should be corrected more
    by diffusion. We use feature interactions as a proxy for degree.
    
    Args:
        p_ar: AR-only predictions
        p_hybrid: AR+Diffusion predictions
        y: True outcomes
        features: Optional (N, d) feature matrix for complexity estimation
        
    Returns:
        Complexity analysis of errors
    """
    p_ar = np.asarray(p_ar, dtype=np.float64).flatten()
    p_hybrid = np.asarray(p_hybrid, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    ar_error = np.abs(p_ar - y)
    hybrid_error = np.abs(p_hybrid - y)
    error_reduction = ar_error - hybrid_error
    
    if features is None:
        # Without features, we can only analyze error distribution
        return {
            "has_features": False,
            "ar_error_variance": float(np.var(ar_error)),
            "hybrid_error_variance": float(np.var(hybrid_error)),
            "error_reduction_variance": float(np.var(error_reduction)),
            "note": "Provide features for complexity analysis",
        }
    
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    n, d = features.shape
    
    # Compute "complexity" as average absolute pairwise feature product
    # This is a proxy for degree-2 interactions
    complexity_scores = np.zeros(n)
    for i in range(min(d, 10)):  # Limit for efficiency
        for j in range(i + 1, min(d, 10)):
            complexity_scores += np.abs(features[:, i] * features[:, j])
    
    complexity_scores /= max(1, (min(d, 10) * (min(d, 10) - 1)) // 2)
    
    # Correlation between complexity and error reduction
    mask = np.isfinite(complexity_scores) & np.isfinite(error_reduction)
    if mask.sum() < 10:
        return {
            "has_features": True,
            "error": "Not enough valid samples for complexity analysis",
        }
    
    corr_complexity_reduction = float(np.corrcoef(
        complexity_scores[mask], error_reduction[mask]
    )[0, 1])
    
    # Stratified by complexity
    complexity_median = float(np.median(complexity_scores))
    high_complexity = complexity_scores > complexity_median
    
    high_complexity_reduction = float(np.mean(error_reduction[high_complexity]))
    low_complexity_reduction = float(np.mean(error_reduction[~high_complexity]))
    
    return {
        "has_features": True,
        "corr_complexity_vs_error_reduction": corr_complexity_reduction,
        "complexity_median": complexity_median,
        "high_complexity_reduction": high_complexity_reduction,
        "low_complexity_reduction": low_complexity_reduction,
        "targets_complex_errors": high_complexity_reduction > low_complexity_reduction,
        "interpretation": (
            f"Correlation between feature complexity and error reduction: {corr_complexity_reduction:.3f}. "
            f"High-complexity samples improve by {high_complexity_reduction:.4f}, "
            f"low-complexity by {low_complexity_reduction:.4f}. "
            f"{'Diffusion targets complex (high-degree) errors.' if high_complexity_reduction > low_complexity_reduction else 'No evidence of targeting complex errors.'}"
        ),
    }


# =============================================================================
# Combined Analysis
# =============================================================================

def run_hybrid_analysis(
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
    *,
    features: Optional[np.ndarray] = None,
    error_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run complete hybrid correction analysis.
    
    Args:
        p_ar: AR-only predictions
        p_hybrid: AR+Diffusion predictions
        y: True outcomes
        features: Optional feature matrix for complexity analysis
        error_threshold: Threshold for "wrong" prediction
        
    Returns:
        Complete analysis results
    """
    # Correction classification
    classification = classify_corrections(
        p_ar=p_ar,
        p_hybrid=p_hybrid,
        y=y,
        error_threshold=error_threshold,
    )
    
    # Error correlations
    correlations = compute_error_correlations(
        p_ar=p_ar,
        p_hybrid=p_hybrid,
        y=y,
    )
    
    # Complexity analysis
    complexity = estimate_error_complexity(
        p_ar=p_ar,
        p_hybrid=p_hybrid,
        y=y,
        features=features,
    )
    
    # Overall assessment
    summary = classification.summary()
    
    diffusion_is_useful = (
        summary.get("diff_helps_rate", 0) > summary.get("diff_hurts_rate", 0) and
        summary.get("ar_wrong_diff_corrects_rate", 0) > summary.get("ar_right_diff_corrupts_rate", 0)
    )
    
    targets_ar_errors = correlations.get("targets_high_error", False)
    
    return {
        "classification": summary,
        "correlations": correlations,
        "complexity": complexity,
        "diffusion_is_useful": diffusion_is_useful,
        "targets_ar_errors": targets_ar_errors,
        "h1_supported": diffusion_is_useful,
        "spectral_hypothesis_supported": (
            targets_ar_errors or 
            complexity.get("targets_complex_errors", False)
        ),
        "overall_interpretation": (
            f"Diffusion {'helps' if diffusion_is_useful else 'does not help'} overall "
            f"(net help rate: {summary.get('net_help_rate', 0):.2%}). "
            f"Correction ratio (fixes/corruptions): {summary.get('correction_ratio', 0):.2f}. "
            f"{'Targets high-error samples.' if targets_ar_errors else 'Does not specifically target high-error samples.'}"
        ),
    }


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_correction_analysis(
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
    output_dir: str,
) -> None:
    """
    Generate diagnostic plots for hybrid correction analysis.
    
    Args:
        p_ar: AR-only predictions
        p_hybrid: AR+Diffusion predictions
        y: True outcomes
        output_dir: Directory for output plots
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    p_ar = np.asarray(p_ar, dtype=np.float64).flatten()
    p_hybrid = np.asarray(p_hybrid, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    ar_error = np.abs(p_ar - y)
    hybrid_error = np.abs(p_hybrid - y)
    correction_magnitude = np.abs(p_ar - p_hybrid)
    error_reduction = ar_error - hybrid_error
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Error comparison scatter
    ax = axes[0, 0]
    ax.scatter(ar_error, hybrid_error, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (no change)')
    ax.set_xlabel('AR Error |p_ar - y|', fontsize=12)
    ax.set_ylabel('Hybrid Error |p_hybrid - y|', fontsize=12)
    ax.set_title('Error Comparison: AR vs Hybrid', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Top-right: Correction magnitude vs AR error
    ax = axes[0, 1]
    ax.scatter(ar_error, correction_magnitude, alpha=0.3, s=10, c=error_reduction, cmap='RdYlGn')
    ax.set_xlabel('AR Error |p_ar - y|', fontsize=12)
    ax.set_ylabel('Correction Magnitude |p_ar - p_hybrid|', fontsize=12)
    ax.set_title('Does Diffusion Target High Errors?', fontsize=14)
    
    # Bottom-left: Error reduction histogram
    ax = axes[1, 0]
    ax.hist(error_reduction, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=np.mean(error_reduction), color='g', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(error_reduction):.3f}')
    ax.set_xlabel('Error Reduction (AR error - Hybrid error)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Error Reduction', fontsize=14)
    ax.legend()
    
    # Bottom-right: Classification pie chart
    ax = axes[1, 1]
    classification = classify_corrections(p_ar, p_hybrid, y)
    summary = classification.summary()
    
    labels = ['Diff Helps', 'Diff Hurts', 'No Change']
    sizes = [
        summary['diff_helps_rate'],
        summary['diff_hurts_rate'],
        1 - summary['diff_helps_rate'] - summary['diff_hurts_rate'],
    ]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Effect of Diffusion on Predictions', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correction_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional: stratified analysis plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bin AR errors and show improvement in each bin
    n_bins = 10
    ar_error_bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (ar_error_bins[:-1] + ar_error_bins[1:]) / 2
    bin_improvements = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (ar_error >= ar_error_bins[i]) & (ar_error < ar_error_bins[i + 1])
        if mask.sum() > 0:
            bin_improvements.append(np.mean(error_reduction[mask]))
            bin_counts.append(mask.sum())
        else:
            bin_improvements.append(0)
            bin_counts.append(0)
    
    bars = ax.bar(bin_centers, bin_improvements, width=0.08, color='steelblue', edgecolor='black')
    
    # Color bars by sign
    for bar, val in zip(bars, bin_improvements):
        if val < 0:
            bar.set_color('#e74c3c')
        else:
            bar.set_color('#2ecc71')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('AR Error Bin', fontsize=12)
    ax.set_ylabel('Mean Error Reduction (positive = improvement)', fontsize=12)
    ax.set_title('Error Reduction by AR Error Level', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stratified_improvement.png", dpi=150, bbox_inches='tight')
    plt.close()


