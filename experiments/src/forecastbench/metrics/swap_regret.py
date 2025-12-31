"""
Swap Regret vs External Regret: Measuring calibration quality.

The theory (§9) establishes:
- External regret controls performance against unconditional comparators
- Swap regret controls forecast-conditional deviations (calibeating resistance)
- Swap regret is the economically relevant quantity for statistical arbitrage

This module implements:
1. External regret computation
2. Swap regret computation (discretized and continuous)
3. Decomposition showing the swap-external gap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def brier_loss_single(q: float, y: float) -> float:
    """Brier loss for a single prediction."""
    return (q - y) ** 2


def compute_external_regret(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """
    Compute external regret: comparison to best constant predictor.
    
    Reg^ext_T = Σ (q_t - y_t)² - min_a Σ (a - y_t)²
    
    For Brier loss, the optimal constant is ȳ (empirical mean).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    
    T = len(predictions)
    if T == 0:
        return {"regret": 0.0, "regret_per_round": 0.0}
    
    # Forecaster's loss
    forecaster_loss = np.sum((predictions - outcomes) ** 2)
    
    # Best constant: ȳ (continuous comparator)
    y_bar = np.mean(outcomes)
    best_constant_loss = np.sum((y_bar - outcomes) ** 2)
    
    # For grid comparator
    grid = np.linspace(0, 1, n_bins + 1)
    grid_centers = (grid[:-1] + grid[1:]) / 2
    grid_losses = [np.sum((a - outcomes) ** 2) for a in grid_centers]
    best_grid_loss = min(grid_losses)
    
    regret_continuous = forecaster_loss - best_constant_loss
    regret_grid = forecaster_loss - best_grid_loss
    
    return {
        "forecaster_loss": float(forecaster_loss),
        "best_constant": float(y_bar),
        "best_constant_loss": float(best_constant_loss),
        "regret_continuous": float(regret_continuous),
        "regret_grid": float(regret_grid),
        "regret_per_round": float(regret_continuous / T),
    }


def compute_swap_regret(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """
    Compute swap regret: comparison to best remapping σ: A_m → A_m.
    
    Reg^swap_T = max_σ Σ [(q_t - y_t)² - (σ(q_t) - y_t)²]
    
    The continuous comparator version: for each bin, remap to E[Y | q ∈ bin].
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    
    T = len(predictions)
    if T == 0:
        return {"regret": 0.0, "regret_per_round": 0.0}
    
    # Bin predictions
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(predictions, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    # For each bin, compute optimal remapping (E[Y | bin])
    forecaster_loss_total = 0.0
    optimal_remap_loss_total = 0.0
    bin_stats = []
    
    for b in range(n_bins):
        mask = bin_idx == b
        N_b = int(mask.sum())
        if N_b == 0:
            continue
        
        q_b = predictions[mask]
        y_b = outcomes[mask]
        
        # Forecaster loss on this bin
        bin_loss = float(np.sum((q_b - y_b) ** 2))
        forecaster_loss_total += bin_loss
        
        # Optimal remapping: predict ȳ_b for all in this bin
        y_bar_b = float(np.mean(y_b))
        optimal_loss = float(np.sum((y_bar_b - y_b) ** 2))
        optimal_remap_loss_total += optimal_loss
        
        # Bin miscalibration
        q_bar_b = float(np.mean(q_b))
        miscal = q_bar_b - y_bar_b
        
        bin_stats.append({
            "bin": b,
            "N": N_b,
            "q_bar": q_bar_b,
            "y_bar": y_bar_b,
            "miscal": miscal,
            "bin_loss": bin_loss,
            "optimal_loss": optimal_loss,
            "improvement": bin_loss - optimal_loss,
        })
    
    swap_regret = forecaster_loss_total - optimal_remap_loss_total
    
    return {
        "forecaster_loss": float(forecaster_loss_total),
        "optimal_remap_loss": float(optimal_remap_loss_total),
        "swap_regret": float(swap_regret),
        "swap_regret_per_round": float(swap_regret / T),
        "n_bins": n_bins,
        "bin_stats": bin_stats,
    }


def swap_external_decomposition(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """
    Decompose swap regret into external regret + heterogeneity term.
    
    From Proposition 11:
    Reg^swap - Reg^ext = Σ_a N_a (ȳ_a - ȳ)²
    
    This is the between-bin variance of conditional means.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    
    ext_result = compute_external_regret(predictions, outcomes, n_bins)
    swap_result = compute_swap_regret(predictions, outcomes, n_bins)
    
    T = len(predictions)
    y_bar = float(np.mean(outcomes))
    
    # Compute heterogeneity term
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(predictions, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    heterogeneity = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        N_b = int(mask.sum())
        if N_b == 0:
            continue
        y_bar_b = float(np.mean(outcomes[mask]))
        heterogeneity += N_b * (y_bar_b - y_bar) ** 2
    
    return {
        "external_regret": ext_result["regret_continuous"],
        "swap_regret": swap_result["swap_regret"],
        "heterogeneity": float(heterogeneity),
        "decomposition_check": abs(
            swap_result["swap_regret"] - ext_result["regret_continuous"] - heterogeneity
        ) < 1e-6,
        "gap": float(swap_result["swap_regret"] - ext_result["regret_continuous"]),
        "gap_per_round": float((swap_result["swap_regret"] - ext_result["regret_continuous"]) / T),
    }


def regret_comparison_experiment(
    predictions_ar: np.ndarray,
    predictions_diff: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """
    Compare AR and diffusion on external and swap regret.
    
    The theory predicts:
    - Diffusion should have lower swap regret (better conditional calibration)
    - AR may have comparable external regret but higher swap regret
    """
    ar_ext = compute_external_regret(predictions_ar, outcomes, n_bins)
    ar_swap = compute_swap_regret(predictions_ar, outcomes, n_bins)
    ar_decomp = swap_external_decomposition(predictions_ar, outcomes, n_bins)
    
    diff_ext = compute_external_regret(predictions_diff, outcomes, n_bins)
    diff_swap = compute_swap_regret(predictions_diff, outcomes, n_bins)
    diff_decomp = swap_external_decomposition(predictions_diff, outcomes, n_bins)
    
    return {
        "ar": {
            "external_regret": ar_ext["regret_continuous"],
            "swap_regret": ar_swap["swap_regret"],
            "gap": ar_decomp["gap"],
            "per_round": {
                "external": ar_ext["regret_per_round"],
                "swap": ar_swap["swap_regret_per_round"],
            },
        },
        "diffusion": {
            "external_regret": diff_ext["regret_continuous"],
            "swap_regret": diff_swap["swap_regret"],
            "gap": diff_decomp["gap"],
            "per_round": {
                "external": diff_ext["regret_per_round"],
                "swap": diff_swap["swap_regret_per_round"],
            },
        },
        "comparison": {
            "diff_wins_external": diff_ext["regret_continuous"] < ar_ext["regret_continuous"],
            "diff_wins_swap": diff_swap["swap_regret"] < ar_swap["swap_regret"],
            "external_ratio": ar_ext["regret_continuous"] / max(diff_ext["regret_continuous"], 1e-9),
            "swap_ratio": ar_swap["swap_regret"] / max(diff_swap["swap_regret"], 1e-9),
        },
    }


def plot_regret_comparison(results: Dict, output_dir: str) -> None:
    """Plot regret comparison bar chart."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Absolute regret
    labels = ['External', 'Swap']
    ar_vals = [results["ar"]["external_regret"], results["ar"]["swap_regret"]]
    diff_vals = [results["diffusion"]["external_regret"], results["diffusion"]["swap_regret"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, ar_vals, width, label='AR', color='#d62728')
    bars2 = axes[0].bar(x + width/2, diff_vals, width, label='Diffusion', color='#2ca02c')
    
    axes[0].set_ylabel('Total Regret', fontsize=12)
    axes[0].set_title('External vs Swap Regret', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Gap (swap - external)
    ar_gap = results["ar"]["gap"]
    diff_gap = results["diffusion"]["gap"]
    
    bars = axes[1].bar(['AR', 'Diffusion'], [ar_gap, diff_gap], 
                       color=['#d62728', '#2ca02c'])
    
    axes[1].set_ylabel('Swap - External Gap', fontsize=12)
    axes[1].set_title('Calibration Defect (Swap-External Gap)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regret_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_swap_regret_experiment(
    d: int = 16,
    k: int = 8,
    alpha: float = 0.8,
    n_samples: int = 50000,
    L_ar: int = 4,
    rho_diff: float = 0.95,
    output_dir: str = "plots",
    seed: int = 0,
) -> Dict:
    """
    Run swap regret comparison between AR and diffusion on parity.
    """
    from forecastbench.benchmarks.parity import ParityMarket, ParitySpec, sample_parity_dataset
    from forecastbench.benchmarks.cliff_fog import l_junta_predictor
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    parity_spec = ParitySpec(d=d, k=k, alpha=alpha, seed=seed)
    dataset = sample_parity_dataset(parity_spec, n_samples)
    z = dataset["z"]
    p_true = dataset["p_true"]
    y = dataset["y"].astype(np.float64)
    S = tuple(int(i) for i in dataset["S"])
    
    mkt = ParityMarket.create(parity_spec)
    
    # AR predictions
    q_ar = l_junta_predictor(z, L_ar, S, alpha)
    
    # Diffusion predictions
    q_diff = mkt.diffusion_analytic(z, rho_diff)
    
    # Compute regrets
    results = regret_comparison_experiment(q_ar, q_diff, y, n_bins=20)
    
    # Add metadata
    results["spec"] = {
        "d": d,
        "k": k,
        "alpha": alpha,
        "n_samples": n_samples,
        "L_ar": L_ar,
        "rho_diff": rho_diff,
    }
    
    plot_regret_comparison(results, output_dir)
    
    return results



