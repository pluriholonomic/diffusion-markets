"""
Cliff vs Fog: Core scaling experiments comparing AR spectral cutoff to diffusion fog.

This implements the signature comparison from main.tex Propositions 6-7:
- AR(L): SCE >= W_{>L}(f) (cliff at L < k)
- Diffusion(ρ): SCE = Σ_S (1-ρ^|S|)² f̂(S)² (continuous recovery)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecastbench.benchmarks.parity import ParityMarket, ParitySpec, sample_parity_dataset
from forecastbench.metrics import brier_loss, expected_calibration_error
from forecastbench.metrics.arbitrage import best_bounded_trader_profit


@dataclass(frozen=True)
class CliffFogSpec:
    """Specification for cliff vs fog experiment."""
    d: int = 16
    k: int = 8
    alpha: float = 0.8
    n_samples: int = 100000
    L_values: Tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    rho_values: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    K_values: Tuple[int, ...] = (1, 4, 16)  # self-consistency width for AR
    seed: int = 0


def l_junta_predictor(z: np.ndarray, L: int, S: Tuple[int, ...], alpha: float) -> np.ndarray:
    """
    Best L-junta predictor for parity market.
    
    If L >= |S|, we can represent the parity exactly.
    Otherwise, the best L-junta is constant 0.5 (zero Fourier mass on S).
    """
    k = len(S)
    if L >= k:
        # Can query all k bits: perfect prediction
        chi = np.prod(z[:, list(S)], axis=1).astype(np.float32)
        return 0.5 + 0.5 * alpha * chi
    else:
        # Cannot see degree-k: constant prediction
        return np.full(z.shape[0], 0.5, dtype=np.float32)


def l_junta_with_sampling(
    z: np.ndarray,
    L: int,
    K: int,
    S: Tuple[int, ...],
    alpha: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate K independent randomized L-query forecasters.
    
    Each sample randomly picks L coordinates to query.
    Mean aggregation across K samples.
    
    Theory: averaging reduces variance but cannot change spectral support.
    """
    if K <= 1:
        return l_junta_predictor(z, L, S, alpha)
    
    rng = np.random.default_rng(seed)
    k = len(S)
    d = z.shape[1]
    n = z.shape[0]
    
    predictions = np.zeros((K, n), dtype=np.float64)
    
    for i in range(K):
        # Random L-query: pick L coordinates uniformly
        query_set = set(rng.choice(d, size=min(L, d), replace=False).tolist())
        
        # Check how many parity bits we capture
        captured = query_set.intersection(set(S))
        
        if len(captured) == k:
            # Got all parity bits: exact prediction
            chi = np.prod(z[:, list(S)], axis=1).astype(np.float64)
            predictions[i] = 0.5 + 0.5 * alpha * chi
        else:
            # Missing some bits: E[chi_S | captured] = 0
            predictions[i] = 0.5
    
    return np.mean(predictions, axis=0).astype(np.float32)


def run_cliff_fog_experiment(spec: CliffFogSpec) -> Dict:
    """
    Run the cliff vs fog experiment.
    
    Returns dict with AR results (varying L, K) and diffusion results (varying ρ).
    """
    parity_spec = ParitySpec(d=spec.d, k=spec.k, alpha=spec.alpha, seed=spec.seed)
    dataset = sample_parity_dataset(parity_spec, spec.n_samples)
    z = dataset["z"]
    p_true = dataset["p_true"]
    y = dataset["y"]
    S = tuple(int(i) for i in dataset["S"])
    
    mkt = ParityMarket.create(parity_spec)
    
    # Theoretical values
    alpha = spec.alpha
    k = spec.k
    
    # AR results: vary L and K
    ar_results = []
    for L in spec.L_values:
        for K in spec.K_values:
            q = l_junta_with_sampling(z, L, K, S, alpha, seed=spec.seed + L * 100 + K)
            
            sce = float(np.mean((q - p_true) ** 2))
            arb = best_bounded_trader_profit(p_true, q, B=1.0)
            ece = expected_calibration_error(q, y)
            brier = brier_loss(q, y)
            
            # Theory: SCE >= (α/2)² when L < k, else 0
            theory_sce = (alpha / 2) ** 2 if L < k else 0.0
            
            ar_results.append({
                "L": int(L),
                "K": int(K),
                "SCE": float(sce),
                "arb_profit": float(arb),
                "ECE": float(ece),
                "brier": float(brier),
                "theory_SCE_lower_bound": float(theory_sce),
                "beats_theory": sce >= theory_sce - 1e-6,
            })
    
    # Diffusion results: vary ρ
    diff_results = []
    for rho in spec.rho_values:
        q = mkt.diffusion_analytic(z, rho)
        
        sce = float(np.mean((q - p_true) ** 2))
        arb = best_bounded_trader_profit(p_true, q, B=1.0)
        ece = expected_calibration_error(q, y)
        brier = brier_loss(q, y)
        
        # Theory: SCE = (α/2)² (1 - ρ^k)²
        theory_sce = (alpha / 2) ** 2 * (1 - rho ** k) ** 2
        
        diff_results.append({
            "rho": float(rho),
            "SCE": float(sce),
            "arb_profit": float(arb),
            "ECE": float(ece),
            "brier": float(brier),
            "theory_SCE": float(theory_sce),
            "matches_theory": abs(sce - theory_sce) < 0.01,
        })
    
    return {
        "spec": {
            "d": spec.d,
            "k": spec.k,
            "alpha": spec.alpha,
            "n_samples": spec.n_samples,
            "seed": spec.seed,
        },
        "S": list(S),
        "ar_results": ar_results,
        "diffusion_results": diff_results,
    }


def compute_matched_comparison(spec: CliffFogSpec) -> pd.DataFrame:
    """
    Compute-matched comparison: same "budget" for AR and diffusion.
    
    Budget mapping (heuristic):
    - AR: L queries (serial depth)
    - Diffusion: T denoising steps → ρ ≈ 1 - 1/(T+1)
    """
    parity_spec = ParitySpec(d=spec.d, k=spec.k, alpha=spec.alpha, seed=spec.seed)
    dataset = sample_parity_dataset(parity_spec, spec.n_samples)
    z = dataset["z"]
    p_true = dataset["p_true"]
    S = tuple(int(i) for i in dataset["S"])
    
    mkt = ParityMarket.create(parity_spec)
    
    compute_budgets = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    
    for budget in compute_budgets:
        # AR: L = budget
        q_ar = l_junta_predictor(z, budget, S, spec.alpha)
        ar_sce = float(np.mean((q_ar - p_true) ** 2))
        ar_arb = best_bounded_trader_profit(p_true, q_ar, B=1.0)
        
        # Diffusion: T = budget steps, ρ = 1 - 1/(T+1)
        rho = 1 - 1 / (budget + 1)
        q_diff = mkt.diffusion_analytic(z, rho)
        diff_sce = float(np.mean((q_diff - p_true) ** 2))
        diff_arb = best_bounded_trader_profit(p_true, q_diff, B=1.0)
        
        rows.append({
            "compute": int(budget),
            "ar_L": int(budget),
            "ar_SCE": float(ar_sce),
            "ar_arb": float(ar_arb),
            "diff_T": int(budget),
            "diff_rho": float(rho),
            "diff_SCE": float(diff_sce),
            "diff_arb": float(diff_arb),
            "diff_wins_sce": diff_sce < ar_sce,
            "diff_wins_arb": diff_arb < ar_arb,
        })
    
    return pd.DataFrame(rows)


def plot_cliff_fog(results: Dict, output_dir: str) -> None:
    """Generate the signature cliff vs fog plot."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ar_df = pd.DataFrame(results["ar_results"])
    diff_df = pd.DataFrame(results["diffusion_results"])
    k = results["spec"]["k"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: AR cliff (K=1 baseline)
    ar_k1 = ar_df[ar_df["K"] == 1]
    axes[0].semilogy(ar_k1["L"], ar_k1["SCE"], 'o-', label="Empirical SCE", markersize=8, linewidth=2)
    axes[0].semilogy(ar_k1["L"], ar_k1["theory_SCE_lower_bound"], 's--', 
                     label="Theory Lower Bound", markersize=6, alpha=0.7)
    axes[0].axvline(x=k, color='r', linestyle=':', linewidth=2, label=f"k={k} (cliff)")
    axes[0].fill_betweenx([1e-6, 1], 0, k, alpha=0.1, color='red')
    axes[0].set_xlabel("Depth L (query budget)", fontsize=12)
    axes[0].set_ylabel("Squared Calibration Error (log)", fontsize=12)
    axes[0].set_title("AR+CoT: Complexity Cliff", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(1e-6, 0.5)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Diffusion fog
    axes[1].semilogy(diff_df["rho"], diff_df["SCE"], 'o-', label="Empirical SCE", markersize=8, linewidth=2)
    axes[1].semilogy(diff_df["rho"], diff_df["theory_SCE"], 's--', 
                     label="Theory Prediction", markersize=6, alpha=0.7)
    axes[1].set_xlabel("Noise Correlation ρ", fontsize=12)
    axes[1].set_ylabel("Squared Calibration Error (log)", fontsize=12)
    axes[1].set_title("Diffusion: Spectral Fog (Continuous Recovery)", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(1e-6, 0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cliff_vs_fog.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional plot: AR with different K (showing K doesn't help)
    fig, ax = plt.subplots(figsize=(8, 5))
    for K in ar_df["K"].unique():
        ar_k = ar_df[ar_df["K"] == K]
        ax.semilogy(ar_k["L"], ar_k["SCE"], 'o-', label=f"K={K}", markersize=6)
    
    ax.axvline(x=k, color='r', linestyle=':', linewidth=2, label=f"k={k}")
    ax.set_xlabel("Depth L", fontsize=12)
    ax.set_ylabel("SCE (log)", fontsize=12)
    ax.set_title("AR: Sampling Width K Cannot Cross the Cliff", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/ar_width_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_cliff_fog_suite(
    k_values: Tuple[int, ...] = (4, 6, 8, 10),
    d: int = 20,
    alpha: float = 0.8,
    n_samples: int = 100000,
    output_dir: str = "plots",
    seed: int = 0,
) -> Dict:
    """
    Run cliff vs fog across multiple k values.
    """
    all_results = {}
    
    for k in k_values:
        L_values = tuple(range(2, k + 4, 2))  # L from 2 to k+3
        rho_values = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
        
        spec = CliffFogSpec(
            d=d, k=k, alpha=alpha, n_samples=n_samples,
            L_values=L_values, rho_values=rho_values,
            seed=seed + k * 1000,
        )
        
        results = run_cliff_fog_experiment(spec)
        all_results[f"k={k}"] = results
        
        plot_cliff_fog(results, f"{output_dir}/k{k}")
    
    return all_results



