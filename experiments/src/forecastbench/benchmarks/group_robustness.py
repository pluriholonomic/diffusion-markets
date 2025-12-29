"""
Group Robustness: Experiments for Propositions 8-9 (diffusion vs AR on small subgroups).

Key claims:
- Prop 8: Diffusion achieves GCal_Gk = (α/2)(1 - ρ^k) on subcubes of size 2^{-k}
- Prop 9: AR(L) with L < k has GCal_Gk >= α/2 (constant lower bound)

This module implements:
1. Finite-sample confidence intervals (Prop 5)
2. Head-to-head comparison on parity subcubes
3. Exponential group scaling experiments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from forecastbench.benchmarks.parity import ParityMarket, ParitySpec, sample_parity_dataset
from forecastbench.benchmarks.subcubes import keys_for_J, worst_group_abs_cond_mean_over_assignments
from forecastbench.benchmarks.cliff_fog import l_junta_predictor


@dataclass(frozen=True)
class GroupRobustnessSpec:
    """Specification for group robustness experiments."""
    d: int = 20
    k: int = 8
    alpha: float = 0.8
    n_samples: int = 500000  # need many samples for small groups
    rho: float = 0.95
    L_ar: Optional[int] = None  # defaults to k-1
    delta: float = 0.05  # confidence level
    seed: int = 0


def worst_group_with_confidence(
    z: np.ndarray,
    residual: np.ndarray,
    J: Sequence[int],
    delta: float = 0.05,
) -> Dict:
    """
    Compute worst-group calibration with finite-sample CI from Proposition 5.
    
    CI: |μ̂_G - μ_G| <= sqrt(2 ln(2/δ) / N_G) w.p. >= 1-δ
    """
    k = len(J)
    keys = keys_for_J(z, J)
    n_groups = 1 << k
    
    results = []
    for key in range(n_groups):
        mask = keys == key
        N_G = int(mask.sum())
        if N_G == 0:
            continue
        
        mu_hat = float(np.mean(residual[mask]))
        # Hoeffding CI width
        ci_width = float(np.sqrt(2 * np.log(2 / delta) / N_G))
        
        results.append({
            "key": int(key),
            "N_G": int(N_G),
            "mu_hat": float(mu_hat),
            "ci_lo": float(mu_hat - ci_width),
            "ci_hi": float(mu_hat + ci_width),
            "ci_width": float(ci_width),
            "significant": abs(mu_hat) > ci_width,
        })
    
    if not results:
        return {
            "worst_group": None,
            "n_groups_seen": 0,
            "n_significant": 0,
        }
    
    worst = max(results, key=lambda r: abs(r["mu_hat"]))
    return {
        "worst_group": worst,
        "n_groups_seen": len(results),
        "n_significant": sum(r["significant"] for r in results),
        "all_groups": results,
    }


def group_robustness_separation(
    spec: GroupRobustnessSpec,
) -> Dict:
    """
    Direct test of Propositions 8 vs 9.
    
    - AR(L) on degree-(L+1) parity: GCal >= α/2
    - Diffusion(ρ): GCal = (α/2)(1 - ρ^k)
    """
    k = spec.k
    L_ar = spec.L_ar if spec.L_ar is not None else (k - 1)
    
    parity_spec = ParitySpec(d=spec.d, k=k, alpha=spec.alpha, seed=spec.seed)
    dataset = sample_parity_dataset(parity_spec, spec.n_samples)
    z = dataset["z"]
    p_true = dataset["p_true"]
    S = tuple(int(i) for i in dataset["S"])
    
    mkt = ParityMarket.create(parity_spec)
    
    # AR prediction (L-junta)
    q_ar = l_junta_predictor(z, L_ar, S, spec.alpha)
    
    # Diffusion prediction
    q_diff = mkt.diffusion_analytic(z, spec.rho)
    
    # Compute residuals
    residual_ar = p_true - q_ar
    residual_diff = p_true - q_diff
    
    # Worst-group calibration on subcubes G_{S,a}
    ar_result = worst_group_with_confidence(z, residual_ar, S, spec.delta)
    diff_result = worst_group_with_confidence(z, residual_diff, S, spec.delta)
    
    # Also compute simple max over assignments
    ar_gcal_simple = worst_group_abs_cond_mean_over_assignments(z=z, residual=residual_ar, J=S)
    diff_gcal_simple = worst_group_abs_cond_mean_over_assignments(z=z, residual=residual_diff, J=S)
    
    # Theory predictions
    alpha = spec.alpha
    ar_theory = alpha / 2  # Prop 9 lower bound
    diff_theory = (alpha / 2) * (1 - spec.rho ** k)  # Prop 8
    
    return {
        "spec": {
            "d": spec.d,
            "k": k,
            "alpha": spec.alpha,
            "n_samples": spec.n_samples,
            "rho": spec.rho,
            "L_ar": L_ar,
            "delta": spec.delta,
        },
        "S": list(S),
        "group_size": float(2 ** (-k)),
        "ar": {
            "L": L_ar,
            "empirical_gcal": float(ar_gcal_simple[0]),
            "worst_group_detail": ar_result["worst_group"],
            "n_significant": ar_result["n_significant"],
            "theory_lower_bound": float(ar_theory),
            "beats_theory": ar_gcal_simple[0] >= ar_theory - 0.02,
        },
        "diffusion": {
            "rho": spec.rho,
            "empirical_gcal": float(diff_gcal_simple[0]),
            "worst_group_detail": diff_result["worst_group"],
            "n_significant": diff_result["n_significant"],
            "theory_prediction": float(diff_theory),
            "matches_theory": abs(diff_gcal_simple[0] - diff_theory) < 0.02,
        },
        "separation": float(ar_gcal_simple[0] - diff_gcal_simple[0]),
        "diffusion_wins": diff_gcal_simple[0] < ar_gcal_simple[0],
    }


def exponential_group_scaling(
    d: int = 24,
    k_values: Tuple[int, ...] = (4, 6, 8, 10, 12),
    alpha: float = 0.8,
    n_samples: int = 1000000,
    rho: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For each k, evaluate worst-group calibration on groups of size 2^{-k}.
    
    Shows that diffusion maintains small error while AR error is constant.
    """
    rows = []
    
    for k in k_values:
        L_ar = k - 1  # AR with L = k-1 cannot see degree-k
        
        spec = GroupRobustnessSpec(
            d=d, k=k, alpha=alpha, n_samples=n_samples,
            rho=rho, L_ar=L_ar, seed=seed + k * 1000,
        )
        
        result = group_robustness_separation(spec)
        
        rows.append({
            "k": k,
            "group_size": result["group_size"],
            "log2_group_size": -k,
            "ar_gcal": result["ar"]["empirical_gcal"],
            "ar_theory": result["ar"]["theory_lower_bound"],
            "diff_gcal": result["diffusion"]["empirical_gcal"],
            "diff_theory": result["diffusion"]["theory_prediction"],
            "separation": result["separation"],
            "diffusion_wins": result["diffusion_wins"],
        })
    
    return pd.DataFrame(rows)


def rho_sweep_for_group_robustness(
    d: int = 20,
    k: int = 8,
    alpha: float = 0.8,
    n_samples: int = 500000,
    rho_values: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Sweep ρ to show diffusion can achieve arbitrarily small GCal.
    """
    parity_spec = ParitySpec(d=d, k=k, alpha=alpha, seed=seed)
    dataset = sample_parity_dataset(parity_spec, n_samples)
    z = dataset["z"]
    p_true = dataset["p_true"]
    S = tuple(int(i) for i in dataset["S"])
    
    mkt = ParityMarket.create(parity_spec)
    
    rows = []
    for rho in rho_values:
        q_diff = mkt.diffusion_analytic(z, rho)
        residual = p_true - q_diff
        
        gcal, _ = worst_group_abs_cond_mean_over_assignments(z=z, residual=residual, J=S)
        theory = (alpha / 2) * (1 - rho ** k)
        
        rows.append({
            "rho": float(rho),
            "empirical_gcal": float(gcal),
            "theory_gcal": float(theory),
            "error": float(abs(gcal - theory)),
        })
    
    return pd.DataFrame(rows)


def plot_group_robustness(results: Dict, output_dir: str) -> None:
    """Generate group robustness plots."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Bar chart: AR vs Diffusion GCal
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ["AR (L=k-1)", "Diffusion (ρ)"]
    empirical = [results["ar"]["empirical_gcal"], results["diffusion"]["empirical_gcal"]]
    theory = [results["ar"]["theory_lower_bound"], results["diffusion"]["theory_prediction"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, empirical, width, label='Empirical', color=['#d62728', '#2ca02c'])
    bars2 = ax.bar(x + width/2, theory, width, label='Theory', alpha=0.5, color=['#ff9896', '#98df8a'])
    
    ax.set_ylabel('Worst-Group Calibration Error', fontsize=12)
    ax.set_title(f'Group Robustness: k={results["spec"]["k"]}, group size=2^{{-{results["spec"]["k"]}}}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, empirical):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/group_robustness_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_exponential_scaling(df: pd.DataFrame, output_dir: str) -> None:
    """Plot group calibration vs group size."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df["k"], df["ar_gcal"], 'o-', label="AR (L=k-1)", markersize=8, linewidth=2, color='#d62728')
    ax.plot(df["k"], df["ar_theory"], 's--', label="AR Theory (α/2)", markersize=6, alpha=0.7, color='#ff9896')
    ax.plot(df["k"], df["diff_gcal"], 'o-', label=f"Diffusion (ρ={df['diff_theory'].iloc[0] / (df['ar_theory'].iloc[0]):.2f})", 
            markersize=8, linewidth=2, color='#2ca02c')
    ax.plot(df["k"], df["diff_theory"], 's--', label="Diffusion Theory", markersize=6, alpha=0.7, color='#98df8a')
    
    ax.set_xlabel("k (group size = 2^{-k})", fontsize=12)
    ax.set_ylabel("Worst-Group Calibration Error", fontsize=12)
    ax.set_title("Exponential Group Scaling: Diffusion vs AR", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add second x-axis showing group size
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df["k"])
    ax2.set_xticklabels([f'2^{{-{k}}}' for k in df["k"]])
    ax2.set_xlabel("Group Size", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exponential_group_scaling.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_group_robustness_suite(
    k_values: Tuple[int, ...] = (4, 6, 8, 10),
    d: int = 24,
    alpha: float = 0.8,
    n_samples: int = 500000,
    rho: float = 0.95,
    output_dir: str = "plots",
    seed: int = 0,
) -> Dict:
    """
    Run full group robustness suite.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Individual experiments for each k
    for k in k_values:
        spec = GroupRobustnessSpec(
            d=d, k=k, alpha=alpha, n_samples=n_samples,
            rho=rho, L_ar=k-1, seed=seed + k * 1000,
        )
        result = group_robustness_separation(spec)
        all_results[f"k={k}"] = result
        plot_group_robustness(result, f"{output_dir}/k{k}")
    
    # Scaling across k
    scaling_df = exponential_group_scaling(
        d=d, k_values=k_values, alpha=alpha, n_samples=n_samples,
        rho=rho, seed=seed,
    )
    all_results["scaling"] = scaling_df.to_dict(orient="records")
    plot_exponential_scaling(scaling_df, output_dir)
    
    # ρ sweep for one k
    k_mid = k_values[len(k_values) // 2]
    rho_df = rho_sweep_for_group_robustness(
        d=d, k=k_mid, alpha=alpha, n_samples=n_samples,
        seed=seed,
    )
    all_results["rho_sweep"] = rho_df.to_dict(orient="records")
    
    return all_results


