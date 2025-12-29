"""
Turtel et al. (2025) Comparison: RLVR Forecasting Paper.

Reference: https://arxiv.org/abs/2505.17989
"Outcome-based Reinforcement Learning to Predict the Future"

This module implements:
1. Direct comparison metrics (Brier, ECE, ROI)
2. Group robustness analysis (not in Turtel et al.)
3. Constraint coherence tests (multi-market)
4. RLVR-as-repair experiment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecastbench.metrics import brier_loss, expected_calibration_error, log_loss
from forecastbench.metrics.trading_sim import KellySimConfig, simulate_kelly_roi
from forecastbench.metrics.swap_regret import compute_swap_regret, compute_external_regret


@dataclass(frozen=True)
class TurtelComparisonSpec:
    """Spec for Turtel comparison experiments."""
    # Turtel et al. reported metrics
    turtel_roi: float = 0.10  # 10% ROI from abstract
    turtel_model_size: str = "14B"
    turtel_training: str = "RLVR"
    
    # Our model specs
    diffusion_param_count: int = 1_000_000  # ~1M params
    ar_baseline: str = "Qwen3-14B (no finetuning)"


def compare_to_turtel_baseline(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str = "y",
    market_prob_col: str = "market_prob",
    transaction_cost: float = 0.0,
) -> Dict:
    """
    Compare model predictions to Turtel et al. reported metrics.
    
    Returns metrics comparable to those in the paper:
    - Brier score
    - ECE (calibration)
    - ROI (trading simulation)
    """
    if pred_col not in df.columns:
        raise ValueError(f"Missing pred_col: {pred_col}")
    if y_col not in df.columns:
        raise ValueError(f"Missing y_col: {y_col}")
    
    predictions = df[pred_col].to_numpy().astype(np.float64)
    outcomes = df[y_col].to_numpy().astype(np.float64)
    
    # Core metrics
    brier = brier_loss(predictions, outcomes)
    ece = expected_calibration_error(predictions, outcomes)
    logloss = log_loss(predictions, outcomes)
    
    result = {
        "n": len(df),
        "brier": float(brier),
        "ece": float(ece),
        "logloss": float(logloss),
    }
    
    # Trading simulation if market prices available
    if market_prob_col in df.columns:
        market_prices = df[market_prob_col].to_numpy().astype(np.float64)
        
        kelly_result = simulate_kelly_roi(
            p=predictions,
            q=market_prices,
            y=outcomes,
            cfg=KellySimConfig(
                initial_bankroll=1.0,
                scale=1.0,
                frac_cap=0.25,
                fee=transaction_cost,
            ),
        )
        
        result["trading"] = {
            "roi": float(kelly_result["roi"]),
            "final_bankroll": float(kelly_result["final_bankroll"]),
            "turtel_reported_roi": 0.10,
            "beats_turtel": kelly_result["roi"] > 0.10,
        }
    
    return result


def group_robustness_advantage(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str = "y",
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Compute group robustness metrics (not analyzed in Turtel et al.).
    
    This highlights an advantage of the diffusion approach:
    theoretical guarantees on subgroup calibration.
    """
    if group_cols is None:
        # Try to infer reasonable group columns
        candidate_cols = ["category", "topic", "source", "month"]
        group_cols = [c for c in candidate_cols if c in df.columns]
    
    predictions = df[pred_col].to_numpy().astype(np.float64)
    outcomes = df[y_col].to_numpy().astype(np.float64)
    
    results = {
        "overall_ece": float(expected_calibration_error(predictions, outcomes)),
    }
    
    for gc in group_cols:
        if gc not in df.columns:
            continue
        
        groups = df[gc].unique()
        group_eces = []
        group_biases = []
        group_sizes = []
        
        for g in groups:
            mask = df[gc] == g
            if mask.sum() < 10:
                continue
            
            p_g = predictions[mask]
            y_g = outcomes[mask]
            
            ece_g = expected_calibration_error(p_g, y_g)
            bias_g = float(np.mean(y_g - p_g))
            
            group_eces.append(ece_g)
            group_biases.append(abs(bias_g))
            group_sizes.append(int(mask.sum()))
        
        if group_eces:
            results[gc] = {
                "n_groups": len(group_eces),
                "worst_ece": float(max(group_eces)),
                "mean_ece": float(np.mean(group_eces)),
                "worst_abs_bias": float(max(group_biases)),
                "mean_abs_bias": float(np.mean(group_biases)),
                "min_group_size": int(min(group_sizes)),
            }
    
    return results


def rlvr_as_repair_analysis(
    base_predictions: np.ndarray,
    rlvr_predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """
    Analyze RLVR as a swap-regret-minimizing repair mechanism.
    
    The theory (§9) suggests RLVR training corresponds to learning
    a repair map σ that fixes conditional deviations (swap regret).
    """
    # Compute regrets for base model
    base_ext = compute_external_regret(base_predictions, outcomes, n_bins)
    base_swap = compute_swap_regret(base_predictions, outcomes, n_bins)
    
    # Compute regrets for RLVR model
    rlvr_ext = compute_external_regret(rlvr_predictions, outcomes, n_bins)
    rlvr_swap = compute_swap_regret(rlvr_predictions, outcomes, n_bins)
    
    return {
        "base": {
            "external_regret": base_ext["regret_continuous"],
            "swap_regret": base_swap["swap_regret"],
            "gap": base_swap["swap_regret"] - base_ext["regret_continuous"],
        },
        "rlvr": {
            "external_regret": rlvr_ext["regret_continuous"],
            "swap_regret": rlvr_swap["swap_regret"],
            "gap": rlvr_swap["swap_regret"] - rlvr_ext["regret_continuous"],
        },
        "interpretation": {
            "rlvr_reduces_swap": rlvr_swap["swap_regret"] < base_swap["swap_regret"],
            "rlvr_reduces_gap": (rlvr_swap["swap_regret"] - rlvr_ext["regret_continuous"]) < 
                               (base_swap["swap_regret"] - base_ext["regret_continuous"]),
            "swap_reduction": base_swap["swap_regret"] - rlvr_swap["swap_regret"],
            "gap_reduction": (base_swap["swap_regret"] - base_ext["regret_continuous"]) -
                            (rlvr_swap["swap_regret"] - rlvr_ext["regret_continuous"]),
        },
        "theory_note": "RLVR = repair map minimizing swap regret (§9.4)",
    }


def diffusion_vs_turtel_summary(
    turtel_metrics: Dict,
    diffusion_metrics: Dict,
) -> Dict:
    """
    Generate a summary comparison table.
    """
    # Helper to safely get numeric value for comparison
    def _get_numeric(d: dict, key: str, default: float) -> float:
        val = d.get(key, default)
        if isinstance(val, (int, float)):
            return float(val)
        return default
    
    turtel_brier = _get_numeric(turtel_metrics, "brier", float("inf"))
    turtel_ece = _get_numeric(turtel_metrics, "ece", float("inf"))
    turtel_roi = _get_numeric(turtel_metrics, "roi", 0.10)
    
    diff_brier = float(diffusion_metrics.get("brier", float("inf")))
    diff_ece = float(diffusion_metrics.get("ece", float("inf")))
    diff_roi = float(diffusion_metrics.get("trading", {}).get("roi", 0))
    
    return {
        "metrics_comparison": {
            "brier": {
                "turtel": turtel_metrics.get("brier", "N/A"),
                "diffusion": diffusion_metrics["brier"],
                "winner": "diffusion" if diff_brier < turtel_brier else "turtel",
            },
            "ece": {
                "turtel": turtel_metrics.get("ece", "N/A"),
                "diffusion": diffusion_metrics["ece"],
                "winner": "diffusion" if diff_ece < turtel_ece else "turtel",
            },
            "roi": {
                "turtel": turtel_metrics.get("roi", 0.10),
                "diffusion": diffusion_metrics.get("trading", {}).get("roi", 0),
                "winner": "diffusion" if diff_roi > turtel_roi else "turtel",
            },
        },
        "advantages_of_our_approach": [
            "Theoretical calibration guarantees (Prop 7): diffusion ≈ noise operator",
            "Group robustness bounds (Props 8-9): works on 2^{-k} sized groups",
            "Lower compute: ~1M params vs 14B params",
            "Multi-market coherence: can enforce Fréchet/logical constraints",
            "RLVR is complementary (not competing): diffusion + light RLVR",
        ],
        "turtel_advantages": [
            "End-to-end trained system",
            "Uses news context effectively",
            "Demonstrated real trading profit",
        ],
    }


def create_turtel_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comparison table for the paper."""
    rows = []
    
    # Model properties
    rows.append({
        "Aspect": "Model Size",
        "Turtel et al.": "14B parameters",
        "Our Approach": "~1M parameters (diffusion) + external embedder",
    })
    rows.append({
        "Aspect": "Training",
        "Turtel et al.": "RLVR (outcome-based RL)",
        "Our Approach": "Supervised (proper scoring) + optional light RLVR",
    })
    rows.append({
        "Aspect": "Calibration Guarantee",
        "Turtel et al.": "Empirical improvement",
        "Our Approach": "Theoretical (Prop 7: T_ρf approximation)",
    })
    rows.append({
        "Aspect": "Group Robustness",
        "Turtel et al.": "Not analyzed",
        "Our Approach": "Bounded (Props 8-9): GCal → 0 as ρ → 1",
    })
    rows.append({
        "Aspect": "Multi-Market",
        "Turtel et al.": "Single-market predictions",
        "Our Approach": "Joint bundles with constraint enforcement",
    })
    
    # Performance (if available)
    if "metrics_comparison" in results:
        mc = results["metrics_comparison"]
        
        if "brier" in mc and mc["brier"]["turtel"] != "N/A":
            rows.append({
                "Aspect": "Brier Score",
                "Turtel et al.": f'{mc["brier"]["turtel"]:.4f}',
                "Our Approach": f'{mc["brier"]["diffusion"]:.4f}',
            })
        
        if "roi" in mc:
            rows.append({
                "Aspect": "Trading ROI",
                "Turtel et al.": f'{mc["roi"]["turtel"]:.1%}',
                "Our Approach": f'{mc["roi"]["diffusion"]:.1%}' if mc["roi"]["diffusion"] else "TBD",
            })
    
    return pd.DataFrame(rows)


def plot_turtel_comparison(results: Dict, output_dir: str) -> None:
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if "metrics_comparison" not in results:
        return
    
    mc = results["metrics_comparison"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Brier score
    if mc.get("brier", {}).get("turtel") != "N/A":
        ax = axes[0]
        vals = [mc["brier"]["turtel"], mc["brier"]["diffusion"]]
        colors = ['#1f77b4', '#2ca02c']
        bars = ax.bar(["Turtel et al.", "Diffusion (Ours)"], vals, color=colors)
        ax.set_ylabel("Brier Score (lower is better)")
        ax.set_title("Prediction Accuracy")
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom')
    
    # ECE
    if mc.get("ece", {}).get("turtel") != "N/A":
        ax = axes[1]
        vals = [mc["ece"]["turtel"], mc["ece"]["diffusion"]]
        colors = ['#1f77b4', '#2ca02c']
        bars = ax.bar(["Turtel et al.", "Diffusion (Ours)"], vals, color=colors)
        ax.set_ylabel("ECE (lower is better)")
        ax.set_title("Calibration")
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom')
    
    # ROI
    ax = axes[2]
    turtel_roi = mc.get("roi", {}).get("turtel", 0.10)
    diff_roi = mc.get("roi", {}).get("diffusion", 0)
    vals = [turtel_roi, diff_roi]
    colors = ['#1f77b4', '#2ca02c']
    bars = ax.bar(["Turtel et al.", "Diffusion (Ours)"], vals, color=colors)
    ax.set_ylabel("ROI")
    ax.set_title("Trading Performance")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/turtel_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_turtel_comparison(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str = "y",
    market_prob_col: str = "market_prob",
    group_cols: Optional[List[str]] = None,
    output_dir: str = "plots",
    turtel_brier: Optional[float] = None,
    turtel_ece: Optional[float] = None,
    turtel_roi: float = 0.10,
) -> Dict:
    """
    Run full Turtel comparison suite.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Our metrics
    our_metrics = compare_to_turtel_baseline(
        df, pred_col, y_col, market_prob_col
    )
    
    # Group robustness
    group_metrics = group_robustness_advantage(
        df, pred_col, y_col, group_cols
    )
    
    # Turtel baseline (from paper or provided)
    turtel_metrics = {
        "brier": turtel_brier if turtel_brier else "N/A",
        "ece": turtel_ece if turtel_ece else "N/A",
        "roi": turtel_roi,
    }
    
    # Summary
    summary = diffusion_vs_turtel_summary(turtel_metrics, our_metrics)
    
    results = {
        "our_metrics": our_metrics,
        "group_robustness": group_metrics,
        "turtel_reported": turtel_metrics,
        **summary,
    }
    
    # Table
    table = create_turtel_comparison_table(results)
    table.to_csv(f"{output_dir}/turtel_comparison_table.csv", index=False)
    
    # Plots
    plot_turtel_comparison(results, output_dir)
    
    return results

