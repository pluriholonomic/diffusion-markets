"""
Multiscale Approachability: Constraint ladder experiments.

The theory (§7.5) claims diffusion inference is a coarse-to-fine approachability procedure:
- Noise level ρ attenuates degree-s components by ρ^s
- Low ρ (high noise) → only low-degree constraints visible
- High ρ (low noise) → higher-degree constraints become reachable

This module implements:
1. Degree-stratified constraint families (parity-based)
2. Multiscale constraint evaluation across ρ
3. Constraint ladder visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from forecastbench.benchmarks.parity import ParityMarket, ParitySpec, sample_parity_dataset, sample_rademacher


def make_degree_s_constraint_family(
    d: int,
    s: int,
    n_constraints: int = 50,
    seed: int = 0,
) -> List[Tuple[Tuple[int, ...], Callable]]:
    """
    Generate random degree-s parity constraints.
    
    Each constraint is h_S(z, q) = χ_S(z) for a random S of size s.
    The violation is E[(Y - q) χ_S(Z)].
    
    Returns list of (S, constraint_fn) tuples.
    """
    rng = np.random.default_rng(seed)
    constraints = []
    
    for _ in range(n_constraints):
        S = tuple(sorted(rng.choice(d, size=s, replace=False).tolist()))
        
        def make_h(S_capture):
            def h(z: np.ndarray, q: np.ndarray) -> np.ndarray:
                chi = np.prod(z[:, list(S_capture)], axis=1).astype(np.float32)
                return chi
            return h
        
        constraints.append((S, make_h(S)))
    
    return constraints


def evaluate_constraints_at_rho(
    z: np.ndarray,
    p_true: np.ndarray,
    q: np.ndarray,
    constraints: List[Tuple[Tuple[int, ...], Callable]],
) -> Dict:
    """
    Evaluate constraint violations for a set of constraints.
    
    Violation = |E[(p_true - q) h(z, q)]|
    """
    residual = p_true - q
    violations = []
    
    for S, h in constraints:
        test_val = h(z, q)
        violation = float(np.abs(np.mean(residual * test_val)))
        violations.append({
            "S": list(S),
            "violation": violation,
        })
    
    violations_arr = [v["violation"] for v in violations]
    return {
        "max_violation": float(np.max(violations_arr)) if violations_arr else 0.0,
        "mean_violation": float(np.mean(violations_arr)) if violations_arr else 0.0,
        "violations": violations,
    }


def multiscale_constraint_evaluation(
    d: int,
    degrees: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    rho_schedule: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
    n_samples: int = 50000,
    n_constraints_per_degree: int = 50,
    seed: int = 0,
) -> Dict:
    """
    For each noise level ρ, compute constraint violations across degrees.
    
    Theory predicts: degree-s constraints become visible when ρ^s is non-negligible.
    """
    rng = np.random.default_rng(seed)
    
    # Generate data with a known parity truth (use max degree)
    max_deg = max(degrees)
    parity_spec = ParitySpec(d=d, k=max_deg, alpha=0.8, seed=seed)
    dataset = sample_parity_dataset(parity_spec, n_samples)
    z = dataset["z"]
    p_true = dataset["p_true"]
    S_true = tuple(int(i) for i in dataset["S"])
    
    mkt = ParityMarket.create(parity_spec)
    
    # Generate constraint families for each degree
    constraint_families = {}
    for s in degrees:
        constraint_families[s] = make_degree_s_constraint_family(
            d=d, s=s, n_constraints=n_constraints_per_degree,
            seed=seed + s * 1000,
        )
    
    # Evaluate at each ρ
    results = {}
    for rho in rho_schedule:
        q_rho = mkt.diffusion_analytic(z, rho)
        
        degree_results = {}
        for s in degrees:
            eval_result = evaluate_constraints_at_rho(
                z, p_true, q_rho, constraint_families[s]
            )
            degree_results[s] = {
                "max_violation": eval_result["max_violation"],
                "mean_violation": eval_result["mean_violation"],
                "effective_attenuation": float(rho ** s),
                "theory_visibility": 1 - rho ** s,  # how much the constraint is "seen"
            }
        
        results[rho] = degree_results
    
    return {
        "d": d,
        "degrees": list(degrees),
        "rho_schedule": list(rho_schedule),
        "n_samples": n_samples,
        "S_true": list(S_true),
        "results": results,
    }


def constraint_ladder_matrix(results: Dict) -> np.ndarray:
    """
    Convert results to a (n_rho, n_degrees) matrix for heatmap plotting.
    """
    rhos = sorted(results["results"].keys())
    degrees = results["degrees"]
    
    matrix = np.zeros((len(rhos), len(degrees)))
    for i, rho in enumerate(rhos):
        for j, s in enumerate(degrees):
            matrix[i, j] = results["results"][rho][s]["max_violation"]
    
    return matrix


def plot_constraint_ladder(results: Dict, output_dir: str) -> None:
    """
    Plot the constraint ladder heatmap.
    
    Expected pattern: diagonal "staircase" where high-degree constraints
    only become visible at low noise.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    matrix = constraint_ladder_matrix(results)
    rhos = sorted(results["results"].keys())
    degrees = results["degrees"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Violation heatmap
    im = axes[0].imshow(matrix.T, aspect='auto', origin='lower', cmap='YlOrRd')
    axes[0].set_xticks(range(len(rhos)))
    axes[0].set_xticklabels([f'{r:.2f}' for r in rhos])
    axes[0].set_yticks(range(len(degrees)))
    axes[0].set_yticklabels([str(s) for s in degrees])
    axes[0].set_xlabel('Noise Correlation ρ', fontsize=12)
    axes[0].set_ylabel('Constraint Degree', fontsize=12)
    axes[0].set_title('Constraint Violations: Coarse-to-Fine', fontsize=14)
    plt.colorbar(im, ax=axes[0], label='Max Violation')
    
    # Right: Theory comparison (effective attenuation)
    theory_matrix = np.zeros_like(matrix)
    for i, rho in enumerate(rhos):
        for j, s in enumerate(degrees):
            theory_matrix[i, j] = 1 - rho ** s  # how much is "visible"
    
    im2 = axes[1].imshow(theory_matrix.T, aspect='auto', origin='lower', cmap='YlOrRd')
    axes[1].set_xticks(range(len(rhos)))
    axes[1].set_xticklabels([f'{r:.2f}' for r in rhos])
    axes[1].set_yticks(range(len(degrees)))
    axes[1].set_yticklabels([str(s) for s in degrees])
    axes[1].set_xlabel('Noise Correlation ρ', fontsize=12)
    axes[1].set_ylabel('Constraint Degree', fontsize=12)
    axes[1].set_title('Theory: 1 - ρ^s (Visibility)', fontsize=14)
    plt.colorbar(im2, ax=axes[1], label='1 - ρ^s')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/constraint_ladder.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional: Line plot showing degree-by-degree recovery
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for j, s in enumerate(degrees):
        violations = [matrix[i, j] for i in range(len(rhos))]
        ax.plot(rhos, violations, 'o-', label=f'Degree {s}', markersize=6)
    
    ax.set_xlabel('Noise Correlation ρ', fontsize=12)
    ax.set_ylabel('Max Constraint Violation', fontsize=12)
    ax.set_title('Constraint Recovery by Degree', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/constraint_recovery_by_degree.png", dpi=150, bbox_inches='tight')
    plt.close()


@dataclass(frozen=True)
class ApproachabilityDynamicsSpec:
    """Spec for tracking approachability dynamics over time."""
    d: int = 16
    k: int = 6
    alpha: float = 0.8
    T: int = 10000
    rho: float = 0.95
    n_constraint_samples: int = 20  # constraints per degree
    seed: int = 0


def approachability_dynamics(spec: ApproachabilityDynamicsSpec) -> Dict:
    """
    Track how AppErr_t evolves as more samples arrive.
    
    This tests the Blackwell approachability hypothesis:
    forecaster predictions should approach C_ε over time.
    """
    rng = np.random.default_rng(spec.seed)
    
    # Generate streaming data
    parity_spec = ParitySpec(d=spec.d, k=spec.k, alpha=spec.alpha, seed=spec.seed)
    mkt = ParityMarket.create(parity_spec)
    S = mkt.S
    
    # Constraint family: group-bin indicators on the parity subcube
    # For simplicity, use the degree-k parity constraint directly
    
    # Cumulative sums for approachability
    cum_residual_chi = 0.0  # E[(Y-q) χ_S]
    
    curve = []
    
    for t in range(spec.T):
        # Sample one point
        z = sample_rademacher(1, spec.d, rng)
        p_true = mkt.p_true(z)[0]
        y = rng.binomial(1, p_true)
        
        # Diffusion prediction
        q = mkt.diffusion_analytic(z, spec.rho)[0]
        
        # Parity constraint test
        chi = float(np.prod(z[0, list(S)]))
        
        # Update cumulative
        residual = float(y) - float(q)
        cum_residual_chi += residual * chi
        
        # Compute approachability error
        if t > 0:
            mean_g = cum_residual_chi / (t + 1)
            app_err = abs(mean_g)  # distance to 0
            
            if (t + 1) % 100 == 0:
                curve.append({
                    "t": t + 1,
                    "app_err": float(app_err),
                    "mean_g": float(mean_g),
                })
    
    return {
        "spec": {
            "d": spec.d,
            "k": spec.k,
            "alpha": spec.alpha,
            "T": spec.T,
            "rho": spec.rho,
        },
        "curve": curve,
    }


def plot_approachability_dynamics(results: Dict, output_dir: str) -> None:
    """Plot AppErr_t over time."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    curve = results["curve"]
    ts = [c["t"] for c in curve]
    app_errs = [c["app_err"] for c in curve]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(ts, app_errs, 'b-', linewidth=2, label='Empirical AppErr')
    
    # Theory: should decay like 1/sqrt(T)
    theory = [1.0 / np.sqrt(t) for t in ts]
    ax.loglog(ts, theory, 'r--', linewidth=2, alpha=0.7, label='1/√T (theory)')
    
    ax.set_xlabel('Time T', fontsize=12)
    ax.set_ylabel('Approachability Error', fontsize=12)
    ax.set_title('Blackwell Approachability Dynamics', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/approachability_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_approachability_suite(
    d: int = 20,
    degrees: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    rho_schedule: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
    n_samples: int = 50000,
    output_dir: str = "plots",
    seed: int = 0,
) -> Dict:
    """Run full approachability suite."""
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Multiscale constraint evaluation
    multiscale_results = multiscale_constraint_evaluation(
        d=d, degrees=degrees, rho_schedule=rho_schedule,
        n_samples=n_samples, seed=seed,
    )
    plot_constraint_ladder(multiscale_results, output_dir)
    
    # Dynamics over time
    dynamics_spec = ApproachabilityDynamicsSpec(
        d=d, k=max(degrees) // 2, alpha=0.8, T=20000,
        rho=0.95, seed=seed,
    )
    dynamics_results = approachability_dynamics(dynamics_spec)
    plot_approachability_dynamics(dynamics_results, output_dir)
    
    return {
        "multiscale": multiscale_results,
        "dynamics": dynamics_results,
    }


# ==============================================================================
# Time-Dependent Blackwell Constraint Tracking
# ==============================================================================

@dataclass
class BlackwellConstraintTracker:
    """
    Track Blackwell approachability constraints over time during training.
    
    This implements the calibration-as-approachability view:
    - Define constraint functions h_g(z, q) for groups g
    - Track cumulative payoff vector: G_T = (1/T) Σ_t (Y_t - q_t) h(z_t, q_t)
    - Approachability requires G_T → C_ε (approachability set)
    
    For forecasting calibration:
    - Groups: (topic, bin) pairs for group-conditional calibration
    - Constraint: E[(Y - q) | group, bin(q)] = 0 for calibration
    - Track: max_g |E[(Y - q) | g]| over time
    
    Hypothesis H3: Diffusion model reduces constraint violations faster than AR-only.
    """
    
    n_groups: int = 10
    n_bins: int = 10
    
    def __post_init__(self):
        # Cumulative sums for each (group, bin) constraint
        self._cum_residual = np.zeros((self.n_groups, self.n_bins))
        self._cum_count = np.zeros((self.n_groups, self.n_bins))
        self._step = 0
        self._history = []
    
    def update(
        self,
        p: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        log_every: int = 100,
    ) -> Dict:
        """
        Update constraint tracking with new batch.
        
        Args:
            p: Predictions (N,)
            y: Outcomes (N,)
            groups: Group indices (N,) in [0, n_groups)
            
        Returns:
            Current constraint violation metrics
        """
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.int64) % self.n_groups
        
        # Bin assignments
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, self.n_bins - 1)
        
        # Update cumulative sums
        residual = y - p
        for i in range(len(p)):
            g, b = groups[i], bin_idx[i]
            self._cum_residual[g, b] += residual[i]
            self._cum_count[g, b] += 1
        
        self._step += len(p)
        
        # Compute current violations
        violations = np.zeros((self.n_groups, self.n_bins))
        for g in range(self.n_groups):
            for b in range(self.n_bins):
                if self._cum_count[g, b] > 0:
                    violations[g, b] = abs(
                        self._cum_residual[g, b] / self._cum_count[g, b]
                    )
        
        # Max constraint violation (distance to approachability set)
        max_violation = float(np.max(violations))
        mean_violation = float(np.mean(violations[self._cum_count > 0]))
        
        # Worst group (for group robustness)
        group_violations = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            mask = self._cum_count[g] > 0
            if mask.sum() > 0:
                group_violations[g] = float(np.max(violations[g][mask]))
        worst_group = int(np.argmax(group_violations))
        worst_group_violation = float(group_violations[worst_group])
        
        metrics = {
            "step": self._step,
            "max_violation": max_violation,
            "mean_violation": mean_violation,
            "worst_group": worst_group,
            "worst_group_violation": worst_group_violation,
            "n_active_constraints": int(np.sum(self._cum_count > 0)),
        }
        
        if self._step % log_every == 0:
            self._history.append(metrics.copy())
        
        return metrics
    
    def get_violation_matrix(self) -> np.ndarray:
        """Get (n_groups, n_bins) matrix of current violations."""
        violations = np.zeros((self.n_groups, self.n_bins))
        for g in range(self.n_groups):
            for b in range(self.n_bins):
                if self._cum_count[g, b] > 0:
                    violations[g, b] = abs(
                        self._cum_residual[g, b] / self._cum_count[g, b]
                    )
        return violations
    
    def get_history(self) -> List[Dict]:
        return self._history
    
    def compute_approachability_rate(self) -> float:
        """
        Estimate the rate at which violations decay.
        
        Theory predicts 1/sqrt(T) decay for approachable games.
        Returns the empirical exponent α where violation ~ T^{-α}.
        """
        if len(self._history) < 10:
            return np.nan
        
        ts = np.array([h["step"] for h in self._history])
        violations = np.array([h["max_violation"] for h in self._history])
        
        # Filter positive violations
        mask = violations > 0
        if mask.sum() < 5:
            return np.nan
        
        # Log-log regression to estimate exponent
        log_t = np.log(ts[mask])
        log_v = np.log(violations[mask])
        
        # Linear fit: log_v = -α * log_t + c
        slope, _ = np.polyfit(log_t, log_v, 1)
        return -float(slope)


def compare_constraint_convergence(
    ar_tracker: BlackwellConstraintTracker,
    diff_tracker: BlackwellConstraintTracker,
) -> Dict:
    """
    Compare Blackwell constraint convergence between AR-only and AR+Diffusion.
    
    This tests Hypothesis H3: Diffusion improving performance implies
    better learning of Blackwell constraints.
    
    Returns statistical comparison of convergence rates and final violations.
    """
    ar_hist = ar_tracker.get_history()
    diff_hist = diff_tracker.get_history()
    
    if len(ar_hist) < 5 or len(diff_hist) < 5:
        return {"error": "Not enough history for comparison"}
    
    # Final violations
    ar_final = ar_hist[-1]["max_violation"]
    diff_final = diff_hist[-1]["max_violation"]
    
    # Convergence rates
    ar_rate = ar_tracker.compute_approachability_rate()
    diff_rate = diff_tracker.compute_approachability_rate()
    
    # Area under violation curve (lower = faster convergence)
    ar_auc = np.trapz(
        [h["max_violation"] for h in ar_hist],
        [h["step"] for h in ar_hist],
    )
    diff_auc = np.trapz(
        [h["max_violation"] for h in diff_hist],
        [h["step"] for h in diff_hist],
    )
    
    # Worst-group comparison
    ar_worst = ar_hist[-1].get("worst_group_violation", ar_final)
    diff_worst = diff_hist[-1].get("worst_group_violation", diff_final)
    
    return {
        "ar_final_violation": ar_final,
        "diff_final_violation": diff_final,
        "violation_reduction": (ar_final - diff_final) / max(ar_final, 1e-6),
        "ar_convergence_rate": ar_rate,
        "diff_convergence_rate": diff_rate,
        "rate_improvement": diff_rate - ar_rate if not (np.isnan(ar_rate) or np.isnan(diff_rate)) else np.nan,
        "ar_auc": ar_auc,
        "diff_auc": diff_auc,
        "auc_reduction": (ar_auc - diff_auc) / max(ar_auc, 1e-6),
        "ar_worst_group": ar_worst,
        "diff_worst_group": diff_worst,
        "worst_group_improvement": (ar_worst - diff_worst) / max(ar_worst, 1e-6),
        "diffusion_helps": diff_final < ar_final,
        "faster_convergence": diff_rate > ar_rate if not (np.isnan(ar_rate) or np.isnan(diff_rate)) else None,
    }


def plot_constraint_convergence_comparison(
    ar_tracker: BlackwellConstraintTracker,
    diff_tracker: BlackwellConstraintTracker,
    output_dir: str,
) -> None:
    """Plot comparative constraint convergence."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ar_hist = ar_tracker.get_history()
    diff_hist = diff_tracker.get_history()
    
    if not ar_hist or not diff_hist:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Max violation over time
    ar_ts = [h["step"] for h in ar_hist]
    ar_viol = [h["max_violation"] for h in ar_hist]
    diff_ts = [h["step"] for h in diff_hist]
    diff_viol = [h["max_violation"] for h in diff_hist]
    
    axes[0].semilogy(ar_ts, ar_viol, 'r-', linewidth=2, label='AR only')
    axes[0].semilogy(diff_ts, diff_viol, 'b-', linewidth=2, label='AR + Diffusion')
    
    # Theory: 1/sqrt(T)
    max_t = max(max(ar_ts), max(diff_ts))
    theory_ts = np.linspace(100, max_t, 100)
    theory_viol = 1 / np.sqrt(theory_ts)
    axes[0].semilogy(theory_ts, theory_viol * ar_viol[0] * np.sqrt(ar_ts[0]), 
                     'k--', alpha=0.5, label='1/√T (theory)')
    
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Max Constraint Violation', fontsize=12)
    axes[0].set_title('Blackwell Approachability Convergence', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Middle: Worst-group violation
    ar_worst = [h.get("worst_group_violation", h["max_violation"]) for h in ar_hist]
    diff_worst = [h.get("worst_group_violation", h["max_violation"]) for h in diff_hist]
    
    axes[1].semilogy(ar_ts, ar_worst, 'r-', linewidth=2, label='AR only')
    axes[1].semilogy(diff_ts, diff_worst, 'b-', linewidth=2, label='AR + Diffusion')
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Worst-Group Violation', fontsize=12)
    axes[1].set_title('Group Robustness (Worst-Group)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Right: Violation heatmaps
    ar_matrix = ar_tracker.get_violation_matrix()
    diff_matrix = diff_tracker.get_violation_matrix()
    
    # Show AR - Diff (positive = Diffusion is better)
    improvement = ar_matrix - diff_matrix
    im = axes[2].imshow(improvement.T, aspect='auto', cmap='RdYlGn', 
                        origin='lower', vmin=-0.1, vmax=0.1)
    axes[2].set_xlabel('Group', fontsize=12)
    axes[2].set_ylabel('Probability Bin', fontsize=12)
    axes[2].set_title('Violation Reduction\n(Green = Diff better)', fontsize=14)
    plt.colorbar(im, ax=axes[2], label='AR - Diff')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/constraint_convergence_comparison.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


@dataclass(frozen=True)
class TimeDependentBlackwellSpec:
    """Spec for time-dependent Blackwell experiment."""
    n_groups: int = 10
    n_bins: int = 10
    n_samples: int = 10000
    eval_every: int = 100
    seed: int = 0


def run_time_dependent_blackwell_comparison(
    ar_predictions: np.ndarray,
    diff_predictions: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    output_dir: str,
    spec: TimeDependentBlackwellSpec = TimeDependentBlackwellSpec(),
) -> Dict:
    """
    Run comparative time-dependent Blackwell analysis.
    
    Tests Hypothesis H3: Diffusion model better learns Blackwell constraints.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ar_tracker = BlackwellConstraintTracker(n_groups=spec.n_groups, n_bins=spec.n_bins)
    diff_tracker = BlackwellConstraintTracker(n_groups=spec.n_groups, n_bins=spec.n_bins)
    
    n = min(len(ar_predictions), len(diff_predictions), len(outcomes))
    
    # Process in batches
    for i in range(0, n, spec.eval_every):
        end = min(i + spec.eval_every, n)
        
        ar_tracker.update(
            ar_predictions[i:end],
            outcomes[i:end],
            groups[i:end],
            log_every=spec.eval_every,
        )
        
        diff_tracker.update(
            diff_predictions[i:end],
            outcomes[i:end],
            groups[i:end],
            log_every=spec.eval_every,
        )
    
    # Compare
    comparison = compare_constraint_convergence(ar_tracker, diff_tracker)
    
    # Plot
    plot_constraint_convergence_comparison(ar_tracker, diff_tracker, output_dir)
    
    # Save results
    import json
    results = {
        "spec": {
            "n_groups": spec.n_groups,
            "n_bins": spec.n_bins,
            "n_samples": n,
        },
        "comparison": comparison,
        "ar_history": ar_tracker.get_history(),
        "diff_history": diff_tracker.get_history(),
    }
    
    with open(f"{output_dir}/blackwell_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


# ==============================================================================
# Statistical Arbitrage Detection via C_t Distance
# ==============================================================================

def compute_arbitrage_bound(
    q_market: np.ndarray,
    p_model: np.ndarray,
    y: np.ndarray,
    *,
    constraint_type: str = "calibration",
    n_bins: int = 10,
) -> Dict:
    """
    Compute statistical arbitrage bounds using distance to C_t.
    
    KEY INSIGHT: The time-dependent Blackwell set C_t encodes the correlation
    structure of the market. The distance d(q, C_t) from market prices q to
    the constraint set C_t provides an upper bound on extractable statistical
    arbitrage.
    
    The model's predictions p define C_t implicitly:
    - C_t = {q : |E[(Y - q) | bin(q)]| ≤ ε for all bins}
    - d(q, C_t) ≈ max_bin |E[(Y - q_market) | bin(q_market)]|
    
    We measure:
    1. Distance to constraints (arbitrage upper bound)
    2. Realized PnL from betting against the market toward C_t
    3. Correlation between distance and realized profit
    
    This directly tests: Does diffusion learn C_t better, enabling
    better arbitrage detection and profit extraction?
    
    Args:
        q_market: Market prices (N,)
        p_model: Model predictions (N,) - defines the target C_t
        y: Realized outcomes (N,)
        constraint_type: "calibration" or "fréchet"
        n_bins: Number of calibration bins
        
    Returns:
        Dict with arbitrage bounds and correlations
    """
    q_market = np.asarray(q_market, dtype=np.float64)
    p_model = np.asarray(p_model, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    
    # Compute per-sample distance to C_t
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx_market = np.clip(np.digitize(q_market, bin_edges) - 1, 0, n_bins - 1)
    bin_idx_model = np.clip(np.digitize(p_model, bin_edges) - 1, 0, n_bins - 1)
    
    # Distance to C_t for market prices (per-bin calibration error)
    market_distance = np.zeros(n)
    model_distance = np.zeros(n)
    
    for b in range(n_bins):
        mask = bin_idx_market == b
        if mask.sum() > 1:
            residual = y[mask] - q_market[mask]
            market_distance[mask] = np.abs(np.mean(residual))
            
        mask = bin_idx_model == b
        if mask.sum() > 1:
            residual = y[mask] - p_model[mask]
            model_distance[mask] = np.abs(np.mean(residual))
    
    # Realized PnL from betting on model vs market
    # Position: b = sign(p_model - q_market) * |p_model - q_market|
    # This bets that the market should move toward the model's prediction
    position = p_model - q_market  # Long if model > market
    pnl = position * (y - q_market)  # Profit = position * (outcome - price)
    
    # Correlation between distance and profit magnitude
    finite_mask = np.isfinite(market_distance) & np.isfinite(pnl)
    if finite_mask.sum() > 10:
        corr_dist_pnl = float(np.corrcoef(market_distance[finite_mask], 
                                           np.abs(pnl[finite_mask]))[0, 1])
    else:
        corr_dist_pnl = 0.0
    
    # Fraction of arbitrage captured
    # Total arbitrage = sum of |y - q| (max possible profit if you knew y)
    total_arbitrage = float(np.sum(np.abs(y - q_market)))
    captured_arbitrage = float(np.sum(pnl[pnl > 0]))
    arbitrage_capture_rate = captured_arbitrage / max(total_arbitrage, 1e-10)
    
    # Sharpe ratio of trading strategy
    if np.std(pnl) > 1e-10:
        sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(252))  # Annualized
    else:
        sharpe = 0.0
    
    return {
        "n": n,
        "market_mean_distance": float(np.mean(market_distance)),
        "market_max_distance": float(np.max(market_distance)),
        "model_mean_distance": float(np.mean(model_distance)),
        "model_max_distance": float(np.max(model_distance)),
        "distance_reduction": float(1 - np.mean(model_distance) / max(np.mean(market_distance), 1e-10)),
        "mean_pnl": float(np.mean(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "sharpe_ratio": sharpe,
        "corr_distance_profit": corr_dist_pnl,
        "arbitrage_capture_rate": arbitrage_capture_rate,
        "interpretation": (
            f"Market distance to C_t: {np.mean(market_distance):.4f} "
            f"(bounds max arb). Model reduces to {np.mean(model_distance):.4f} "
            f"({100*(1-np.mean(model_distance)/max(np.mean(market_distance),1e-10)):.1f}% reduction). "
            f"Distance-profit correlation: {corr_dist_pnl:.2f}"
        ),
    }


def compare_arbitrage_detection(
    q_market: np.ndarray,
    p_ar: np.ndarray,
    p_hybrid: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict:
    """
    Compare AR-only vs AR+Diffusion for statistical arbitrage detection.
    
    This directly tests H3: Does diffusion improve learning of C_t,
    measured by arbitrage detection and profit extraction?
    
    Key metrics:
    - Distance reduction: How much closer to C_t does each model get?
    - Arbitrage capture: What fraction of exploitable arbitrage do we extract?
    - Distance-profit correlation: Does knowing d(q, C_t) predict profit?
    
    Args:
        q_market: Market prices
        p_ar: AR-only predictions
        p_hybrid: AR+Diffusion predictions
        y: Realized outcomes
        n_bins: Calibration bins
        
    Returns:
        Comparative arbitrage detection results
    """
    ar_arb = compute_arbitrage_bound(q_market, p_ar, y, n_bins=n_bins)
    hybrid_arb = compute_arbitrage_bound(q_market, p_hybrid, y, n_bins=n_bins)
    
    return {
        "ar": ar_arb,
        "hybrid": hybrid_arb,
        "improvement": {
            "distance_reduction_gain": hybrid_arb["distance_reduction"] - ar_arb["distance_reduction"],
            "pnl_improvement": hybrid_arb["mean_pnl"] - ar_arb["mean_pnl"],
            "pnl_improvement_pct": (hybrid_arb["mean_pnl"] - ar_arb["mean_pnl"]) / max(abs(ar_arb["mean_pnl"]), 1e-10),
            "capture_rate_improvement": hybrid_arb["arbitrage_capture_rate"] - ar_arb["arbitrage_capture_rate"],
            "sharpe_improvement": hybrid_arb["sharpe_ratio"] - ar_arb["sharpe_ratio"],
            "correlation_improvement": hybrid_arb["corr_distance_profit"] - ar_arb["corr_distance_profit"],
        },
        "summary": (
            f"AR captures {ar_arb['arbitrage_capture_rate']*100:.1f}% of arbitrage, "
            f"Hybrid captures {hybrid_arb['arbitrage_capture_rate']*100:.1f}% "
            f"(+{(hybrid_arb['arbitrage_capture_rate']-ar_arb['arbitrage_capture_rate'])*100:.1f}pp). "
            f"Sharpe: AR={ar_arb['sharpe_ratio']:.2f}, Hybrid={hybrid_arb['sharpe_ratio']:.2f}"
        ),
        "hypothesis_h3_supported": (
            hybrid_arb["arbitrage_capture_rate"] > ar_arb["arbitrage_capture_rate"] and
            hybrid_arb["distance_reduction"] > ar_arb["distance_reduction"]
        ),
    }


def constraint_ladder_analysis(
    z: np.ndarray,
    p_true: np.ndarray,
    diffusion_samples: Dict[float, np.ndarray],
    *,
    degrees: Tuple[int, ...] = (1, 2, 4, 6, 8),
) -> Dict:
    """
    Analyze the "constraint ladder" at different noise levels.
    
    HYPOTHESIS: At noise level σ, diffusion learns constraints up to degree
    inversely proportional to σ. Higher noise = coarser constraints (marginals),
    lower noise = finer constraints (higher-order moments).
    
    This validates the interpretation that diffusion's noise schedule
    corresponds to a sequence C_σ₁ ⊃ C_σ₂ ⊃ ... ⊃ C_0 of nested constraint sets.
    
    Args:
        z: Context vectors (N, d)
        p_true: True probabilities (N,)
        diffusion_samples: Dict mapping noise level σ to predictions at that σ
        degrees: Polynomial degrees to test
        
    Returns:
        Constraint satisfaction at each (σ, degree) pair
    """
    results = {}
    
    for sigma, p_sigma in diffusion_samples.items():
        p_sigma = np.asarray(p_sigma, dtype=np.float64)
        sigma_results = {}
        
        for deg in degrees:
            # For each degree, compute constraint violation
            # (This is a simplified version - real implementation would use
            # the parity market framework from ParityMarket)
            
            # Simple proxy: higher-degree "calibration" using z features
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            
            d = min(deg, z.shape[1])
            
            # Create degree-d polynomial features from z
            # For simplicity, use product of first d features
            if d > 0 and z.shape[1] >= d:
                feature = np.prod(z[:, :d], axis=1)
            else:
                feature = np.ones(len(z))
            
            # Constraint: E[(Y - p) * feature] = 0 for calibration
            residual = p_true - p_sigma
            weighted_residual = residual * feature
            constraint_violation = float(np.abs(np.mean(weighted_residual)))
            
            sigma_results[f"degree_{deg}"] = {
                "violation": constraint_violation,
                "satisfied": constraint_violation < 0.1,
            }
        
        results[f"sigma_{sigma}"] = sigma_results
    
    # Check if constraint satisfaction follows expected pattern
    # (higher σ should satisfy low-degree, fail high-degree)
    pattern_check = {}
    sigmas = sorted(diffusion_samples.keys(), reverse=True)  # High to low
    
    for deg in degrees:
        violations = [
            results[f"sigma_{s}"][f"degree_{deg}"]["violation"]
            for s in sigmas
        ]
        # Should be decreasing (better at low noise)
        monotonic = all(violations[i] >= violations[i+1] for i in range(len(violations)-1))
        pattern_check[f"degree_{deg}_monotonic"] = monotonic
    
    return {
        "per_sigma_degree": results,
        "pattern_check": pattern_check,
        "ladder_hypothesis_supported": all(pattern_check.values()),
        "interpretation": (
            "Constraint ladder hypothesis: At high noise, only low-degree constraints "
            "are satisfied. As noise decreases, higher-degree constraints are learned. "
            f"Pattern check: {sum(pattern_check.values())}/{len(pattern_check)} degrees show monotonic improvement."
        ),
    }

