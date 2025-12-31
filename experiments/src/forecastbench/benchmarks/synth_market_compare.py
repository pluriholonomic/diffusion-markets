"""
Synthetic Prediction Market Benchmark: AR vs Diffusion

Tests forecasters on realistic prediction market structures:
1. Correlated markets (factor models)
2. Logical constraints (Fréchet, implications)
3. Multi-market bundles
4. Different noise levels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass(frozen=True)
class SynthMarketSpec:
    """Specification for synthetic prediction market benchmark."""
    
    d: int = 32  # Context dimension
    m: int = 8   # Number of markets
    n_train: int = 10000
    n_test: int = 2000
    
    # Correlation structure
    structure: Literal["independent", "factor", "chain", "frechet", "hierarchical"] = "factor"
    n_factors: int = 3  # For factor model
    factor_strength: float = 0.7  # How much variance explained by factors
    
    # Noise
    noise: float = 0.3
    
    seed: int = 0


def sample_factor_model(spec: SynthMarketSpec, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factor model: p_i = sigmoid(w_i @ x + loading_i @ factors + noise)
    
    This creates realistic correlation between markets via shared factors.
    """
    d, m = spec.d, spec.m
    
    # Context
    X = rng.standard_normal((n, d)).astype(np.float32)
    
    # Market-specific weights
    W = rng.standard_normal((d, m)).astype(np.float32) / np.sqrt(d)
    
    # Factor loadings (each market loads on shared factors)
    n_factors = min(spec.n_factors, m)
    factor_loadings = rng.standard_normal((n_factors, m)).astype(np.float32) / np.sqrt(n_factors)
    
    # Sample factors for each datapoint
    factors = rng.standard_normal((n, n_factors)).astype(np.float32)
    
    # Combine
    market_logits = X @ W  # (n, m) - context contribution
    factor_contrib = factors @ factor_loadings  # (n, m) - shared factor contribution
    noise = spec.noise * rng.standard_normal((n, m)).astype(np.float32)
    
    # Mix based on factor_strength
    logits = (1 - spec.factor_strength) * market_logits + spec.factor_strength * factor_contrib + noise
    
    P = _sigmoid(logits).astype(np.float32)
    return X, P


def sample_chain_model(spec: SynthMarketSpec, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chain implication: p_0 <= p_1 <= ... <= p_{m-1}
    
    Markets form a logical chain of implications.
    """
    d, m = spec.d, spec.m
    
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, m)).astype(np.float32) / np.sqrt(d)
    
    raw_logits = X @ W + spec.noise * rng.standard_normal((n, m)).astype(np.float32)
    raw_p = _sigmoid(raw_logits)
    
    # Sort to enforce chain constraint
    P = np.sort(raw_p, axis=1).astype(np.float32)
    return X, P


def sample_frechet_model(spec: SynthMarketSpec, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fréchet constraints for bundles: markets come in pairs (A_i, B_i, A_i ∧ B_i)
    
    For each triple, p_AB must satisfy: max(0, p_A + p_B - 1) <= p_AB <= min(p_A, p_B)
    """
    d = spec.d
    n_triples = spec.m // 3
    m = n_triples * 3
    
    X = rng.standard_normal((n, d)).astype(np.float32)
    
    P_list = []
    for t in range(n_triples):
        # Sample p_A, p_B independently
        w_a = rng.standard_normal(d).astype(np.float32) / np.sqrt(d)
        w_b = rng.standard_normal(d).astype(np.float32) / np.sqrt(d)
        w_mix = rng.standard_normal(d).astype(np.float32) / np.sqrt(d)
        
        p_a = _sigmoid(X @ w_a + spec.noise * rng.standard_normal(n))
        p_b = _sigmoid(X @ w_b + spec.noise * rng.standard_normal(n))
        
        # Sample p_AB in Fréchet interval
        lo = np.maximum(0, p_a + p_b - 1)
        hi = np.minimum(p_a, p_b)
        mix = _sigmoid(X @ w_mix)  # ∈ (0, 1)
        p_ab = lo + mix * (hi - lo)
        
        P_list.extend([p_a, p_b, p_ab])
    
    P = np.stack(P_list, axis=1).astype(np.float32)
    return X, P


def sample_hierarchical_model(spec: SynthMarketSpec, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical: Top-level events imply sub-events.
    
    Structure: root -> [child_1, child_2, ...] with p_root <= p_child_i
    """
    d, m = spec.d, spec.m
    
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, m)).astype(np.float32) / np.sqrt(d)
    
    raw_logits = X @ W + spec.noise * rng.standard_normal((n, m)).astype(np.float32)
    raw_p = _sigmoid(raw_logits)
    
    # First market is root, rest are children (root implies each child)
    # Constraint: p_root <= p_child_i
    # Implementation: p_root = raw_p[:, 0], p_child_i = p_root + (1 - p_root) * raw_p[:, i]
    P = np.zeros_like(raw_p)
    P[:, 0] = raw_p[:, 0]
    for i in range(1, m):
        P[:, i] = P[:, 0] + (1 - P[:, 0]) * raw_p[:, i]
    
    return X, P.astype(np.float32)


def sample_independent_model(spec: SynthMarketSpec, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Independent markets - no correlation structure."""
    d, m = spec.d, spec.m
    
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, m)).astype(np.float32) / np.sqrt(d)
    
    logits = X @ W + spec.noise * rng.standard_normal((n, m)).astype(np.float32)
    P = _sigmoid(logits).astype(np.float32)
    return X, P


def sample_data(spec: SynthMarketSpec) -> Dict[str, np.ndarray]:
    """Sample train/test data according to spec."""
    rng = np.random.default_rng(spec.seed)
    
    sampler = {
        "independent": sample_independent_model,
        "factor": sample_factor_model,
        "chain": sample_chain_model,
        "frechet": sample_frechet_model,
        "hierarchical": sample_hierarchical_model,
    }[spec.structure]
    
    X_train, P_train = sampler(spec, spec.n_train, rng)
    X_test, P_test = sampler(spec, spec.n_test, rng)
    
    # Sample outcomes from probabilities
    Y_train = (rng.uniform(size=P_train.shape) < P_train).astype(np.float32)
    Y_test = (rng.uniform(size=P_test.shape) < P_test).astype(np.float32)
    
    return {
        "X_train": X_train,
        "P_train": P_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "P_test": P_test,
        "Y_test": Y_test,
    }


def compute_constraint_violations(P_pred: np.ndarray, structure: str) -> Dict:
    """Compute constraint violations for the given structure."""
    n, m = P_pred.shape
    
    if structure == "chain":
        # p_i <= p_{i+1}
        violations = []
        for i in range(m - 1):
            v = np.maximum(0, P_pred[:, i] - P_pred[:, i + 1])
            violations.append(v)
        V = np.stack(violations, axis=1)
        
    elif structure == "hierarchical":
        # p_0 <= p_i for i > 0
        violations = []
        for i in range(1, m):
            v = np.maximum(0, P_pred[:, 0] - P_pred[:, i])
            violations.append(v)
        V = np.stack(violations, axis=1)
        
    elif structure == "frechet":
        n_triples = m // 3
        violations = []
        for t in range(n_triples):
            p_a = P_pred[:, t * 3]
            p_b = P_pred[:, t * 3 + 1]
            p_ab = P_pred[:, t * 3 + 2]
            
            # Fréchet lower bound: p_ab >= max(0, p_a + p_b - 1)
            lo = np.maximum(0, p_a + p_b - 1)
            v_lo = np.maximum(0, lo - p_ab)
            
            # Fréchet upper bound: p_ab <= min(p_a, p_b)
            hi = np.minimum(p_a, p_b)
            v_hi = np.maximum(0, p_ab - hi)
            
            violations.extend([v_lo, v_hi])
        V = np.stack(violations, axis=1) if violations else np.zeros((n, 0))
        
    else:
        # No constraints for independent
        V = np.zeros((n, 0))
    
    return {
        "mean_violation": float(np.mean(V)) if V.size > 0 else 0.0,
        "max_violation": float(np.max(V)) if V.size > 0 else 0.0,
        "frac_any_violated": float(np.mean(np.any(V > 0.01, axis=1))) if V.size > 0 else 0.0,
        "n_constraints": V.shape[1],
    }


def ar_predictor_oracle(X: np.ndarray, P_true: np.ndarray, L: int) -> np.ndarray:
    """
    Simulate L-query AR predictor.
    
    AR with query depth L can only use L coordinates of X.
    We simulate this by training on X[:, :L] only.
    """
    from sklearn.linear_model import LogisticRegression
    
    n, m = P_true.shape
    d = X.shape[1]
    
    # Use only first L coordinates (simulating L-query limitation)
    X_limited = X[:, :min(L, d)]
    
    # Fit per-market logistic regression
    P_pred = np.zeros_like(P_true)
    for i in range(m):
        model = LogisticRegression(max_iter=500, solver='lbfgs')
        y_binary = (P_true[:, i] > 0.5).astype(int)
        model.fit(X_limited, y_binary)
        P_pred[:, i] = model.predict_proba(X_limited)[:, 1]
    
    return P_pred.astype(np.float32)


def diffusion_predictor_analytic(X: np.ndarray, P_true: np.ndarray, rho: float) -> np.ndarray:
    """
    Simulate diffusion predictor with noise parameter rho.
    
    Diffusion smoothly attenuates signal: pred = rho * P_true + (1 - rho) * 0.5
    This is a simplification of the spectral attenuation.
    """
    # Diffusion converges to truth as rho -> 1
    P_pred = rho * P_true + (1 - rho) * 0.5
    return P_pred.astype(np.float32)


def run_synth_market_benchmark(spec: SynthMarketSpec) -> Dict:
    """
    Run the full synthetic market benchmark.
    
    Compares AR (various L) vs Diffusion (various rho) on:
    1. MSE to true probabilities
    2. Brier score against outcomes
    3. Constraint violations
    4. Calibration
    """
    data = sample_data(spec)
    
    X_train, P_train = data["X_train"], data["P_train"]
    X_test, P_test, Y_test = data["X_test"], data["P_test"], data["Y_test"]
    
    results = {
        "spec": {
            "d": spec.d,
            "m": spec.m,
            "n_train": spec.n_train,
            "n_test": spec.n_test,
            "structure": spec.structure,
            "noise": spec.noise,
        },
        "ar_results": [],
        "diffusion_results": [],
    }
    
    # Test AR with various query depths
    for L in [2, 4, 8, 16, 32]:
        if L > spec.d:
            continue
            
        P_pred_ar = ar_predictor_oracle(X_test, P_test, L=L)
        
        # Metrics
        mse = float(np.mean((P_pred_ar - P_test) ** 2))
        brier = float(np.mean((P_pred_ar - Y_test) ** 2))
        violations = compute_constraint_violations(P_pred_ar, spec.structure)
        
        results["ar_results"].append({
            "L": L,
            "mse": mse,
            "brier": brier,
            **violations,
        })
    
    # Test Diffusion with various rho
    for rho in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        P_pred_diff = diffusion_predictor_analytic(X_test, P_test, rho=rho)
        
        mse = float(np.mean((P_pred_diff - P_test) ** 2))
        brier = float(np.mean((P_pred_diff - Y_test) ** 2))
        violations = compute_constraint_violations(P_pred_diff, spec.structure)
        
        results["diffusion_results"].append({
            "rho": rho,
            "mse": mse,
            "brier": brier,
            **violations,
        })
    
    # Oracle (perfect prediction)
    violations_oracle = compute_constraint_violations(P_test, spec.structure)
    results["oracle"] = {
        "mse": 0.0,
        "brier": float(np.mean((P_test - Y_test) ** 2)),  # Irreducible Bayes error
        **violations_oracle,
    }
    
    # Constant 0.5 baseline
    P_const = np.full_like(P_test, 0.5)
    violations_const = compute_constraint_violations(P_const, spec.structure)
    results["constant_baseline"] = {
        "mse": float(np.mean((P_const - P_test) ** 2)),
        "brier": float(np.mean((P_const - Y_test) ** 2)),
        **violations_const,
    }
    
    return results


def create_comparison_table(results: Dict) -> str:
    """Create a markdown comparison table from results."""
    lines = []
    lines.append(f"## Synthetic Market Benchmark: {results['spec']['structure']}")
    lines.append(f"- d={results['spec']['d']}, m={results['spec']['m']}, n_test={results['spec']['n_test']}")
    lines.append("")
    lines.append("| Model | MSE | Brier | Constraint Violations |")
    lines.append("|-------|-----|-------|----------------------|")
    
    # Oracle
    o = results["oracle"]
    lines.append(f"| Oracle | 0.000 | {o['brier']:.4f} | {o['frac_any_violated']:.1%} |")
    
    # Constant
    c = results["constant_baseline"]
    lines.append(f"| Constant 0.5 | {c['mse']:.4f} | {c['brier']:.4f} | {c['frac_any_violated']:.1%} |")
    
    # AR
    for ar in results["ar_results"]:
        lines.append(f"| AR L={ar['L']} | {ar['mse']:.4f} | {ar['brier']:.4f} | {ar['frac_any_violated']:.1%} |")
    
    # Diffusion
    for diff in results["diffusion_results"]:
        lines.append(f"| Diff ρ={diff['rho']} | {diff['mse']:.4f} | {diff['brier']:.4f} | {diff['frac_any_violated']:.1%} |")
    
    return "\n".join(lines)



