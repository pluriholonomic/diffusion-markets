"""
Synthetic Headlines Benchmark: Controlled Text with Known Structure

Tests whether models learn REAL correlation structure vs overfit to text patterns.

Experimental Design:
1. Generate realistic headlines with KNOWN underlying probability structure
2. Compare model performance on:
   - Synthetic structured (ground truth is f(hidden_vars))
   - Synthetic unstructured (ground truth is random)
   - Real Polymarket (ground truth is actual outcomes)

Type I Error: Model "learns" on unstructured data → overfitting
Type II Error: Model fails on structured data → inadequate learning

Key insight: If model performs similarly on structured vs unstructured,
it's likely overfitting to surface text patterns rather than learning
true semantic relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np


# =============================================================================
# Headline Templates by Category
# =============================================================================

SPORTS_TEMPLATES = [
    "Will {team_a} beat {team_b} in {tournament}?",
    "Will {team_a} win {tournament}?",
    "Will {team_a} vs. {team_b} end in a draw?",
    "Will {team_a} score more than {score_threshold} goals against {team_b}?",
    "Will {player} score in {team_a} vs {team_b}?",
]

CRYPTO_TEMPLATES = [
    "Will {crypto} reach ${price_target} by {date}?",
    "Will {crypto} be above ${price_threshold} on {date}?",
    "Will {crypto} outperform {crypto_b} this {period}?",
    "Will {crypto} market cap exceed ${market_cap}B?",
]

POLITICS_TEMPLATES = [
    "Will {candidate} win {election}?",
    "Will {bill} pass the {chamber}?",
    "Will {country} {action} by {date}?",
    "Will {politician} {event} before {date}?",
]

ENTERTAINMENT_TEMPLATES = [
    "Will {movie} gross over ${box_office}M?",
    "Will {artist} win {award}?",
    "Will {show} be renewed for season {season}?",
]

# Entity pools for template filling
TEAMS = ["Lakers", "Celtics", "Warriors", "Heat", "Bucks", "Suns", "Nuggets", "76ers",
         "Manchester United", "Liverpool", "Chelsea", "Arsenal", "Barcelona", "Real Madrid",
         "SAW", "FaZe", "Navi", "G2", "Cloud9", "Vitality", "MOUZ", "Astralis"]

TOURNAMENTS = ["NBA Finals", "Premier League", "Champions League", "World Cup", 
               "ESL Katowice", "BLAST Premier", "IEM Cologne", "Major Championship"]

CRYPTOS = ["Bitcoin", "Ethereum", "Solana", "XRP", "Cardano", "Dogecoin", "BNB"]

CANDIDATES = ["Trump", "Biden", "Harris", "DeSantis", "Newsom", "Haley"]
ELECTIONS = ["2024 Presidential Election", "GOP Primary", "Democratic Primary", "midterm elections"]

MOVIES = ["Avatar 3", "Dune Part 3", "Spider-Man 4", "The Batman 2", "Oppenheimer 2"]
ARTISTS = ["Taylor Swift", "Drake", "Beyoncé", "Bad Bunny", "The Weeknd"]
AWARDS = ["Grammy", "Oscar", "Emmy", "Golden Globe"]


@dataclass(frozen=True)
class SyntheticHeadlineSpec:
    """Specification for synthetic headline generation."""
    
    n_samples: int = 2000
    n_test: int = 500
    
    # Structure type
    structure: Literal["factor", "independent", "adversarial"] = "factor"
    # factor: Hidden factors determine outcomes (learnable structure)
    # independent: Random outcomes (no structure)
    # adversarial: Structure exists but contradicts surface patterns
    
    n_factors: int = 3
    noise: float = 0.2
    seed: int = 0
    
    # Category mix
    category_weights: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)  # sports, crypto, politics, entertainment


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_headline(
    category: str,
    rng: np.random.Generator,
) -> Tuple[str, Dict[str, str]]:
    """Generate a single headline with its entity assignments."""
    
    if category == "sports":
        template = rng.choice(SPORTS_TEMPLATES)
        entities = {
            "team_a": rng.choice(TEAMS),
            "team_b": rng.choice(TEAMS),
            "tournament": rng.choice(TOURNAMENTS),
            "player": f"Player_{rng.integers(1, 100)}",
            "score_threshold": str(rng.integers(1, 5)),
        }
    elif category == "crypto":
        template = rng.choice(CRYPTO_TEMPLATES)
        crypto = rng.choice(CRYPTOS)
        entities = {
            "crypto": crypto,
            "crypto_b": rng.choice([c for c in CRYPTOS if c != crypto]),
            "price_target": str(rng.integers(10, 200) * 1000),
            "price_threshold": str(rng.integers(10, 200) * 1000),
            "market_cap": str(rng.integers(100, 2000)),
            "date": f"202{rng.integers(4,6)}-{rng.integers(1,13):02d}",
            "period": rng.choice(["week", "month", "quarter"]),
        }
    elif category == "politics":
        template = rng.choice(POLITICS_TEMPLATES)
        entities = {
            "candidate": rng.choice(CANDIDATES),
            "election": rng.choice(ELECTIONS),
            "bill": f"H.R. {rng.integers(100, 9999)}",
            "chamber": rng.choice(["House", "Senate"]),
            "country": rng.choice(["US", "China", "Russia", "EU"]),
            "action": rng.choice(["impose sanctions", "sign treaty", "hold elections"]),
            "politician": rng.choice(CANDIDATES),
            "event": rng.choice(["resign", "announce candidacy", "face indictment"]),
            "date": f"202{rng.integers(4,6)}-{rng.integers(1,13):02d}",
        }
    else:  # entertainment
        template = rng.choice(ENTERTAINMENT_TEMPLATES)
        entities = {
            "movie": rng.choice(MOVIES),
            "box_office": str(rng.integers(100, 1000)),
            "artist": rng.choice(ARTISTS),
            "award": rng.choice(AWARDS),
            "show": f"Show_{rng.integers(1, 50)}",
            "season": str(rng.integers(2, 10)),
        }
    
    # Fill template
    headline = template
    for key, value in entities.items():
        headline = headline.replace("{" + key + "}", value)
    
    return headline, entities


def create_entity_features(entities: Dict[str, str], category: str) -> np.ndarray:
    """
    Convert entity assignments to numerical features.
    
    This creates a "hidden variable" representation that determines outcomes.
    """
    features = []
    
    # Hash entities to get consistent numerical features
    for key, value in sorted(entities.items()):
        # Simple hash to float in [-1, 1]
        h = hash(value) % 10000 / 5000 - 1.0
        features.append(h)
    
    # Category one-hot
    cat_idx = {"sports": 0, "crypto": 1, "politics": 2, "entertainment": 3}.get(category, 0)
    cat_onehot = [0.0] * 4
    cat_onehot[cat_idx] = 1.0
    features.extend(cat_onehot)
    
    # Pad to fixed size
    while len(features) < 20:
        features.append(0.0)
    
    return np.array(features[:20], dtype=np.float32)


def generate_synthetic_dataset(spec: SyntheticHeadlineSpec) -> Dict:
    """
    Generate synthetic headlines with controlled probability structure.
    
    Returns dict with:
    - headlines: List[str]
    - features: (n, d) hidden features
    - p_true: (n,) true probabilities
    - y: (n,) sampled outcomes
    - categories: List[str]
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n_samples
    
    categories = ["sports", "crypto", "politics", "entertainment"]
    weights = np.array(spec.category_weights[:len(categories)])
    weights = weights / weights.sum()
    
    headlines = []
    all_entities = []
    all_categories = []
    
    for _ in range(n):
        cat = rng.choice(categories, p=weights)
        headline, entities = generate_headline(cat, rng)
        headlines.append(headline)
        all_entities.append(entities)
        all_categories.append(cat)
    
    # Create hidden features
    features = np.stack([
        create_entity_features(ent, cat) 
        for ent, cat in zip(all_entities, all_categories)
    ])
    
    # Generate probabilities based on structure
    if spec.structure == "factor":
        # True structure: p = sigmoid(factors @ loadings + noise)
        # Factors are linear combinations of entity features
        n_factors = spec.n_factors
        factor_weights = rng.standard_normal((features.shape[1], n_factors)).astype(np.float32)
        factors = features @ factor_weights / np.sqrt(features.shape[1])
        
        # Market-specific loadings (single output)
        loadings = rng.standard_normal(n_factors).astype(np.float32)
        logits = factors @ loadings + spec.noise * rng.standard_normal(n).astype(np.float32)
        p_true = _sigmoid(logits).astype(np.float32)
        
    elif spec.structure == "independent":
        # No structure: random probabilities
        p_true = rng.uniform(0.2, 0.8, size=n).astype(np.float32)
        
    elif spec.structure == "adversarial":
        # Structure that contradicts surface patterns
        # High-prestige entities (popular teams, high prices) have LOWER probability
        prestige_score = np.zeros(n, dtype=np.float32)
        for i, (ent, cat) in enumerate(zip(all_entities, all_categories)):
            # Popular teams/cryptos get high prestige
            if cat == "sports" and ent.get("team_a") in ["Lakers", "Barcelona", "FaZe"]:
                prestige_score[i] += 1.0
            if cat == "crypto" and ent.get("crypto") in ["Bitcoin", "Ethereum"]:
                prestige_score[i] += 1.0
        
        # Adversarial: high prestige → LOW probability
        logits = -prestige_score + spec.noise * rng.standard_normal(n).astype(np.float32)
        p_true = _sigmoid(logits).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown structure: {spec.structure}")
    
    # Sample outcomes
    y = (rng.uniform(size=n) < p_true).astype(np.float32)
    
    return {
        "headlines": headlines,
        "features": features,
        "p_true": p_true,
        "y": y,
        "categories": all_categories,
        "spec": {
            "structure": spec.structure,
            "n_samples": spec.n_samples,
            "n_factors": spec.n_factors,
            "noise": spec.noise,
        },
    }


def compute_structure_recoverability(
    p_pred: np.ndarray,
    p_true: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """
    Compute metrics that distinguish structure learning from overfitting.
    
    Key metrics:
    - MSE to p_true: Did we learn the TRUE generative model?
    - Brier to y: Did we predict outcomes well?
    - Calibration: Are probabilities well-calibrated?
    
    Type I detection: High Brier, low MSE → overfitting to noise
    Type II detection: Low Brier, high MSE → missing structure
    """
    p_pred = np.clip(p_pred, 1e-6, 1 - 1e-6)
    
    # MSE to true probabilities (structure recovery)
    mse_to_truth = float(np.mean((p_pred - p_true) ** 2))
    
    # Brier score (outcome prediction)
    brier = float(np.mean((p_pred - y) ** 2))
    
    # Irreducible Bayes error
    bayes_brier = float(np.mean((p_true - y) ** 2))
    
    # Calibration ECE
    n_bins = 10
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y[mask].mean() - p_pred[mask].mean())
    ece = float(ece / len(y))
    
    # Correlation with true probabilities
    corr = float(np.corrcoef(p_pred, p_true)[0, 1]) if len(p_pred) > 1 else 0.0
    
    return {
        "mse_to_truth": mse_to_truth,
        "brier": brier,
        "bayes_brier": bayes_brier,
        "excess_brier": brier - bayes_brier,  # How much worse than optimal
        "ece": ece,
        "correlation_with_truth": corr,
        "n": len(y),
    }


def run_synthetic_headlines_benchmark(spec: SyntheticHeadlineSpec) -> Dict:
    """
    Run the full synthetic headlines benchmark.
    
    Generates data and evaluates baseline predictors.
    For full AR/Diffusion comparison, this returns the dataset
    for external model evaluation.
    """
    data = generate_synthetic_dataset(spec)
    
    n = len(data["y"])
    n_test = min(spec.n_test, n // 5)
    n_train = n - n_test
    
    # Split
    y_test = data["y"][n_train:]
    p_true_test = data["p_true"][n_train:]
    headlines_test = data["headlines"][n_train:]
    
    results = {
        "spec": data["spec"],
        "n_train": n_train,
        "n_test": n_test,
        "headlines_sample": headlines_test[:5],
        "baselines": {},
    }
    
    # Baseline: constant 0.5
    p_const = np.full(n_test, 0.5)
    results["baselines"]["constant_0.5"] = compute_structure_recoverability(
        p_const, p_true_test, y_test
    )
    
    # Baseline: training set prior
    y_train = data["y"][:n_train]
    prior = float(np.mean(y_train))
    p_prior = np.full(n_test, prior)
    results["baselines"]["prior"] = compute_structure_recoverability(
        p_prior, p_true_test, y_test
    )
    results["baselines"]["prior"]["base_rate"] = prior
    
    # Oracle: true probabilities
    results["baselines"]["oracle"] = compute_structure_recoverability(
        p_true_test, p_true_test, y_test
    )
    
    # Store data for external model evaluation
    results["test_data"] = {
        "headlines": headlines_test,
        "p_true": p_true_test.tolist(),
        "y": y_test.tolist(),
    }
    
    return results


def create_type_error_analysis(
    results_structured: Dict,
    results_independent: Dict,
    results_real: Optional[Dict] = None,
) -> str:
    """
    Analyze Type I and Type II errors across conditions.
    """
    lines = []
    lines.append("# Type I / Type II Error Analysis")
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("- **Type I (False Positive)**: Claims structure where none exists")
    lines.append("  - Test: Performance on INDEPENDENT should equal prior baseline")
    lines.append("- **Type II (False Negative)**: Misses real structure")
    lines.append("  - Test: Performance on FACTOR should approach oracle")
    lines.append("")
    
    lines.append("## Results")
    lines.append("")
    lines.append("| Condition | MSE to Truth | Correlation | Excess Brier |")
    lines.append("|-----------|--------------|-------------|--------------|")
    
    for name, res in [
        ("Structured (Factor)", results_structured),
        ("Unstructured (Independent)", results_independent),
    ]:
        if "baselines" in res and "prior" in res["baselines"]:
            prior = res["baselines"]["prior"]
            lines.append(
                f"| {name} (Prior) | {prior['mse_to_truth']:.4f} | "
                f"{prior['correlation_with_truth']:.3f} | {prior['excess_brier']:.4f} |"
            )
    
    if results_real:
        lines.append(f"| Real Polymarket | ? | ? | ? |")
    
    lines.append("")
    lines.append("## Type I Test")
    lines.append("If model beats prior on INDEPENDENT data → **Type I Error (overfitting)**")
    lines.append("")
    lines.append("## Type II Test")
    lines.append("If model equals prior on STRUCTURED data → **Type II Error (underfitting)**")
    
    return "\n".join(lines)


@dataclass
class CrossEvalSpec:
    """Spec for cross-evaluation: train on one structure, test on all."""
    
    n_train: int = 5000
    n_test: int = 1000
    train_structure: Literal["factor", "independent", "adversarial"] = "factor"
    n_factors: int = 5
    noise: float = 0.2
    seed: int = 0
    
    # Diffusion hyperparameters
    diff_hidden_dim: int = 256
    diff_depth: int = 4
    diff_T: int = 50
    diff_train_steps: int = 2000
    diff_batch_size: int = 256
    diff_lr: float = 1e-4
    diff_samples: int = 16
    
    # Use embedder or raw features
    use_embedder: bool = False
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def run_cross_evaluation(spec: CrossEvalSpec, device: str = "cuda") -> Dict:
    """
    Train diffusion model on one structure, evaluate on all structures.
    
    Returns:
    - train_metrics: Performance on training structure
    - transfer_metrics: Dict[structure, metrics] for each test structure
    - type_error_analysis: Summary of Type I/II errors
    """
    import torch
    
    rng = np.random.default_rng(spec.seed)
    
    # Generate training data
    train_spec = SyntheticHeadlineSpec(
        n_samples=spec.n_train,
        n_test=0,
        structure=spec.train_structure,
        n_factors=spec.n_factors,
        noise=spec.noise,
        seed=spec.seed,
    )
    train_data = generate_synthetic_dataset(train_spec)
    
    # Generate test data for all structures
    test_datasets = {}
    for structure in ["factor", "independent", "adversarial"]:
        test_spec = SyntheticHeadlineSpec(
            n_samples=spec.n_test,
            n_test=0,
            structure=structure,  # type: ignore
            n_factors=spec.n_factors,
            noise=spec.noise,
            seed=spec.seed + 1000,  # Different seed for test
        )
        test_datasets[structure] = generate_synthetic_dataset(test_spec)
    
    # Prepare training data
    X_train = train_data["features"]  # (n_train, d)
    y_train = train_data["y"]  # (n_train,)
    
    # Train a simple MLP diffusion model on (features -> probability)
    # This tests if diffusion can learn structure from features
    from forecastbench.models.diffusion_core import (
        DiffusionModelSpec,
        ContinuousDiffusionForecaster,
    )
    
    diff_spec = DiffusionModelSpec(
        out_dim=1,
        cond_dim=X_train.shape[1],
        hidden_dim=spec.diff_hidden_dim,
        depth=spec.diff_depth,
        T=spec.diff_T,
        beta_start=1e-4,
        beta_end=0.02,
        time_dim=32,
    )
    
    forecaster = ContinuousDiffusionForecaster(diff_spec, device=device)
    
    # Train on logit(p_true) to learn the true structure
    # For simplicity, we train on outcomes y (which are noisy samples from p_true)
    y_logit = np.log((y_train + 1e-6) / (1 - y_train + 1e-6))
    y_logit = np.clip(y_logit, -5, 5)  # Clip extreme values
    
    forecaster.train_mse_eps(
        x_train=y_logit[:, None],  # (n, 1)
        cond_train=X_train,
        n_steps=spec.diff_train_steps,
        batch_size=spec.diff_batch_size,
        lr=spec.diff_lr,
        log_every=500,
    )
    
    # Evaluate on all structures
    results = {
        "train_structure": spec.train_structure,
        "transfer_metrics": {},
    }
    
    for structure, test_data in test_datasets.items():
        X_test = test_data["features"]
        y_test = test_data["y"]
        p_true_test = test_data["p_true"]
        
        # Sample predictions
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        
        # Monte Carlo sampling
        preds_all = []
        for _ in range(spec.diff_samples):
            samples = forecaster.sample_x(
                cond=X_test_t,
                n_steps=spec.diff_T,
                eta=0.0,
            )  # (n_test, 1)
            preds_all.append(samples.cpu().numpy())
        
        preds_logit = np.mean(preds_all, axis=0)[:, 0]  # (n_test,)
        preds = 1.0 / (1.0 + np.exp(-preds_logit))
        preds = np.clip(preds, 0.01, 0.99)
        
        # Compute metrics
        metrics = compute_structure_recoverability(preds, p_true_test, y_test)
        metrics["structure"] = structure
        metrics["is_training_structure"] = (structure == spec.train_structure)
        
        results["transfer_metrics"][structure] = metrics
    
    # Type I/II error analysis
    factor_metrics = results["transfer_metrics"]["factor"]
    indep_metrics = results["transfer_metrics"]["independent"]
    
    # Type I: Does model "find" structure in independent data?
    # Measured by: correlation_with_truth should be ~0 for independent
    type_i_score = abs(indep_metrics.get("correlation_with_truth", 0))
    type_i_flag = type_i_score > 0.1  # Threshold for Type I error
    
    # Type II: Does model miss structure in factor data?
    # Measured by: correlation_with_truth should be high for factor
    type_ii_score = 1.0 - factor_metrics.get("correlation_with_truth", 0)
    type_ii_flag = type_ii_score > 0.5  # Threshold for Type II error
    
    results["type_error_analysis"] = {
        "type_i_score": type_i_score,
        "type_i_flag": type_i_flag,
        "type_ii_score": type_ii_score,
        "type_ii_flag": type_ii_flag,
        "interpretation": {
            "type_i": "HIGH correlation on independent data → overfitting to noise/surface patterns",
            "type_ii": "LOW correlation on factor data → failing to learn true structure",
        },
    }
    
    return results

