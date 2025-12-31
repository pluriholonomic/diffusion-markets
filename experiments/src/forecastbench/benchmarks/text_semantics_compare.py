"""
Text Semantics Benchmark: AR vs Diffusion on Real Headlines

Tests whether models can leverage semantic meaning in prediction market questions.

Key question: Does chain-of-thought reasoning (AR) beat embedding-based diffusion
when the text contains meaningful information?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TextSemanticsSpec:
    """Specification for text semantics benchmark."""
    
    dataset_path: str
    text_cols: List[str]
    y_col: str = "y"
    market_prob_col: str = "market_prob"
    
    n_samples: int = 500  # Per-model evaluation size
    train_frac: float = 0.8
    seed: int = 0
    
    # AR settings
    ar_model: str = "Qwen/Qwen3-14B"
    ar_K: int = 3  # Self-consistency samples
    ar_max_tokens: int = 128
    
    # Diffusion settings
    embed_model: str = "Qwen/Qwen3-14B"
    diff_hidden: int = 256
    diff_depth: int = 4
    diff_steps: int = 1000


def load_polymarket_data(
    path: str,
    text_cols: List[str],
    y_col: str,
    market_prob_col: str,
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load and prepare Polymarket data."""
    df = pd.read_parquet(path)
    
    # Filter to rows with outcomes
    df = df[df[y_col].notna()].copy()
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    if max_rows:
        df = df.head(max_rows)
    
    # Create combined text
    def combine_text(row):
        parts = []
        for col in text_cols:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return " ".join(parts)
    
    df["text"] = df.apply(combine_text, axis=1)
    
    return df


def baseline_constant(n: int) -> np.ndarray:
    """Constant 0.5 prediction."""
    return np.full(n, 0.5, dtype=np.float32)


def baseline_market(df: pd.DataFrame, market_prob_col: str) -> np.ndarray:
    """Use market probability as prediction."""
    return df[market_prob_col].values.astype(np.float32)


def baseline_prior(df: pd.DataFrame, y_col: str, train_frac: float) -> Tuple[float, np.ndarray]:
    """Use training set base rate as prediction."""
    n_train = int(len(df) * train_frac)
    y_train = df[y_col].values[:n_train]
    base_rate = float(np.mean(y_train))
    return base_rate, np.full(len(df) - n_train, base_rate, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute forecasting metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 1e-6, 1 - 1e-6)
    
    # Brier score
    brier = float(np.mean((y_pred - y_true) ** 2))
    
    # Log loss
    logloss = float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    
    # Calibration (ECE)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    ece = float(ece / len(y_true))
    
    # Accuracy (at 0.5 threshold)
    accuracy = float(np.mean((y_pred > 0.5) == y_true))
    
    return {
        "brier": brier,
        "logloss": logloss,
        "ece": ece,
        "accuracy": accuracy,
        "n": len(y_true),
    }


def analyze_text_difficulty(df: pd.DataFrame, y_col: str) -> Dict:
    """Analyze what makes questions hard/easy."""
    # Categorize by topic keywords
    categories = {
        "sports": ["win", "beat", "game", "match", "championship", "score"],
        "crypto": ["bitcoin", "btc", "eth", "crypto", "token", "price"],
        "politics": ["trump", "biden", "election", "vote", "president", "congress"],
        "entertainment": ["movie", "album", "award", "oscar", "grammy"],
    }
    
    def categorize(text):
        text_lower = text.lower()
        for cat, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return cat
        return "other"
    
    df = df.copy()
    df["category"] = df["text"].apply(categorize)
    
    stats = {}
    for cat in df["category"].unique():
        mask = df["category"] == cat
        stats[cat] = {
            "n": int(mask.sum()),
            "base_rate": float(df.loc[mask, y_col].mean()),
        }
    
    return stats


def run_text_semantics_analysis(
    df: pd.DataFrame,
    y_col: str,
    market_prob_col: str,
    train_frac: float = 0.8,
) -> Dict:
    """
    Run analysis of how well different baselines perform.
    
    This doesn't require running AR/Diffusion - just analyzes the dataset
    and baseline performance.
    """
    n = len(df)
    n_train = int(n * train_frac)
    
    y_all = df[y_col].values.astype(np.float32)
    y_test = y_all[n_train:]
    
    results = {
        "n_total": n,
        "n_train": n_train,
        "n_test": n - n_train,
        "baselines": {},
    }
    
    # Constant 0.5
    p_const = baseline_constant(len(y_test))
    results["baselines"]["constant_0.5"] = compute_metrics(y_test, p_const)
    
    # Market probability (if available)
    if market_prob_col in df.columns:
        p_market = df[market_prob_col].values[n_train:].astype(np.float32)
        results["baselines"]["market"] = compute_metrics(y_test, p_market)
    
    # Base rate prior
    base_rate, p_prior = baseline_prior(df, y_col, train_frac)
    results["baselines"]["prior"] = {
        "base_rate": base_rate,
        **compute_metrics(y_test, p_prior),
    }
    
    # Text analysis
    results["text_analysis"] = analyze_text_difficulty(df, y_col)
    
    return results


def create_comparison_prompts(texts: List[str]) -> List[str]:
    """Create prompts for AR model."""
    prompts = []
    for text in texts:
        prompt = f"""You are a forecaster predicting binary outcomes for prediction markets.

Question: {text}

Think step by step about the factors that would influence this outcome.
Then provide your probability estimate (0-100%) that this resolves YES.

Reasoning:"""
        prompts.append(prompt)
    return prompts


def summarize_text_semantics_results(results: Dict) -> str:
    """Create markdown summary."""
    lines = []
    lines.append("# Text Semantics Benchmark Results")
    lines.append("")
    lines.append(f"- Total samples: {results['n_total']}")
    lines.append(f"- Train: {results['n_train']}, Test: {results['n_test']}")
    lines.append("")
    
    lines.append("## Baseline Performance")
    lines.append("")
    lines.append("| Model | Brier | LogLoss | ECE | Accuracy |")
    lines.append("|-------|-------|---------|-----|----------|")
    
    for name, metrics in results["baselines"].items():
        lines.append(
            f"| {name} | {metrics['brier']:.4f} | {metrics['logloss']:.4f} | "
            f"{metrics['ece']:.4f} | {metrics['accuracy']:.1%} |"
        )
    
    lines.append("")
    lines.append("## Category Analysis")
    lines.append("")
    for cat, stats in results["text_analysis"].items():
        lines.append(f"- **{cat}**: n={stats['n']}, base_rate={stats['base_rate']:.1%}")
    
    return "\n".join(lines)



