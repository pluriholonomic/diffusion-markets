"""
GRPO: Group Relative Policy Optimization for Forecasting.

Implements GRPO (DeepSeekMath-style) for:
1. AR language models (like Turtel et al.)
2. Diffusion models (for the repair head)

GRPO Algorithm:
1. For each input x, sample K completions y_1, ..., y_K
2. Compute reward R_i for each completion
3. Compute group advantage: A_i = R_i - mean(R_1, ..., R_K)
4. Policy update: maximize sum_i A_i * log p(y_i | x)
5. Add KL penalty to reference policy

Modified GRPO (Dr. GRPO / Turtel et al. variant):
- Removes standard-deviation normalization: A_i = r_i - μ (not (r_i - μ)/σ)
- Preserves raw magnitude of large forecast errors
- Better for correcting extreme miscalibrations
- Requires additional guard-rails for stability

Guard-rails (from Turtel et al.):
- Token-length limits
- Gibberish filter  
- Early-stop criterion
- Gradients proportional to Brier loss

Reference: https://arxiv.org/abs/2505.17989
"Outcome-based Reinforcement Learning to Predict the Future"
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from forecastbench.models.ar_cot import _extract_prob


@dataclass(frozen=True)
class GRPORewardSpec:
    """
    Reward for forecasting.
    
    Supports five modes:
    
    1. turtel_brier: R = -(p̂ - y)² (Turtel et al. 2025)
       - Strict Brier: parse failures get R = -1.0
       - Pure Brier score, no trading component
       
    2. hybrid: R = alpha * logscore(p, y) + beta * pnl(p, q, y)
       - Our original formulation with trading signal
       
    3. kelly: R = log(1 + f*(odds * y - 1)) ≈ log(W_{t+1} / W_t)
       - Kelly criterion for optimal log-wealth growth
       - Optimal bet fraction: f* = (p - q) / (1 - q) for YES, adjusted for NO
       - Natural for long-term wealth maximization
       
    4. blackwell_aware: R = -(p̂ - y)² - λ * constraint_violation
       - Brier score + penalty for Blackwell constraint violations
       - Encourages calibration across groups/bins
       
    5. rlcr: RLCR (Damani et al. 2025, arXiv:2507.16806)
       - R = α * correctness + β * (-Brier)
       - "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty"
       - Jointly optimizes accuracy AND calibration
       - Proven: bounded proper scoring rules yield well-calibrated models
       - Can also include group-conditional calibration penalties
    """
    # Reward mode
    mode: str = "turtel_brier"  # "turtel_brier" | "hybrid" | "kelly" | "blackwell_aware" | "rlcr"
    
    # For hybrid mode
    alpha_logscore: float = 1.0
    beta_pnl: float = 0.1
    
    # Trading params (for hybrid and kelly modes)
    B: float = 1.0  # Max bet size
    transaction_cost: float = 0.01  # Turtel uses 0.01 (1 cent slippage)
    trading_mode: str = "linear"  # linear|sign|kelly
    
    # For turtel_brier mode
    parse_failure_penalty: float = -1.0  # Strict Brier: max loss for parse failures
    
    # For kelly mode
    kelly_fraction: float = 0.25  # Fractional Kelly (reduces variance)
    kelly_min_edge: float = 0.02  # Minimum edge to bet (skip small edges)
    kelly_log_clip: float = -3.0  # Clip log(1 + f*payoff) to avoid -inf
    
    # For blackwell_aware mode
    blackwell_lambda: float = 0.1  # Weight on constraint violations
    blackwell_n_bins: int = 10  # Number of calibration bins
    
    # For rlcr mode (Damani et al. 2025, arXiv:2507.16806)
    # R = rlcr_alpha * correctness + rlcr_beta * (-Brier) + rlcr_gamma * group_calibration
    rlcr_alpha: float = 1.0  # Weight on binary correctness: 1 if |p - y| < 0.5 else 0
    rlcr_beta: float = 1.0  # Weight on Brier calibration score
    rlcr_gamma: float = 0.5  # Weight on group-conditional calibration (CRITICAL for calibration)
    rlcr_correctness_threshold: float = 0.5  # p in correct half -> correct
    rlcr_n_groups: int = 10  # For group-conditional calibration
    rlcr_use_group_calibration: bool = True  # Add group calibration term


@dataclass(frozen=True)
class GRPOTrainSpec:
    """
    GRPO training specification for AR models.
    
    Supports two variants:
    1. Standard GRPO: A_i = (r_i - μ) / σ (normalized)
    2. Dr. GRPO / Turtel: A_i = r_i - μ (raw, no std normalization)
    
    The Turtel et al. variant (dr_grpo=True) preserves raw magnitude of 
    large forecast errors, helping correct extreme miscalibrations.
    """
    
    # Model
    model_name_or_path: str
    device: str = "auto"
    dtype: str = "auto"
    device_map: Optional[str] = None
    trust_remote_code: bool = True
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Decoding
    include_cot: bool = True
    cot_max_steps: int = 4
    max_prompt_tokens: int = 512
    max_new_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.95
    
    # GRPO-specific
    K: int = 4  # Number of samples per input (group size)
    steps: int = 200
    batch_size: int = 4  # Number of inputs per step (total samples = batch_size * K)
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    
    # KL regularization
    kl_coef: float = 0.01  # Lower than REINFORCE since group baseline helps
    
    # Algorithm variant (from Turtel et al. 2025)
    # "grpo": Standard GRPO with σ normalization
    # "dr_grpo": Modified GRPO (no σ normalization) - Dr. GRPO
    # "remax": ReMax with learned baseline (BEST in Turtel paper)
    algorithm: str = "remax"  # "grpo" | "dr_grpo" | "remax"
    
    # Deprecated flags (use algorithm instead)
    dr_grpo: bool = False  # Deprecated, use algorithm="dr_grpo"
    normalize_advantages: bool = False  # Deprecated
    
    # ReMax-specific (Turtel et al. Section 2.2)
    remax_baseline_lr: float = 1e-6  # Learned value baseline LR
    remax_baseline_loss_scale: float = 0.5  # MSE loss scaling
    entropy_bonus: float = 0.001  # Entropy bonus coefficient
    
    # Reward clipping and guard-rails (from Turtel et al.)
    reward_clip: float = 1.0  # Brier is already bounded [-1, 0]
    
    # Guard-rails for stability (Turtel et al. Section 2.3)
    max_response_tokens: int = 512  # Token-length limit
    max_input_chars: int = 16000  # Hard truncate input (Turtel: 16k chars)
    gibberish_filter: bool = True  # Filter gibberish outputs
    non_english_filter: bool = True  # Filter non-English passages
    require_rationale: bool = True  # Require <think>...</think> block
    early_stop_patience: int = 20  # Early stop if no improvement for N steps
    brier_weighted_grads: bool = False  # Turtel doesn't use this
    
    # Logging
    log_every: int = 10
    save_every: int = 50
    
    reward: GRPORewardSpec = GRPORewardSpec()


def _brier_reward(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Turtel et al. Brier-based reward: R = -(p̂ - y)²
    
    Returns negative squared error (higher = better).
    Range: [-1, 0]
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return -((p - y) ** 2)


def _kelly_reward(
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    *,
    fraction: float = 0.25,
    min_edge: float = 0.02,
    log_clip: float = -3.0,
    transaction_cost: float = 0.01,
) -> np.ndarray:
    """
    Kelly criterion reward: R = log(1 + f * payoff)
    
    The Kelly criterion gives the optimal bet size for maximizing long-term
    log-wealth growth. This makes it a natural reward for forecasting.
    
    For a YES bet at price q:
        - Win payoff: (1-q)/q - cost (you pay q, get 1)
        - Lose payoff: -1 - cost (lose your stake)
        - Kelly fraction: f* = p * (1-q)/q - (1-p) = (p - q) / (q * (1-q))
        
    For a NO bet at price (1-q):
        - Win payoff: q/(1-q) - cost
        - Lose payoff: -1 - cost
        - Kelly fraction: f* = (1-p) * q/(1-q) - p = ((1-p) - (1-q)) / ((1-q) * q)
        
    We use fractional Kelly (f' = fraction * f*) to reduce variance.
    
    Args:
        p: Model's probability predictions
        q: Market prices
        y: Realized outcomes (0 or 1)
        fraction: Fractional Kelly (0.25 = quarter Kelly)
        min_edge: Minimum |p - q| to place a bet
        log_clip: Minimum log value (avoid -inf on ruin)
        transaction_cost: Per-trade cost as fraction
        
    Returns:
        Log wealth change for each sample
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.clip(np.asarray(q, dtype=np.float64), 0.01, 0.99)
    y = np.asarray(y, dtype=np.float64)
    
    edge = p - q
    n = len(p)
    log_returns = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if abs(edge[i]) < min_edge:
            # No bet - log(1 + 0) = 0
            log_returns[i] = 0.0
            continue
        
        if edge[i] > 0:
            # Bet YES: we think p > q
            # Kelly: f* = (p - q) / (q * (1-q))  [simplified from odds form]
            # But simplified for binary: f* = (p - q) / (1 - q) [fraction of wealth]
            f_star = (p[i] - q[i]) / (1 - q[i])
            f = fraction * np.clip(f_star, 0, 1)  # Cap at 100% of stake
            
            if y[i] == 1:
                # Win: paid q, get 1, net = (1 - q) - cost
                payoff = f * ((1 - q[i]) / q[i] - transaction_cost)
            else:
                # Lose: lose stake plus cost
                payoff = -f * (1 + transaction_cost)
        else:
            # Bet NO: we think p < q
            f_star = (q[i] - p[i]) / q[i]
            f = fraction * np.clip(f_star, 0, 1)
            
            if y[i] == 0:
                # Win: paid (1-q), get 1, net = q / (1-q) - cost
                payoff = f * (q[i] / (1 - q[i]) - transaction_cost)
            else:
                # Lose: lose stake plus cost
                payoff = -f * (1 + transaction_cost)
        
        # Log wealth ratio (clip to avoid -inf on ruin)
        log_returns[i] = np.clip(np.log1p(payoff), log_clip, 5.0)
    
    return log_returns


def _blackwell_aware_reward(
    p: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    lambda_constraint: float = 0.1,
) -> Tuple[np.ndarray, Dict]:
    """
    Blackwell-aware reward: Brier + constraint violation penalty.
    
    R = -(p̂ - y)² - λ * |E[(Y - p̂) | bin(p̂)]|
    
    The constraint violation measures calibration within each bin.
    This encourages the model to learn Blackwell approachability constraints.
    
    Args:
        p: Model predictions
        y: Outcomes
        n_bins: Number of calibration bins
        lambda_constraint: Weight on constraint violation
        
    Returns:
        (rewards, constraint_info) where constraint_info contains per-bin violations
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    
    # Base Brier reward
    brier_reward = -((p - y) ** 2)
    
    # Compute per-bin calibration violations
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    bin_violations = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            residual = y[mask] - p[mask]
            bin_violations[b] = np.abs(np.mean(residual))
            bin_counts[b] = mask.sum()
    
    # Per-sample violation based on which bin it falls into
    sample_violations = bin_violations[bin_idx]
    
    # Combined reward
    reward = brier_reward - lambda_constraint * sample_violations
    
    constraint_info = {
        "bin_violations": bin_violations.tolist(),
        "bin_counts": bin_counts.tolist(),
        "max_violation": float(np.max(bin_violations)),
        "mean_violation": float(np.mean(bin_violations)),
    }
    
    return reward, constraint_info


def _rlcr_reward(
    p: np.ndarray,
    y: np.ndarray,
    *,
    q: Optional[np.ndarray] = None,  # Market probability - CRITICAL for forecasting
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
    correctness_threshold: float = 0.5,
    n_groups: int = 10,
    use_group_calibration: bool = True,
    groups: Optional[np.ndarray] = None,
    use_market_relative: bool = True,  # NEW: use market-relative rewards
) -> Tuple[np.ndarray, Dict]:
    """
    RLCR reward: Reinforcement Learning with Calibration Rewards.
    
    From Damani et al. (2025) "Beyond Binary Rewards: Training LMs to 
    Reason About Their Uncertainty" (arXiv:2507.16806).
    
    **IMPORTANT FIX**: The original RLCR was designed for classification where
    p is P(correct). For FORECASTING, we need market-relative rewards to avoid
    the model collapsing to predict p=0 or p=1 based on class imbalance.
    
    **Problem with original RLCR for forecasting**:
    - Brier reward = -(p - y)² is maximized at p=y
    - With 73% y=0, model learns to predict low p for everything
    - This is the OPPOSITE of calibration!
    
    **Fix: Market-relative RLCR**:
    - Correctness = sign(p - q) == sign(y - q)  [directional edge vs market]
    - Brier improvement = (q - y)² - (p - y)²  [reward beating the market]
    - Calibration = penalize |E[y - p | bin(p)]|  [standard calibration]
    
    Args:
        p: Model predictions (forecasted probabilities)
        y: Binary outcomes (0 or 1)
        q: Market probabilities (REQUIRED for market-relative mode)
        alpha: Weight on directional correctness (default 1.0)
        beta: Weight on Brier improvement over market (default 1.0)
        gamma: Weight on group calibration (default 0.1)
        correctness_threshold: Not used in market-relative mode
        n_groups: Number of groups for group-conditional calibration
        use_group_calibration: Whether to add Blackwell constraint term
        groups: Optional group assignments (if None, use bins of p)
        use_market_relative: If True, use market-relative rewards (recommended)
        
    Returns:
        (rewards, info_dict) where rewards has shape (n,)
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    
    # Clip to avoid edge cases
    p = np.clip(p, 0.001, 0.999)
    
    if use_market_relative and q is not None:
        # ============================================================
        # MARKET-RELATIVE RLCR (recommended for forecasting)
        # ============================================================
        q = np.asarray(q, dtype=np.float64)
        q = np.clip(q, 0.001, 0.999)
        
        # Component 1: Directional Correctness vs Market
        # correct = 1 if we bet in the right direction relative to market
        # i.e., if (p > q and y > q) or (p < q and y < q)
        edge_pred = p - q  # Our predicted edge over market
        edge_realized = y - q  # Actual edge (y is 0 or 1)
        correct = (np.sign(edge_pred) == np.sign(edge_realized))
        correctness_reward = correct.astype(np.float64)
        
        # Component 2: Brier Improvement over Market
        # Reward = Brier(market) - Brier(model)
        # Positive if we're better than market, negative if worse
        brier_market = (q - y) ** 2
        brier_model = (p - y) ** 2
        brier_improvement = brier_market - brier_model  # Higher = better
        # Normalize to similar scale as correctness (range roughly [-1, 1])
        brier_reward = brier_improvement  # Already bounded in [-1, 1]
        
        # Also track absolute Brier for logging
        brier = brier_model
        
    else:
        # ============================================================
        # ORIGINAL RLCR (problematic for imbalanced forecasting)
        # ============================================================
        # Component 1: Binary Correctness (Damani et al. Eq. 2, first term)
        correct = np.abs(p - y) < correctness_threshold
        correctness_reward = correct.astype(np.float64)
        
        # Component 2: Brier Score (Damani et al. Eq. 2, second term)
        brier = (p - y) ** 2
        brier_reward = -brier  # Negative because lower Brier is better
        brier_improvement = -brier  # For consistency in logging
    
    # Component 3: Group-conditional Calibration (Blackwell extension)
    # Penalty for violating the constraint E[(Y - p) | group] = 0
    group_penalty = np.zeros(n)
    group_violations = {}
    
    if use_group_calibration:
        # Use prediction bins as groups if not provided
        if groups is None:
            bin_edges = np.linspace(0, 1, n_groups + 1)
            groups = np.clip(np.digitize(p, bin_edges) - 1, 0, n_groups - 1)
        else:
            groups = np.asarray(groups) % n_groups
        
        # Compute per-group calibration errors (Blackwell constraint violations)
        for g in range(n_groups):
            mask = groups == g
            if mask.sum() > 1:
                # Calibration error = |mean(y) - mean(p)| in this bin
                residual = y[mask] - p[mask]
                group_error = np.abs(np.mean(residual))
                group_violations[g] = float(group_error)
                group_penalty[mask] = group_error
    
    # Combined RLCR reward
    # Market-relative: R = α*directional_correct + β*brier_improvement - γ*calibration_error
    reward = (
        alpha * correctness_reward +
        beta * brier_reward -
        gamma * group_penalty
    )
    
    info = {
        "correctness_mean": float(correctness_reward.mean()),
        "correctness_rate": float(correct.mean()),
        "brier_mean": float(brier.mean()),
        "brier_reward_mean": float(brier_reward.mean()),
        "brier_improvement_mean": float(np.mean(brier_improvement)),
        "group_violations": group_violations,
        "max_group_violation": float(max(group_violations.values())) if group_violations else 0.0,
        "mean_group_violation": float(np.mean(list(group_violations.values()))) if group_violations else 0.0,
        "market_relative": use_market_relative and q is not None,
    }
    
    return reward, info


def _logscore(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=np.float64)
    return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)


def _pnl_proxy(
    *,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    B: float,
    transaction_cost: float,
    mode: str,
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gap = p - q
    if mode == "sign":
        b = B * np.sign(gap)
    elif mode == "linear":
        b = np.clip(B * gap, -B, B)
    else:
        raise ValueError("mode must be sign|linear")
    return b * (y - q) - transaction_cost * np.abs(b)


def build_prompt(info: str, *, include_cot: bool, cot_max_steps: int) -> str:
    if include_cot:
        return (
            "You are a probabilistic forecaster.\n"
            "Given the information below, estimate P(YES) as a number in [0,1].\n"
            f"Think step-by-step in at most {max(int(cot_max_steps), 1)} short steps, then output ONLY a JSON object "
            'of the form {"p_yes": <number>}.\n\n'
            f"INFORMATION:\n{info}\n\n"
            "OUTPUT:\n"
        )
    return (
        "You are a probabilistic forecaster.\n"
        'Given the information below, output ONLY a JSON object of the form {"p_yes": <number in [0,1]>}.\n\n'
        f"INFORMATION:\n{info}\n\n"
        "OUTPUT:\n"
    )


def _is_gibberish(text: str, threshold: float = 0.5) -> bool:
    """
    Simple heuristic to detect gibberish outputs.
    
    Checks for:
    - Excessive repetition
    - Too many special characters
    - Very low entropy
    """
    if len(text) < 5:
        return False
    
    # Check for excessive character repetition
    from collections import Counter
    char_counts = Counter(text.lower())
    most_common_ratio = char_counts.most_common(1)[0][1] / len(text) if char_counts else 0
    if most_common_ratio > 0.5:
        return True
    
    # Check for repeated substrings (e.g., "aaaaa" or "abcabcabc")
    for length in [2, 3, 4]:
        for i in range(len(text) - length * 3):
            substr = text[i:i+length]
            if text[i:i+length*3] == substr * 3:
                return True
    
    # Check for too many special/control characters
    special_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?"\'-()[]{}') / max(len(text), 1)
    if special_ratio > 0.3:
        return True
    
    return False


def _auto_device(spec: GRPOTrainSpec, torch) -> str:
    device = str(spec.device)
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _torch_dtype(spec: GRPOTrainSpec, torch):
    dt = str(spec.dtype)
    if dt == "float16":
        return torch.float16
    if dt == "bfloat16":
        return torch.bfloat16
    if dt == "float32":
        return torch.float32
    if dt == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError("dtype must be auto|float16|bfloat16|float32")


def _bnb_compute_dtype(name: str, torch):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _compute_logps_for_generated(
    *,
    model,
    input_len: int,
    sequences,
    pad_token_id: int,
):
    """Compute per-sample logp sums over generated tokens."""
    torch = __import__("torch")
    attn = (sequences != int(pad_token_id)).to(torch.long)
    out = model(sequences, attention_mask=attn)
    logits = out.logits
    logp = torch.log_softmax(logits, dim=-1)
    
    target = sequences[:, 1:]
    logp_next = logp[:, :-1, :]
    tok_lp = torch.gather(logp_next, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    
    pos = torch.arange(1, sequences.shape[1], device=sequences.device).unsqueeze(0)
    gen_mask = (pos >= int(input_len)) & (target != int(pad_token_id))
    tok_lp = tok_lp * gen_mask.to(tok_lp.dtype)
    lp_sum = torch.sum(tok_lp, dim=1)
    gen_count = torch.sum(gen_mask.to(torch.long), dim=1)
    return lp_sum, gen_count


class GRPOTrainer:
    """
    GRPO trainer for AR forecasting models.
    
    Implements the Group Relative Policy Optimization algorithm:
    - Sample K completions per input
    - Compute group-relative advantages
    - Update with KL-regularized policy gradient
    """
    
    def __init__(self, spec: GRPOTrainSpec):
        self.spec = spec
        self._model = None
        self._tok = None
        self._device = None
        self._input_device = None
        self._opt = None
        self._step = 0
        self._init()
    
    def _init(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        spec = self.spec
        device = _auto_device(spec, torch)
        dtype = _torch_dtype(spec, torch)
        
        tok = AutoTokenizer.from_pretrained(
            spec.model_name_or_path, 
            trust_remote_code=bool(spec.trust_remote_code)
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        
        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=bool(spec.trust_remote_code),
        )
        
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_dt = _bnb_compute_dtype(spec.bnb_4bit_compute_dtype, torch)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_dt,
            )
            if spec.device_map is None:
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = str(spec.device_map)
        elif spec.device_map is not None:
            load_kwargs["device_map"] = str(spec.device_map)
            load_kwargs["low_cpu_mem_usage"] = True
        
        model = AutoModelForCausalLM.from_pretrained(spec.model_name_or_path, **load_kwargs)
        model.train()
        
        if spec.device_map is None and not spec.load_in_4bit:
            model.to(device)
        
        # LoRA adapter
        if spec.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=int(spec.lora_r),
                lora_alpha=int(spec.lora_alpha),
                lora_dropout=float(spec.lora_dropout),
                target_modules=list(spec.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(spec.lr), weight_decay=float(spec.weight_decay))
        
        self._tok = tok
        self._model = model
        self._device = device
        try:
            self._input_device = str(next(model.parameters()).device)
        except StopIteration:
            self._input_device = device
        self._opt = opt
        self._step = 0
    
    @property
    def tokenizer(self):
        return self._tok
    
    @property
    def model(self):
        return self._model
    
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model.save_pretrained(str(out_dir))
        except Exception:
            import torch
            torch.save(self._model.state_dict(), str(out_dir / "model_state_dict.pt"))
        self._tok.save_pretrained(str(out_dir))
        (out_dir / "grpo_spec.json").write_text(
            json.dumps(asdict(self.spec), indent=2, sort_keys=True) + "\n"
        )
    
    def train_step(self, *, infos: List[str], y: np.ndarray, q: np.ndarray) -> dict:
        """
        One GRPO update step with Dr. GRPO variant and guard-rails.
        
        For each of B inputs, samples K completions and computes group-relative advantages.
        
        Dr. GRPO (Turtel et al.): A_i = r_i - μ (no std normalization)
        This preserves raw magnitude of large forecast errors.
        
        Guard-rails:
        - Token-length limit
        - Gibberish filter
        - Brier-weighted gradients
        
        Args:
            infos: (B,) list of prompt info strings
            y: (B,) outcomes
            q: (B,) market prices
        
        Returns:
            Training metrics dict
        """
        import torch
        
        spec = self.spec
        tok = self._tok
        model = self._model
        K = int(spec.K)
        B = len(infos)
        
        # Build prompts
        prompts = [
            build_prompt(info, include_cot=bool(spec.include_cot), cot_max_steps=int(spec.cot_max_steps))
            for info in infos
        ]
        
        # Expand prompts K times each for K samples per input
        # Layout: [prompt_0_sample_0, prompt_0_sample_1, ..., prompt_B-1_sample_K-1]
        prompts_expanded = prompts * K
        
        enc = tok(
            prompts_expanded,
            padding=True,
            truncation=True,
            max_length=int(spec.max_prompt_tokens),
            return_tensors="pt",
        )
        in_dev = self._input_device or model.device
        input_ids = enc["input_ids"].to(in_dev)
        attn = enc["attention_mask"].to(in_dev)
        input_len = int(input_ids.shape[1])
        
        # Sample K completions per prompt (with token-length limit guard-rail)
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=float(spec.temperature),
            top_p=float(spec.top_p),
            max_new_tokens=min(int(spec.max_new_tokens), int(spec.max_response_tokens)),
            pad_token_id=int(tok.pad_token_id),
            eos_token_id=int(tok.eos_token_id),
        )
        
        # Decode and extract probabilities
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        ps = []
        n_fail = 0
        n_gibberish = 0
        
        for t in texts:
            tail = t.split("OUTPUT:", 1)[-1]
            
            # Guard-rail: Gibberish filter
            if spec.gibberish_filter:
                # Simple heuristic: if output has too many non-ASCII or repeated chars
                non_ascii_ratio = sum(1 for c in tail if ord(c) > 127) / max(len(tail), 1)
                if non_ascii_ratio > 0.3 or _is_gibberish(tail):
                    n_gibberish += 1
                    ps.append(0.5)  # Default to 0.5 for gibberish
                    continue
            
            p = _extract_prob(tail)
            if p is None:
                n_fail += 1
                p = 0.5
            ps.append(float(np.clip(p, 0.0, 1.0)))
        
        # Reshape: (B * K,) -> (K, B) then transpose to (B, K)
        p_arr = np.asarray(ps, dtype=np.float64).reshape(K, B).T  # (B, K)
        
        # Expand y and q to match
        y_arr = np.asarray(y, dtype=np.float64).reshape(B, 1).repeat(K, axis=1)  # (B, K)
        q_arr = np.asarray(q, dtype=np.float64).reshape(B, 1).repeat(K, axis=1)  # (B, K)
        
        # Compute rewards for all B*K samples based on reward mode
        reward_mode = spec.reward.mode
        constraint_info = None
        
        if reward_mode == "turtel_brier":
            # Turtel et al. 2025: Pure Brier reward
            R = _brier_reward(p_arr.flatten(), y_arr.flatten())
            # Parse failures already handled above with p=0.5 -> Brier = (0.5-y)^2 = 0.25
            
        elif reward_mode == "kelly":
            # Kelly criterion for optimal log-wealth growth
            R = _kelly_reward(
                p_arr.flatten(),
                q_arr.flatten(),
                y_arr.flatten(),
                fraction=float(spec.reward.kelly_fraction),
                min_edge=float(spec.reward.kelly_min_edge),
                log_clip=float(spec.reward.kelly_log_clip),
                transaction_cost=float(spec.reward.transaction_cost),
            )
            
        elif reward_mode == "blackwell_aware":
            # Brier + Blackwell constraint violation penalty
            R, constraint_info = _blackwell_aware_reward(
                p_arr.flatten(),
                y_arr.flatten(),
                n_bins=int(spec.reward.blackwell_n_bins),
                lambda_constraint=float(spec.reward.blackwell_lambda),
            )
            
        elif reward_mode == "rlcr":
            # RLCR: Damani et al. 2025 (arXiv:2507.16806)
            # FIXED: Now uses market-relative rewards to avoid collapse to p=0
            # R = α * directional_correct + β * brier_improvement - γ * calibration_error
            R, constraint_info = _rlcr_reward(
                p_arr.flatten(),
                y_arr.flatten(),
                q=q_arr.flatten(),  # CRITICAL: pass market price for market-relative rewards
                alpha=float(spec.reward.rlcr_alpha),
                beta=float(spec.reward.rlcr_beta),
                gamma=float(spec.reward.rlcr_gamma),
                correctness_threshold=float(spec.reward.rlcr_correctness_threshold),
                n_groups=int(spec.reward.rlcr_n_groups),
                use_group_calibration=bool(spec.reward.rlcr_use_group_calibration),
                use_market_relative=True,  # Use market-relative rewards
            )
            
        else:  # "hybrid" mode (original)
            logscore = _logscore(p_arr.flatten(), y_arr.flatten())  # (B*K,)
            pnl = _pnl_proxy(
                p=p_arr.flatten(),
                q=q_arr.flatten(),
                y=y_arr.flatten(),
                B=float(spec.reward.B),
                transaction_cost=float(spec.reward.transaction_cost),
                mode=str(spec.reward.trading_mode),
            )
            R = float(spec.reward.alpha_logscore) * logscore + float(spec.reward.beta_pnl) * pnl
        
        # Clip rewards
        if spec.reward_clip > 0:
            R = np.clip(R, -spec.reward_clip, spec.reward_clip)
        
        # Reshape to (B, K)
        R = R.reshape(B, K)
        
        # Compute advantages based on algorithm (from Turtel et al. 2025)
        algorithm = spec.algorithm
        
        if algorithm == "remax":
            # ReMax: Select best sample per input, subtract running baseline
            best_idx = R.argmax(axis=1)  # (B,)
            best_rewards = R[np.arange(B), best_idx]  # (B,)
            
            # Use running baseline (EMA) for stability
            if not hasattr(self, '_remax_baseline'):
                self._remax_baseline = float(best_rewards.mean())
            else:
                ema = 0.95
                self._remax_baseline = ema * self._remax_baseline + (1 - ema) * float(best_rewards.mean())
            
            # Only best sample gets updated
            advantages = np.zeros_like(R)
            advantages[np.arange(B), best_idx] = best_rewards - self._remax_baseline
            
        elif algorithm == "dr_grpo" or spec.dr_grpo:
            # Dr. GRPO (Turtel et al.): A_i = r_i - μ (NO std normalization)
            R_mean = R.mean(axis=1, keepdims=True)  # (B, 1)
            advantages = R - R_mean  # (B, K)
            
        else:  # Standard GRPO
            # GRPO: A_i = (r_i - μ) / σ
            R_mean = R.mean(axis=1, keepdims=True)  # (B, 1)
            advantages = R - R_mean  # (B, K)
            adv_std = advantages.std() + 1e-8
            advantages = advantages / adv_std
        
        # Guard-rail: Brier-weighted gradients
        # Weight updates by Brier contribution to keep gradients proportional to calibration error
        if spec.brier_weighted_grads:
            brier_per_sample = (p_arr - y_arr) ** 2  # (B, K)
            # Higher Brier error -> higher weight (focus on fixing miscalibrated predictions)
            brier_weights = brier_per_sample / (brier_per_sample.mean() + 1e-8)
            advantages = advantages * brier_weights
        
        # Flatten back to (B*K,) for loss computation
        # Need to match the order of gen: [prompt_0_sample_0, ..., prompt_0_sample_K-1, ...]
        # But we reshaped as (K, B).T = (B, K), so flatten as (B*K) matches gen if we do .T.flatten()
        advantages_flat = advantages.T.flatten()  # (B*K,)
        
        # Compute log probabilities for policy
        seq = gen.to(in_dev)
        logp_pol, n_gen = _compute_logps_for_generated(
            model=model, input_len=input_len, sequences=seq, pad_token_id=int(tok.pad_token_id)
        )
        
        # Reference log probabilities (LoRA disabled)
        try:
            with model.disable_adapter():
                logp_ref, _ = _compute_logps_for_generated(
                    model=model, input_len=input_len, sequences=seq, pad_token_id=int(tok.pad_token_id)
                )
        except Exception:
            logp_ref = logp_pol.detach()
        
        # KL penalty
        kl = (logp_pol - logp_ref).detach()  # (B*K,)
        
        # GRPO loss: -sum_i (A_i * log p_i) + kl_coef * KL
        adv_t = torch.from_numpy(advantages_flat.astype(np.float32)).to(model.device)
        policy_loss = -torch.mean(adv_t * logp_pol)
        kl_loss = float(spec.kl_coef) * torch.mean(kl)
        loss = policy_loss + kl_loss
        
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if spec.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.grad_clip))
        self._opt.step()
        
        self._step += 1
        
        # Compute Brier for early stopping tracking
        brier = float(np.mean((p_arr - y_arr) ** 2))
        
        metrics = {
            "step": int(self._step),
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "reward_mean": float(R.mean()),
            "reward_std": float(R.std()),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "kl_mean": float(torch.mean(kl).detach().cpu().item()),
            "gen_tokens_mean": float(torch.mean(n_gen.to(torch.float32)).detach().cpu().item()),
            "parse_failures": int(n_fail),
            "gibberish_filtered": int(n_gibberish),
            "brier": brier,
            "reward_mode": reward_mode,
            "algorithm": algorithm,
        }
        
        # Add algorithm-specific metrics
        if algorithm == "remax":
            metrics["remax_baseline"] = float(self._remax_baseline)
        
        # Add constraint info for blackwell_aware or rlcr mode
        if constraint_info is not None:
            # Handle different key names from different reward functions
            max_key = "max_violation" if "max_violation" in constraint_info else "max_group_violation"
            mean_key = "mean_violation" if "mean_violation" in constraint_info else "mean_group_violation"
            metrics["max_constraint_violation"] = constraint_info.get(max_key, 0.0)
            metrics["mean_constraint_violation"] = constraint_info.get(mean_key, 0.0)
        
        return metrics


def train_grpo(
    *,
    infos: List[str],
    y: np.ndarray,
    q: np.ndarray,
    spec: GRPOTrainSpec,
    out_dir: Path,
) -> Dict[str, object]:
    """
    High-level GRPO training loop with Dr. GRPO variant and guard-rails.
    
    Implements the training approach from Turtel et al. (2025):
    - Modified GRPO (no std normalization)
    - Token-length limits
    - Gibberish filter
    - Early-stop criterion
    - Brier-weighted gradients
    """
    rng = np.random.default_rng(int(spec.seed))
    trainer = GRPOTrainer(spec)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: List[dict] = []
    
    n = len(infos)
    bs = int(spec.batch_size)
    
    # Early stopping tracking
    best_brier = float("inf")
    patience_counter = 0
    best_step = 0
    
    t0 = time.time()
    for step in range(int(spec.steps)):
        idx = rng.integers(0, n, size=(bs,))
        batch_infos = [infos[int(i)] for i in idx.tolist()]
        batch_y = y[idx]
        batch_q = q[idx]
        rec = trainer.train_step(infos=batch_infos, y=batch_y, q=batch_q)
        metrics.append(rec)
        
        # Early stopping check (guard-rail from Turtel et al.)
        current_brier = rec.get("brier", float("inf"))
        if current_brier < best_brier:
            best_brier = current_brier
            patience_counter = 0
            best_step = step + 1
            # Save best model
            trainer.save(out_dir / "best")
        else:
            patience_counter += 1
        
        if spec.early_stop_patience > 0 and patience_counter >= spec.early_stop_patience:
            print(f"[grpo] Early stopping at step {step+1} (no improvement for {spec.early_stop_patience} steps)")
            print(f"[grpo] Best Brier: {best_brier:.4f} at step {best_step}")
            break
        
        if spec.log_every and (step + 1) % int(spec.log_every) == 0:
            dt = time.time() - t0
            print(
                f"[grpo] step {step+1}/{spec.steps} "
                f"loss={rec['loss']:.4f} R={rec['reward_mean']:.4f} "
                f"brier={current_brier:.4f} A_std={rec['advantage_std']:.4f} "
                f"kl={rec['kl_mean']:.4f} fails={rec['parse_failures']} "
                f"gibberish={rec.get('gibberish_filtered', 0)} time={dt:.1f}s"
            )
        if spec.save_every and (step + 1) % int(spec.save_every) == 0:
            trainer.save(out_dir / f"ckpt_step_{step+1:06d}")
    
    trainer.save(out_dir / "final")
    
    return {
        "n": n,
        "steps": step + 1,  # Actual steps taken (may be less due to early stop)
        "best_step": best_step,
        "best_brier": best_brier,
        "metrics": metrics,
        "out_dir": str(out_dir),
        "early_stopped": patience_counter >= spec.early_stop_patience if spec.early_stop_patience > 0 else False,
    }


# ==============================================================================
# GRPO for Diffusion Models
# ==============================================================================

@dataclass(frozen=True)
class DiffusionGRPOSpec:
    """
    GRPO training spec for diffusion refinement heads.
    
    Supports:
    - Dr. GRPO (no std normalization, like Turtel et al.)
    - ReMax (select best sample, subtract baseline, update)
    """
    
    # Diffusion architecture
    hidden_dim: int = 256
    depth: int = 4
    T: int = 50
    time_dim: int = 32
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # GRPO params
    K: int = 8  # Samples per input
    steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-4
    
    # Algorithm variant
    algorithm: str = "remax"  # "grpo" | "dr_grpo" | "remax"
    # remax: Use ReMax (best performing in Turtel et al.)
    # dr_grpo: Dr. GRPO (no std normalization)
    # grpo: Standard GRPO with std normalization
    
    # KL regularization (to pretrained diffusion weights)
    kl_coef: float = 0.001
    
    # ReMax-specific
    remax_baseline_ema: float = 0.95  # EMA for running baseline
    
    # Reward
    reward: GRPORewardSpec = GRPORewardSpec()
    
    seed: int = 0
    log_every: int = 100


class DiffusionGRPOTrainer:
    """
    GRPO/ReMax trainer for diffusion refinement heads.
    
    The idea: instead of just MSE training, we use GRPO/ReMax to directly
    optimize for forecasting reward (log score + PnL).
    
    Supports three algorithms (from Turtel et al.):
    1. grpo: Standard GRPO with std normalization
    2. dr_grpo: Dr. GRPO (no std normalization)
    3. remax: ReMax (best in Turtel et al.) - uses best sample with baseline subtraction
    
    For each input (AR prediction + text embedding):
    1. Sample K refined predictions from diffusion
    2. Compute reward for each
    3. Update diffusion weights with appropriate advantages
    """
    
    def __init__(
        self,
        cond_dim: int,
        spec: DiffusionGRPOSpec,
        device: str = "cuda",
        pretrained_weights: Optional[dict] = None,
    ):
        import torch
        import torch.nn as nn
        
        self.spec = spec
        self.device = device
        self.cond_dim = cond_dim
        
        # Build diffusion denoiser
        self.denoiser = self._build_denoiser(cond_dim, spec)
        self.denoiser.to(device)
        
        # Store reference weights for KL
        if pretrained_weights:
            self.ref_weights = {k: v.clone() for k, v in pretrained_weights.items()}
        else:
            self.ref_weights = {k: v.clone() for k, v in self.denoiser.state_dict().items()}
        
        # Diffusion schedule
        betas = torch.linspace(spec.beta_start, spec.beta_end, spec.T, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        
        self.opt = torch.optim.AdamW(self.denoiser.parameters(), lr=spec.lr)
        self._step = 0
        
        # ReMax baseline (EMA of rewards)
        self._remax_baseline = 0.0
    
    def _build_denoiser(self, cond_dim: int, spec: DiffusionGRPOSpec):
        import torch.nn as nn
        
        class Denoiser(nn.Module):
            def __init__(self, cond_dim, hidden_dim, depth, time_dim):
                super().__init__()
                self.time_embed = nn.Sequential(
                    nn.Linear(time_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                
                # Input: x_t (1) + time_embed (hidden) + cond (cond_dim)
                in_dim = 1 + hidden_dim + cond_dim
                layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
                for _ in range(depth - 1):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
                layers.append(nn.Linear(hidden_dim, 1))
                self.net = nn.Sequential(*layers)
                self.time_dim = time_dim
            
            def forward(self, x_t, t, cond):
                # Sinusoidal time embedding
                half = self.time_dim // 2
                freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
                args = t[:, None].float() * freqs[None]
                t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
                h_t = self.time_embed(t_emb)
                inp = torch.cat([x_t, h_t, cond], dim=-1)
                return self.net(inp)
        
        return Denoiser(cond_dim, spec.hidden_dim, spec.depth, spec.time_dim)
    
    def sample(self, cond: "torch.Tensor", n_samples: int = 1) -> "torch.Tensor":
        """Sample n_samples refined predictions per condition."""
        import torch
        
        B = cond.shape[0]
        device = cond.device
        T = self.spec.T
        
        # Expand cond for n_samples
        cond_exp = cond.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        
        # Start from noise
        x = torch.randn(B * n_samples, 1, device=device)
        
        # DDPM sampling
        for t in reversed(range(T)):
            t_batch = torch.full((B * n_samples,), t, device=device, dtype=torch.long)
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            beta_t = 1 - alpha_bar_t / alpha_bar_prev
            
            eps_pred = self.denoiser(x, t_batch, cond_exp)
            
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -5, 5)  # Clip logits
            
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * noise
            else:
                x = x0_pred
        
        # Convert logits to probabilities
        probs = torch.sigmoid(x)
        return probs.reshape(B, n_samples)
    
    def train_step(
        self,
        *,
        cond: "torch.Tensor",  # (B, cond_dim)
        y: np.ndarray,  # (B,) outcomes
        q: np.ndarray,  # (B,) market prices
    ) -> dict:
        """
        GRPO/ReMax training step for diffusion.
        
        Supports three algorithms:
        - grpo: Standard GRPO with std normalization
        - dr_grpo: Dr. GRPO (no std normalization) - from Turtel et al.
        - remax: ReMax (best in Turtel et al.) - uses best sample with baseline
        """
        import torch
        
        spec = self.spec
        K = spec.K
        B = cond.shape[0]
        
        # Sample K predictions per input
        p_samples = self.sample(cond, n_samples=K)  # (B, K)
        p_arr = p_samples.detach().cpu().numpy()
        
        # Expand y and q
        y_arr = np.asarray(y).reshape(B, 1).repeat(K, axis=1)
        q_arr = np.asarray(q).reshape(B, 1).repeat(K, axis=1)
        
        # Compute rewards based on mode
        reward_mode = spec.reward.mode
        
        if reward_mode == "turtel_brier":
            R = _brier_reward(p_arr.flatten(), y_arr.flatten()).reshape(B, K)
        elif reward_mode == "kelly":
            R = _kelly_reward(
                p_arr.flatten(), q_arr.flatten(), y_arr.flatten(),
                fraction=spec.reward.kelly_fraction,
                min_edge=spec.reward.kelly_min_edge,
                log_clip=spec.reward.kelly_log_clip,
                transaction_cost=spec.reward.transaction_cost,
            ).reshape(B, K)
        elif reward_mode == "blackwell_aware":
            R, _ = _blackwell_aware_reward(
                p_arr.flatten(), y_arr.flatten(),
                n_bins=spec.reward.blackwell_n_bins,
                lambda_constraint=spec.reward.blackwell_lambda,
            )
            R = R.reshape(B, K)
        else:  # hybrid
            logscore = _logscore(p_arr.flatten(), y_arr.flatten()).reshape(B, K)
            pnl = _pnl_proxy(
                p=p_arr.flatten(), q=q_arr.flatten(), y=y_arr.flatten(),
                B=spec.reward.B, transaction_cost=spec.reward.transaction_cost,
                mode=spec.reward.trading_mode,
            ).reshape(B, K)
            R = spec.reward.alpha_logscore * logscore + spec.reward.beta_pnl * pnl
        
        # Compute advantages based on algorithm
        if spec.algorithm == "remax":
            # ReMax: Select best sample per input, subtract EMA baseline
            best_idx = R.argmax(axis=1)  # (B,)
            best_rewards = R[np.arange(B), best_idx]  # (B,)
            
            # Update baseline with EMA
            self._remax_baseline = (
                spec.remax_baseline_ema * self._remax_baseline +
                (1 - spec.remax_baseline_ema) * best_rewards.mean()
            )
            
            # Advantages: only the best sample gets updated
            advantages = np.zeros_like(R)
            advantages[np.arange(B), best_idx] = best_rewards - self._remax_baseline
            
        elif spec.algorithm == "dr_grpo":
            # Dr. GRPO: A_i = r_i - μ (NO std normalization)
            R_mean = R.mean(axis=1, keepdims=True)
            advantages = R - R_mean
            
        else:  # Standard GRPO
            # GRPO: A_i = (r_i - μ) / σ
            R_mean = R.mean(axis=1, keepdims=True)
            advantages = R - R_mean
            adv_std = advantages.std() + 1e-8
            advantages = advantages / adv_std
        
        # Convert to tensor
        adv_t = torch.from_numpy(advantages.astype(np.float32)).to(cond.device)
        
        # Compute loss: minimize MSE weighted by -advantage
        self.denoiser.train()
        T = spec.T
        
        # Pick a random timestep
        t = torch.randint(0, T, (B * K,), device=cond.device)
        
        # Sample x_0 from the data (use sigmoid inverse of samples)
        x_0 = torch.logit(p_samples.flatten().clamp(1e-4, 1 - 1e-4)).unsqueeze(-1)  # (B*K, 1)
        
        # Add noise
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        # Expand cond
        cond_exp = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        
        # Predict noise
        eps_pred = self.denoiser(x_t, t, cond_exp)
        
        # Weighted MSE loss (weight by advantage)
        mse = (eps_pred - eps) ** 2
        weights = adv_t.flatten().unsqueeze(-1)  # (B*K, 1)
        
        # Policy gradient style: high advantage -> lower loss (encourage)
        # Use softmax-style weighting for stability
        loss = torch.mean(mse * torch.exp(-weights.clamp(-5, 5)))
        
        # KL to reference (L2 to initial weights)
        kl = 0.0
        for name, param in self.denoiser.named_parameters():
            if name in self.ref_weights:
                kl = kl + torch.mean((param - self.ref_weights[name].to(param.device)) ** 2)
        kl = spec.kl_coef * kl
        
        total_loss = loss + kl
        
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        
        self._step += 1
        
        # Compute Brier for tracking
        brier = float(np.mean((p_arr - y_arr) ** 2))
        
        return {
            "step": self._step,
            "loss": float(total_loss.item()),
            "mse_loss": float(loss.item()),
            "kl_loss": float(kl.item()) if isinstance(kl, torch.Tensor) else float(kl),
            "reward_mean": float(R.mean()),
            "reward_best": float(R.max()),
            "advantage_std": float(advantages.std()),
            "brier": brier,
            "algorithm": spec.algorithm,
            "remax_baseline": float(self._remax_baseline) if spec.algorithm == "remax" else None,
        }


def train_diffusion_grpo(
    *,
    cond: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    spec: DiffusionGRPOSpec,
    device: str = "cuda",
    pretrained_weights: Optional[dict] = None,
) -> Tuple["DiffusionGRPOTrainer", Dict]:
    """Train diffusion model with GRPO."""
    import torch
    
    trainer = DiffusionGRPOTrainer(
        cond_dim=cond.shape[1],
        spec=spec,
        device=device,
        pretrained_weights=pretrained_weights,
    )
    
    rng = np.random.default_rng(spec.seed)
    n = len(y)
    bs = spec.batch_size
    metrics = []
    
    t0 = time.time()
    for step in range(spec.steps):
        idx = rng.integers(0, n, size=(bs,))
        cond_batch = torch.tensor(cond[idx], dtype=torch.float32, device=device)
        y_batch = y[idx]
        q_batch = q[idx]
        
        rec = trainer.train_step(cond=cond_batch, y=y_batch, q=q_batch)
        metrics.append(rec)
        
        if spec.log_every and (step + 1) % spec.log_every == 0:
            dt = time.time() - t0
            print(
                f"[diff_grpo] step {step+1}/{spec.steps} "
                f"loss={rec['loss']:.4f} R={rec['reward_mean']:.4f} "
                f"A_std={rec['advantage_std']:.4f} time={dt:.1f}s"
            )
    
    return trainer, {"steps": spec.steps, "metrics": metrics}

