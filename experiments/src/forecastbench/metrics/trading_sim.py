from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Statistical Arbitrage via Calibration
# =============================================================================
# 
# KEY INSIGHT: Calibration errors are predictable mispricing.
# If E[Y | bin(q)] ≠ q, we have a systematic edge that can be exploited.
#
# STRATEGY IMPROVEMENTS OVER FLAT BETTING:
# 1. Size by d(q, C_t) — bet more when calibration error is large
# 2. Portfolio aggregation — exploit correlations across events  
# 3. Early exit — exit when market corrects, don't wait for resolution
# 4. Fractional Kelly with distance weighting


@dataclass(frozen=True)
class KellySimConfig:
    """
    Simple bankroll-based Kelly simulator for binary markets.

    We interpret `q` as the traded price (cost to buy a YES share),
    with payout 1 if YES else 0.

    At each step, we choose a stake fraction f of current bankroll on either YES or NO:
      - if p > q: bet on YES with f_yes = scale * (p-q)/(1-q)
      - if p < q: bet on NO  with f_no  = scale * (q-p)/q

    Then cap |f| <= frac_cap and update bankroll using standard Kelly odds.

    This is a stylized simulator intended for *relative* comparison, not a perfect market microstructure model.
    """

    initial_bankroll: float = 1.0
    scale: float = 1.0
    frac_cap: float = 0.25
    fee: float = 0.0  # proportional fee on stake (approximate)
    eps: float = 1e-9


def simulate_kelly_roi(
    *,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    cfg: KellySimConfig = KellySimConfig(),
    return_curve: bool = False,
) -> Dict[str, object]:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if p.shape != q.shape or p.shape != y.shape:
        raise ValueError("p,q,y must have the same shape")
    if p.size == 0:
        return {"n": 0, "roi": 0.0, "final_bankroll": float(cfg.initial_bankroll), "curve": [] if return_curve else None}

    br = float(cfg.initial_bankroll)
    curve = [br]
    eps = float(cfg.eps)
    cap = float(cfg.frac_cap)
    scale = float(cfg.scale)
    fee = float(cfg.fee)

    for pi, qi, yi in zip(p.tolist(), q.tolist(), y.tolist()):
        pi = float(np.clip(pi, eps, 1.0 - eps))
        qi = float(np.clip(qi, eps, 1.0 - eps))
        yi = float(yi)
        if yi not in (0.0, 1.0):
            yi = float(np.clip(yi, 0.0, 1.0))

        if pi > qi:
            # bet YES
            f = scale * (pi - qi) / max(1.0 - qi, eps)
            f = float(np.clip(f, 0.0, cap))
            if f > 0:
                stake = f * br
                br = br - fee * stake
                odds = (1.0 - qi) / qi  # net odds on stake if YES happens
                br = br * (1.0 - f) + br * f * (1.0 + odds) if yi >= 0.5 else br * (1.0 - f)
        elif pi < qi:
            # bet NO (equivalently buy NO at price 1-q)
            f = scale * (qi - pi) / max(qi, eps)
            f = float(np.clip(f, 0.0, cap))
            if f > 0:
                stake = f * br
                br = br - fee * stake
                odds = qi / (1.0 - qi)  # net odds if NO happens
                br = br * (1.0 - f) + br * f * (1.0 + odds) if yi < 0.5 else br * (1.0 - f)
        # else: no bet
        curve.append(float(br))

    roi = (br - float(cfg.initial_bankroll)) / max(float(cfg.initial_bankroll), eps)
    out: Dict[str, object] = {"n": int(p.size), "roi": float(roi), "final_bankroll": float(br)}
    if return_curve:
        out["curve"] = curve
    return out


# =============================================================================
# Calibration-Aware Statistical Arbitrage
# =============================================================================

@dataclass(frozen=True)
class CalibrationArbConfig:
    """
    Configuration for calibration-aware statistical arbitrage.
    
    This exploits the insight that calibration errors = predictable mispricing.
    We use the Blackwell constraint set C_t to:
    1. Size positions by distance d(q, C_t) — more edge = larger position
    2. Aggregate across bins for portfolio-level diversification
    3. Exit early when market corrects toward C_t
    
    The theory: diffusion models learn C_t, so their predictions define
    the "fair value" that markets mean-revert toward.
    """
    
    initial_bankroll: float = 1.0
    
    # Kelly parameters
    kelly_scale: float = 0.25  # Fractional Kelly (1/4 Kelly is safer)
    kelly_cap: float = 0.10   # Max fraction per bet
    
    # Calibration-aware sizing
    use_distance_weighting: bool = True  # Weight by d(q, C_t)
    distance_scale: float = 2.0  # Multiplier for distance-based sizing
    min_edge: float = 0.02  # Minimum edge to enter position
    
    # Portfolio parameters
    n_bins: int = 10
    aggregate_bins: bool = True  # Aggregate positions within bins
    
    # Early exit parameters
    early_exit: bool = True
    exit_threshold: float = 0.5  # Exit when price moves this fraction toward model
    
    # Costs
    fee: float = 0.01  # 1% transaction cost
    
    eps: float = 1e-9


@dataclass
class Position:
    """Represents an open position."""
    idx: int
    direction: int  # +1 for long YES, -1 for short (long NO)
    entry_price: float
    model_price: float
    size: float
    entry_time: int


def simulate_calibration_arb(
    *,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    q_history: Optional[np.ndarray] = None,  # (n_events, n_timesteps) price history
    calibration_stats: Optional[Dict] = None,  # Pre-computed calibration for lookhead-free
    cfg: CalibrationArbConfig = CalibrationArbConfig(),
    return_details: bool = False,
) -> Dict:
    """
    Simulate calibration-aware statistical arbitrage.
    
    This strategy uses the model's predictions to define the constraint set C_t,
    and trades toward that set with position sizing based on distance.
    
    KEY IMPROVEMENTS OVER FLAT BETTING:
    1. Distance-weighted Kelly: f ∝ d(q, C_t) * Kelly_fraction
    2. Bin aggregation: Pool positions within calibration bins
    3. Early exit: Exit when market corrects (don't need resolution)
    
    IMPORTANT: For proper backtesting, pass `calibration_stats` computed from
    a training set. Otherwise, calibration is computed with lookahead bias.
    
    Args:
        p: Model predictions (N,) - defines C_t
        q: Market prices (N,) at entry
        y: Realized outcomes (N,)
        q_history: Optional (N, T) matrix of price evolution for early exit
        calibration_stats: Pre-computed {bin_idx: {"bias": float, "confidence": float}}
        cfg: Configuration
        
    Returns:
        Trading metrics including ROI, Sharpe, and per-bin breakdown
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    
    eps = float(cfg.eps)
    
    # Compute bin assignments based on MODEL predictions (not market!)
    # This is the key insight: bin by model prediction to exploit calibration
    bin_edges = np.linspace(0, 1, cfg.n_bins + 1)
    bin_idx_model = np.clip(np.digitize(p, bin_edges) - 1, 0, cfg.n_bins - 1)
    bin_idx_market = np.clip(np.digitize(q, bin_edges) - 1, 0, cfg.n_bins - 1)
    
    # Use model bins for grouping (we trust model calibration)
    bin_idx = bin_idx_model
    
    # Compute calibration statistics per bin
    if calibration_stats is not None:
        # Use pre-computed stats (no lookahead)
        bin_calibration_error = np.zeros(cfg.n_bins)
        bin_bias = np.zeros(cfg.n_bins)  # E[Y] - E[p] per bin
        for b in range(cfg.n_bins):
            if str(b) in calibration_stats:
                bin_calibration_error[b] = abs(calibration_stats[str(b)].get("bias", 0))
                bin_bias[b] = calibration_stats[str(b)].get("bias", 0)
    else:
        # Compute from data (has lookahead bias - for analysis only)
        bin_calibration_error = np.zeros(cfg.n_bins)
        bin_bias = np.zeros(cfg.n_bins)
        
        for b in range(cfg.n_bins):
            mask = bin_idx == b
            if mask.sum() > 5:
                # Calibration error = |E[Y] - E[p]| within bin
                # Positive bias = model underestimates, negative = overestimates
                bin_bias[b] = np.mean(y[mask]) - np.mean(p[mask])
                bin_calibration_error[b] = abs(bin_bias[b])
    
    # Per-event distance to C_t (use bin's calibration error)
    distance_to_Ct = np.array([bin_calibration_error[b] for b in bin_idx])
    per_event_bias = np.array([bin_bias[b] for b in bin_idx])
    
    # Initialize bankroll and tracking
    br = float(cfg.initial_bankroll)
    curve = [br]
    
    trades = []
    bin_pnls = {b: [] for b in range(cfg.n_bins)}
    
    for i in range(n):
        pi = float(np.clip(p[i], eps, 1.0 - eps))
        qi = float(np.clip(q[i], eps, 1.0 - eps))
        yi = float(y[i])
        bi = int(bin_idx[i])
        dist = float(distance_to_Ct[i])
        bias = float(per_event_bias[i])
        
        # Edge = model's expected profit, adjusted by calibration bias
        # If model is well-calibrated (bias ≈ 0), use model prediction directly
        # If model underestimates (bias > 0), adjust upward
        # This incorporates the learned C_t structure
        adjusted_p = pi + bias * 0.5  # Partial correction toward calibrated
        adjusted_p = float(np.clip(adjusted_p, eps, 1.0 - eps))
        
        edge = adjusted_p - qi  # Positive if adjusted model > market
        
        if abs(edge) < cfg.min_edge:
            continue  # Skip if edge too small
        
        # Direction based on adjusted edge
        direction = 1 if edge > 0 else -1
        
        # Kelly fraction using adjusted probabilities
        if direction > 0:
            kelly_f = cfg.kelly_scale * (adjusted_p - qi) / max(1.0 - qi, eps)
        else:
            kelly_f = cfg.kelly_scale * (qi - adjusted_p) / max(qi, eps)
        
        kelly_f = float(np.clip(kelly_f, 0.0, cfg.kelly_cap))
        
        # Distance weighting: scale position by d(q, C_t)
        if cfg.use_distance_weighting:
            # More distance = more edge = larger position
            # Cap at 2x the base Kelly fraction
            distance_mult = min(1.0 + cfg.distance_scale * dist, 2.0)
            kelly_f *= distance_mult
            kelly_f = float(np.clip(kelly_f, 0.0, cfg.kelly_cap * 2))
        
        if kelly_f < eps:
            continue
        
        # Stake
        stake = kelly_f * br
        entry_cost = stake * (1 + cfg.fee)
        
        # Check for early exit if we have price history
        exit_early = False
        exit_price = qi
        
        if cfg.early_exit and q_history is not None and i < q_history.shape[0]:
            prices = q_history[i]
            # Check if price ever moves toward our model prediction
            for t, qt in enumerate(prices):
                if np.isnan(qt):
                    continue
                # Market moved toward model?
                correction = (qt - qi) / max(abs(pi - qi), eps)
                if direction > 0:  # We're long, want price to go up toward pi
                    if correction >= cfg.exit_threshold:
                        exit_early = True
                        exit_price = qt
                        break
                else:  # We're short, want price to go down toward pi  
                    if correction <= -cfg.exit_threshold:
                        exit_early = True
                        exit_price = qt
                        break
        
        # Compute PnL
        if exit_early:
            # Exit at intermediate price
            pnl = direction * (exit_price - qi) * stake - cfg.fee * stake
        else:
            # Hold to resolution
            if direction > 0:
                # Long YES: pay qi, receive yi
                pnl = stake * (yi / qi - 1) if yi > 0.5 else -stake
            else:
                # Long NO: pay (1-qi), receive (1-yi)  
                pnl = stake * ((1 - yi) / (1 - qi) - 1) if yi < 0.5 else -stake
            pnl -= cfg.fee * stake
        
        # Update bankroll
        br = max(br + pnl, eps)  # Prevent negative bankroll
        curve.append(float(br))
        
        # Track
        trades.append({
            "idx": i,
            "direction": direction,
            "edge": abs(edge),
            "distance": dist,
            "kelly_f": kelly_f,
            "stake": stake,
            "pnl": pnl,
            "exit_early": exit_early,
            "bin": bi,
        })
        bin_pnls[bi].append(pnl)
    
    # Aggregate statistics
    total_pnl = br - cfg.initial_bankroll
    roi = total_pnl / max(cfg.initial_bankroll, eps)
    
    # Per-bin performance
    bin_stats = {}
    for b in range(cfg.n_bins):
        pnls = bin_pnls[b]
        if pnls:
            bin_stats[b] = {
                "n_trades": len(pnls),
                "total_pnl": sum(pnls),
                "mean_pnl": np.mean(pnls),
                "win_rate": np.mean([p > 0 for p in pnls]),
                "calibration_error": bin_calibration_error[b],
            }
    
    # Sharpe ratio
    if trades:
        pnl_array = np.array([t["pnl"] for t in trades])
        if np.std(pnl_array) > eps:
            sharpe = float(np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    # Early exit stats
    if trades:
        n_early_exit = sum(1 for t in trades if t["exit_early"])
        early_exit_rate = n_early_exit / len(trades)
        early_exit_pnl = sum(t["pnl"] for t in trades if t["exit_early"])
    else:
        n_early_exit = 0
        early_exit_rate = 0.0
        early_exit_pnl = 0.0
    
    out = {
        "n": n,
        "n_trades": len(trades),
        "initial_bankroll": cfg.initial_bankroll,
        "final_bankroll": br,
        "total_pnl": total_pnl,
        "roi": roi,
        "sharpe": sharpe,
        "win_rate": np.mean([t["pnl"] > 0 for t in trades]) if trades else 0.0,
        "mean_edge": np.mean([t["edge"] for t in trades]) if trades else 0.0,
        "mean_distance": np.mean([t["distance"] for t in trades]) if trades else 0.0,
        "early_exit_rate": early_exit_rate,
        "early_exit_pnl": early_exit_pnl,
        "bin_stats": bin_stats,
    }
    
    if return_details:
        out["trades"] = trades
        out["curve"] = curve
    
    return out


def simulate_portfolio_arbitrage(
    *,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: CalibrationArbConfig = CalibrationArbConfig(),
) -> Dict:
    """
    Simulate portfolio-level arbitrage using group correlations.
    
    This constructs portfolios within groups (e.g., same category, same topic)
    to exploit cross-event correlations. The portfolio approach reduces variance
    compared to individual bets.
    
    KEY INSIGHT: Within-group calibration errors are more predictable because
    they reflect systematic model biases on correlated events.
    
    Args:
        p: Model predictions (N,)
        q: Market prices (N,)
        y: Realized outcomes (N,)
        groups: Group assignments (N,) 
        cfg: Configuration
        
    Returns:
        Portfolio-level trading metrics
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    eps = float(cfg.eps)
    br = float(cfg.initial_bankroll)
    
    group_results = {}
    
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        
        if n_g < 3:
            continue
        
        p_g = p[mask]
        q_g = q[mask]
        y_g = y[mask]
        
        # Group-level calibration error (systematic bias)
        group_calib_error = np.abs(np.mean(p_g) - np.mean(q_g))
        
        # Portfolio: aggregate positions within group
        # Long-short portfolio: long where p > q, short where p < q
        positions = p_g - q_g  # Signed edge per event
        
        # Portfolio weight = sum of absolute positions (L1 norm)
        # Each position is sized by its edge
        portfolio_size = np.sum(np.abs(positions))
        
        if portfolio_size < eps:
            continue
        
        # Normalized positions (sum of |weights| = 1)
        weights = positions / portfolio_size
        
        # Kelly sizing for portfolio
        expected_return = np.sum(weights * (p_g - q_g))  # Weighted expected edge
        kelly_f = cfg.kelly_scale * abs(expected_return) / max(1.0 - abs(expected_return), eps)
        kelly_f = float(np.clip(kelly_f, 0.0, cfg.kelly_cap))
        
        # Distance weighting
        if cfg.use_distance_weighting:
            kelly_f *= min(1.0 + cfg.distance_scale * group_calib_error, 2.0)
            kelly_f = float(np.clip(kelly_f, 0.0, cfg.kelly_cap * 2))
        
        if kelly_f < eps:
            continue
        
        stake = kelly_f * br
        
        # Portfolio PnL: sum of weighted individual PnLs
        individual_pnls = positions * (y_g - q_g)  # Position * (outcome - entry)
        portfolio_pnl = np.sum(individual_pnls) * stake / portfolio_size - cfg.fee * stake
        
        br = max(br + portfolio_pnl, eps)
        
        group_results[int(g)] = {
            "n_events": int(n_g),
            "group_calib_error": float(group_calib_error),
            "portfolio_size": float(portfolio_size),
            "expected_return": float(expected_return),
            "kelly_f": float(kelly_f),
            "stake": float(stake),
            "pnl": float(portfolio_pnl),
        }
    
    total_pnl = br - cfg.initial_bankroll
    roi = total_pnl / max(cfg.initial_bankroll, eps)
    
    return {
        "n_groups": n_groups,
        "n_groups_traded": len(group_results),
        "initial_bankroll": cfg.initial_bankroll,
        "final_bankroll": br,
        "total_pnl": total_pnl,
        "roi": roi,
        "group_results": group_results,
    }


def compute_calibration_stats(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Compute calibration statistics for lookahead-free backtesting.
    
    Call this on a TRAINING set, then pass the result to simulate_calibration_arb
    when backtesting on a TEST set.
    
    This captures the systematic biases in each prediction bin:
    - If E[Y | p ∈ bin] > E[p | p ∈ bin], model underestimates in that bin
    - We can profit by betting more on YES in those bins
    
    Args:
        p: Model predictions from training set
        y: Outcomes from training set
        n_bins: Number of calibration bins
        
    Returns:
        Dict mapping bin_idx -> {bias, confidence, count}
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, n_bins - 1)
    
    stats = {}
    for b in range(n_bins):
        mask = bin_idx == b
        count = mask.sum()
        
        if count > 5:
            mean_y = np.mean(y[mask])
            mean_p = np.mean(p[mask])
            std_y = np.std(y[mask])
            
            # Bias: positive = model underestimates
            bias = mean_y - mean_p
            
            # Confidence: how reliable is this estimate?
            # Use standard error of the mean
            se = std_y / np.sqrt(count) if count > 1 else 1.0
            confidence = 1.0 / (1.0 + se)  # Higher confidence if lower SE
            
            stats[str(b)] = {
                "bias": float(bias),
                "confidence": float(confidence),
                "count": int(count),
                "mean_y": float(mean_y),
                "mean_p": float(mean_p),
                "bin_center": float((bin_edges[b] + bin_edges[b+1]) / 2),
            }
        else:
            stats[str(b)] = {
                "bias": 0.0,
                "confidence": 0.0,
                "count": int(count),
            }
    
    return stats


def run_walk_forward_backtest(
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.5,
    n_folds: int = 5,
    cfg: CalibrationArbConfig = CalibrationArbConfig(),
) -> Dict:
    """
    Walk-forward backtest with expanding training window.
    
    This is the proper way to test calibration-aware strategies:
    1. Train calibration stats on historical data
    2. Trade on out-of-sample data
    3. Expand window and repeat
    
    Avoids lookahead bias that plagued the simple calibration_arb simulation.
    
    Args:
        p: Model predictions (chronologically ordered!)
        q: Market prices
        y: Outcomes
        train_frac: Initial training fraction
        n_folds: Number of walk-forward periods
        cfg: Trading config
        
    Returns:
        Aggregate metrics across all out-of-sample periods
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    
    # Determine fold boundaries
    initial_train = int(n * train_frac)
    remaining = n - initial_train
    fold_size = remaining // n_folds
    
    all_pnl = []
    fold_results = []
    
    cumulative_bankroll = 1.0
    
    for fold in range(n_folds):
        # Training set: all data up to this fold
        train_end = initial_train + fold * fold_size
        test_start = train_end
        test_end = train_end + fold_size if fold < n_folds - 1 else n
        
        if test_end <= test_start:
            continue
        
        # Compute calibration stats on training set
        calib_stats = compute_calibration_stats(
            p[:train_end], y[:train_end], n_bins=cfg.n_bins
        )
        
        # Trade on test set using training calibration
        fold_result = simulate_calibration_arb(
            p=p[test_start:test_end],
            q=q[test_start:test_end],
            y=y[test_start:test_end],
            calibration_stats=calib_stats,
            cfg=CalibrationArbConfig(
                initial_bankroll=cumulative_bankroll,
                kelly_scale=cfg.kelly_scale,
                kelly_cap=cfg.kelly_cap,
                use_distance_weighting=cfg.use_distance_weighting,
                distance_scale=cfg.distance_scale,
                min_edge=cfg.min_edge,
                n_bins=cfg.n_bins,
                early_exit=False,  # No price history in simple version
                fee=cfg.fee,
            ),
            return_details=True,
        )
        
        cumulative_bankroll = fold_result["final_bankroll"]
        fold_results.append({
            "fold": fold,
            "train_size": train_end,
            "test_size": test_end - test_start,
            "roi": fold_result["roi"],
            "final_bankroll": fold_result["final_bankroll"],
            "n_trades": fold_result["n_trades"],
            "win_rate": fold_result["win_rate"],
        })
        
        if "trades" in fold_result:
            all_pnl.extend([t["pnl"] for t in fold_result["trades"]])
    
    # Aggregate
    total_roi = cumulative_bankroll - 1.0
    if all_pnl:
        sharpe = float(np.mean(all_pnl) / np.std(all_pnl) * np.sqrt(252)) if np.std(all_pnl) > 0 else 0.0
        win_rate = float(np.mean([p > 0 for p in all_pnl]))
    else:
        sharpe = 0.0
        win_rate = 0.0
    
    return {
        "n_folds": n_folds,
        "initial_train_frac": train_frac,
        "total_roi": total_roi,
        "final_bankroll": cumulative_bankroll,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_trades": len(all_pnl),
        "fold_results": fold_results,
    }


def compare_trading_strategies(
    *,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    q_history: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compare different trading strategies on the same data.
    
    Strategies:
    1. Flat: Simple sign-based, hold to resolution
    2. Kelly: Standard Kelly sizing, hold to resolution  
    3. Calibration-aware: Distance-weighted Kelly
    4. With early exit: Exit when market corrects
    5. Portfolio: Group-level aggregation
    
    Returns comparative metrics to show improvement from calibration awareness.
    """
    from forecastbench.benchmarks.polymarket_eval import realized_trading_pnl
    
    results = {}
    
    # 1. Flat betting
    flat_pnl = realized_trading_pnl(y=y, market_prob=q, pred_prob=p, B=1.0)
    results["flat"] = {
        "pnl_per_event": flat_pnl,
        "total_pnl": flat_pnl * len(y),
        "roi": flat_pnl,  # Approximate
    }
    
    # 2. Standard Kelly
    kelly_result = simulate_kelly_roi(
        p=p, q=q, y=y,
        cfg=KellySimConfig(scale=1.0, frac_cap=0.25),
    )
    results["kelly"] = {
        "roi": kelly_result["roi"],
        "final_bankroll": kelly_result["final_bankroll"],
    }
    
    # 3. Fractional Kelly (safer)
    frac_kelly_result = simulate_kelly_roi(
        p=p, q=q, y=y,
        cfg=KellySimConfig(scale=0.25, frac_cap=0.10),
    )
    results["frac_kelly"] = {
        "roi": frac_kelly_result["roi"],
        "final_bankroll": frac_kelly_result["final_bankroll"],
    }
    
    # 4. Calibration-aware (no early exit)
    calib_result = simulate_calibration_arb(
        p=p, q=q, y=y,
        cfg=CalibrationArbConfig(
            kelly_scale=0.25,
            kelly_cap=0.10,
            use_distance_weighting=True,
            early_exit=False,
        ),
    )
    results["calibration_aware"] = {
        "roi": calib_result["roi"],
        "final_bankroll": calib_result["final_bankroll"],
        "sharpe": calib_result["sharpe"],
        "n_trades": calib_result["n_trades"],
    }
    
    # 5. With early exit (if price history available)
    if q_history is not None:
        early_exit_result = simulate_calibration_arb(
            p=p, q=q, y=y, q_history=q_history,
            cfg=CalibrationArbConfig(
                kelly_scale=0.25,
                kelly_cap=0.10,
                use_distance_weighting=True,
                early_exit=True,
                exit_threshold=0.5,
            ),
        )
        results["early_exit"] = {
            "roi": early_exit_result["roi"],
            "final_bankroll": early_exit_result["final_bankroll"],
            "sharpe": early_exit_result["sharpe"],
            "early_exit_rate": early_exit_result["early_exit_rate"],
        }
    
    # 6. Portfolio aggregation (if groups available)
    if groups is not None:
        portfolio_result = simulate_portfolio_arbitrage(
            p=p, q=q, y=y, groups=groups,
            cfg=CalibrationArbConfig(
                kelly_scale=0.25,
                kelly_cap=0.10,
                use_distance_weighting=True,
            ),
        )
        results["portfolio"] = {
            "roi": portfolio_result["roi"],
            "final_bankroll": portfolio_result["final_bankroll"],
            "n_groups_traded": portfolio_result["n_groups_traded"],
        }
    
    # Summary
    results["summary"] = {
        "best_strategy": max(
            [(k, v.get("roi", v.get("pnl_per_event", 0))) for k, v in results.items() if k != "summary"],
            key=lambda x: x[1]
        )[0],
        "improvement_over_flat": (
            results.get("calibration_aware", {}).get("roi", 0) - 
            results["flat"]["pnl_per_event"]
        ),
    }
    
    return results


