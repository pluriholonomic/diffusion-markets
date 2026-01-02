"""
Tail Risk Metrics for Prediction Market Strategies

Prediction market resolutions are like credit events/defaults:
- Binary outcomes (win/lose entire position)
- Jump risk at resolution
- Standard Sharpe underestimates this risk

This module provides:
- Expected Shortfall (ES) Sharpe
- Sortino Ratio
- VaR metrics
- Resolution-aware risk measures
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TailRiskMetrics:
    """Container for tail risk metrics."""
    # Standard metrics
    sharpe: float
    mean_return: float
    std_return: float
    
    # Tail risk metrics
    var_5pct: float  # 5th percentile (Value at Risk)
    var_1pct: float  # 1st percentile
    es_5pct: float   # Expected Shortfall at 5%
    es_1pct: float   # Expected Shortfall at 1%
    es_sharpe_5pct: float  # ES-adjusted Sharpe
    es_sharpe_1pct: float
    
    # Downside metrics
    sortino: float
    downside_deviation: float
    
    # Extreme metrics  
    max_loss: float
    max_gain: float
    calmar: float  # Return / Max Drawdown proxy
    
    # Distribution shape
    skewness: float
    kurtosis: float
    win_rate: float
    
    # Sample info
    n_observations: int


def compute_tail_risk_metrics(
    returns: np.ndarray,
    annualization_factor: float = np.sqrt(252),
    risk_free_rate: float = 0.0,
) -> TailRiskMetrics:
    """
    Compute comprehensive tail risk metrics for a return series.
    
    Args:
        returns: Array of returns (can be trade-level or daily)
        annualization_factor: sqrt(periods_per_year), default sqrt(252) for daily
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        TailRiskMetrics dataclass with all metrics
    """
    n = len(returns)
    if n < 10:
        return TailRiskMetrics(
            sharpe=0, mean_return=0, std_return=0,
            var_5pct=0, var_1pct=0, es_5pct=0, es_1pct=0,
            es_sharpe_5pct=0, es_sharpe_1pct=0,
            sortino=0, downside_deviation=0,
            max_loss=0, max_gain=0, calmar=0,
            skewness=0, kurtosis=0, win_rate=0,
            n_observations=n
        )
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    # Standard Sharpe
    sharpe = (mean_ret - risk_free_rate) / std_ret * annualization_factor if std_ret > 0 else 0
    
    # VaR (percentile of losses)
    var_5pct = np.percentile(returns, 5)
    var_1pct = np.percentile(returns, 1)
    
    # Expected Shortfall (conditional VaR)
    tail_5 = returns[returns <= var_5pct]
    tail_1 = returns[returns <= var_1pct]
    es_5pct = np.mean(tail_5) if len(tail_5) > 0 else var_5pct
    es_1pct = np.mean(tail_1) if len(tail_1) > 0 else var_1pct
    
    # ES-adjusted Sharpe (penalizes tail risk)
    es_sharpe_5pct = (mean_ret - risk_free_rate) / abs(es_5pct) * annualization_factor if es_5pct != 0 else 0
    es_sharpe_1pct = (mean_ret - risk_free_rate) / abs(es_1pct) * annualization_factor if es_1pct != 0 else 0
    
    # Sortino (only penalizes downside)
    downside = returns[returns < 0]
    downside_dev = np.std(downside) if len(downside) > 1 else std_ret
    sortino = (mean_ret - risk_free_rate) / downside_dev * annualization_factor if downside_dev > 0 else 0
    
    # Extreme values
    max_loss = np.min(returns)
    max_gain = np.max(returns)
    
    # Calmar-style (return / max loss)
    calmar = mean_ret / abs(max_loss) * 252 if max_loss < 0 else float('inf')
    
    # Distribution shape
    # Skewness: negative = left tail, positive = right tail
    skewness = ((returns - mean_ret) ** 3).mean() / (std_ret ** 3) if std_ret > 0 else 0
    # Kurtosis: >3 = fat tails, <3 = thin tails
    kurtosis = ((returns - mean_ret) ** 4).mean() / (std_ret ** 4) if std_ret > 0 else 0
    
    win_rate = np.mean(returns > 0)
    
    return TailRiskMetrics(
        sharpe=sharpe,
        mean_return=mean_ret,
        std_return=std_ret,
        var_5pct=var_5pct,
        var_1pct=var_1pct,
        es_5pct=es_5pct,
        es_1pct=es_1pct,
        es_sharpe_5pct=es_sharpe_5pct,
        es_sharpe_1pct=es_sharpe_1pct,
        sortino=sortino,
        downside_deviation=downside_dev,
        max_loss=max_loss,
        max_gain=max_gain,
        calmar=calmar,
        skewness=skewness,
        kurtosis=kurtosis,
        win_rate=win_rate,
        n_observations=n
    )


def compute_resolution_risk_metrics(
    trade_outcomes: List[Dict[str, Any]],
    annualization_factor: float = np.sqrt(252),
) -> Dict[str, float]:
    """
    Compute risk metrics specific to prediction market resolutions.
    
    Treats each resolution as a potential 'default event' where
    the position either pays off or goes to zero.
    
    Args:
        trade_outcomes: List of dicts with keys:
            - 'pnl': Trade PnL
            - 'position_size': Size of position
            - 'outcome': 0 or 1 (market resolution)
            - 'predicted_prob': Our predicted probability
            - 'market_price': Price we paid
            
    Returns:
        Dict with resolution-specific risk metrics
    """
    if not trade_outcomes:
        return {}
    
    pnls = np.array([t['pnl'] for t in trade_outcomes])
    sizes = np.array([t.get('position_size', 1) for t in trade_outcomes])
    outcomes = np.array([t.get('outcome', 0) for t in trade_outcomes])
    
    # Standard tail metrics
    base_metrics = compute_tail_risk_metrics(pnls, annualization_factor)
    
    # Resolution-specific metrics
    
    # 1. Loss given default (LGD) - average loss when wrong
    wrong_trades = pnls[pnls < 0]
    lgd = np.mean(wrong_trades) if len(wrong_trades) > 0 else 0
    
    # 2. Recovery rate - how much we keep when wrong (relative to position)
    if len(wrong_trades) > 0:
        wrong_sizes = sizes[pnls < 0]
        recovery_rate = 1 - np.mean(np.abs(wrong_trades) / wrong_sizes)
    else:
        recovery_rate = 1.0
    
    # 3. Jump risk - volatility conditional on resolution
    # This captures the 'event risk' aspect
    pnl_volatility = np.std(pnls)
    normalized_pnl = pnls / sizes if np.all(sizes > 0) else pnls
    jump_volatility = np.std(normalized_pnl)  # Per-dollar volatility
    
    # 4. Tail concentration ratio
    # What fraction of total loss comes from worst 5%?
    sorted_pnl = np.sort(pnls)
    n_tail = max(1, int(len(pnls) * 0.05))
    tail_loss = np.sum(sorted_pnl[:n_tail])
    total_loss = np.sum(pnls[pnls < 0])
    tail_concentration = tail_loss / total_loss if total_loss < 0 else 0
    
    # 5. Brier score (calibration metric)
    predicted_probs = np.array([t.get('predicted_prob', 0.5) for t in trade_outcomes])
    brier_score = np.mean((predicted_probs - outcomes) ** 2)
    
    return {
        **base_metrics.__dict__,
        'loss_given_default': lgd,
        'recovery_rate': recovery_rate,
        'jump_volatility': jump_volatility,
        'tail_concentration_5pct': tail_concentration,
        'brier_score': brier_score,
        'default_rate': 1 - base_metrics.win_rate,  # Probability of loss
    }


def compare_risk_profiles(
    strategy_results: Dict[str, List[float]],
    annualization_factor: float = np.sqrt(252),
) -> Dict[str, Dict[str, float]]:
    """
    Compare tail risk profiles across multiple strategies.
    
    Args:
        strategy_results: Dict mapping strategy name to list of returns
        
    Returns:
        Dict mapping strategy name to risk metrics
    """
    profiles = {}
    
    for name, returns in strategy_results.items():
        metrics = compute_tail_risk_metrics(np.array(returns), annualization_factor)
        
        # Compute relative risk indicators
        profiles[name] = {
            'sharpe': metrics.sharpe,
            'es_sharpe': metrics.es_sharpe_5pct,
            'sortino': metrics.sortino,
            'tail_risk_ratio': metrics.sharpe / metrics.es_sharpe_5pct if metrics.es_sharpe_5pct != 0 else float('inf'),
            'skewness': metrics.skewness,
            'kurtosis': metrics.kurtosis,
            'win_rate': metrics.win_rate,
            'var_5pct': metrics.var_5pct,
            'es_5pct': metrics.es_5pct,
        }
        
        # Flag concerning patterns
        profiles[name]['has_fat_tails'] = metrics.kurtosis > 4
        profiles[name]['has_left_skew'] = metrics.skewness < -0.5
        profiles[name]['tail_risk_elevated'] = metrics.sharpe > metrics.es_sharpe_5pct * 1.2
    
    return profiles


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    # Simulate prediction market returns (fat-tailed, slightly negative skew)
    n_trades = 200
    
    # Base returns: mixture of wins and losses
    wins = np.random.exponential(50, int(n_trades * 0.7))  # 70% win rate
    losses = -np.random.exponential(80, int(n_trades * 0.3))  # Losses are larger
    returns = np.concatenate([wins, losses])
    np.random.shuffle(returns)
    
    print("=" * 60)
    print("Tail Risk Metrics Demo")
    print("=" * 60)
    
    metrics = compute_tail_risk_metrics(returns)
    
    print(f"\nStandard Metrics:")
    print(f"  Sharpe:     {metrics.sharpe:.2f}")
    print(f"  Mean:       ${metrics.mean_return:.2f}")
    print(f"  Std:        ${metrics.std_return:.2f}")
    
    print(f"\nTail Risk Metrics:")
    print(f"  VaR 5%:     ${metrics.var_5pct:.2f}")
    print(f"  ES 5%:      ${metrics.es_5pct:.2f}")
    print(f"  ES Sharpe:  {metrics.es_sharpe_5pct:.2f}")
    
    print(f"\nDownside Metrics:")
    print(f"  Sortino:    {metrics.sortino:.2f}")
    print(f"  Max Loss:   ${metrics.max_loss:.2f}")
    print(f"  Win Rate:   {metrics.win_rate:.1%}")
    
    print(f"\nDistribution Shape:")
    print(f"  Skewness:   {metrics.skewness:.2f} {'(left tail)' if metrics.skewness < 0 else '(right tail)'}")
    print(f"  Kurtosis:   {metrics.kurtosis:.2f} {'(fat tails)' if metrics.kurtosis > 3 else '(thin tails)'}")
    
    # Risk interpretation
    print(f"\nRisk Interpretation:")
    if metrics.sharpe > metrics.es_sharpe_5pct * 1.1:
        print("  ⚠️  Sharpe > ES_Sharpe: Standard Sharpe may understate tail risk")
    else:
        print("  ✓  ES_Sharpe ≈ Sharpe: Tails appear well-behaved")
