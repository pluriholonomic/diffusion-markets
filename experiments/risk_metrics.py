#!/usr/bin/env python3
"""Advanced risk metrics with Cornish-Fisher and Expected Shortfall."""

import pandas as pd
import numpy as np
from scipy import stats

def cornish_fisher_var(returns, alpha=0.05):
    """VaR with Cornish-Fisher adjustment for skewness and kurtosis."""
    mu = returns.mean()
    sigma = returns.std()
    S = stats.skew(returns)  # Skewness
    K = stats.kurtosis(returns)  # Excess kurtosis
    
    z = stats.norm.ppf(alpha)  # Standard normal quantile
    
    # Cornish-Fisher expansion
    z_cf = (z + (z**2 - 1) * S / 6 
            + (z**3 - 3*z) * K / 24 
            - (2*z**3 - 5*z) * S**2 / 36)
    
    var_cf = mu + sigma * z_cf
    var_normal = mu + sigma * z
    
    return var_cf, var_normal, S, K

def expected_shortfall(returns, alpha=0.05):
    """Expected Shortfall (CVaR) - average loss beyond VaR."""
    var = np.percentile(returns, alpha * 100)
    es = returns[returns <= var].mean()
    return es, var

def sortino_ratio(returns, target=0):
    """Sortino ratio - Sharpe but only penalizes downside volatility."""
    excess = returns - target
    downside = returns[returns < target]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-8
    return excess.mean() / downside_std

def omega_ratio(returns, threshold=0):
    """Omega ratio - probability-weighted gain/loss ratio."""
    gains = returns[returns > threshold].sum()
    losses = -returns[returns <= threshold].sum()
    return gains / losses if losses > 0 else np.inf


print("=" * 85)
print("ADVANCED RISK METRICS: Cornish-Fisher & Expected Shortfall")
print("=" * 85)

runs = [
    ("AR Baseline", "runs/20251230_153746_eval_ar_baseline_2k/predictions.parquet"),
    ("RLCR HighGamma", "runs/20251230_190124_eval_rlcr_highgamma_pnl_2k/predictions.parquet"),
    ("RLCR Broken", "runs/20251230_204837_eval_rlcr_longrun_pnl_2k/predictions.parquet"),
    ("RLCR Fixed", "runs/20251231_051122_eval_rlcr_fixed_v1/predictions.parquet"),
]

print("\n" + "-" * 85)
print(f"{'Model':<16} | {'Skew':>6} | {'Kurt':>6} | {'VaR_N':>8} | {'VaR_CF':>8} | {'ES(5%)':>8} | {'Sortino':>7} | {'Omega':>6}")
print("-" * 85)

results = []
for name, path in runs:
    try:
        df = pd.read_parquet(path)
        p = df["pred_prob"].values
        y = df["y"].values
        q = df["market_prob"].values
        
        # Calculate per-trade returns
        positions = np.sign(p - q)
        returns = positions * (y - q) - 0.02 * np.abs(positions)
        
        # Standard metrics
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / std_ret
        
        # Cornish-Fisher VaR
        var_cf, var_normal, skew, kurt = cornish_fisher_var(returns, alpha=0.05)
        
        # Expected Shortfall
        es, var_empirical = expected_shortfall(returns, alpha=0.05)
        
        # Sortino and Omega
        sortino = sortino_ratio(returns)
        omega = omega_ratio(returns)
        
        print(f"{name:<16} | {skew:>6.2f} | {kurt:>6.2f} | {var_normal:>8.4f} | {var_cf:>8.4f} | {es:>8.4f} | {sortino:>7.3f} | {omega:>6.2f}")
        
        results.append({
            "name": name,
            "mean_ret": mean_ret,
            "std_ret": std_ret,
            "sharpe": sharpe,
            "es": es,
            "skew": skew,
            "kurt": kurt,
            "sortino": sortino,
            "omega": omega,
        })
        
    except Exception as e:
        print(f"{name}: Error - {e}")

print("-" * 85)

print("""
Metric Definitions:
- Skew     : Distribution asymmetry (negative = left tail heavier)
- Kurt     : Excess kurtosis (>0 = fat tails, normal=0)
- VaR_N    : 5% Value-at-Risk assuming normality
- VaR_CF   : 5% VaR with Cornish-Fisher adjustment for skew/kurtosis
- ES(5%)   : Expected Shortfall - mean loss when in worst 5% of outcomes
- Sortino  : Like Sharpe but only penalizes downside volatility
- Omega    : Ratio of probability-weighted gains to losses (>1 is good)
""")

# Now calculate ES-adjusted Sharpe
print("\n" + "=" * 85)
print("EXPECTED SHORTFALL SHARPE (ES-Sharpe) & MODIFIED SHARPE")
print("=" * 85)
print("""
ES-Sharpe = Mean Return / |Expected Shortfall at 5%|
  -> Measures return per unit of tail risk

Modified Sharpe (Cornish-Fisher) = Mean / Modified_StdDev
  -> Adjusts for skewness and kurtosis in the denominator
""")

trades_per_year = 20  # Based on ~18 day holding period

print(f"{'Model':<16} | {'Sharpe':>7} | {'Ann.Sharpe':>10} | {'ES-Sharpe':>9} | {'Ann.ES-Sh':>9} | {'Sortino':>7} | {'Ann.Sort':>8}")
print("-" * 85)

for r in results:
    es_sharpe = r["mean_ret"] / abs(r["es"]) if r["es"] != 0 else np.inf
    ann_sharpe = r["sharpe"] * np.sqrt(trades_per_year)
    ann_es_sharpe = es_sharpe * np.sqrt(trades_per_year)
    ann_sortino = r["sortino"] * np.sqrt(trades_per_year)
    
    print(f"{r['name']:<16} | {r['sharpe']:>7.3f} | {ann_sharpe:>10.2f} | {es_sharpe:>9.3f} | {ann_es_sharpe:>9.2f} | {r['sortino']:>7.3f} | {ann_sortino:>8.2f}")

print("-" * 85)
print("""
Key Insight:
- If ES-Sharpe >> Sharpe: Returns are concentrated, tail risk is low
- If ES-Sharpe << Sharpe: Fat tails! Normal Sharpe understates risk
- Sortino > Sharpe: More upside variance than downside (good!)
""")
