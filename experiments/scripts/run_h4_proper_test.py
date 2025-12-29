#!/usr/bin/env python3
"""
Proper H4 Test: Distance-Profit Correlation using d(q, C_t).

This test implements the actual H4 hypothesis:
- d(q, C_t) = distance from market prices to the learned Blackwell set
- Theory: larger d(q, C_t) should predict larger arbitrage opportunity
- Uses projection direction for trade sizing (not naive "bet toward 0.5")
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_data(
    clob_path: Path,
    resolution_path: Path,
    embeddings_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Load all required data."""
    print("Loading data...")
    
    # CLOB data
    clob = pd.read_parquet(clob_path)
    print(f"  CLOB: {len(clob):,} rows, {clob['market_id'].nunique()} markets")
    
    # Resolutions
    res = pd.read_parquet(resolution_path)
    print(f"  Resolutions: {len(res):,} markets")
    
    # Embeddings
    embeds_df = pd.read_parquet(embeddings_path)
    embeddings = {
        row['market_id']: np.array(row['embedding'], dtype=np.float32)
        for _, row in embeds_df.iterrows()
    }
    print(f"  Embeddings: {len(embeddings):,} markets")
    
    return clob, res, embeddings


def project_to_convex_hull(q: np.ndarray, samples: np.ndarray) -> Dict:
    """
    Project point q onto the convex hull of samples.
    
    Uses QP formulation: find weights w ≥ 0, sum(w) = 1
    minimizing ||sum(w_i * samples[i]) - q||^2
    
    Returns dict with:
        - projected: closest point in conv(samples) to q
        - distance: ||q - projected||
        - direction: unit vector pointing from projected to q
    """
    from scipy.optimize import minimize
    
    n_samples, n_dim = samples.shape
    
    # Objective: minimize ||samples.T @ w - q||^2
    def objective(w):
        proj = samples.T @ w
        return np.sum((proj - q) ** 2)
    
    def grad(w):
        proj = samples.T @ w
        return 2 * samples @ (proj - q)
    
    # Constraints: sum(w) = 1, w >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1) for _ in range(n_samples)]
    
    # Initial guess: uniform weights
    w0 = np.ones(n_samples) / n_samples
    
    result = minimize(
        objective,
        w0,
        jac=grad,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-8, 'maxiter': 500},
    )
    
    # Compute projected point
    projected = samples.T @ result.x
    
    # Distance and direction
    diff = q - projected
    distance = np.linalg.norm(diff)
    direction = diff / (distance + 1e-12)  # Unit vector
    
    return {
        'projected': projected,
        'distance': distance,
        'direction': direction,
        'weights': result.x,
        'success': result.success,
    }


def run_h4_test(
    model,
    clob: pd.DataFrame,
    resolutions: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    n_ct_samples: int = 64,
    n_ddim_steps: int = 32,
    slippage_bps: float = 100.0,
    seed: int = 42,
) -> Dict:
    """
    Run the proper H4 test.
    
    Strategy:
    1. For each market, get its embedding and current price q
    2. Generate C_t samples from diffusion model
    3. Project q onto conv(C_t) to get d(q, C_t) and direction
    4. Trade in the direction suggested by projection
    5. Measure correlation between d(q, C_t) and realized PnL
    """
    print(f"\nRunning H4 test with {n_ct_samples} C_t samples...")
    
    # Get overlapping markets
    clob_markets = set(clob['market_id'].unique())
    res_markets = set(resolutions['market_id'].unique())
    embed_markets = set(embeddings.keys())
    
    overlap = clob_markets & res_markets & embed_markets
    print(f"  Markets with CLOB + resolution + embedding: {len(overlap)}")
    
    if len(overlap) == 0:
        return {"error": "No overlapping markets"}
    
    # Get last price for each market
    last_prices = clob.groupby('market_id').last()['mid_price'].to_dict()
    
    # Get outcomes
    outcomes_map = resolutions.set_index('market_id')['outcome'].to_dict()
    
    # Process each market
    results = []
    rng = np.random.default_rng(seed)
    
    markets_to_test = list(overlap)
    print(f"  Testing {len(markets_to_test)} markets...")
    
    for i, market_id in enumerate(markets_to_test):
        if market_id not in last_prices or market_id not in outcomes_map:
            continue
        
        price = last_prices[market_id]
        outcome = outcomes_map[market_id]
        
        if pd.isna(price) or pd.isna(outcome):
            continue
        
        # Get embedding
        embed = embeddings[market_id].reshape(1, -1)
        
        # Generate C_t samples from diffusion model
        ct_samples = model.sample_proba(
            embed, 
            n_samples=n_ct_samples, 
            n_steps=n_ddim_steps,
            seed=seed + i * 100,
        )  # (n_samples, 1) for single market
        
        # For single market, samples is (n_samples, 1)
        # We need to compute projection in 1D (trivial case)
        # Project scalar q onto convex hull of scalars = just clip to [min, max]
        q = np.array([price])
        samples_flat = ct_samples.flatten()
        
        # 1D projection is simple: closest point in [min, max] to q
        ct_min = samples_flat.min()
        ct_max = samples_flat.max()
        
        if price < ct_min:
            projected = ct_min
            distance = ct_min - price
            direction = 1.0  # Price is below C_t, buy
        elif price > ct_max:
            projected = ct_max
            distance = price - ct_max
            direction = -1.0  # Price is above C_t, sell
        else:
            projected = price  # Inside C_t
            distance = 0.0
            direction = 0.0  # No trade
        
        # Trade based on direction
        # direction > 0: go LONG (expect price to go up toward C_t)
        # direction < 0: go SHORT (expect price to go down toward C_t)
        position = np.sign(direction)
        
        # PnL calculation
        # Long position: profit = outcome - price
        # Short position: profit = price - outcome
        if position > 0:
            raw_pnl = outcome - price
        elif position < 0:
            raw_pnl = price - outcome
        else:
            raw_pnl = 0.0  # No trade if inside C_t
        
        # Apply costs
        cost = (slippage_bps / 10000) * 2 if position != 0 else 0.0
        net_pnl = raw_pnl - cost
        
        results.append({
            'market_id': market_id,
            'price': price,
            'outcome': outcome,
            'ct_min': ct_min,
            'ct_max': ct_max,
            'ct_mean': samples_flat.mean(),
            'ct_std': samples_flat.std(),
            'distance': distance,
            'direction': direction,
            'position': position,
            'raw_pnl': raw_pnl,
            'cost': cost,
            'net_pnl': net_pnl,
            'inside_ct': distance == 0,
        })
        
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(markets_to_test)} markets...")
    
    print(f"  Completed {len(results)} trades")
    return results


def analyze_h4_results(results: List[Dict]) -> Dict:
    """Analyze H4 test results."""
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("H4 TEST RESULTS")
    print("=" * 60)
    
    # Basic stats
    n_total = len(df)
    n_traded = (df['position'] != 0).sum()
    n_inside_ct = df['inside_ct'].sum()
    
    print(f"\n1. Trade Statistics:")
    print(f"   Total markets: {n_total}")
    print(f"   Traded: {n_traded} ({n_traded/n_total:.1%})")
    print(f"   Inside C_t (no trade): {n_inside_ct} ({n_inside_ct/n_total:.1%})")
    
    # Filter to traded markets for correlation analysis
    traded = df[df['position'] != 0].copy()
    
    if len(traded) < 10:
        print(f"\n   WARNING: Too few trades ({len(traded)}) for meaningful analysis")
        return {"error": "Too few trades", "n_traded": len(traded)}
    
    # PnL stats
    total_pnl = traded['net_pnl'].sum()
    gross_pnl = traded['raw_pnl'].sum()
    win_rate = (traded['net_pnl'] > 0).mean()
    
    print(f"\n2. PnL Statistics:")
    print(f"   Total PnL: {total_pnl:+.4f}")
    print(f"   Gross PnL: {gross_pnl:+.4f}")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Avg PnL/trade: {traded['net_pnl'].mean():+.4f}")
    
    # THE KEY TEST: correlation between d(q, C_t) and |PnL|
    print(f"\n3. H4 Core Test: Correlation of d(q, C_t) with PnL")
    
    # Compute correlations
    corr_dist_pnl = traded['distance'].corr(traded['net_pnl'])
    corr_dist_abs_pnl = traded['distance'].corr(traded['net_pnl'].abs())
    
    # Statistical significance
    r, p_value = stats.pearsonr(traded['distance'], traded['net_pnl'])
    
    print(f"   corr(distance, net_pnl) = {corr_dist_pnl:.4f}")
    print(f"   corr(distance, |net_pnl|) = {corr_dist_abs_pnl:.4f}")
    print(f"   p-value: {p_value:.4e}")
    
    # H4 test result
    h4_pass = corr_dist_pnl > 0.1 and p_value < 0.05
    print(f"\n   H4 Test: {'PASS ✓' if h4_pass else 'FAIL ✗'}")
    print(f"   (Criterion: corr > 0.1 and p < 0.05)")
    
    # Compare with naive baseline
    print(f"\n4. Comparison with Naive Strategy:")
    
    # Naive: bet toward 0.5
    naive_pnl = []
    for _, row in traded.iterrows():
        naive_pos = 1 if row['price'] < 0.5 else -1
        if naive_pos > 0:
            pnl = row['outcome'] - row['price']
        else:
            pnl = row['price'] - row['outcome']
        naive_pnl.append(pnl - row['cost'])
    
    traded['naive_pnl'] = naive_pnl
    
    ct_win_rate = (traded['net_pnl'] > 0).mean()
    naive_win_rate = (traded['naive_pnl'] > 0).mean()
    ct_total = traded['net_pnl'].sum()
    naive_total = traded['naive_pnl'].sum()
    
    print(f"   C_t strategy: win={ct_win_rate:.1%}, PnL={ct_total:+.4f}")
    print(f"   Naive (→0.5): win={naive_win_rate:.1%}, PnL={naive_total:+.4f}")
    print(f"   Improvement: {ct_total - naive_total:+.4f}")
    
    # Distance analysis by bucket
    print(f"\n5. Performance by Distance Bucket:")
    
    traded['dist_bucket'] = pd.cut(
        traded['distance'], 
        bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
        labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5', '0.5+']
    )
    
    for bucket in ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5', '0.5+']:
        mask = traded['dist_bucket'] == bucket
        if mask.sum() > 0:
            bucket_df = traded[mask]
            avg_pnl = bucket_df['net_pnl'].mean()
            win = (bucket_df['net_pnl'] > 0).mean()
            print(f"   {bucket}: n={mask.sum():3d}, win={win:.0%}, avg_pnl={avg_pnl:+.4f}")
    
    return {
        "n_total": int(n_total),
        "n_traded": int(n_traded),
        "n_inside_ct": int(n_inside_ct),
        "total_pnl": float(total_pnl),
        "gross_pnl": float(gross_pnl),
        "win_rate": float(win_rate),
        "h4_correlation": float(corr_dist_pnl),
        "h4_p_value": float(p_value),
        "h4_pass": bool(h4_pass),
        "ct_strategy_pnl": float(ct_total),
        "naive_strategy_pnl": float(naive_total),
        "improvement": float(ct_total - naive_total),
    }


def main():
    parser = argparse.ArgumentParser(description="Run proper H4 test")
    parser.add_argument("--ct-samples", type=int, default=64)
    parser.add_argument("--ddim-steps", type=int, default=32)
    parser.add_argument("--slippage-bps", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROPER H4 TEST: Distance-Profit Correlation")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  C_t samples: {args.ct_samples}")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Slippage: {args.slippage_bps} bps per side")
    
    # Load model
    print("\nLoading diffusion model...")
    from backtest.ct_loader_legacy import LegacyDiffusionForecaster
    
    model = LegacyDiffusionForecaster.load(
        "runs/20251225_235102_pm_suite_difftrain_fixed_ready/model.pt",
        device="cpu"
    )
    print(f"  ✓ Model loaded: cond_dim={model.spec.cond_dim}")
    
    # Load data
    clob, resolutions, embeddings = load_data(
        Path("data/backtest/clob_merged.parquet"),
        Path("data/backtest/resolutions.parquet"),
        Path("data/embeddings_cache.parquet"),
    )
    
    # Run test
    results = run_h4_test(
        model,
        clob,
        resolutions,
        embeddings,
        n_ct_samples=args.ct_samples,
        n_ddim_steps=args.ddim_steps,
        slippage_bps=args.slippage_bps,
        seed=args.seed,
    )
    
    if isinstance(results, dict) and "error" in results:
        print(f"\nError: {results['error']}")
        return
    
    # Analyze
    analysis = analyze_h4_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(f"runs/h4_test_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trades
    pd.DataFrame(results).to_csv(output_dir / "trades.csv", index=False)
    
    # Save analysis
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

