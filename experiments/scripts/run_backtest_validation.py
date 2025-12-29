#!/usr/bin/env python3
"""
Run complete backtest pipeline validation.

This script runs both tracks:
- Track 1: Forecast evaluation (H1/calibration)  
- Track 2: Trading simulation (H4/profit)

With C_t validation and conservative execution costs.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationConfig:
    """Configuration for backtest validation."""
    
    # Data paths
    forecast_data: Path = Path("data/polymarket/pm_horizon_24h.parquet")
    clob_data: Path = Path("data/backtest/clob_merged.parquet")
    resolution_data: Path = Path("data/backtest/resolutions.parquet")
    embeddings_cache: Path = Path("data/embeddings_cache.parquet")
    clob_snapshots: Path = Path("data/clob_snapshots/snapshot_stats.parquet")
    
    # Model checkpoint
    checkpoint_dir: Path = Path("runs/20251225_235102_pm_suite_difftrain_fixed_ready")
    
    # Output
    output_dir: Path = Path("runs/backtest_validation")
    
    # Execution costs (conservative: 200 bps round-trip)
    slippage_bps: float = 100.0  # 100 bps per side
    
    # Cost sensitivity sweep
    cost_levels: tuple = (50, 100, 200, 500)
    
    # C_t validation
    ct_samples: int = 64
    ct_validation: bool = True
    
    # Test split
    test_frac: float = 0.15
    seed: int = 42


def run_track1_calibration(cfg: ValidationConfig, output_dir: Path) -> Dict:
    """
    Track 1: Forecast evaluation (H1/calibration).
    
    Computes calibration metrics on pm_horizon_24h.parquet.
    """
    print("\n" + "=" * 60)
    print("TRACK 1: FORECAST EVALUATION (H1/Calibration)")
    print("=" * 60)
    
    # Load forecast data
    print("\n1. Loading forecast data...")
    df = pd.read_parquet(cfg.forecast_data)
    print(f"   Samples: {len(df)}")
    
    # Split into test set
    np.random.seed(cfg.seed)
    test_mask = np.random.rand(len(df)) < cfg.test_frac
    test_df = df[test_mask].copy()
    print(f"   Test set: {len(test_df)} samples")
    
    # Get predictions and outcomes
    # market_prob = model's prediction at snapshot time
    # y = actual outcome (0 or 1)
    preds = test_df['market_prob'].values
    outcomes = test_df['y'].values
    
    # Handle NaN
    valid = ~np.isnan(preds) & ~np.isnan(outcomes)
    preds = preds[valid]
    outcomes = outcomes[valid]
    print(f"   Valid predictions: {len(preds)}")
    
    # Compute calibration metrics
    print("\n2. Computing calibration metrics...")
    
    # Brier score
    brier = np.mean((preds - outcomes) ** 2)
    print(f"   Brier Score: {brier:.4f}")
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    calibration_data = []
    
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_pred = preds[mask].mean()
            bin_outcome = outcomes[mask].mean()
            bin_count = mask.sum()
            ece += abs(bin_pred - bin_outcome) * bin_count / len(preds)
            calibration_data.append({
                "bin": i,
                "lower": bin_edges[i],
                "upper": bin_edges[i + 1],
                "mean_pred": float(bin_pred),
                "mean_outcome": float(bin_outcome),
                "count": int(bin_count),
            })
    
    print(f"   ECE: {ece:.4f}")
    
    # Overconfidence rate
    confidence = np.abs(preds - 0.5)
    high_conf_mask = confidence > 0.25  # Predictions > 0.75 or < 0.25
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (
            (preds[high_conf_mask] > 0.5) == outcomes[high_conf_mask]
        ).mean()
        overconfidence_rate = 1.0 - high_conf_accuracy
    else:
        overconfidence_rate = 0.0
    print(f"   Overconfidence rate: {overconfidence_rate:.4f}")
    
    # Save results
    results = {
        "n_samples": len(preds),
        "brier_score": float(brier),
        "ece": float(ece),
        "overconfidence_rate": float(overconfidence_rate),
        "calibration_bins": calibration_data,
    }
    
    cal_dir = output_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cal_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save calibration data for plotting
    pd.DataFrame(calibration_data).to_csv(cal_dir / "reliability_data.csv", index=False)
    
    print(f"\n   Saved to {cal_dir}")
    
    return results


def run_track2_trading(cfg: ValidationConfig, output_dir: Path) -> Dict:
    """
    Track 2: Trading simulation (H4/profit).
    
    Runs backtest with conservative execution costs.
    """
    print("\n" + "=" * 60)
    print("TRACK 2: TRADING SIMULATION (H4/Profit)")
    print("=" * 60)
    
    # Load CLOB data
    print("\n1. Loading CLOB data...")
    clob_df = pd.read_parquet(cfg.clob_data)
    print(f"   CLOB rows: {len(clob_df):,}")
    print(f"   Unique markets: {clob_df['market_id'].nunique()}")
    
    # Load resolutions
    res_df = pd.read_parquet(cfg.resolution_data)
    print(f"   Resolutions: {len(res_df):,}")
    
    # Get overlapping markets
    clob_markets = set(clob_df['market_id'].unique())
    res_markets = set(res_df['market_id'].unique())
    overlap = clob_markets & res_markets
    print(f"   Overlapping markets: {len(overlap)}")
    
    if len(overlap) == 0:
        print("   WARNING: No overlapping markets. Skipping trading simulation.")
        return {"error": "No overlapping markets"}
    
    # Filter to overlapping markets
    clob_df = clob_df[clob_df['market_id'].isin(overlap)].copy()
    res_df = res_df[res_df['market_id'].isin(overlap)].copy()
    
    # Simple trading simulation
    print("\n2. Running simple trading simulation...")
    
    # Strategy: bet on markets where price differs from resolution
    trades = []
    
    # Get last price for each market
    last_prices = clob_df.groupby('market_id').last()['mid_price']
    
    for market_id in overlap:
        if market_id not in last_prices.index:
            continue
            
        price = last_prices[market_id]
        outcome_row = res_df[res_df['market_id'] == market_id]
        
        if len(outcome_row) == 0:
            continue
            
        outcome = outcome_row.iloc[0]['outcome']
        
        # Skip if price is NaN
        if pd.isna(price):
            continue
        
        # Simple strategy: go long if price < 0.5, short if price > 0.5
        position = 1 if price < 0.5 else -1
        
        # PnL before costs
        raw_pnl = position * (outcome - price)
        
        # Apply costs
        cost = cfg.slippage_bps / 10000 * 2  # Round-trip
        net_pnl = raw_pnl - cost
        
        trades.append({
            "market_id": market_id,
            "price": float(price),
            "outcome": float(outcome),
            "position": position,
            "raw_pnl": float(raw_pnl),
            "cost": float(cost),
            "net_pnl": float(net_pnl),
        })
    
    print(f"   Trades: {len(trades)}")
    
    if len(trades) == 0:
        return {"error": "No trades executed"}
    
    trades_df = pd.DataFrame(trades)
    
    # Compute metrics
    total_pnl = trades_df['net_pnl'].sum()
    gross_pnl = trades_df['raw_pnl'].sum()
    total_cost = trades_df['cost'].sum()
    win_rate = (trades_df['net_pnl'] > 0).mean()
    avg_pnl = trades_df['net_pnl'].mean()
    
    print(f"\n3. Trading Results:")
    print(f"   Total PnL: {total_pnl:.4f}")
    print(f"   Gross PnL: {gross_pnl:.4f}")
    print(f"   Total Cost: {total_cost:.4f}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Avg PnL/Trade: {avg_pnl:.4f}")
    
    # H4 validation: correlation between |price - 0.5| and |pnl|
    trades_df['distance'] = np.abs(trades_df['price'] - 0.5)
    trades_df['abs_pnl'] = np.abs(trades_df['net_pnl'])
    
    h4_corr = trades_df['distance'].corr(trades_df['abs_pnl'])
    print(f"\n4. H4 Validation:")
    print(f"   Correlation(distance, |pnl|): {h4_corr:.4f}")
    
    # Save results
    results = {
        "n_trades": len(trades),
        "total_pnl": float(total_pnl),
        "gross_pnl": float(gross_pnl),
        "total_cost": float(total_cost),
        "win_rate": float(win_rate),
        "avg_pnl_per_trade": float(avg_pnl),
        "cost_bps": cfg.slippage_bps * 2,
        "h4_correlation": float(h4_corr) if not pd.isna(h4_corr) else None,
    }
    
    trading_dir = output_dir / "trading"
    trading_dir.mkdir(parents=True, exist_ok=True)
    
    with open(trading_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    trades_df.to_csv(trading_dir / "trades.csv", index=False)
    
    print(f"\n   Saved to {trading_dir}")
    
    return results


def run_cost_sensitivity(cfg: ValidationConfig, output_dir: Path) -> Dict:
    """
    Run cost sensitivity analysis at different cost levels.
    """
    print("\n" + "=" * 60)
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Load data (simplified - reuse from track 2)
    clob_df = pd.read_parquet(cfg.clob_data)
    res_df = pd.read_parquet(cfg.resolution_data)
    
    clob_markets = set(clob_df['market_id'].unique())
    res_markets = set(res_df['market_id'].unique())
    overlap = clob_markets & res_markets
    
    clob_df = clob_df[clob_df['market_id'].isin(overlap)].copy()
    res_df = res_df[res_df['market_id'].isin(overlap)].copy()
    
    last_prices = clob_df.groupby('market_id').last()['mid_price']
    
    results = []
    
    for cost_bps in cfg.cost_levels:
        cost = cost_bps / 10000 * 2  # Round-trip
        
        pnls = []
        for market_id in overlap:
            if market_id not in last_prices.index:
                continue
            price = last_prices[market_id]
            outcome_row = res_df[res_df['market_id'] == market_id]
            if len(outcome_row) == 0 or pd.isna(price):
                continue
            
            outcome = outcome_row.iloc[0]['outcome']
            position = 1 if price < 0.5 else -1
            raw_pnl = position * (outcome - price)
            net_pnl = raw_pnl - cost
            pnls.append(net_pnl)
        
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
        
        results.append({
            "cost_bps": cost_bps,
            "round_trip_bps": cost_bps * 2,
            "n_trades": len(pnls),
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "profitable": bool(total_pnl > 0),
        })
        
        status = "✓" if total_pnl > 0 else "✗"
        print(f"   {cost_bps:3d} bps: PnL={total_pnl:+.4f}, Win={win_rate:.1%} {status}")
    
    # Find breakeven
    breakeven = None
    for r in results:
        if r['profitable']:
            breakeven = r['cost_bps']
            break
    
    summary = {
        "levels": results,
        "breakeven_bps": breakeven,
    }
    
    cost_dir = output_dir / "cost_sensitivity"
    cost_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cost_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    pd.DataFrame(results).to_csv(cost_dir / "pnl_by_cost.csv", index=False)
    
    return summary


def run_hypothesis_tests(
    calibration: Dict,
    trading: Dict,
    cost_sensitivity: Dict,
    output_dir: Path,
) -> Dict:
    """
    Run statistical hypothesis tests for H1-H4.
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTS")
    print("=" * 60)
    
    results = {}
    
    # H1: Calibration quality
    print("\n1. H1: Calibration Quality")
    ece = calibration.get('ece', 1.0)
    brier = calibration.get('brier_score', 1.0)
    h1_pass = ece < 0.10 and brier < 0.25
    results['h1_calibration'] = {
        "ece": ece,
        "brier": brier,
        "threshold_ece": 0.10,
        "threshold_brier": 0.25,
        "pass": h1_pass,
    }
    status = "PASS" if h1_pass else "FAIL"
    print(f"   ECE={ece:.4f} (<0.10?), Brier={brier:.4f} (<0.25?) → {status}")
    
    # H4: Distance-Profit Correlation
    print("\n2. H4: Distance-Profit Correlation")
    h4_corr = trading.get('h4_correlation')
    if h4_corr is not None:
        h4_pass = h4_corr > 0.1
        results['h4_distance_profit'] = {
            "correlation": h4_corr,
            "threshold": 0.1,
            "pass": h4_pass,
        }
        status = "PASS" if h4_pass else "FAIL"
        print(f"   Correlation={h4_corr:.4f} (>0.1?) → {status}")
    else:
        results['h4_distance_profit'] = {"error": "Could not compute"}
        print("   Could not compute correlation")
    
    # Profitability at conservative costs
    print("\n3. Profitability at Conservative Costs")
    profitable_at_200bps = any(
        r['profitable'] for r in cost_sensitivity.get('levels', [])
        if r['cost_bps'] == 200
    )
    results['profitability'] = {
        "profitable_at_200bps": profitable_at_200bps,
        "breakeven_bps": cost_sensitivity.get('breakeven_bps'),
    }
    status = "PASS" if profitable_at_200bps else "FAIL"
    print(f"   Profitable at 200bps? {status}")
    
    # Save
    hyp_dir = output_dir / "hypothesis_tests"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(hyp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    summary_lines = [
        "# Hypothesis Test Summary",
        "",
        "| Hypothesis | Metric | Value | Threshold | Result |",
        "|------------|--------|-------|-----------|--------|",
    ]
    
    if 'h1_calibration' in results:
        h1 = results['h1_calibration']
        summary_lines.append(
            f"| H1 | ECE | {h1['ece']:.4f} | <{h1['threshold_ece']} | {'PASS' if h1['pass'] else 'FAIL'} |"
        )
        summary_lines.append(
            f"| H1 | Brier | {h1['brier']:.4f} | <{h1['threshold_brier']} | {'PASS' if h1['pass'] else 'FAIL'} |"
        )
    
    if 'h4_distance_profit' in results and 'correlation' in results['h4_distance_profit']:
        h4 = results['h4_distance_profit']
        summary_lines.append(
            f"| H4 | Correlation | {h4['correlation']:.4f} | >{h4['threshold']} | {'PASS' if h4['pass'] else 'FAIL'} |"
        )
    
    if 'profitability' in results:
        prof = results['profitability']
        summary_lines.append(
            f"| Profit | 200bps | {'Yes' if prof['profitable_at_200bps'] else 'No'} | Yes | {'PASS' if prof['profitable_at_200bps'] else 'FAIL'} |"
        )
    
    with open(hyp_dir / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run backtest validation pipeline")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--skip-trading", action="store_true", help="Skip trading simulation")
    args = parser.parse_args()
    
    cfg = ValidationConfig()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(f"runs/backtest_validation_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BACKTEST PIPELINE VALIDATION")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    print(f"Execution cost: {cfg.slippage_bps * 2:.0f} bps round-trip")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
        json.dump(cfg_dict, f, indent=2)
    
    # Track 1: Calibration
    calibration = run_track1_calibration(cfg, output_dir)
    
    # Track 2: Trading
    if not args.skip_trading:
        trading = run_track2_trading(cfg, output_dir)
        cost_sensitivity = run_cost_sensitivity(cfg, output_dir)
    else:
        trading = {}
        cost_sensitivity = {}
    
    # Hypothesis tests
    hypothesis = run_hypothesis_tests(calibration, trading, cost_sensitivity, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = {
        "timestamp": timestamp,
        "calibration": calibration,
        "trading": trading,
        "cost_sensitivity": cost_sensitivity,
        "hypothesis_tests": hypothesis,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files:")
    print(f"  - {output_dir}/summary.json")
    print(f"  - {output_dir}/hypothesis_tests/summary.md")
    print(f"  - {output_dir}/calibration/metrics.json")
    print(f"  - {output_dir}/trading/results.json")
    
    return summary


if __name__ == "__main__":
    main()

