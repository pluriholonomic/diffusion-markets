#!/usr/bin/env python3
"""Analyze all experiment runs."""

import json
from pathlib import Path
import pandas as pd

print("=" * 90)
print("EXPERIMENT ANALYSIS - ALL EVAL RUNS WITH METRICS")
print("=" * 90)

runs_dir = Path("runs")
results = []

for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    
    metrics_file = run_dir / "metrics.json"
    pred_file = run_dir / "predictions.parquet"
    
    if not metrics_file.exists():
        continue
    
    try:
        with open(metrics_file) as f:
            m = json.load(f)
        
        row = {"run": run_dir.name}
        row["brier"] = m.get("metrics", {}).get("brier")
        row["ece"] = m.get("metrics", {}).get("ece")
        
        trading = m.get("metrics", {}).get("trading", {})
        row["pnl"] = trading.get("pnl") if trading else None
        row["sharpe"] = trading.get("sharpe") if trading else None
        
        if pred_file.exists():
            df = pd.read_parquet(pred_file)
            row["pred_mean"] = df["pred_prob"].mean()
            row["y_mean"] = df["y"].mean()
        
        results.append(row)
    except Exception as e:
        print(f"Error processing {run_dir.name}: {e}")

df = pd.DataFrame(results)
df = df[df["brier"].notna()]
df["calib_gap"] = abs(df["pred_mean"] - df["y_mean"])
df = df.sort_values("brier")

# Focus on Dec 30-31 runs
recent = df[df["run"].str.contains("20251230|20251231", regex=True)]

print("\n📊 RECENT EVALUATIONS (Dec 30-31):")
print("-" * 90)
for idx, r in recent.iterrows():
    name = r["run"][9:50]  # Remove date prefix
    b = f"{r['brier']:.4f}" if pd.notna(r["brier"]) else "-"
    e = f"{r['ece']:.4f}" if pd.notna(r["ece"]) else "-"
    p = f"{r['pred_mean']:.3f}" if pd.notna(r.get("pred_mean")) else "-"
    y = f"{r['y_mean']:.3f}" if pd.notna(r.get("y_mean")) else "-"
    g = f"{r['calib_gap']:.3f}" if pd.notna(r.get("calib_gap")) else "-"
    pnl_val = r.get("pnl")
    pnl = f"{pnl_val:.2f}" if pnl_val is not None and pd.notna(pnl_val) else "-"
    sh_val = r.get("sharpe")
    sh = f"{sh_val:.2f}" if sh_val is not None and pd.notna(sh_val) else "-"
    print(f"{name:40s} Brier={b} ECE={e} | pred={p} y={y} gap={g} | PNL={pnl} Sharpe={sh}")

# GRPO training runs
print("\n" + "=" * 90)
print("📈 GRPO/RLCR TRAINING RUNS")
print("-" * 90)

for run_dir in sorted(runs_dir.iterdir()):
    results_file = run_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                m = json.load(f)
            name = run_dir.name[:60]
            b = m.get("best_brier", 0)
            s = m.get("best_step", "?")
            t = m.get("steps", "?")
            e = "early-stop" if m.get("early_stopped") else ""
            print(f"{name:60s} best_brier={b:.4f} @ step {s}/{t} {e}")
        except Exception as ex:
            print(f"Error: {run_dir.name}: {ex}")

# Summary
print("\n" + "=" * 90)
print("🎯 KEY INSIGHTS")
print("-" * 90)

# Best/worst
if len(recent) > 0:
    best = recent.iloc[0]
    print(f"\n✅ BEST Brier: {best['run'][9:]}")
    print(f"   Brier={best['brier']:.4f}, ECE={best['ece']:.4f}")
    
    if "calib_gap" in recent.columns and recent["calib_gap"].notna().any():
        cal_sorted = recent.dropna(subset=["calib_gap"]).sort_values("calib_gap")
        if len(cal_sorted) > 0:
            best_cal = cal_sorted.iloc[0]
            worst_cal = cal_sorted.iloc[-1]
            print(f"\n✅ BEST Calibration: {best_cal['run'][9:]}")
            print(f"   pred_mean={best_cal['pred_mean']:.3f}, y_mean={best_cal['y_mean']:.3f}, gap={best_cal['calib_gap']:.3f}")
            print(f"\n❌ WORST Calibration: {worst_cal['run'][9:]}")
            print(f"   pred_mean={worst_cal['pred_mean']:.3f}, y_mean={worst_cal['y_mean']:.3f}, gap={worst_cal['calib_gap']:.3f}")
