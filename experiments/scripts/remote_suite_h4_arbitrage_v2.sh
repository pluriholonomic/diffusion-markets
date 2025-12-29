#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# H4 ARBITRAGE EXPERIMENTS SUITE (v2)
# =============================================================================
#
# This suite tests the arbitrage hypothesis H4:
#   "Distance d(q, C_t) to learned constraint set C_t bounds extractable profit"
#
# Key improvements over v1:
#   - Hierarchical constraints: multicalibration (inner) + Frechet (outer)
#   - Bootstrap CIs on approachability rate
#   - Per-sample hybrid correction analysis
#   - Multi-market Frechet constraint validation
#
# Estimated time: ~3-4 hours
# =============================================================================

cd /root/diffusion-markets/experiments
export PYTHONPATH=src

mkdir -p remote_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="remote_logs/${TIMESTAMP}_h4_arbitrage_v2"
mkdir -p "$LOG_DIR"

echo "[suite] =============================================="
echo "[suite] H4 ARBITRAGE EXPERIMENTS SUITE (v2)"
echo "[suite] Start: $(date -u)"
echo "[suite] Log dir: $LOG_DIR"
echo "[suite] =============================================="

# Check GPU availability
echo "[suite] GPU status:"
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("gpus", torch.cuda.device_count())'

# =============================================================================
# PHASE 1: SYNTHETIC MULTI-MARKET VALIDATION
# =============================================================================
# Validate that d(q, C_t) predicts profit on synthetic data with known structure
# Estimated time: ~30 min

echo "[suite] =============================================="
echo "[suite] PHASE 1: SYNTHETIC MULTI-MARKET VALIDATION"
echo "[suite] =============================================="

# 1.1 Multi-market Frechet structure benchmark
echo "[suite] Running synthetic multi-market Frechet benchmark..."
.venv/bin/python -m forecastbench multimarket \
  --n-samples 50000 \
  --m 9 \
  --structure frechet \
  --noise 0.2 \
  --rho 0.95 \
  --run-name "h4_v2_synth_frechet" \
  2>&1 | tee "$LOG_DIR/synth_frechet.log"

# 1.2 Multi-market chain structure (implication constraints)
echo "[suite] Running synthetic chain structure benchmark..."
.venv/bin/python -m forecastbench multimarket \
  --n-samples 50000 \
  --m 6 \
  --structure chain \
  --noise 0.2 \
  --rho 0.95 \
  --run-name "h4_v2_synth_chain" \
  2>&1 | tee "$LOG_DIR/synth_chain.log"

# 1.3 Independent structure (Type I error test - should have no arbitrage)
echo "[suite] Running synthetic independent benchmark (Type I test)..."
.venv/bin/python -m forecastbench multimarket \
  --n-samples 50000 \
  --m 6 \
  --structure independent \
  --noise 0.2 \
  --rho 0.95 \
  --run-name "h4_v2_synth_independent" \
  2>&1 | tee "$LOG_DIR/synth_independent.log"

# =============================================================================
# PHASE 2: HIERARCHICAL CONSTRAINT EVALUATION
# =============================================================================
# Run pm_eval_v2 on existing predictions with full hierarchical analysis
# Estimated time: ~1 hour

echo "[suite] =============================================="
echo "[suite] PHASE 2: HIERARCHICAL CONSTRAINT EVALUATION"
echo "[suite] =============================================="

# Find latest prediction files
DERIVED_DIR="${DERIVED_DIR:-/root/polymarket_data/derived}"
RUNS_DIR="runs"

# Look for existing predictions
DIFF_PRED=""
HYBRID_PRED=""
AR_PRED=""

for f in $(ls -1t $RUNS_DIR/*/predictions.parquet 2>/dev/null | head -20); do
  dir=$(dirname "$f")
  name=$(basename "$dir")
  if [[ "$name" == *"diffusion"* ]] && [ -z "$DIFF_PRED" ]; then
    DIFF_PRED="$f"
  elif [[ "$name" == *"hybrid"* ]] && [ -z "$HYBRID_PRED" ]; then
    HYBRID_PRED="$f"
  elif [[ "$name" == *"ar"* ]] && [ -z "$AR_PRED" ]; then
    AR_PRED="$f"
  fi
done

echo "[suite] Found predictions:"
echo "  Diffusion: $DIFF_PRED"
echo "  Hybrid: $HYBRID_PRED"
echo "  AR: $AR_PRED"

# 2.1 Evaluate diffusion predictions with hierarchical constraints
if [ -n "$DIFF_PRED" ]; then
  echo "[suite] Evaluating diffusion with hierarchical constraints..."
  .venv/bin/python -m forecastbench pm_eval_v2 \
    --dataset-path "$DIFF_PRED" \
    --run-name "h4_v2_eval_diffusion" \
    --hierarchical-constraints \
    --multicalib-groups "topic,volume_q5" \
    --frechet-bundle-col "category" \
    --bundle-size 3 \
    --approachability-rate \
    --bootstrap-n 1000 \
    --seed 0 \
    2>&1 | tee "$LOG_DIR/eval_diffusion.log"
fi

# 2.2 Evaluate hybrid predictions with hierarchical constraints
if [ -n "$HYBRID_PRED" ]; then
  echo "[suite] Evaluating hybrid with hierarchical constraints..."
  
  # Check if AR predictions available for correction analysis
  AR_COL_ARG=""
  if [ -n "$AR_PRED" ]; then
    AR_COL_ARG="--ar-pred-col pred_ar"
  fi
  
  .venv/bin/python -m forecastbench pm_eval_v2 \
    --dataset-path "$HYBRID_PRED" \
    --run-name "h4_v2_eval_hybrid" \
    --hierarchical-constraints \
    --multicalib-groups "topic,volume_q5" \
    --frechet-bundle-col "category" \
    --bundle-size 3 \
    --approachability-rate \
    --bootstrap-n 1000 \
    $AR_COL_ARG \
    --seed 0 \
    2>&1 | tee "$LOG_DIR/eval_hybrid.log"
fi

# =============================================================================
# PHASE 3: MULTI-MARKET FRECHET ANALYSIS
# =============================================================================
# Analyze Frechet constraint violations across market categories
# Estimated time: ~30 min

echo "[suite] =============================================="
echo "[suite] PHASE 3: MULTI-MARKET FRECHET ANALYSIS"
echo "[suite] =============================================="

# Find a suitable dataset with category information
EVAL_DATA=""
if [ -n "$DIFF_PRED" ]; then
  EVAL_DATA="$DIFF_PRED"
elif [ -n "$HYBRID_PRED" ]; then
  EVAL_DATA="$HYBRID_PRED"
elif [ -f "$DERIVED_DIR/gamma_yesno_ready.parquet" ]; then
  EVAL_DATA="$DERIVED_DIR/gamma_yesno_ready.parquet"
fi

if [ -n "$EVAL_DATA" ]; then
  echo "[suite] Running multimarket arbitrage analysis on: $EVAL_DATA"
  
  # Frechet constraints
  .venv/bin/python -m forecastbench multimarket_arb \
    --dataset-path "$EVAL_DATA" \
    --run-name "h4_v2_frechet_arb" \
    --bundle-col "category" \
    --bundle-size 3 \
    --constraint-type "frechet" \
    --seed 0 \
    2>&1 | tee "$LOG_DIR/frechet_arb.log"
  
  # Mutual exclusion constraints (alternative test)
  .venv/bin/python -m forecastbench multimarket_arb \
    --dataset-path "$EVAL_DATA" \
    --run-name "h4_v2_mutex_arb" \
    --bundle-col "category" \
    --bundle-size 4 \
    --constraint-type "mutual_exclusion" \
    --seed 0 \
    2>&1 | tee "$LOG_DIR/mutex_arb.log"
else
  echo "[suite] WARN: No suitable dataset found for multimarket analysis"
fi

# =============================================================================
# PHASE 4: PER-SAMPLE HYBRID CORRECTION ANALYSIS
# =============================================================================
# Analyze how diffusion corrects/corrupts AR predictions
# Estimated time: ~15 min

echo "[suite] =============================================="
echo "[suite] PHASE 4: PER-SAMPLE HYBRID CORRECTION ANALYSIS"
echo "[suite] =============================================="

if [ -n "$HYBRID_PRED" ] && [ -n "$AR_PRED" ]; then
  echo "[suite] Running hybrid correction analysis..."
  
  .venv/bin/python - <<'PYTHON_SCRIPT'
import json
from pathlib import Path
import numpy as np
import pandas as pd

from forecastbench.benchmarks.hybrid_analysis import (
    run_hybrid_analysis,
    plot_correction_analysis,
)

runs_dir = Path("runs")

# Find hybrid and AR predictions
hybrid_files = list(runs_dir.glob("*hybrid*/predictions.parquet"))
ar_files = list(runs_dir.glob("*ar*/predictions.parquet"))

if not hybrid_files:
    print("[correction] No hybrid predictions found")
    exit(0)

hybrid_path = sorted(hybrid_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
print(f"[correction] Using hybrid: {hybrid_path}")

df_hybrid = pd.read_parquet(hybrid_path)

# Check for AR predictions in same file or separate
if "pred_ar" in df_hybrid.columns:
    p_ar = df_hybrid["pred_ar"].values
    p_hybrid = df_hybrid["pred_prob"].values
    y = df_hybrid["y"].values
elif ar_files:
    ar_path = sorted(ar_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    df_ar = pd.read_parquet(ar_path)
    
    # Match by ID
    common_ids = set(df_hybrid["id"]) & set(df_ar["id"])
    print(f"[correction] Matching {len(common_ids)} samples between hybrid and AR")
    
    df_hybrid = df_hybrid[df_hybrid["id"].isin(common_ids)].sort_values("id")
    df_ar = df_ar[df_ar["id"].isin(common_ids)].sort_values("id")
    
    p_ar = df_ar["pred_prob"].values
    p_hybrid = df_hybrid["pred_prob"].values
    y = df_hybrid["y"].values
else:
    print("[correction] No AR predictions found")
    exit(0)

# Run analysis
analysis = run_hybrid_analysis(p_ar=p_ar, p_hybrid=p_hybrid, y=y)

print("\n[correction] ============================================")
print("[correction] HYBRID CORRECTION ANALYSIS")
print("[correction] ============================================")
print(f"  Diff helps rate: {analysis['classification']['diff_helps_rate']:.2%}")
print(f"  Diff hurts rate: {analysis['classification']['diff_hurts_rate']:.2%}")
print(f"  Net help rate: {analysis['classification']['net_help_rate']:.2%}")
print(f"  Correction ratio: {analysis['classification']['correction_ratio']:.2f}")
print(f"  Targets AR errors: {analysis['targets_ar_errors']}")
print(f"  H1 supported: {analysis['h1_supported']}")
print(f"  Spectral hypothesis: {analysis['spectral_hypothesis_supported']}")
print("[correction] ============================================\n")

# Save results
out_dir = runs_dir / "h4_v2_correction_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, default=str)

# Generate plots
plot_correction_analysis(p_ar, p_hybrid, y, str(out_dir))

print(f"[correction] Results saved to: {out_dir}")
PYTHON_SCRIPT
else
  echo "[suite] WARN: Need both hybrid and AR predictions for correction analysis"
fi

# =============================================================================
# PHASE 5: FULL COMPARISON AND SUMMARY
# =============================================================================
# Compare all models and generate summary report
# Estimated time: ~15 min

echo "[suite] =============================================="
echo "[suite] PHASE 5: FULL COMPARISON AND SUMMARY"
echo "[suite] =============================================="

.venv/bin/python - <<'PYTHON_SCRIPT'
import json
from pathlib import Path
import pandas as pd

print("\n" + "="*70)
print("H4 ARBITRAGE EXPERIMENTS SUITE (v2) - SUMMARY")
print("="*70)

runs_dir = Path("runs")

# Collect metrics from all h4_v2 runs
results = {}
for run_dir in sorted(runs_dir.glob("*h4_v2*")):
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        results[run_dir.name] = metrics

if not results:
    print("No h4_v2 results found yet.")
    exit(0)

# Print hierarchical constraint results
print("\n### Hierarchical Constraint Analysis")
print("-"*60)
for name, m in sorted(results.items()):
    if "hierarchical_constraints" in m:
        hc = m["hierarchical_constraints"]
        print(f"  {name}:")
        print(f"    Distance to C_t: {hc.get('distance_to_C', 'N/A'):.4f}")
        print(f"    - Multicalib:    {hc.get('d_multicalib', 'N/A'):.4f}")
        print(f"    - Frechet:       {hc.get('d_frechet', 'N/A'):.4f}")

# Print approachability rate results
print("\n### Approachability Rate Analysis (H2)")
print("-"*60)
print("  Theory: decay rate α = 0.5 for Blackwell approachability")
for name, m in sorted(results.items()):
    if "approachability_rate" in m:
        ar = m["approachability_rate"]
        rate = ar.get("rate", float("nan"))
        ci_lo = ar.get("rate_ci_lo", float("nan"))
        ci_hi = ar.get("rate_ci_hi", float("nan"))
        consistent = ar.get("consistent_with_theory", False)
        print(f"  {name}:")
        print(f"    Rate: {rate:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
        print(f"    Consistent with 1/sqrt(T): {consistent}")

# Print arbitrage results
print("\n### Arbitrage Bound Analysis (H4)")
print("-"*60)
for name, m in sorted(results.items()):
    if "arbitrage_bound" in m:
        ab = m["arbitrage_bound"]
        print(f"  {name}:")
        print(f"    Market distance to C_t: {ab.get('market_distance_to_C', 'N/A'):.4f}")
        print(f"    Model distance to C_t:  {ab.get('model_distance_to_C', 'N/A'):.4f}")
        print(f"    Distance reduction:     {ab.get('distance_reduction', 0)*100:.1f}%")
        print(f"    Arbitrage capture:      {ab.get('arbitrage_capture_rate', 0)*100:.1f}%")
        print(f"    Sharpe ratio:           {ab.get('sharpe_ratio', 0):.2f}")

# Print hybrid analysis results
print("\n### Hybrid Correction Analysis (H1/H3)")
print("-"*60)
hybrid_analysis_file = runs_dir / "h4_v2_correction_analysis" / "analysis.json"
if hybrid_analysis_file.exists():
    with open(hybrid_analysis_file) as f:
        ha = json.load(f)
    print(f"  Diff helps rate: {ha['classification']['diff_helps_rate']:.2%}")
    print(f"  Net help rate: {ha['classification']['net_help_rate']:.2%}")
    print(f"  Correction ratio: {ha['classification']['correction_ratio']:.2f}")
    print(f"  Targets AR errors: {ha['targets_ar_errors']}")
    print(f"  H1 (diffusion useful): {ha['h1_supported']}")
    print(f"  Spectral hypothesis: {ha['spectral_hypothesis_supported']}")
else:
    print("  [Not computed - need both AR and hybrid predictions]")

# Print overall hypothesis assessment
print("\n" + "="*70)
print("HYPOTHESIS ASSESSMENT")
print("="*70)

print("""
H1 (Diffusion improves AR):
  → See Hybrid Correction Analysis: diffusion should help more than hurt

H2 (Blackwell approachability rate):
  → See Approachability Rate Analysis: rate should be ~0.5 (1/sqrt(T))

H3 (Diffusion learns C_t better):
  → Compare model distances to C_t between AR-only and hybrid
  → Hybrid should have smaller distance to constraint set

H4 (d(q, C_t) bounds extractable arbitrage):
  → See Arbitrage Bound Analysis: distance should correlate with profit
  → Lower distance should mean better calibrated predictions
""")

print("="*70 + "\n")

# Save summary to JSON
summary_path = runs_dir / "h4_v2_summary.json"
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Summary saved to: {summary_path}")
PYTHON_SCRIPT

echo "[suite] =============================================="
echo "[suite] SUITE COMPLETE"
echo "[suite] End: $(date -u)"
echo "[suite] Logs: $LOG_DIR"
echo "[suite] =============================================="


