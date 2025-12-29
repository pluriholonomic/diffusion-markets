#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# COMPREHENSIVE EXPERIMENT SUITE: Turtel/RLCR/Blackwell Comparison
# =============================================================================
#
# Tests three hypotheses:
#   H1: Diffusion repair module improves AR performance
#   H2: Setup can be viewed through Blackwell approachability lens
#   H3: Diffusion improving performance → better Blackwell constraint learning
#
# Comparisons:
#   - Turtel et al. (2025): RLVR with ReMax, Brier reward
#   - Damani et al. (2025): RLCR with correctness + Brier
#   - Ours: AR + Diffusion repair with Blackwell constraints
#
# Estimated time breakdown (H200 8x80GB):
#   - Synthetic experiments: ~30 min
#   - GRPO training (per config): ~2-4 hours
#   - Evaluation: ~30 min per model
#   - Total: ~8-12 hours
# =============================================================================

cd /root/diffusion-markets/experiments
export PYTHONPATH=src

mkdir -p remote_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="remote_logs/${TIMESTAMP}_turtel_rlcr_blackwell"
mkdir -p "$LOG_DIR"

echo "[suite] =============================================="
echo "[suite] TURTEL/RLCR/BLACKWELL COMPARISON SUITE"
echo "[suite] Start: $(date -u)"
echo "[suite] Log dir: $LOG_DIR"
echo "[suite] =============================================="

# Check GPU availability
echo "[suite] GPU status:"
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("gpus", torch.cuda.device_count()); [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'

# =============================================================================
# PHASE 1: SYNTHETIC EXPERIMENTS (Blackwell Approachability Theory)
# =============================================================================
# These support H2 and H3 - the theoretical foundation
# Estimated time: ~30 min

echo "[suite] =============================================="
echo "[suite] PHASE 1: SYNTHETIC EXPERIMENTS"
echo "[suite] =============================================="

# 1.1 Parity benchmark with Blackwell diagnostics
for k in 4 6 8 10 12; do
  echo "[suite] parity --blackwell k=$k"
  .venv/bin/python -m forecastbench parity \
    --d 24 --k "$k" --alpha 0.8 --n 100000 --rho 0.95 --L 4 \
    --blackwell --bw-group chi_S --bw-curve-every 500 \
    --run-name "turtel_suite_parity_k${k}" \
    2>&1 | tee "$LOG_DIR/parity_k${k}.log"
done

# 1.2 Approachability suite with constraint tracking
echo "[suite] approachability_suite"
.venv/bin/python -m forecastbench approachability_suite \
  --d 24 --degrees "4,6,8,10,12" --n-per-degree 50000 \
  --rho 0.95 --L 4 --seed 0 \
  --run-name "turtel_suite_approachability" \
  2>&1 | tee "$LOG_DIR/approachability_suite.log"

# 1.3 Cliff vs Fog (AR spectral cliff vs diffusion continuous recovery)
echo "[suite] cliff_fog"
.venv/bin/python -m forecastbench cliff_fog \
  --d 24 --k 8 --alpha 0.8 --n 50000 \
  --L-values "2,4,6,8" --rho-values "0.7,0.8,0.9,0.95,0.99" \
  --run-name "turtel_suite_cliff_fog" \
  2>&1 | tee "$LOG_DIR/cliff_fog.log"

# 1.4 Group robustness (Propositions 8-9)
echo "[suite] group_robustness"
.venv/bin/python -m forecastbench group_robustness \
  --d 20 --k 10 --alpha 0.8 --n 100000 \
  --rho 0.95 --L 4 \
  --run-name "turtel_suite_group_robustness" \
  2>&1 | tee "$LOG_DIR/group_robustness.log"

# 1.5 Swap regret comparison
echo "[suite] swap_regret"
.venv/bin/python -m forecastbench swap_regret \
  --d 24 --k 8 --alpha 0.8 --n 50000 \
  --rho 0.95 --L 4 \
  --run-name "turtel_suite_swap_regret" \
  2>&1 | tee "$LOG_DIR/swap_regret.log"

# =============================================================================
# PHASE 2: DATA PREPARATION
# =============================================================================
# Prepare Polymarket data for training and evaluation
# Estimated time: ~15 min

echo "[suite] =============================================="
echo "[suite] PHASE 2: DATA PREPARATION"
echo "[suite] =============================================="

DERIVED_DIR="${DERIVED_DIR:-/root/polymarket_data/derived}"
RAW_GZ="${RAW_GZ:-/root/polymarket_data/gamma/markets_raw.jsonl.gz}"
OUT_READY="$DERIVED_DIR/gamma_yesno_ready.parquet"

# Check if data already exists
if [ -f "$OUT_READY" ]; then
  echo "[suite] Data already prepared at $OUT_READY"
else
  echo "[suite] Building dataset from Gamma dump..."
  # Use existing polymarket suite for data prep
  MIN_VOLUME=100 CREATED_AFTER=2024-01-01T00:00:00Z SAMPLE_ROWS=10000 \
    bash scripts/remote_suite_polymarket.sh 2>&1 | tee "$LOG_DIR/data_prep.log" || true
fi

# Verify data exists
if [ ! -f "$OUT_READY" ]; then
  echo "[suite] ERROR: Could not prepare data at $OUT_READY"
  echo "[suite] Trying alternative path..."
  OUT_READY=$(ls -1t "$DERIVED_DIR"/*.parquet 2>/dev/null | head -n 1 || echo "")
  if [ -z "$OUT_READY" ]; then
    echo "[suite] FATAL: No parquet files found in $DERIVED_DIR"
    exit 1
  fi
fi

echo "[suite] Using dataset: $OUT_READY"
N_SAMPLES=$(.venv/bin/python -c "import pandas as pd; print(len(pd.read_parquet('$OUT_READY')))")
echo "[suite] Dataset has $N_SAMPLES samples"

# =============================================================================
# PHASE 3: GRPO TRAINING EXPERIMENTS
# =============================================================================
# Train models with different algorithms and reward functions
# Estimated time: ~6-8 hours (4 configs x 1.5-2 hours each)

echo "[suite] =============================================="
echo "[suite] PHASE 3: GRPO TRAINING EXPERIMENTS"
echo "[suite] =============================================="

# Training parameters (matching Turtel et al. as closely as possible)
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
STEPS=500  # Reduced for initial run; increase for full comparison
BATCH_SIZE=4
K=4
LR="2e-6"

# 3.1 Turtel-style: ReMax + Pure Brier (baseline)
echo "[suite] GRPO: ReMax + Turtel Brier"
.venv/bin/python -m forecastbench grpo_train \
  --data-path "$OUT_READY" \
  --out-dir "runs/grpo" \
  --model "$MODEL" \
  --algorithm remax \
  --reward-mode turtel_brier \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --K "$K" \
  --lr "$LR" \
  --run-name "turtel_remax_brier" \
  2>&1 | tee "$LOG_DIR/grpo_remax_brier.log"

# 3.2 Damani-style: Dr.GRPO + RLCR (correctness + Brier)
echo "[suite] GRPO: Dr.GRPO + RLCR"
.venv/bin/python -m forecastbench grpo_train \
  --data-path "$OUT_READY" \
  --out-dir "runs/grpo" \
  --model "$MODEL" \
  --algorithm dr_grpo \
  --reward-mode rlcr \
  --rlcr-alpha 1.0 \
  --rlcr-beta 1.0 \
  --rlcr-gamma 0.1 \
  --rlcr-use-groups \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --K "$K" \
  --lr "$LR" \
  --run-name "damani_drgrpo_rlcr" \
  2>&1 | tee "$LOG_DIR/grpo_drgrpo_rlcr.log"

# 3.3 Ours: ReMax + Blackwell-aware reward
echo "[suite] GRPO: ReMax + Blackwell-aware"
.venv/bin/python -m forecastbench grpo_train \
  --data-path "$OUT_READY" \
  --out-dir "runs/grpo" \
  --model "$MODEL" \
  --algorithm remax \
  --reward-mode blackwell_aware \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --K "$K" \
  --lr "$LR" \
  --run-name "ours_remax_blackwell" \
  2>&1 | tee "$LOG_DIR/grpo_remax_blackwell.log"

# 3.4 Kelly criterion reward (economic value)
echo "[suite] GRPO: ReMax + Kelly"
.venv/bin/python -m forecastbench grpo_train \
  --data-path "$OUT_READY" \
  --out-dir "runs/grpo" \
  --model "$MODEL" \
  --algorithm remax \
  --reward-mode kelly \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --K "$K" \
  --lr "$LR" \
  --run-name "ours_remax_kelly" \
  2>&1 | tee "$LOG_DIR/grpo_remax_kelly.log"

# =============================================================================
# PHASE 4: DIFFUSION TRAINING (Our Repair Module)
# =============================================================================
# Train diffusion model for calibration repair
# Estimated time: ~1 hour

echo "[suite] =============================================="
echo "[suite] PHASE 4: DIFFUSION TRAINING"
echo "[suite] =============================================="

echo "[suite] Diffusion baseline training"
.venv/bin/python -m forecastbench pm_difftrain \
  --dataset-path "$OUT_READY" \
  --run-name "turtel_suite_diffusion" \
  --max-rows 5000 \
  --text-cols "question,description" \
  --train-frac 0.8 \
  --seed 0 \
  --embed-model "sentence-transformers/all-MiniLM-L6-v2" \
  --embed-device "cuda" \
  --embed-dtype "float16" \
  --embed-batch-size 128 \
  --device "cuda" \
  --train-steps 2000 \
  --batch-size 512 \
  --sample-steps 32 \
  --mc 16 \
  --agg "mean" \
  2>&1 | tee "$LOG_DIR/diffusion_train.log"

# =============================================================================
# PHASE 5: AR + DIFFUSION HYBRID TRAINING
# =============================================================================
# Train the combined AR + Diffusion model (our main contribution)
# Estimated time: ~2 hours

echo "[suite] =============================================="
echo "[suite] PHASE 5: AR + DIFFUSION HYBRID TRAINING"
echo "[suite] =============================================="

echo "[suite] AR + Diffusion hybrid"
.venv/bin/python -m forecastbench pm_hybrid_train \
  --dataset-path "$OUT_READY" \
  --run-name "turtel_suite_hybrid" \
  --max-rows 5000 \
  --text-cols "question,description" \
  --ar-model "Qwen/Qwen3-14B" \
  --diff-train-steps 1500 \
  --diff-batch-size 256 \
  --diff-T 64 \
  --diff-sample-steps 32 \
  --diff-n-heads 8 \
  --diff-d-model 512 \
  --diff-n-layers 6 \
  --joint-weight-ar 0.3 \
  --joint-weight-diff 0.7 \
  --device "cuda" \
  --seed 0 \
  2>&1 | tee "$LOG_DIR/hybrid_train.log"

# =============================================================================
# PHASE 6: EVALUATION AND COMPARISON
# =============================================================================
# Evaluate all models and compare to Turtel baseline
# Estimated time: ~1 hour

echo "[suite] =============================================="
echo "[suite] PHASE 6: EVALUATION AND COMPARISON"
echo "[suite] =============================================="

# Get the latest run directories
DIFF_RUN=$(ls -1td runs/*_turtel_suite_diffusion 2>/dev/null | head -n 1 || echo "")
HYBRID_RUN=$(ls -1td runs/*_turtel_suite_hybrid 2>/dev/null | head -n 1 || echo "")

# Evaluate diffusion model
if [ -n "$DIFF_RUN" ] && [ -f "$DIFF_RUN/predictions.parquet" ]; then
  echo "[suite] Evaluating diffusion model: $DIFF_RUN"
  .venv/bin/python -m forecastbench pm_eval \
    --dataset-path "$DIFF_RUN/predictions.parquet" \
    --pred-col "pred_prob" \
    --text-cols "question,description" \
    --run-name "turtel_suite_eval_diffusion" \
    --group-cols "volume_q5,ttc_q5" \
    --approachability \
    --trading-mode "sign" \
    2>&1 | tee "$LOG_DIR/eval_diffusion.log"
fi

# Evaluate hybrid model
if [ -n "$HYBRID_RUN" ] && [ -f "$HYBRID_RUN/predictions.parquet" ]; then
  echo "[suite] Evaluating hybrid model: $HYBRID_RUN"
  .venv/bin/python -m forecastbench pm_eval \
    --dataset-path "$HYBRID_RUN/predictions.parquet" \
    --pred-col "pred_prob" \
    --text-cols "question,description" \
    --run-name "turtel_suite_eval_hybrid" \
    --group-cols "volume_q5,ttc_q5" \
    --approachability \
    --trading-mode "sign" \
    2>&1 | tee "$LOG_DIR/eval_hybrid.log"
fi

# Run Turtel comparison with reported baseline numbers
echo "[suite] Turtel comparison analysis"
for pred_file in "$DIFF_RUN/predictions.parquet" "$HYBRID_RUN/predictions.parquet"; do
  if [ -f "$pred_file" ]; then
    run_name=$(echo "$pred_file" | sed 's/.*runs\/[0-9_]*_\(.*\)\/predictions.parquet/\1/')
    .venv/bin/python -m forecastbench turtel_compare \
      --dataset-path "$pred_file" \
      --pred-col "pred_prob" \
      --y-col "y" \
      --market-prob-col "market_prob" \
      --turtel-brier 0.19 \
      --turtel-ece 0.05 \
      --turtel-roi 0.10 \
      --run-name "turtel_suite_compare_${run_name}" \
      2>&1 | tee "$LOG_DIR/turtel_compare_${run_name}.log"
  fi
done

# =============================================================================
# PHASE 7: BLACKWELL CONSTRAINT ANALYSIS
# =============================================================================
# Compare constraint convergence between models
# Estimated time: ~30 min

echo "[suite] =============================================="
echo "[suite] PHASE 7: BLACKWELL CONSTRAINT ANALYSIS"
echo "[suite] =============================================="

# Run constraint convergence comparison on predictions
.venv/bin/python - <<'PYTHON_SCRIPT'
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from forecastbench.metrics.multiscale_approachability import (
        BlackwellConstraintTracker,
        BlackwellConstraintTrackerSpec,
        compare_constraint_convergence,
    )
except ImportError as e:
    print(f"[blackwell] Import error: {e}")
    exit(0)

# Find prediction files
runs_dir = Path("runs")
pred_files = {}

for pattern in ["diffusion", "hybrid"]:
    matches = list(runs_dir.glob(f"*turtel_suite_{pattern}/predictions.parquet"))
    if matches:
        pred_files[pattern] = sorted(matches, reverse=True)[0]

if len(pred_files) < 2:
    print(f"[blackwell] Not enough prediction files found: {pred_files}")
    exit(0)

print(f"[blackwell] Comparing: {list(pred_files.keys())}")

# Load predictions
dfs = {}
for name, path in pred_files.items():
    df = pd.read_parquet(path)
    dfs[name] = df
    print(f"[blackwell] Loaded {name}: {len(df)} samples")

# Use common samples
common_ids = set(dfs["diffusion"]["id"]) & set(dfs["hybrid"]["id"])
print(f"[blackwell] Common samples: {len(common_ids)}")

# Compare constraint convergence
results = compare_constraint_convergence(
    p_ar=dfs["diffusion"][dfs["diffusion"]["id"].isin(common_ids)]["pred_prob"].values,
    p_hybrid=dfs["hybrid"][dfs["hybrid"]["id"].isin(common_ids)]["pred_prob"].values,
    y=dfs["diffusion"][dfs["diffusion"]["id"].isin(common_ids)]["y"].values,
    n_bins=10,
    n_groups=5,
)

print("\n[blackwell] ============================================")
print("[blackwell] CONSTRAINT CONVERGENCE COMPARISON")
print("[blackwell] ============================================")
print(f"  Violation reduction (Diff→Hybrid): {results['violation_reduction']:.2%}")
print(f"  Rate improvement: {results['rate_improvement']:.2%}")
print(f"  Worst group improvement: {results['worst_group_improvement']:.2%}")
print(f"  Conclusion: {results['interpretation']}")
print("[blackwell] ============================================\n")

# Save results
out_path = Path("runs") / "blackwell_constraint_comparison.json"
out_path.write_text(json.dumps(results, indent=2, default=str))
print(f"[blackwell] Results saved to: {out_path}")
PYTHON_SCRIPT

# =============================================================================
# PHASE 8: SUMMARY REPORT
# =============================================================================

echo "[suite] =============================================="
echo "[suite] PHASE 8: SUMMARY REPORT"
echo "[suite] =============================================="

.venv/bin/python - <<'PYTHON_SCRIPT'
import json
from pathlib import Path
import pandas as pd

print("\n" + "="*60)
print("TURTEL/RLCR/BLACKWELL COMPARISON SUITE - SUMMARY")
print("="*60)

runs_dir = Path("runs")

# Collect all metrics
results = {}

for run_dir in sorted(runs_dir.glob("*turtel_suite*")):
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        run_name = run_dir.name.split("_", 2)[-1] if "_" in run_dir.name else run_dir.name
        results[run_name] = metrics

# Print comparison table
print("\n### Brier Score (lower is better)")
print("-"*50)
for name, m in sorted(results.items()):
    if "brier" in m:
        print(f"  {name:30s}: {m['brier']:.4f}")
    elif "our_metrics" in m and "brier" in m["our_metrics"]:
        print(f"  {name:30s}: {m['our_metrics']['brier']:.4f}")

print("\n### ECE (lower is better)")
print("-"*50)
for name, m in sorted(results.items()):
    if "ece" in m:
        print(f"  {name:30s}: {m['ece']:.4f}")
    elif "our_metrics" in m and "ece" in m["our_metrics"]:
        print(f"  {name:30s}: {m['our_metrics']['ece']:.4f}")

print("\n### ROI (higher is better)")
print("-"*50)
print(f"  {'Turtel et al. (reported)':30s}: 10.00%")
for name, m in sorted(results.items()):
    if "trading" in m and "roi" in m["trading"]:
        print(f"  {name:30s}: {m['trading']['roi']*100:.2f}%")
    elif "our_metrics" in m and "trading" in m["our_metrics"]:
        print(f"  {name:30s}: {m['our_metrics']['trading']['roi']*100:.2f}%")

# Blackwell comparison
blackwell_file = runs_dir / "blackwell_constraint_comparison.json"
if blackwell_file.exists():
    with open(blackwell_file) as f:
        bc = json.load(f)
    print("\n### Blackwell Constraint Analysis")
    print("-"*50)
    print(f"  Violation reduction (Diff→Hybrid): {bc.get('violation_reduction', 0)*100:.1f}%")
    print(f"  Rate improvement: {bc.get('rate_improvement', 0)*100:.1f}%")
    print(f"  Interpretation: {bc.get('interpretation', 'N/A')}")

print("\n" + "="*60)
print("HYPOTHESIS ASSESSMENT")
print("="*60)
print("""
H1 (Diffusion improves AR):
  → Compare hybrid ROI/Brier to AR-only baselines above

H2 (Blackwell approachability lens):
  → See approachability_suite and parity --blackwell results

H3 (Diffusion → better constraints):
  → See Blackwell Constraint Analysis above
""")
print("="*60 + "\n")
PYTHON_SCRIPT

echo "[suite] =============================================="
echo "[suite] SUITE COMPLETE"
echo "[suite] End: $(date -u)"
echo "[suite] Logs: $LOG_DIR"
echo "[suite] =============================================="


