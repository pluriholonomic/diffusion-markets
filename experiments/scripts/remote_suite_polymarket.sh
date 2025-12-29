#!/usr/bin/env bash
set -euo pipefail

# Polymarket evaluation suite aligned with the *evaluation* aspects of:
#   Outcome-based Reinforcement Learning to Predict the Future (arXiv:2505.17989)
# https://arxiv.org/abs/2505.17989
#
# This suite is evaluation-only (no RL training loop). It produces:
# - a labeled dataset from Gamma dump (pm_build_gamma)
# - a criterion-based market price via full CLOB history (pm_download_clob_history + pm_build_criterion_prices)
# - diffusion baseline (pm_difftrain)
# - AR baseline via HF LLM (pm_eval with --llm-model and median-of-K)

cd /root/diffusion-markets/experiments
export PYTHONPATH=src

RAW_GZ="${RAW_GZ:-/root/polymarket_data/gamma/markets_raw.jsonl.gz}"
DERIVED_DIR="${DERIVED_DIR:-/root/polymarket_data/derived}"
mkdir -p "$DERIVED_DIR" remote_logs

MIN_VOLUME="${MIN_VOLUME:-100}"
CREATED_AFTER="${CREATED_AFTER:-2024-01-01T00:00:00Z}"

# Sizes (keep modest by default; you can scale later)
SAMPLE_ROWS="${SAMPLE_ROWS:-20000}"
ENRICH_MAX_ROWS="${ENRICH_MAX_ROWS:-5000}"  # used as max rows for CLOB history download
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-512}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-14B}"
LLM_K="${LLM_K:-5}"

# Themis/brier.fyi-style market probability criterion:
#   midpoint | time-average | before-close-hours-24 | before-close-days-7 | before-close-days-30 | ...
CRITERION="${CRITERION:-midpoint}"

# Diffusion-side text embedder (keep lightweight; does not need to match the AR LLM).
DIFF_EMBED_MODEL="${DIFF_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
DIFF_EMBED_DTYPE="${DIFF_EMBED_DTYPE:-float16}"
DIFF_EMBED_BATCH_SIZE="${DIFF_EMBED_BATCH_SIZE:-128}"

# CLOB history download fidelity (passed through to /prices-history).
# Larger values are coarser but faster (e.g. 60, 1440).
CLOB_FIDELITY="${CLOB_FIDELITY:-60}"

echo "[pm_suite] start: $(date -u)"
echo "[pm_suite] raw_gz=$RAW_GZ"
echo "[pm_suite] derived_dir=$DERIVED_DIR"
echo "[pm_suite] min_volume=$MIN_VOLUME created_after=$CREATED_AFTER"
echo "[pm_suite] sample_rows=$SAMPLE_ROWS enrich_max_rows=$ENRICH_MAX_ROWS eval_max_examples=$EVAL_MAX_EXAMPLES"
echo "[pm_suite] llm_model=$LLM_MODEL llm_K=$LLM_K"
echo "[pm_suite] diff_embed_model=$DIFF_EMBED_MODEL diff_embed_dtype=$DIFF_EMBED_DTYPE diff_embed_batch_size=$DIFF_EMBED_BATCH_SIZE"
echo "[pm_suite] criterion=$CRITERION clob_fidelity=$CLOB_FIDELITY"

OUT_ALL="$DERIVED_DIR/gamma_yesno_resolved.parquet"
OUT_SAMPLE="$DERIVED_DIR/gamma_yesno_sample.parquet"
OUT_READY="$DERIVED_DIR/gamma_yesno_ready.parquet"
CLOB_HISTORY_DIR="$DERIVED_DIR/clob_history_yes_f${CLOB_FIDELITY}"

echo "[pm_suite] (1) build dataset from Gamma dump -> $OUT_ALL"
.venv/bin/python -m forecastbench pm_build_gamma \
  --input "$RAW_GZ" \
  --out "$OUT_ALL" \
  --min-volume 0 \
  --chunk-rows 20000

echo "[pm_suite] (2) filter createdAt >= $CREATED_AFTER and volumeNum >= $MIN_VOLUME; sample $SAMPLE_ROWS -> $OUT_SAMPLE"
.venv/bin/python - <<PY
import pandas as pd
from pathlib import Path

inp = Path("$OUT_ALL")
out = Path("$OUT_SAMPLE")
min_vol = float("$MIN_VOLUME")
created_after = "$CREATED_AFTER"
sample_rows = int("$SAMPLE_ROWS")

df = pd.read_parquet(inp)
df["createdAt_dt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
cut = pd.to_datetime(created_after, utc=True)
df = df[df["createdAt_dt"].notna() & (df["createdAt_dt"] >= cut)].copy()
df = df[df["volumeNum"] >= min_vol].copy()

if len(df) > sample_rows:
    df = df.sample(n=sample_rows, random_state=0)

df = df.drop(columns=["createdAt_dt"])
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print({"kept": int(len(df)), "out": str(out)})
PY

echo "[pm_suite] (3) download full CLOB history for YES tokens -> $CLOB_HISTORY_DIR (max_rows=$ENRICH_MAX_ROWS)"
.venv/bin/python -m forecastbench pm_download_clob_history \
  --input "$OUT_SAMPLE" \
  --out-dir "$CLOB_HISTORY_DIR" \
  --fidelity "$CLOB_FIDELITY" \
  --earliest-timestamp 1704096000 \
  --sleep-s 0.05 \
  --max-rows "$ENRICH_MAX_ROWS"

echo "[pm_suite] (4) build criterion-based market_prob ($CRITERION) -> $OUT_READY"
.venv/bin/python -m forecastbench pm_build_criterion_prices \
  --input "$OUT_SAMPLE" \
  --clob-history-dir "$CLOB_HISTORY_DIR" \
  --out "$OUT_READY" \
  --criterion "$CRITERION" \
  --max-rows "$ENRICH_MAX_ROWS"

echo "[pm_suite] (5) diffusion baseline: pm_difftrain on $OUT_READY"
RUN_DIFF="pm_suite_difftrain"
.venv/bin/python -m forecastbench pm_difftrain \
  --dataset-path "$OUT_READY" \
  --run-name "$RUN_DIFF" \
  --max-rows "$ENRICH_MAX_ROWS" \
  --text-cols "question,description" \
  --train-frac 0.8 \
  --seed 0 \
  --embed-model "$DIFF_EMBED_MODEL" \
  --embed-device "cuda" \
  --embed-dtype "$DIFF_EMBED_DTYPE" \
  --embed-batch-size "$DIFF_EMBED_BATCH_SIZE" \
  --device "cuda" \
  --train-steps 2000 \
  --batch-size 512 \
  --sample-steps 32 \
  --mc 16 \
  --agg "mean"

# Locate latest run dir for pm_difftrain to get predictions.parquet
RUN_DIR_DIFF=$(ls -1td runs/*_"$RUN_DIFF" | head -n 1 || true)
TEST_PARQUET="$RUN_DIR_DIFF/predictions.parquet"
echo "[pm_suite] diffusion run_dir=$RUN_DIR_DIFF"
echo "[pm_suite] test_parquet=$TEST_PARQUET"

echo "[pm_suite] (6) evaluate diffusion on same held-out test set"
.venv/bin/python -m forecastbench pm_eval \
  --dataset-path "$TEST_PARQUET" \
  --pred-col "pred_prob" \
  --text-cols "question,description" \
  --run-name "pm_suite_eval_diff" \
  --group-cols "volume_q5,ttc_q5" \
  --approachability \
  --trading-mode "sign"

echo "[pm_suite] (7) evaluate AR baseline (median-of-K) on same test set (max_examples=$EVAL_MAX_EXAMPLES)"
.venv/bin/python -m forecastbench pm_eval \
  --dataset-path "$TEST_PARQUET" \
  --max-examples "$EVAL_MAX_EXAMPLES" \
  --pred-col "pred_prob" \
  --text-cols "question,description" \
  --run-name "pm_suite_eval_ar" \
  --group-cols "volume_q5,ttc_q5" \
  --approachability \
  --trading-mode "sign" \
  --llm-model "$LLM_MODEL" \
  --K "$LLM_K" \
  --L 4 \
  --seed 0 \
  --llm-device "cuda" \
  --llm-device-map "auto" \
  --llm-max-new-tokens 128 \
  --llm-agg "median"

echo "[pm_suite] (8) market baseline (use market_prob as predictor on same test set)"
.venv/bin/python -m forecastbench pm_eval \
  --dataset-path "$TEST_PARQUET" \
  --pred-col "market_prob" \
  --text-cols "question,description" \
  --run-name "pm_suite_eval_market" \
  --group-cols "volume_q5,ttc_q5" \
  --approachability \
  --trading-mode "sign"

echo "[pm_suite] done: $(date -u)"


