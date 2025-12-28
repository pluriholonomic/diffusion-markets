#!/bin/bash
# =============================================================================
# Exact Turtel Comparison Training (Headlines Pre-Fetched)
# =============================================================================
# Assumes headlines have already been fetched via fetch_exa_headlines.sh
# This script runs the training experiments using pre-enriched data.
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJ_ROOT}/data"
RUNS_DIR="${PROJ_ROOT}/runs/turtel_exa_comparison"

mkdir -p "${RUNS_DIR}"

cd "${PROJ_ROOT}/experiments"

# =============================================================================
# Step 1: Check that headlines have been fetched
# =============================================================================
log_info "Step 1: Checking for pre-enriched headlines data..."

HEADLINES_OUT="${DATA_DIR}/polymarket/turtel_exa_enriched.parquet"

if [[ ! -f "${HEADLINES_OUT}" ]]; then
    log_error "Headlines not found at: ${HEADLINES_OUT}"
    log_error "Run fetch_exa_headlines.sh first!"
    exit 1
fi

log_info "Using pre-enriched data: ${HEADLINES_OUT}"

# =============================================================================
# Step 3: Train AR model with ReMax + Brier (Turtel baseline)
# =============================================================================
log_info "Step 3: Training AR model with ReMax + Brier reward (Turtel baseline)..."

AR_REMAX_DIR="${RUNS_DIR}/ar_remax_brier"
mkdir -p "${AR_REMAX_DIR}"

forecastbench grpo_train \
    --data-path "${HEADLINES_OUT}" \
    --out-dir "${AR_REMAX_DIR}" \
    --model "Qwen/Qwen3-1.7B" \
    --algorithm remax \
    --reward-mode turtel_brier \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-6 \
    --samples-per-question 4 \
    --prompt-col turtel_prompt \
    --question-col question \
    --outcome-col outcome \
    --seed 42

# =============================================================================
# Step 4: Train AR+Diffusion Hybrid with ReMax + Brier
# =============================================================================
log_info "Step 4: Training AR+Diffusion hybrid with ReMax + Brier..."

HYBRID_REMAX_DIR="${RUNS_DIR}/hybrid_remax_brier"
mkdir -p "${HYBRID_REMAX_DIR}"

forecastbench grpo_train \
    --data-path "${HEADLINES_OUT}" \
    --out-dir "${HYBRID_REMAX_DIR}" \
    --model "Qwen/Qwen3-1.7B" \
    --algorithm remax \
    --reward-mode turtel_brier \
    --use-diffusion \
    --diffusion-steps 50 \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-6 \
    --samples-per-question 4 \
    --prompt-col turtel_prompt \
    --question-col question \
    --outcome-col outcome \
    --seed 42

# =============================================================================
# Step 5: Train with Blackwell-Aware Reward
# =============================================================================
log_info "Step 5: Training with Blackwell-aware reward..."

AR_BLACKWELL_DIR="${RUNS_DIR}/ar_blackwell_aware"
mkdir -p "${AR_BLACKWELL_DIR}"

forecastbench grpo_train \
    --data-path "${HEADLINES_OUT}" \
    --out-dir "${AR_BLACKWELL_DIR}" \
    --model "Qwen/Qwen3-1.7B" \
    --algorithm remax \
    --reward-mode blackwell_aware \
    --blackwell-lambda 0.1 \
    --blackwell-n-bins 10 \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-6 \
    --samples-per-question 4 \
    --prompt-col turtel_prompt \
    --question-col question \
    --outcome-col outcome \
    --seed 42

HYBRID_BLACKWELL_DIR="${RUNS_DIR}/hybrid_blackwell_aware"
mkdir -p "${HYBRID_BLACKWELL_DIR}"

forecastbench grpo_train \
    --data-path "${HEADLINES_OUT}" \
    --out-dir "${HYBRID_BLACKWELL_DIR}" \
    --model "Qwen/Qwen3-1.7B" \
    --algorithm remax \
    --reward-mode blackwell_aware \
    --blackwell-lambda 0.1 \
    --blackwell-n-bins 10 \
    --use-diffusion \
    --diffusion-steps 50 \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-6 \
    --samples-per-question 4 \
    --prompt-col turtel_prompt \
    --question-col question \
    --outcome-col outcome \
    --seed 42

# =============================================================================
# Step 6: Evaluate All Models
# =============================================================================
log_info "Step 6: Evaluating all models..."

EVAL_DIR="${RUNS_DIR}/eval"
mkdir -p "${EVAL_DIR}"

for model_dir in "${AR_REMAX_DIR}" "${HYBRID_REMAX_DIR}" "${AR_BLACKWELL_DIR}" "${HYBRID_BLACKWELL_DIR}"; do
    model_name=$(basename "${model_dir}")
    log_info "Evaluating ${model_name}..."
    
    forecastbench pm_eval \
        --model-path "${model_dir}/checkpoint-best" \
        --data-path "${HEADLINES_OUT}" \
        --out "${EVAL_DIR}/${model_name}_eval.json" \
        --prompt-col turtel_prompt \
        --question-col question \
        --outcome-col outcome \
        --compute-blackwell-metrics \
        --compute-arbitrage-metrics || log_warn "Eval failed for ${model_name}"
done

# =============================================================================
# Step 7: Generate Comparison Report
# =============================================================================
log_info "Step 7: Generating comparison report..."

REPORT_FILE="${RUNS_DIR}/turtel_exa_comparison_report.md"

cat > "${REPORT_FILE}" << 'EOF'
# Turtel Exa Comparison Report

**Generated:** $(date)

## Experimental Setup

This experiment uses the exact same setup as Turtel et al. (2025):
- **Headlines Source:** Exa.ai
- **Temporal Controls:** Prediction date sampled uniformly between market open/close
- **No Leakage:** Headlines strictly before prediction date
- **Training Algorithm:** ReMax
- **Reward Function:** Brier Score (primary), Blackwell-aware (comparison)

## Models Compared

| Model | Description | Hypothesis |
|-------|-------------|------------|
| AR + ReMax + Brier | Turtel baseline (AR only) | H1 baseline |
| AR+Diff + ReMax + Brier | Our hybrid model | H1: Diffusion improves AR |
| AR + ReMax + Blackwell | Blackwell-aware reward | H2/H3: Constraint learning |
| AR+Diff + ReMax + Blackwell | Hybrid + Blackwell | H3: Diffusion learns C_t |

## Results

(Results will be populated by pm_compare)

## Files

- Evaluation results: `eval/`
- Model checkpoints: `{model_name}/`
- Enriched data: `data/polymarket/turtel_exa_enriched.parquet`

EOF

log_info "Report template created: ${REPORT_FILE}"

# =============================================================================
# Step 8: Run Blackwell Constraint Comparison
# =============================================================================
log_info "Step 8: Running Blackwell constraint convergence comparison..."

forecastbench blackwell_compare \
    --ar-model "${AR_REMAX_DIR}/checkpoint-best" \
    --hybrid-model "${HYBRID_REMAX_DIR}/checkpoint-best" \
    --data-path "${HEADLINES_OUT}" \
    --out-dir "${RUNS_DIR}/blackwell_comparison" \
    --n-groups 10 \
    --n-bins 10 || log_warn "Blackwell comparison failed"

# =============================================================================
# Done!
# =============================================================================
log_info "=============================================="
log_info "Turtel Exa Comparison Complete!"
log_info "=============================================="
log_info "Results: ${RUNS_DIR}"
log_info "Report:  ${REPORT_FILE}"
log_info ""
log_info "Key files:"
log_info "  - Evaluations: ${EVAL_DIR}/"
log_info "  - Blackwell:   ${RUNS_DIR}/blackwell_comparison/"
log_info "  - Data:        ${HEADLINES_OUT}"

