#!/bin/bash
# =============================================================================
# Fetch Exa Headlines (CPU-only, no GPU needed)
# =============================================================================
# Run this ONCE to enrich Polymarket data with Turtel-style headlines.
# The enriched data is then reused by all training experiments.
# =============================================================================

set -euo pipefail

export EXA_API_KEY="${EXA_API_KEY:-6e0dcafc-9ab0-4035-9c3e-250c22ae3715}"

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJ_ROOT}/data"

cd "${PROJ_ROOT}/experiments"

# Activate venv if available
if [[ -f "${PROJ_ROOT}/experiments/.venv/bin/activate" ]]; then
    source "${PROJ_ROOT}/experiments/.venv/bin/activate"
fi

# Use python -m forecastbench if forecastbench not in PATH
if ! command -v forecastbench &> /dev/null; then
    alias forecastbench=".venv/bin/python -m forecastbench"
    FORECASTBENCH=".venv/bin/python -m forecastbench"
else
    FORECASTBENCH="forecastbench"
fi

echo "[fetch_exa] Starting Exa headline enrichment..."
echo "[fetch_exa] EXA_API_KEY set: ${EXA_API_KEY:0:8}..."

# Find existing Polymarket data
PM_BASE=""
for candidate in \
    "/root/polymarket_data/derived/gamma_yesno_ready.parquet" \
    "${DATA_DIR}/polymarket/criterion_prices.parquet" \
    "${DATA_DIR}/polymarket/gamma_base.parquet"; do
    if [[ -f "$candidate" ]]; then
        PM_BASE="$candidate"
        break
    fi
done

if [[ -z "${PM_BASE}" ]]; then
    echo "[fetch_exa] No existing Polymarket data found. Building..."
    
    $FORECASTBENCH pm_build_gamma \
        --out "${DATA_DIR}/polymarket/gamma_base.parquet" \
        --num-workers 8 \
        --min-volume 50000 || true
    
    $FORECASTBENCH pm_build_criterion_prices \
        --input-parquet "${DATA_DIR}/polymarket/gamma_base.parquet" \
        --out "${DATA_DIR}/polymarket/criterion_prices.parquet" \
        --final-trade-col "lastTradePrice" \
        --volume-col "volume"
    
    PM_BASE="${DATA_DIR}/polymarket/criterion_prices.parquet"
fi

echo "[fetch_exa] Using base data: ${PM_BASE}"

# Enrich with Exa headlines
OUT_FILE="${DATA_DIR}/polymarket/turtel_exa_enriched.parquet"
CACHE_DIR="${DATA_DIR}/headlines_cache"

mkdir -p "${CACHE_DIR}"

echo "[fetch_exa] Enriching with Exa headlines..."
echo "[fetch_exa] Output: ${OUT_FILE}"
echo "[fetch_exa] Cache:  ${CACHE_DIR}"

$FORECASTBENCH pm_turtel_headlines \
    --input "${PM_BASE}" \
    --out "${OUT_FILE}" \
    --news-source exa \
    --sample-prediction-date \
    --window-days 7 \
    --max-articles 10 \
    --cache-dir "${CACHE_DIR}" \
    --seed 42

echo "[fetch_exa] Done! Enriched data saved to: ${OUT_FILE}"
echo "[fetch_exa] Headlines cached in: ${CACHE_DIR}"

# Print stats
python3 -c "
import pandas as pd
df = pd.read_parquet('${OUT_FILE}')
print(f'Total rows: {len(df)}')
print(f'With headlines: {(df[\"n_headlines\"] > 0).sum()}')
print(f'Coverage: {(df[\"n_headlines\"] > 0).mean():.1%}')
"

