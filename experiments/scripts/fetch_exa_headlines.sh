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

echo "[fetch_exa] Starting Exa headline enrichment..."
echo "[fetch_exa] EXA_API_KEY set: ${EXA_API_KEY:0:8}..."

# Ensure we have base Polymarket data
PM_BASE="${DATA_DIR}/polymarket/criterion_prices.parquet"
if [[ ! -f "${PM_BASE}" ]]; then
    echo "[fetch_exa] Building base Polymarket data first..."
    
    forecastbench pm_gamma_build \
        --out "${DATA_DIR}/polymarket/gamma_base.parquet" \
        --num-workers 8 \
        --min-volume 50000 || true
    
    forecastbench pm_criterion_build \
        --input-parquet "${DATA_DIR}/polymarket/gamma_base.parquet" \
        --out "${PM_BASE}" \
        --final-trade-col "lastTradePrice" \
        --volume-col "volume"
fi

echo "[fetch_exa] Using base data: ${PM_BASE}"

# Enrich with Exa headlines
OUT_FILE="${DATA_DIR}/polymarket/turtel_exa_enriched.parquet"
CACHE_DIR="${DATA_DIR}/headlines_cache"

mkdir -p "${CACHE_DIR}"

echo "[fetch_exa] Enriching with Exa headlines..."
echo "[fetch_exa] Output: ${OUT_FILE}"
echo "[fetch_exa] Cache:  ${CACHE_DIR}"

forecastbench pm_turtel_headlines \
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

