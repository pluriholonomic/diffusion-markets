#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <ssh_host> <remote_experiments_dir> <local_experiments_dir> [interval_seconds]" >&2
  echo "example: $0 root@95.133.252.72 /root/diffusion-markets/experiments /Users/tarunchitra/repos/diffusion-markets/experiments 120" >&2
  exit 2
fi

HOST="$1"
REMOTE_EXP="$2"
LOCAL_EXP="$3"
INTERVAL="${4:-300}"

mkdir -p "$LOCAL_EXP/data/polymarket" "$LOCAL_EXP/remote_logs"
mkdir -p "$LOCAL_EXP/data/polymarket/derived"
mkdir -p "$LOCAL_EXP/data/polymarket/gdelt_cache_h24" "$LOCAL_EXP/data/polymarket/gdelt_cache_d7"

SSH="ssh -p 22 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "[pull] host=$HOST remote_exp=$REMOTE_EXP local_exp=$LOCAL_EXP interval_s=$INTERVAL"

while true; do
  date -u

  # Resolved yes/no dataset built from Gamma dump (pm_build_gamma)
  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/data/polymarket/gamma_yesno_resolved.parquet" \
    "$LOCAL_EXP/data/polymarket/" || true

  # Derived datasets (e.g. horizon snapshots + news-enriched variants)
  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/data/polymarket/derived/" \
    "$LOCAL_EXP/data/polymarket/derived/" || true

  # GDELT caches (for reproducibility + resume; best-effort)
  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/data/polymarket/gdelt_cache_h24/" \
    "$LOCAL_EXP/data/polymarket/gdelt_cache_h24/" || true

  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/data/polymarket/gdelt_cache_d7/" \
    "$LOCAL_EXP/data/polymarket/gdelt_cache_d7/" || true

  # High-frequency YES-token CLOB history (pm_download_clob_history)
  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/data/polymarket/clob_history_yes_f1/" \
    "$LOCAL_EXP/data/polymarket/clob_history_yes_f1/" || true

  # Collector logs (best-effort)
  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/remote_logs/pm_download_clob_hist.log" \
    "$LOCAL_EXP/remote_logs/" || true

  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/remote_logs/pm_news_h24.log" \
    "$LOCAL_EXP/remote_logs/" || true

  rsync -az -e "$SSH" \
    "$HOST:$REMOTE_EXP/remote_logs/pm_news_d7.log" \
    "$LOCAL_EXP/remote_logs/" || true

  sleep "$INTERVAL"
done


