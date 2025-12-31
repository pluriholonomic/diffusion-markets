#!/bin/bash
# Pull completed results from remote GPU server
# Usage: ./pull_results_loop.sh [interval_seconds]

REMOTE_HOST="root@95.133.252.72"
REMOTE_DIR="/root/diffusion-markets/experiments/runs"
LOCAL_DIR="/Users/tarunchitra/repos/diffusion-markets/experiments/runs"
INTERVAL=${1:-60}

mkdir -p "$LOCAL_DIR"

echo "[pull_results] Starting sync loop (interval=${INTERVAL}s)"
echo "[pull_results] Remote: $REMOTE_HOST:$REMOTE_DIR"
echo "[pull_results] Local: $LOCAL_DIR"

while true; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Syncing results..."
    
    # Sync only completed runs (those with predictions.parquet or model.pt)
    rsync -avz --progress \
        --include='*/' \
        --include='*.parquet' \
        --include='*.pt' \
        --include='*.json' \
        --include='*.txt' \
        --include='*.log' \
        --exclude='*' \
        "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync complete. Sleeping ${INTERVAL}s..."
    sleep "$INTERVAL"
done
