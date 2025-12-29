#!/bin/bash
# Pull completed results from remote GPU server

REMOTE_HOST="root@95.133.252.72"
REMOTE_DIR="/root/diffusion-markets/experiments/runs"
LOCAL_DIR="/Users/tarunchitra/repos/diffusion-markets/experiments/runs"
INTERVAL=${1:-60}

mkdir -p "$LOCAL_DIR"

echo "[pull_results] Starting loop: remote=$REMOTE_HOST:$REMOTE_DIR local=$LOCAL_DIR interval=${INTERVAL}s"

while true; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Syncing results..."
    
    # Sync all run directories
    rsync -avz --progress \
        --exclude='*.pt' \
        --exclude='checkpoint-*' \
        "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/" 2>&1 | tail -5
    
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync complete. Waiting ${INTERVAL}s..."
    sleep "$INTERVAL"
done
