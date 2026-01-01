#!/bin/bash
# Monitor for 20K eval completion and sync results

REMOTE="root@95.133.252.72"
PORT=22
REMOTE_DIR="/root/diffusion-markets/experiments"
LOCAL_DIR="/Users/tarunchitra/repos/diffusion-markets/experiments/remote_results"

echo "[$(date)] Starting completion monitor..."

while true; do
    # Check if 20K eval jobs are still running
    RUNNING=$(ssh -p $PORT $REMOTE 'ps aux | grep "eval.*20k" | grep -v grep | wc -l' 2>/dev/null)
    
    if [ "$RUNNING" = "0" ]; then
        echo "[$(date)] 20K evals completed! Syncing results..."
        
        # Sync lightweight files
        rsync -avz --progress -e "ssh -p $PORT" \
            --include='*/' \
            --include='*.parquet' \
            --include='*.json' \
            --include='*.jsonl' \
            --include='*.log' \
            --include='*.txt' \
            --include='*.csv' \
            --exclude='*.safetensors' \
            --exclude='*.bin' \
            --exclude='*.pt' \
            --exclude='*.pth' \
            --exclude='*' \
            $REMOTE:$REMOTE_DIR/runs/ \
            $LOCAL_DIR/
        
        echo "[$(date)] Sync complete!"
        
        # Show what was synced
        echo ""
        echo "=== New 20K Results ==="
        ls -la $LOCAL_DIR | grep "20k"
        
        # Send notification (macOS)
        osascript -e 'display notification "20K evaluations complete and synced!" with title "Experiment Results"' 2>/dev/null
        
        break
    else
        echo "[$(date)] $RUNNING eval jobs still running. Checking again in 5 minutes..."
        sleep 300
    fi
done
