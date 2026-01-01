#!/bin/bash
# Periodic sync of stress test results from remote to local
# Runs every 10 minutes until stopped

REMOTE_HOST="root@95.133.252.72"
REMOTE_PORT="22"
REMOTE_DIR="/root/diffusion-markets/experiments"
LOCAL_DIR="/Users/tarunchitra/repos/diffusion-markets/experiments"

SYNC_INTERVAL=600  # 10 minutes

echo "========================================"
echo "STRESS TEST RESULTS SYNC"
echo "========================================"
echo "Remote: $REMOTE_HOST:$REMOTE_DIR"
echo "Local:  $LOCAL_DIR"
echo "Interval: ${SYNC_INTERVAL}s"
echo "Started: $(date)"
echo "========================================"
echo ""

sync_results() {
    echo ""
    echo "=== SYNC $(date) ==="
    
    # Create local directories
    mkdir -p "$LOCAL_DIR/runs/stress_test_impact"
    mkdir -p "$LOCAL_DIR/logs/stress_test"
    
    # Sync master results logs (most important)
    echo "Syncing master results..."
    rsync -avz --progress \
        -e "ssh -p $REMOTE_PORT" \
        "$REMOTE_HOST:$REMOTE_DIR/runs/stress_test_impact/*/master_results.jsonl" \
        "$LOCAL_DIR/runs/stress_test_impact/" 2>/dev/null || true
    
    # Sync model-specific results
    echo "Syncing model results..."
    rsync -avz \
        -e "ssh -p $REMOTE_PORT" \
        --include='*/' \
        --include='*_results.json' \
        --exclude='*' \
        "$REMOTE_HOST:$REMOTE_DIR/runs/stress_test_impact/" \
        "$LOCAL_DIR/runs/stress_test_impact/" 2>/dev/null || true
    
    # Sync portfolio results too
    echo "Syncing portfolio results..."
    rsync -avz \
        -e "ssh -p $REMOTE_PORT" \
        "$REMOTE_HOST:$REMOTE_DIR/runs/portfolio_optimization/" \
        "$LOCAL_DIR/runs/portfolio_optimization/" 2>/dev/null || true
    
    # Sync convergence results
    rsync -avz \
        -e "ssh -p $REMOTE_PORT" \
        "$REMOTE_HOST:$REMOTE_DIR/runs/convergence_parallel/" \
        "$LOCAL_DIR/runs/convergence_parallel/" 2>/dev/null || true
    
    # Quick status check
    echo ""
    echo "=== REMOTE STATUS ==="
    ssh -p $REMOTE_PORT $REMOTE_HOST << 'REMOTE_EOF'
cd /root/diffusion-markets/experiments
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "Stress workers: $(ps aux | grep stress_test | grep -v grep | wc -l)"
echo "Portfolio workers: $(ps aux | grep portfolio_strategies | grep -v grep | wc -l)"

total=0
for w in $(seq 1 60); do
    log="runs/stress_test_impact/worker_$w/master_results.jsonl"
    if [ -f "$log" ]; then
        count=$(wc -l < "$log" 2>/dev/null || echo 0)
        total=$((total + count))
    fi
done
echo "Total models tested: $total"
REMOTE_EOF

    # Local analysis
    echo ""
    echo "=== LOCAL ANALYSIS ==="
    local_results="$LOCAL_DIR/runs/stress_test_impact"
    if [ -d "$local_results" ]; then
        total_local=$(cat "$local_results"/worker_*/master_results.jsonl 2>/dev/null | wc -l || echo 0)
        echo "Local results: $total_local models"
        
        # Quick summary if we have results
        if [ "$total_local" -gt 0 ]; then
            python3 -c "
import json
import sys
from pathlib import Path

results = []
for f in Path('$local_results').glob('worker_*/master_results.jsonl'):
    with open(f) as fp:
        for line in fp:
            try:
                results.append(json.loads(line))
            except:
                pass

if results:
    avg = [r for r in results if r.get('regime') == 'average']
    adv = [r for r in results if r.get('regime') == 'adversarial']
    
    print(f'Average regime: {len(avg)} models, mean Sharpe={sum(r[\"final_mean_sharpe\"] for r in avg)/max(len(avg),1):.2f}')
    print(f'Adversarial:    {len(adv)} models, mean Sharpe={sum(r[\"final_mean_sharpe\"] for r in adv)/max(len(adv),1):.2f}')
    
    if results:
        min_r = min(results, key=lambda x: x.get('final_min_sharpe', 0))
        print(f'Worst case: {min_r[\"model_name\"]} with min Sharpe={min_r[\"final_min_sharpe\"]:.2f}')
" 2>/dev/null || echo "Analysis pending..."
        fi
    fi
    
    echo ""
    echo "Next sync in ${SYNC_INTERVAL}s..."
}

# Initial sync
sync_results

# Continuous sync loop
while true; do
    sleep $SYNC_INTERVAL
    sync_results
done
