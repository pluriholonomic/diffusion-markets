#!/bin/bash
# Check status of diffusion training and Exa data

echo "============================================================"
echo "STATUS CHECK: $(date)"
echo "============================================================"

# Check if training is running
TRAIN_PID=$(pgrep -f "train_proper_diffusion.py")
if [ -n "$TRAIN_PID" ]; then
    echo ""
    echo "✓ Training is RUNNING (PID: $TRAIN_PID)"
    echo ""
    echo "Latest training output:"
    tail -20 runs/training_log_*.log 2>/dev/null | grep -E "step|loss|Final|VALIDATION|corr"
else
    echo ""
    echo "✗ Training is NOT running"
    
    # Check if it completed
    LATEST_RUN=$(ls -1td runs/proper_diffusion_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "  Latest completed run: $LATEST_RUN"
        if [ -f "$LATEST_RUN/validation.json" ]; then
            echo "  Validation results:"
            cat "$LATEST_RUN/validation.json"
        fi
    fi
fi

# Check Exa data status
echo ""
echo "------------------------------------------------------------"
echo "EXA DATA STATUS:"
if [ -f "data/polymarket/turtel_exa_enriched.parquet" ]; then
    echo "✓ Exa enriched data EXISTS"
    python3 -c "import pandas as pd; df = pd.read_parquet('data/polymarket/turtel_exa_enriched.parquet'); print(f'  Rows: {len(df):,}')"
else
    echo "✗ Exa enriched data NOT YET READY"
    echo "  Waiting for: data/polymarket/turtel_exa_enriched.parquet"
fi

echo ""
echo "------------------------------------------------------------"
echo "AVAILABLE DATA:"
for f in data/polymarket/*.parquet; do
    if [ -f "$f" ]; then
        rows=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$f')))" 2>/dev/null)
        echo "  $(basename $f): $rows rows"
    fi
done | head -10

echo "============================================================"



