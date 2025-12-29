#!/bin/bash
# Continuous CLOB snapshot collection
#
# Run in background:
#   nohup ./scripts/collect_clob_continuous.sh > logs/clob_collection.log 2>&1 &
#
# Collects snapshots every hour for the top 2000 tokens

cd "$(dirname "$0")/.."

INTERVAL=3600  # 1 hour
OUTPUT_DIR="data/clob_snapshots_historical"
TOKENS_FILE="data/tokens_to_fetch.txt"

mkdir -p "$OUTPUT_DIR" logs

echo "Starting continuous CLOB collection..."
echo "  Interval: ${INTERVAL}s"
echo "  Output: $OUTPUT_DIR"
echo "  Tokens: $(wc -l < $TOKENS_FILE) tokens"

while true; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DATE_DIR="$OUTPUT_DIR/$(date +%Y%m%d)"
    
    echo ""
    echo "=== Collecting at $TIMESTAMP ==="
    
    python3 << EOF
from py_clob_client.client import ClobClient
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

tokens = open('$TOKENS_FILE').read().strip().split('\n')
client = ClobClient(host="https://clob.polymarket.com")
out_dir = Path('$DATE_DIR')
out_dir.mkdir(parents=True, exist_ok=True)

results = []
ts = time.time()

for i, token_id in enumerate(tokens):
    try:
        book = client.get_order_book(token_id)
        bids = [(float(b.price), float(b.size)) for b in book.bids]
        asks = [(float(a.price), float(a.size)) for a in book.asks]
        
        snapshot = {
            'token_id': token_id,
            'timestamp': ts,
            'bids': [{'price': p, 'size': s} for p, s in sorted(bids, key=lambda x: -x[0])],
            'asks': [{'price': p, 'size': s} for p, s in sorted(asks, key=lambda x: x[0])],
        }
        
        with open(out_dir / f'{token_id}_{int(ts)}.json', 'w') as f:
            json.dump(snapshot, f)
        
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 1
        mid = (best_bid + best_ask) / 2
        
        results.append({
            'token_id': token_id,
            'timestamp': ts,
            'mid_price': mid,
            'spread_bps': 10000 * (best_ask - best_bid) / mid if mid > 0 else None,
        })
    except:
        pass
    
    time.sleep(0.05)

# Save summary
df = pd.DataFrame(results)
df.to_parquet(out_dir / f'summary_{int(ts)}.parquet', index=False)
print(f"Collected {len(results)} snapshots, saved to {out_dir}")
EOF

    echo "Sleeping ${INTERVAL}s..."
    sleep $INTERVAL
done


