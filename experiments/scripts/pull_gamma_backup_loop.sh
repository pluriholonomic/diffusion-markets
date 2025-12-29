#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <ssh_host> <remote_gamma_dir> <local_dir> [interval_seconds]" >&2
  echo "example: $0 root@95.133.252.72 /root/polymarket_data/gamma polymarket_backups/gamma 300" >&2
  exit 2
fi

HOST="$1"
REMOTE_DIR="$2"
LOCAL_DIR="$3"
INTERVAL="${4:-300}"

mkdir -p "$LOCAL_DIR"

SSH="ssh -p 22 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "[backup] host=$HOST remote_dir=$REMOTE_DIR local_dir=$LOCAL_DIR interval_s=$INTERVAL"

while true; do
  date -u

  # Small files: sync normally (overwrite).
  rsync -az -e "$SSH" "$HOST:$REMOTE_DIR/progress.json" "$LOCAL_DIR/" || true
  rsync -az -e "$SSH" "$HOST:$REMOTE_DIR/download.log" "$LOCAL_DIR/" || true
  rsync -az -e "$SSH" "$HOST:$REMOTE_DIR/download.pid" "$LOCAL_DIR/" || true

  # Large append-only file: use --append so we only fetch new bytes.
  rsync -az --append -e "$SSH" "$HOST:$REMOTE_DIR/markets_raw.jsonl" "$LOCAL_DIR/" || true

  sleep "$INTERVAL"
done



