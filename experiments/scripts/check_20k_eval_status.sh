#!/bin/bash
# Quick status check for 20K evaluations

echo "=== 20K Evaluation Status ==="
echo "Time: $(date)"
echo ""

ssh -p 22 root@95.133.252.72 "cd diffusion-markets/experiments && \
echo '=== Running Processes ===' && \
ps aux | grep '.venv/bin/python' | grep -v grep | awk '{print \$2, \$10, \$12}' | head -5 && \
echo '' && \
echo '=== GPU Usage ===' && \
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv && \
echo '' && \
echo '=== Check for completed predictions ===' && \
for name in eval_rlcr_fixed_20k eval_ar_baseline_20k; do
  f=\$(find runs -maxdepth 2 -name 'predictions.parquet' -path \"*\${name}*\" 2>/dev/null | head -1)
  if [ -n \"\$f\" ]; then
    echo \"\$name: COMPLETE (\$f)\"
    ls -lh \"\$f\"
  else
    echo \"\$name: Still running...\"
  fi
done && \
echo '' && \
echo '=== Queue worker log (last 5 lines) ===' && \
tail -5 remote_logs/queue_worker.log 2>/dev/null"
