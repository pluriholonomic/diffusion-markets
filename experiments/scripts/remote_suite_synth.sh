#!/usr/bin/env bash
set -euo pipefail

cd /root/diffusion-markets/experiments
export PYTHONPATH=src

mkdir -p remote_logs

echo "[suite] start: $(date -u)"
echo "[suite] host: $(hostname)"
echo "[suite] python: $(.venv/bin/python -V)"
echo "[suite] torch/cuda:"
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("gpus", torch.cuda.device_count())'

# Parity sweep (analytic + junta baseline)
for k in 4 6 8 10; do
  echo "[suite] parity k=$k"
  .venv/bin/python -m forecastbench parity \
    --d 24 --k "$k" --alpha 0.8 --n 80000 --rho 0.9 --L 4 \
    --run-name "suite_parity_k${k}"
done

# Intrinsic vs post-processing control (no LLM)
for k in 6 8 10; do
  echo "[suite] intrinsic_post k=$k"
  .venv/bin/python -m forecastbench intrinsic_post \
    --d 24 --k "$k" --alpha 0.8 --rho 0.9 --L 4 \
    --n-train 20000 --n-test 5000 \
    --post-bins 20 --post-prior 5.0 \
    --run-name "suite_intrinsic_post_k${k}_n20k"
done

# Learned diffusion sanity sweeps (GPU)
echo "[suite] difftrain (binary logit diffusion)"
.venv/bin/python -m forecastbench difftrain \
  --d 24 --k 12 --alpha 0.8 \
  --n-train 50000 --n-test 10000 \
  --train-steps 1000 --batch-size 1024 \
  --T 64 --sample-steps 32 --device cuda \
  --run-name suite_difftrain_bin

echo "[suite] difftrain_simplex (ALR simplex diffusion)"
.venv/bin/python -m forecastbench difftrain_simplex \
  --d 24 --k 12 --n-outcomes 4 --alpha 1.0 \
  --n-train 50000 --n-test 10000 \
  --train-steps 1000 --batch-size 1024 \
  --T 64 --sample-steps 32 --device cuda \
  --run-name suite_difftrain_simplex

echo "[suite] done: $(date -u)"


