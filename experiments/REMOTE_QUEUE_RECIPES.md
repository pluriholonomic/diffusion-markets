# Remote queue recipes (2×H200)

The repo ships a small **local queue worker** that dispatches jobs to the remote GPU box over SSH:
- `scripts/remote_gpu_enqueue.py`: appends tasks to `remote_queue.jsonl`
- `scripts/remote_gpu_queue.py`: watches the queue, checks GPU idleness, and launches tasks with `CUDA_VISIBLE_DEVICES=<gpu>`

Important constraint: **each queued task sees exactly one GPU** (via `CUDA_VISIBLE_DEVICES`), so “use both H200s” means **enqueue two tasks pinned to GPU 0 and GPU 1**.

## 0) Start / verify the queue worker (on your laptop)

```bash
cd experiments
nohup python3 scripts/remote_gpu_queue.py \
  --host root@95.133.252.72 --port 22 \
  --remote-workdir /root/diffusion-markets/experiments \
  --queue-file /Users/tarunchitra/repos/diffusion-markets/experiments/remote_queue.jsonl \
  --state-file /Users/tarunchitra/repos/diffusion-markets/experiments/remote_logs/remote_gpu_queue_state.jsonl \
  --poll-s 30 \
  > remote_logs/local_remote_gpu_queue.log 2>&1 &
```

## 1) Keep both H200s busy: diffusion/learned-$C_t$ vs RLVR

### GPU0: learned-$C_t$ + arb diagnostics (bundle diffusion)

Example (requires a trained bundle diffusion checkpoint and a dataset with `market_prob`):

```bash
python3 scripts/remote_gpu_enqueue.py \
  --queue-file remote_queue.jsonl \
  --id pm_learnedCt_arb_gpu0 --gpu 0 \
  --cmd '.venv/bin/python -m forecastbench pm_learnedCt_arb \
    --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
    --model-path runs/*_pm_bundle_difftrain/model.pt \
    --bundle-col topic --bundle-size 8 --bundle-drop-last \
    --embed-model sentence-transformers/all-MiniLM-L6-v2 --embed-device cuda --embed-dtype float16 \
    --device cuda --sample-steps 32 --mc 32 --agg mean \
    --transaction-cost 0.0 --B 1.0 \
    --repair-at-resolution --repair-group-cols volume_q5,ttc_q5 \
    --run-name pm_learnedCt_arb'
```

Notes:
- `pm_learnedCt_arb` computes **bundle-level learned-$C_t$** and online arb estimates (Hedge + optional neural witness), then also reports standard scalar metrics + repair baseline.
- You may want to replace the wildcard `runs/*_pm_bundle_difftrain/model.pt` with an explicit path (recommended).

### GPU1: RLVR training (LoRA-first)

```bash
python3 scripts/remote_gpu_enqueue.py \
  --queue-file remote_queue.jsonl \
  --id pm_rlvr_train_gpu1 --gpu 1 \
  --cmd '.venv/bin/python -m forecastbench pm_rlvr_train \
    --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
    --text-cols question,description \
    --model-name-or-path Qwen/Qwen3-14B \
    --device cuda --dtype auto --device-map auto \
    --steps 200 --batch-size 4 --lr 1e-4 \
    --alpha-logscore 1.0 --beta-pnl 0.1 --trading-mode linear \
    --transaction-cost 0.0 --B 1.0 \
    --kl-coef 0.02 --reward-clip 10.0 --baseline-ema 0.95 \
    --run-name pm_rlvr_train'
```

## 2) Evaluate the trained RLVR adapter (after training finishes)

You’ll need the adapter output directory. By default `pm_rlvr_train` writes:
- `runs/<timestamp>_pm_rlvr_train/rlvr/final/` (adapter + tokenizer files)

Example evaluation:

```bash
python3 scripts/remote_gpu_enqueue.py \
  --queue-file remote_queue.jsonl \
  --id pm_rlvr_eval_gpu1 --gpu 1 \
  --cmd '.venv/bin/python -m forecastbench pm_rlvr_eval \
    --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
    --max-examples 512 \
    --text-cols question,description \
    --base-model Qwen/Qwen3-14B \
    --adapter-path runs/*_pm_rlvr_train/rlvr/final \
    --K 5 --agg median \
    --device cuda --device-map auto \
    --transaction-cost 0.0 --B 1.0 --trading-mode sign \
    --group-cols volume_q5,ttc_q5 \
    --approachability \
    --run-name pm_rlvr_eval'
```

## 3) Compare runs with bootstrap CIs

Once you have multiple evaluation run directories (each with `predictions.parquet`), you can compare:

```bash
python3 scripts/remote_gpu_enqueue.py \
  --queue-file remote_queue.jsonl \
  --id pm_compare_cpu --gpu any \
  --cmd '.venv/bin/python -m forecastbench pm_compare \
    --run-name pm_compare_suite \
    --model diffusion,runs/*_pm_suite_eval_diff \
    --model ar_base,runs/*_pm_suite_eval_ar \
    --model market,runs/*_pm_suite_eval_market,market_prob \
    --model rlvr,runs/*_pm_rlvr_eval \
    --baseline market \
    --n-boot 200 \
    --transaction-cost 0.0 --B 1.0 --trading-mode sign'
```

This writes a run with:\n
- `summary.csv` (point estimates + 95% bootstrap CIs)\n
- `plots/compare_bootstrap.png`\n
\n


