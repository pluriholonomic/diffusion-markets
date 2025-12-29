# Diffusion Markets Experiments

This folder contains a **reproducible benchmarking harness** for the experiments proposed in `main.tex`:

- **Synthetic Truth functions** on the Boolean cube (e.g. parity markets) and their predicted separations (AR depth “cliff” vs diffusion “fog”).
- **Small-group stress tests** over subcubes \(G_{S,a}\), including an explicit check of the “only \(S=J\) contributes” fact in the parity proof (see `main.tex` around the step “Since \(|S|=|J|=k\), this requires \(S=J\)”).
- **Real-data evaluation (no training loop)** on Polymarket-style datasets, aligned to the evaluation-only intent in `main.tex` and the paper it cites.

## Setup

### Local (CPU / laptop)

```bash
cd experiments
python -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"
```

### GPU box (multi-GPU H100)

```bash
cd experiments
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu]"
```

Notes:
- The `gpu` extra enables `torch`, `transformers`, and `accelerate`. You’ll still want a CUDA-enabled torch build.
- For large LLMs (e.g. 14B+), pass `--llm-device-map auto` to shard across multiple GPUs (requires `accelerate`).

## Quickstart (synthetic, local)

Parity benchmark (runs without any LLM weights; uses analytic predictors + explicit L-query baselines):

```bash
forecastbench parity --d 16 --k 8 --alpha 0.8 --n 50000 --rho 0.95 --L 4
```

Small-group stress test (enumerates subcubes of codimension `k` and computes worst-group calibration):

```bash
forecastbench groupstress --d 14 --k 7 --alpha 0.8 --n 200000 --rho 0.9 --L 3
```

Artifacts are written to `experiments/runs/<timestamp>_<run_name>/`.

## Polymarket data sources

This harness supports multiple ways to obtain Polymarket market metadata and timeseries:

- **Polymarket subgraphs (Goldsky-hosted GraphQL)**: see the official docs at
  [docs.polymarket.com — Subgraph overview](https://docs.polymarket.com/developers/subgraph/overview?utm_source=chatgpt.com).
- **PolyData Explorer downloads** (JSON exports you download manually):
  [polydataexplore.org](https://polydataexplore.org/?utm_source=chatgpt.com).
- **Python odds fetcher** (`polymarket` on PyPI):
  [pypi.org/project/polymarket](https://pypi.org/project/polymarket/?utm_source=chatgpt.com).

The CLI expects an **offline dataset** (Parquet/JSONL) for evaluation; ingestion helpers can build that dataset from any of the above sources.

## Commands

- `forecastbench parity`: synthetic parity markets + score/calibration/arbitrage metrics.
- `forecastbench groupstress`: subcube group calibration and the parity “only \(S=J\)” diagnostic.
- `forecastbench intrinsic_post`: synthetic control comparing intrinsic calibration vs post-processing (group/bin wrapper on the true parity partition).
- `forecastbench difftrain`: train a tiny learned logit-diffusion model on parity (local sanity check).
- `forecastbench difftrain_simplex`: train a tiny learned diffusion model for **simplex outputs** using the **ALR transform** (multi-outcome sanity check).
- `forecastbench pm_build_polydata`: convert a PolyData Explorer JSON export to the minimal dataset schema.
- `forecastbench pm_build_subgraph`: run a user-supplied GraphQL query against a Polymarket subgraph and coerce to the minimal schema.
- `forecastbench pm_download_gamma`: download Polymarket market metadata via the public Gamma API into an append-only JSONL (good for long runs + checkpointing).
- `forecastbench pm_build_gamma`: build a Parquet yes/no dataset (with labels + CLOB token IDs) from a Gamma dump.
- `forecastbench pm_enrich_clob`: add a **first-available post-open market probability** from the Polymarket CLOB (useful for a quick market baseline, but not aligned to brier.fyi criteria).
- `forecastbench pm_download_clob_history`: download full CLOB price histories (per token) for later criterion/horizon sampling.
- `forecastbench pm_build_horizon_prices`: build a dataset where `market_prob` is sampled at a fixed horizon before close (e.g. 7d before close).
- `forecastbench pm_build_criterion_prices`: build a dataset where `market_prob` is computed using a **Themis/brier.fyi-style criterion** (e.g. `midpoint`, `time-average`, `before-close-days-30`).
- `forecastbench pm_difftrain`: diffusion-side baseline on Polymarket dataset (text→prob), evaluated on a held-out split.
- `forecastbench pm_eval`: evaluate a dataset (optionally run an HF LLM to produce `pred_prob` first; evaluation-only).

## Remote GPU box workflow (recommended)

- **Download Polymarket data to a path outside `experiments/`** (so it won’t get clobbered by code sync):

```bash
PYTHONPATH=src .venv/bin/python -m forecastbench pm_download_gamma --out-dir /root/polymarket_data/gamma
```

- **Back up to your local machine every 5 minutes** (works with macOS’s older `rsync`):

```bash
bash scripts/pull_gamma_backup_loop.sh \
  root@95.133.252.72 \
  /root/polymarket_data/gamma \
  polymarket_backups/gamma
```

### Remote queue (2×GPU) workflow

This repo includes a simple SSH-based queue to keep a multi-GPU box saturated using one job per GPU.

See: [`REMOTE_QUEUE_RECIPES.md`](REMOTE_QUEUE_RECIPES.md).


