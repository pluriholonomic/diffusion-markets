### Criterion-based `market_prob` (brier.fyi / Themis alignment)

This repo evaluates Polymarket-style datasets where each row represents a resolved market with:

- `y`: realized outcome / resolution in \([0,1]\)
- `pred_prob`: a model prediction (LLM / diffusion)
- `market_prob`: a market-implied probability used as a baseline (and as the “traded price” for the simple trading proxy)

Historically in this codebase, `market_prob` could mean **different snapshots** depending on how the dataset was built
(`pm_enrich_clob` vs `pm_build_horizon_prices`). That made it easy to accidentally compare results that used different
“market probability at time \(t\)” conventions.

The brier.fyi pipeline (Project Themis) makes this explicit: it computes market scores using **criterion probabilities**
derived from the full probability time-series, such as:

- `midpoint`
- `time-average`
- `before-close-days-7`
- `before-close-days-30`

See Themis source: `extract/src/criteria.rs` and `grader/src/scores.rs` in [`wasabipesto/themis`](https://github.com/wasabipesto/themis).

#### What changed

We now support building datasets where `market_prob` is computed from full CLOB history using a **Themis-style criterion name**:

- New CLI: `forecastbench pm_build_criterion_prices`
- New module: `forecastbench.data.criterion_build`

The output dataset includes:

- `criterion`: the criterion name used (e.g. `midpoint`)
- `history_start_ts`, `history_end_ts`: open/close timestamps inferred from the history segments
- `market_prob_target_ts`: the criterion timestamp (e.g. midpoint time)
- `market_prob_ts`: the actual segment-start timestamp used for the sampled value (for piecewise-constant histories)
- `market_event_ts`: alias of `history_end_ts` (used by repair-at-resolution simulation)
- `market_prob`: the criterion probability (this is what pm_eval uses for the market baseline)

#### Dataset built locally in this repo

Built a midpoint-criterion dataset from the existing local CLOB histories:

```bash
cd /Users/tarunchitra/repos/diffusion-markets/experiments
PYTHONPATH=src python -m forecastbench pm_build_criterion_prices \
  --input data/polymarket/gamma_yesno_resolved.parquet \
  --clob-history-dir data/polymarket/clob_history_yes_f1 \
  --out data/polymarket/pm_criterion_midpoint_f1.parquet \
  --criterion midpoint
```

This produced **5,000 rows** (the subset of the 50k-row Gamma file that had downloaded CLOB histories).

#### Do we need to rerun experiments?

- **No retraining is required** for diffusion/LLM models.
- If you want “market baseline Brier” comparable to brier.fyi’s “brier-midpoint”, you should **re-evaluate** using a dataset
  built with `--criterion midpoint` (or your chosen criterion).




