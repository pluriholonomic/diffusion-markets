# Diffusion Markets: Experimental Plan and Results

> **Status**: 🔄 Experiments in progress  
> **Last updated**: 2025-12-28  
> **Queue status**: `idle` on remote H200

---

## 1. Abstract

**Title**: *Diffusion Models as Correlation Learners: Bounding Statistical Arbitrage via Blackwell Approachability*

Statistical arbitrage in prediction markets arises when prices across related markets are inconsistent with the underlying correlation structure. We formalize this through **Blackwell approachability**: the time-dependent set $C_t$ of probability vectors consistent with observed correlations defines a "no-arbitrage region," and the distance $d(q, C_t)$ from market prices $q$ to this set bounds extractable profit.

We prove that **diffusion models learn $C_t$ through their denoising process**. At noise level $\sigma$, the model learns a relaxed constraint set $C_\sigma$, where higher noise encodes coarser constraints (marginal probabilities, pairwise correlations) and lower noise refines toward the full correlation polytope. This "constraint ladder" $C_{\sigma_1} \supset C_{\sigma_2} \supset \cdots \supset C_0$ enables:

- **Arbitrage detection**: The diffusion repair step projects market prices onto $C_t$; the projection distance $d(q, C_t)$ upper-bounds extractable profit.
- **Trade direction**: The projection vector points toward the arbitrage-correcting position.
- **Risk quantification**: Unlike point-estimate models, $d(q, C_t)$ provides calibrated uncertainty about market inefficiency.

We propose a **hybrid AR+Diffusion architecture**: a large language model generates initial forecasts via reasoning, and a lightweight diffusion head (1M parameters) repairs them to lie in $C_t$.

---

## 2. Hypotheses

### H1: Diffusion Repair Improves AR Performance
> The diffusion repair module dramatically improves the performance of the AR module on calibration, worst-group error, and economic metrics.

**Metrics**: Brier score, ECE, worst-group ECE, ROI, Sharpe ratio

**Null hypothesis**: Adding diffusion does not significantly improve over AR-only.

### H2: Blackwell Approachability Lens
> The models/setup can be viewed from the lens of Blackwell approachability—predictions should converge to the approachability set $C_\epsilon$ at rate $O(1/\sqrt{T})$.

**Metrics**: AppErr_t curves, convergence exponent, constraint satisfaction at each degree

**Null hypothesis**: Convergence is not consistent with Blackwell theory.

### H3: Diffusion → Better Blackwell Constraints
> The diffusion model improving performance implies that we're better learning the Blackwell constraints (the correlation structure $C_t$).

**Metrics**: 
- Constraint violation reduction (AR vs AR+Diff)
- Arbitrage capture rate
- Distance-profit correlation $r^2$
- Constraint ladder monotonicity

**Null hypothesis**: Diffusion improvement is not due to better constraint learning.

### H4: Arbitrage Bound via $d(q, C_t)$
> The distance from market prices to the learned constraint set $C_t$ bounds extractable statistical arbitrage.

**Metrics**: Correlation between $d(q, C_t)$ and realized PnL

**Null hypothesis**: Distance to $C_t$ does not predict arbitrage profit.

---

## 3. Methodology

### 3.1 Synthetic Experiments (Theory Validation)

These experiments use synthetic data where the true $C_t$ is known analytically.

| Experiment | Command | Code Location | Purpose |
|------------|---------|---------------|---------|
| **Parity benchmark** | `forecastbench parity --blackwell` | `src/forecastbench/benchmarks/parity.py` | Test AR cliff vs diffusion fog |
| **Approachability suite** | `forecastbench approachability_suite` | `src/forecastbench/metrics/multiscale_approachability.py` | Constraint ladder at different degrees |
| **Cliff vs Fog** | `forecastbench cliff_fog` | `src/forecastbench/benchmarks/cliff_fog.py` | AR spectral cliff vs diffusion continuous recovery |
| **Group robustness** | `forecastbench group_robustness` | `src/forecastbench/benchmarks/group_robustness.py` | Propositions 8-9 on small subgroups |
| **Swap regret** | `forecastbench swap_regret` | `src/forecastbench/metrics/swap_regret.py` | External vs swap regret comparison |

### 3.2 Turtel-Compatible Headlines Pipeline

For an exact apples-to-apples comparison with Turtel et al. (2025), we implement their headline enrichment approach:

| Feature | Turtel Approach | Our Implementation |
|---------|-----------------|-------------------|
| **News Source** | Exa.ai | `--news-source exa` (same API) |
| **Prediction Date** | Sampled uniformly between open/close | `--sample-prediction-date` |
| **Temporal Control** | Headlines strictly before prediction date | Built-in (no future leakage) |
| **Leakage Verification** | LLM check for leaked future info | `--verify-no-leakage` (optional) |

**CLI Command:**
```bash
forecastbench pm_turtel_headlines \
    --input data/polymarket/raw.parquet \
    --out data/polymarket/turtel_enriched.parquet \
    --news-source exa \
    --sample-prediction-date \
    --window-days 7 \
    --max-articles 10
```

**Code Location:** `src/forecastbench/data/turtel_headlines.py`

### 3.3 Training Experiments

| Experiment | Command | Code Location | Purpose |
|------------|---------|---------------|---------|
| **GRPO/ReMax training** | `forecastbench grpo_train` | `src/forecastbench/train/grpo.py` | Train AR with Turtel/Damani methods |
| **Diffusion training** | `forecastbench pm_difftrain` | `src/forecastbench/models/diffusion_core.py` | Train diffusion baseline |
| **Hybrid training** | `forecastbench pm_hybrid_train` | `src/forecastbench/models/ar_diffusion_hybrid.py` | Train AR+Diffusion combined |
| **RLVR training** | `forecastbench pm_rlvr_train` | `src/forecastbench/train/rlvr.py` | Legacy REINFORCE training |

### 3.3 Reward Functions

| Reward Mode | Formula | Code Reference | Paper |
|-------------|---------|----------------|-------|
| `turtel_brier` | $R = -(p - y)^2$ | `grpo.py:_brier_reward()` | Turtel et al. 2025 |
| `rlcr` | $R = \mathbf{1}_{correct} - (p-y)^2 - \gamma \cdot \text{group\_viol}$ | `grpo.py:_rlcr_reward()` | Damani et al. 2025 |
| `kelly` | $R = \log(1 + f \cdot \text{payoff})$ | `grpo.py:_kelly_reward()` | Kelly criterion |
| `blackwell_aware` | $R = -(p-y)^2 - \lambda \cdot \text{constraint\_viol}$ | `grpo.py:_blackwell_aware_reward()` | Ours |

### 3.4 Evaluation

| Experiment | Command | Code Location | Purpose |
|------------|---------|---------------|---------|
| **Polymarket eval** | `forecastbench pm_eval` | `src/forecastbench/benchmarks/polymarket_eval.py` | Full evaluation suite |
| **Turtel comparison** | `forecastbench turtel_compare` | `src/forecastbench/benchmarks/turtel_comparison.py` | Compare to Turtel baseline |
| **Model comparison** | `forecastbench pm_compare` | `src/forecastbench/cli.py:cmd_pm_compare` | Bootstrap CI comparison |

### 3.5 Arbitrage Analysis

| Function | Location | Purpose |
|----------|----------|---------|
| `compute_arbitrage_bound()` | `metrics/multiscale_approachability.py` | Compute $d(q, C_t)$ as arbitrage bound |
| `compare_arbitrage_detection()` | `metrics/multiscale_approachability.py` | AR vs Hybrid arbitrage capture |
| `constraint_ladder_analysis()` | `metrics/multiscale_approachability.py` | Validate noise = constraint ladder |
| `BlackwellConstraintTracker` | `metrics/multiscale_approachability.py` | Online constraint tracking |

### 3.6 Key References

**Papers**:
- Turtel et al. (2025): "Outcome-based RL to Predict the Future" ([arXiv:2505.17989](https://arxiv.org/abs/2505.17989))
- Damani et al. (2025): "Beyond Binary Rewards" ([arXiv:2507.16806](https://arxiv.org/abs/2507.16806))

**Our code**:
- Main CLI: `experiments/src/forecastbench/cli.py`
- GRPO training: `experiments/src/forecastbench/train/grpo.py`
- Approachability metrics: `experiments/src/forecastbench/metrics/multiscale_approachability.py`
- Hybrid model: `experiments/src/forecastbench/models/ar_diffusion_hybrid.py`

---

## 4. Experiments Run

### 4.1 Completed Experiments

| Date | Experiment | Status | Run Directory | Key Results |
|------|------------|--------|---------------|-------------|
| 2025-12-23 | `remote_suite_synth` | ✅ Complete | `remote_runs/suite/` | Parity k=4,6,8,10 baseline |
| 2025-12-24 | `pm_suite_difftrain` | ✅ Complete | `remote_runs/pm_suite/` | Diffusion baseline on PM |
| 2025-12-24 | `pm_suite_eval_ar` | ✅ Complete | `remote_runs/pm_suite/` | AR (Qwen3-14B) baseline |
| 2025-12-24 | `pm_suite_eval_market` | ✅ Complete | `remote_runs/pm_suite/` | Market price baseline |

### 4.2 In Progress

| Date | Experiment | Status | Queue ID | Estimated Completion |
|------|------------|--------|----------|---------------------|
| 2025-12-28 | `git_pull_and_install` | 🔄 Queued | `git_pull_and_install` | ~5 min |
| 2025-12-28 | `turtel_rlcr_blackwell_suite` | 🔄 Queued | `turtel_rlcr_blackwell_suite` | ~10-14 hours |

**Suite breakdown**:
- Phase 1: Synthetic (parity, approachability, cliff_fog) — ~45 min
- Phase 2: Data prep — ~15 min
- Phase 3: GRPO training (4 configs) — ~6-8 hours
- Phase 4: Diffusion training — ~1 hour
- Phase 5: Hybrid training — ~2 hours
- Phase 6: Evaluation — ~1 hour
- Phase 7: Blackwell analysis — ~30 min

### 4.3 Pending Experiments

| Experiment | Depends On | Purpose |
|------------|------------|---------|
| Constraint ladder validation | Diffusion training | Verify $C_\sigma$ nesting |
| Cross-market arbitrage | Hybrid training | Multi-market $C_t$ |
| RLCR vs Turtel head-to-head | GRPO training | Calibration reward comparison |

---

## 5. Current Conclusions

### 5.1 What We Can Conclude (Preliminary)

Based on completed experiments (`pm_suite` from 2025-12-24):

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Diffusion reduces Brier score vs market | `pm_suite_eval_diff`: Brier improvement observed | Medium |
| AR reasoning helps over baseline | `pm_suite_eval_ar` vs `pm_suite_eval_market` | Medium |
| Synthetic parity shows AR cliff | `suite_parity_k*` runs | High |

### 5.2 What We Cannot Yet Conclude

| Claim | Missing Evidence | When Available |
|-------|------------------|----------------|
| **H1**: Diffusion improves AR | Need AR+Diff hybrid results | After `pm_hybrid_train` |
| **H2**: Blackwell rate $O(1/\sqrt{T})$ | Need full approachability suite | After `turtel_rlcr_blackwell_suite` |
| **H3**: Diffusion learns $C_t$ better | Need `compare_constraint_convergence()` | After suite |
| **H4**: $d(q, C_t)$ bounds arb profit | Need `compute_arbitrage_bound()` on real data | After suite |
| RLCR > Turtel for calibration | Need GRPO training comparison | After suite |
| Constraint ladder = noise schedule | Need `constraint_ladder_analysis()` | After diffusion training |

### 5.3 Preliminary Numbers (from pm_suite 2025-12-24)

```
┌─────────────────────────────────────────────────────┐
│ Model            │ Brier  │ ECE    │ ROI    │ Notes │
├─────────────────────────────────────────────────────┤
│ Market baseline  │ 0.22   │ 0.08   │ 0.0%   │ —     │
│ Diffusion        │ 0.19   │ 0.05   │ ~5%    │ 1M params │
│ AR (Qwen3-14B)   │ 0.21   │ 0.07   │ ~3%    │ Median-of-5 │
│ Turtel (reported)│ 0.19   │ 0.05   │ 10%    │ 14B RLVR │
└─────────────────────────────────────────────────────┘
```

**Note**: These are preliminary from a single run. Bootstrap CIs pending.

---


### Auto-Updated Status (2025-12-28T02:14:50Z)

**Headlines Fetch**: ⏳ Pending


**GPU Status**:
- GPU 0: 100% util, 58407/143771 MB
- GPU 1: 100% util, 58387/143771 MB

**Running Experiments**:
- PID 151728: `tmux new -d -s pm_beta0p2_gpu1 cd /root/diffusion-markets/ex...`
- PID 898501: `.venv/bin/python -m forecastbench pm_hybrid_train --dataset-...`
- PID 898631: `.venv/bin/python -m forecastbench pm_hybrid_train --dataset-...`
- PID 899497: `.venv/bin/python -m forecastbench pm_hybrid_train --dataset-...`
- PID 899664: `.venv/bin/python -m forecastbench pm_hybrid_train --dataset-...`

---

## 6. Monitoring & Updates

### Check experiment status

```bash
# Queue status
tail -20 experiments/remote_logs/remote_gpu_queue_state.jsonl

# Live log (when running)
ssh root@95.133.252.72 "tail -f /root/diffusion-markets/experiments/remote_logs/turtel_rlcr_blackwell_suite.log"

# GPU utilization
ssh root@95.133.252.72 "nvidia-smi"
```

### Pull results

```bash
# After completion
rsync -avz root@95.133.252.72:/root/diffusion-markets/experiments/runs/*turtel_suite* experiments/remote_runs/

# Blackwell comparison
rsync -avz root@95.133.252.72:/root/diffusion-markets/experiments/runs/blackwell_constraint_comparison.json experiments/remote_runs/
```

### Re-run failed experiments

```bash
cd experiments
python3 scripts/remote_gpu_enqueue.py \
  --queue-file remote_queue.jsonl \
  --id <experiment_id> --gpu any \
  --cmd '<command>'
```

---

## 7. File Index

| File | Purpose |
|------|---------|
| `experiments/EXPERIMENTS.md` | This document |
| `experiments/scripts/remote_suite_turtel_rlcr_blackwell.sh` | Main experiment script |
| `experiments/src/forecastbench/train/grpo.py` | GRPO/ReMax/RLCR training |
| `experiments/src/forecastbench/metrics/multiscale_approachability.py` | Blackwell metrics + arbitrage |
| `experiments/src/forecastbench/models/ar_diffusion_hybrid.py` | Hybrid architecture |
| `experiments/src/forecastbench/benchmarks/turtel_comparison.py` | Turtel baseline comparison |

---

## 8. Timeline

| Date | Milestone |
|------|-----------|
| 2025-12-28 | Suite queued |
| 2025-12-29 (est.) | Suite complete, initial results |
| 2025-12-30 (est.) | Analysis and conclusions |
| TBD | Paper draft update |

---

*This document is auto-updated. Last modified: 2025-12-28T02:14:50Z*

