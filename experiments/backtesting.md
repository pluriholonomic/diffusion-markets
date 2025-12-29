# Backtesting Framework Documentation

## Overview

This document describes the backtesting framework and the experiments conducted to validate Blackwell Approachability-based trading.

**Current Status**: The initial implementation was incorrect - it implemented prediction-based trading rather than true Blackwell approachability. This document records:
1. What was built (prediction-based backtest)
2. What the results show (model bias, not Blackwell validation)
3. What correct implementation should look like

## Architecture

```
backtest/
├── config.py           # BacktestConfig, StrategyConfig dataclasses
├── engine.py           # Main orchestration loop
├── market_state.py     # MarketStateManager for position tracking
├── ct_loader.py        # Loads model checkpoints and samples C_t
├── data/
│   ├── clob_loader.py  # CLOB time-series loading
│   └── group_registry.py  # Market-to-topic mapping for robustness
├── strategies/
│   ├── base.py         # BaseStrategy protocol
│   ├── online_max_arb.py  # O(1/√T) regret Hedge-based strategy
│   ├── stat_arb.py     # Statistical arbitrage using C_t covariance
│   ├── conditional_graph.py  # Conditional dependency trading
│   └── confidence_gated.py   # Kalai-inspired abstention strategy
├── metrics/
│   ├── pnl.py          # PnL, Sharpe, drawdown tracking
│   ├── regret.py       # Online regret curves
│   ├── h4_validation.py  # Distance-profit correlation (H4)
│   └── ct_validation.py  # Sample convergence diagnostics
└── execution/
    ├── cost_model.py   # Spread/impact estimation
    └── clob_fetcher.py # Live CLOB snapshot fetching
```

## Data Pipeline

### 1. Exa Headline Enrichment

We enriched the Polymarket gamma data with Exa-sourced news headlines to provide better conditioning for the diffusion model.

#### Exa Cache Structure
- Location: `data/headlines_cache/`
- Format: `exa_<question_slug>_<YYYYMMDD>.json`
- Total files: **15,950**
- Each file contains up to 10 headlines with title, URL, date, and source

#### Matching Algorithm

**v1: Word Overlap (Jaccard)**
```python
# Extract key words from question and cache filename
q_words = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
cache_words = set(cache_filename.split('_'))
jaccard = len(q_words & cache_words) / len(q_words | cache_words)
# Match if jaccard > 0.3
```
- **Result**: 45,295 / 50,000 matched (90.6%)
- **Problem**: Imprecise matches (e.g., Solana question → XRP headlines)

**v2: Strict Slug Matching**
```python
# Require >70% word overlap on slugs
overlap = len(cache_words & slug_words) / max(len(cache_words), len(slug_words))
# Match only if overlap > 0.7
```
- **Result**: 4,006 / 10,784 matched (37.1% of 2024+ data)
- **Quality**: Much better semantic alignment

#### Generated Datasets

| Dataset | Path | Rows | Description |
|---------|------|------|-------------|
| Loose Match | `data/polymarket/turtel_exa_enriched.parquet` | 45,295 | Word-overlap matching |
| Strict Match | `data/polymarket/exa_strict_matched.parquet` | 4,006 | Slug-based matching, 2024+ only |

### 2. Pre-computed Embeddings

- Location: `data/embeddings_cache.parquet`
- Model: `all-MiniLM-L6-v2` (384-dim)
- Coverage: All markets in gamma_yesno_resolved.parquet

---

## Training Experiments

### Model Architecture

```python
DiffusionModelSpec(
    out_dim=1,           # Logit of probability
    cond_dim=384,        # MiniLM embedding dimension
    hidden_dim=384,      # Hidden layer size
    depth=4,             # Number of residual blocks
    time_dim=64,         # Time embedding dimension
)
```

### Experiment 1: Baseline (No Headlines)

**Run**: `runs/proper_diffusion_20251228_231929/`
- **Data**: 30,000 rows from gamma_yesno_resolved.parquet
- **Steps**: 15,000
- **Final Loss**: 0.0065

**Results**:
| Metric | Value |
|--------|-------|
| corr(pred, true) | 0.21 |
| Pred std | 0.0008 |
| Conditioning | Weak |

**Diagnosis**: Model collapsed to near-constant predictions. The embeddings alone (without headlines) don't provide enough signal for market-specific conditioning.

### Experiment 2: Loose Exa Matching

**Run**: `runs/exa_diffusion_20251229_100724/`
- **Data**: 40,000 rows with word-overlap matched headlines
- **Steps**: 20,000
- **Final Loss**: 0.0615

**Results**:
| Metric | Value |
|--------|-------|
| corr(pred, true) | -0.05 |
| Pred std | 0.25 |
| Pred range | [0.02, 0.99] |
| Conditioning | Moderate |

**Diagnosis**: Model now produces diverse outputs (good pred_std) but correlation is near-zero. The imprecise headline matching likely introduces noise.

### Experiment 3: Fresh Embeddings with Headlines

**Run**: `runs/exa_diffusion_fresh_20251229_101752/`
- **Data**: 35,000 rows, embeddings generated from question + headlines
- **Steps**: 25,000
- **Final Loss**: 0.0084

**Results**:
| Metric | Value |
|--------|-------|
| corr(pred, true) | -0.04 |
| Pred std | 0.27 |
| Pred range | [0.02, 0.99] |

**Diagnosis**: Same issue - diverse outputs but no predictive signal. The headline matching quality is the bottleneck.

### Experiment 4: Strict Matching (Current Best)

**Run**: `runs/exa_strict_20251229_104108/`
- **Data**: 4,006 rows with strict slug-based matching
- **Steps**: 20,000
- **Final Loss**: 0.121

**Results**:
| Metric | Value |
|--------|-------|
| corr(pred, true) | 0.037 |
| Pred std | 0.22 |
| Win rate | 88.3% |
| Total PnL | ~0 |

**Diagnosis**: Slight improvement in correlation but still weak. The 4K samples may be insufficient for learning complex conditioning.

---

## H4 Hypothesis Validation

**H4**: The correlation between distance \( d(q, C_t) \) and realized PnL should be positive (larger mispricing → larger profit).

### Test Methodology

1. For each market, sample 64 points from \( C_t \) using the diffusion model
2. Compute \( d(q, C_t) \) = distance from market price to convex hull of samples
3. Trade in the direction of projection onto \( C_t \)
4. Measure correlation between distance and realized PnL

### Results Summary

| Experiment | corr(d, PnL) | Win Rate | Status |
|------------|--------------|----------|--------|
| Baseline | N/A | N/A | No trades (d ≈ 0 everywhere) |
| Loose Match | 0.39 | 0% | Distance correlates but model wrong |
| Strict Match | - | 88.3% | PnL ≈ 0 (offsetting wins/losses) |

### ⚠️ Critical Finding: Implementation Bug

**We were NOT implementing Blackwell approachability correctly.**

#### What Blackwell Approachability Actually Is

From the paper (Section 4.2), the constraint set C is:
```
C_ε := [-ε, ε]^M
```

Where the payoff vector is:
```
g_t(i) := (Y_t - q_t) · h^i(X_t, q_t)
```

- `h^i` are **test functions** (group indicators, conditional predictors, etc.)
- `g_t(i)` measures correlation between prediction error and test function i
- **No-arbitrage** = average payoff g̅_T stays inside C (all correlations small)

#### What We Were Doing (WRONG)

We were:
1. Training a diffusion model to predict P(outcome | context)
2. Treating the model's prediction as "C_t"
3. Trading toward the model's prediction

**This is just prediction-based trading, not Blackwell approachability!**

The "0.39 correlation with inverted direction" finding is simply: the model is biased +37% high, and fading a biased model is profitable. This has nothing to do with the Blackwell framework.

#### What Correct Implementation Would Look Like

1. Define test functions H = {h^1, ..., h^M}:
   - Group indicators: h^i(x) = 1[market ∈ group i]
   - Conditional predictors: h^i(x, q) = f(embedding(x))
   - Bin indicators: h^i(q) = 1[q ∈ bin i]

2. For each time t, compute payoff vector:
   ```
   g_t(i) = (outcome_t - price_t) · h^i(x_t, price_t)
   ```

3. Track cumulative average g̅_T = (1/T) Σ g_t

4. Arbitrage exists when d(g̅_T, C_ε) > 0

5. The Blackwell response is to adjust prices to project g̅_T toward C

#### Current Status

**H4 is NOT validated.** The experiments tested prediction-based trading, not Blackwell approachability. A correct implementation requires:
- Defining the test function family H
- Computing payoff vectors g_t, not predictions
- Measuring distance to constraint set C_ε, not distance to model predictions

---

## Key Learnings

### 1. ⚠️ Implementation Was Not Blackwell Approachability
The backtest implemented **prediction-based trading**, not the Blackwell framework:
- We trained a model to predict outcomes
- We traded toward model predictions
- This is standard ML forecasting, not approachability

The "100% win rate when inverted" finding simply exploits model bias, not Blackwell structure.

### 2. Model Has Systematic Bias
The diffusion model learned a +37% bias toward YES outcomes:
- Pred mean: 62.7%
- True mean: 25.3%

This makes predictions useless for direct trading but could be corrected with debiasing.

### 3. Headline Matching Quality is Critical
The Jaccard-based word overlap matching produces high coverage (90%) but many mismatches. For example:
- Solana price question → XRP price headlines
- 2020 election question → 2024 election headlines

Strict slug matching reduces this but sacrifices coverage.

### 4. Conditioning Signal vs. Diversity Trade-off
- Without headlines: Model collapses to mean (pred_std ≈ 0)
- With noisy headlines: Model produces diversity but no signal (corr ≈ 0)
- Need: High-quality conditioning that both diversifies AND correlates with outcomes

### 5. Data Volume vs. Quality
- 45K loosely matched samples → good diversity, inverted signal works
- 4K strictly matched samples → weak correlation (0.037)
- Suggests volume helps even with noisy matches

---

## Next Steps

### 1. 🔧 Implement Correct Blackwell Approachability

The current implementation is wrong. Here's what correct implementation should look like:

```python
# Define test function family
test_functions = [
    lambda x, q: 1.0,                          # Unconditional
    lambda x, q: x['is_crypto'],               # Crypto group
    lambda x, q: x['is_sports'],               # Sports group  
    lambda x, q: 1 if q < 0.3 else 0,          # Low-price bin
    lambda x, q: 1 if q > 0.7 else 0,          # High-price bin
    # ... more test functions
]

# Compute payoff vectors over time
g_cumsum = np.zeros(len(test_functions))
for t, (outcome, price, context) in enumerate(data):
    g_t = np.array([
        (outcome - price) * h(context, price) 
        for h in test_functions
    ])
    g_cumsum += g_t
    g_bar = g_cumsum / (t + 1)
    
    # Check if outside constraint set
    epsilon = 0.05  # No-arbitrage tolerance
    violations = np.abs(g_bar) > epsilon
    
    if violations.any():
        # Arbitrage exists! Trade to project back into C
        # Direction depends on WHICH test function violated
        pass
```

### 2. Investigate Model Bias Source
The +37% systematic bias needs investigation:
- Is it in the training data distribution?
- Is it in the logit → probability transform?
- Can we debias without retraining?

### 3. Better Headline Matching
- Use embedding similarity instead of word overlap
- Filter to headlines published BEFORE market end date
- Ensure topic alignment (crypto → crypto, sports → sports)

### 4. Alternative Conditioning
- Use market category as additional conditioning
- Add market final_prob (observed price) as context
- Experiment with hierarchical conditioning

### 5. Out-of-Sample Validation
- Run the inverted strategy on truly held-out data
- Test on different time periods
- Validate on live markets (paper trading)

---

## File Locations

### Trained Models
```
runs/
├── proper_diffusion_20251228_231929/model.pt  # Baseline
├── exa_diffusion_20251229_100724/model.pt     # Loose match
├── exa_diffusion_fresh_20251229_101752/model.pt  # Fresh embeds
└── exa_strict_20251229_104108/model.pt        # Strict match
```

### Data Files
```
data/
├── polymarket/
│   ├── gamma_yesno_resolved.parquet           # 50K markets
│   ├── turtel_exa_enriched.parquet            # 45K loose matches
│   └── exa_strict_matched.parquet             # 4K strict matches
├── embeddings_cache.parquet                   # Pre-computed embeddings
└── backtest/
    ├── clob_merged.parquet                    # CLOB time-series
    └── resolutions.parquet                    # Market outcomes
```

### Exa Cache
```
../data/headlines_cache/
├── exa_*.json                                 # 15,950 headline files
```

---

## Usage

### Training a New Model
```python
from forecastbench.models.diffusion_core import (
    ContinuousDiffusionForecaster,
    DiffusionModelSpec,
)

spec = DiffusionModelSpec(
    out_dim=1,
    cond_dim=384,
    hidden_dim=384,
    depth=4,
)

model = ContinuousDiffusionForecaster(spec)
model.train_mse_eps(
    x0=logits.reshape(-1, 1),  # Target: logit(outcome)
    cond=embeddings,            # Conditioning: MiniLM embeddings
    steps=20000,
    batch_size=128,
    lr=3e-4,
)
model.save("runs/my_model/model.pt")
```

### Loading and Sampling
```python
model = ContinuousDiffusionForecaster.load("runs/my_model/model.pt", device="cpu")

# Sample from C_t
logit = model.sample_x(cond=embedding, n_steps=32, seed=42)
prob = 1.0 / (1.0 + np.exp(-logit[0, 0]))
```

### Running H4 Validation
```python
# See scripts/run_h4_proper_test.py
PYTHONPATH=$(pwd) python scripts/run_h4_proper_test.py \
    --model runs/exa_strict_20251229_104108/model.pt
```

---

## References

- Blackwell, D. (1956). An analog of the minimax theorem for vector payoffs.
- Ho, J., et al. (2020). Denoising diffusion probabilistic models.
- Turtel source: `turtel-source/confidence.tex`
- Damani source: `damani-source/arXiv-2507.16806v1.tar.gz`

