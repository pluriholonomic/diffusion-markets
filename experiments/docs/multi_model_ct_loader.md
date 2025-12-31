# Multi-Model C_t Loader

This document describes the multi-model C_t (constraint set) loader architecture for the Blackwell approachability backtesting framework.

## Overview

The backtesting engine now supports multiple model types for generating C_t samples, the constraint set used in Blackwell approachability-based trading. Each model type offers different tradeoffs between computational cost, calibration accuracy, and sample diversity.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BacktestConfig                                   │
│  ct_mode: "single" | "union" | "legacy"                                 │
│  ct_model: "ar_diffusion" | "rlcr" | "bundle" | "legacy"               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        UnifiedModelLoader                                │
│  - Manages multiple model types                                          │
│  - Routes sampling based on ct_mode                                      │
│  - Computes text embeddings                                              │
└─────────────────────────────────────────────────────────────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │  ARDiffusionLoader│ │    RLCRLoader     │ │BundleDiffusionLoader
        │                   │ │                   │ │                   │
        │ - Qwen3-14B AR    │ │ - Fine-tuned LLM  │ │ - Joint diffusion │
        │ - Diffusion head  │ │ - K samples/text  │ │ - Bundle of mkts  │
        │ - 16 samples      │ │ - Temperature     │ │ - 16 samples      │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │                    │                    │
                    └────────────────────┴────────────────────┘
                                         │
                                         ▼
                              ┌───────────────────┐
                              │   C_t Samples     │
                              │  (n_samples, k)   │
                              └───────────────────┘
```

## C_t Generation Modes

### 1. SINGLE Mode

Uses one specific model to generate C_t samples.

```python
cfg = BacktestConfig(
    ct_mode="single",
    ct_model="rlcr",  # or "ar_diffusion", "bundle"
    rlcr_model="runs/grpo/rlcr_best/best",
    rlcr_K=5,
)
```

### 2. UNION Mode

Combines samples from all enabled models. The resulting C_t is the union (concatenation) of samples from each model, providing a more diverse constraint set.

```python
cfg = BacktestConfig(
    ct_mode="union",
    ar_diffusion_checkpoint=Path("runs/hybrid_v1/"),
    rlcr_model="runs/grpo/rlcr_best/",
    bundle_checkpoint=Path("runs/bundle_v1/"),
    union_models=("ar_diffusion", "rlcr", "bundle"),
)
```

### 3. LEGACY Mode (Default)

Uses the original `CtCheckpointLoader` for backward compatibility.

```python
cfg = BacktestConfig(
    ct_mode="legacy",
    checkpoint_dir=Path("runs/diffusion_checkpoints/"),
)
```

## Model Types

### AR+Diffusion Hybrid

**File:** `backtest/model_loader.py::ARDiffusionLoader`

Combines autoregressive reasoning with diffusion refinement:

1. **AR Stage:** Qwen3-14B generates K chain-of-thought samples → aggregate to q_AR
2. **Diffusion Stage:** Small diffusion head refines q_AR conditioned on text embeddings
3. **Output:** n_samples probability estimates

**Configuration:**
```python
ar_diffusion_checkpoint: Path  # Checkpoint with diff_head.pt
ar_diffusion_samples: int = 16  # Number of diffusion samples
```

**Theoretical Basis:** AR captures serial reasoning with spectral cutoff at depth L; diffusion provides smooth spectral attenuation for calibration.

### RLCR (Reinforcement Learning with Calibration Rewards)

**File:** `backtest/model_loader.py::RLCRLoader`

Fine-tuned LLM optimized for calibrated probability predictions:

1. **Model:** Qwen3-14B with LoRA adapter trained via GRPO
2. **Sampling:** K independent forward passes with temperature sampling
3. **Output:** K probability samples per market

**Configuration:**
```python
rlcr_model: str  # Path to fine-tuned model or HF identifier
rlcr_K: int = 5  # Self-consistency samples
```

**Training Reward:**
```
R = α * directional_correct + β * brier_improvement - γ * calibration_penalty
```

### Bundle Diffusion

**File:** `backtest/model_loader.py::BundleDiffusionLoader`

Joint diffusion model over multiple correlated markets:

1. **Input:** Bundle of k markets with text embeddings
2. **Model:** Transformer-based denoiser
3. **Output:** Joint probability samples (n_samples, k)

**Configuration:**
```python
bundle_checkpoint: Path  # Checkpoint directory
bundle_samples: int = 16  # MC samples
```

**Advantage:** Captures correlations between markets in the same bundle.

## Configuration Reference

### BacktestConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ct_mode` | str | "legacy" | "single", "union", or "legacy" |
| `ct_model` | str | "legacy" | Model for SINGLE mode |
| `ar_diffusion_checkpoint` | Path | None | AR+Diffusion checkpoint |
| `rlcr_model` | str | None | RLCR model path |
| `bundle_checkpoint` | Path | None | Bundle diffusion checkpoint |
| `ar_diffusion_samples` | int | 16 | Samples from AR+Diffusion |
| `rlcr_K` | int | 5 | Self-consistency samples |
| `bundle_samples` | int | 16 | Bundle diffusion samples |
| `union_models` | Tuple[str] | ("ar_diffusion", "rlcr", "bundle") | Models for UNION |
| `ct_embed_model` | str | "all-MiniLM-L6-v2" | Text embedding model |
| `ct_embed_dim` | int | 384 | Embedding dimension |

## Usage Examples

### Example 1: RLCR Only

```python
from backtest.config import BacktestConfig

cfg = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-03-31",
    clob_data_path=Path("data/clob.parquet"),
    checkpoint_dir=Path("runs/dummy"),  # Required but unused
    
    # C_t configuration
    ct_mode="single",
    ct_model="rlcr",
    rlcr_model="runs/ar_rlcr_20k_longrun/best",
    rlcr_K=5,
    
    strategies=("blackwell_calibration",),
)

engine = BacktestEngine(cfg)
results = engine.run()
```

### Example 2: Union of RLCR + AR+Diffusion

```python
cfg = BacktestConfig(
    # ... date and data config ...
    
    ct_mode="union",
    rlcr_model="runs/grpo/rlcr_best/",
    ar_diffusion_checkpoint=Path("runs/hybrid_v1/"),
    union_models=("rlcr", "ar_diffusion"),  # Exclude bundle
    
    rlcr_K=5,
    ar_diffusion_samples=16,
)
```

### Example 3: Programmatic Model Loader

```python
from backtest.model_loader import (
    UnifiedModelLoader, ModelLoaderConfig, CtMode, CtModel
)

cfg = ModelLoaderConfig(
    ct_mode=CtMode.UNION,
    rlcr_model_path="runs/rlcr/best",
    rlcr_K=5,
    enabled_models=("rlcr",),
)

loader = UnifiedModelLoader(cfg)

samples, valid_ids = loader.sample_ct(
    market_ids=["m1", "m2", "m3"],
    texts={"m1": "Will BTC exceed 100k?", "m2": "...", "m3": "..."},
    seed=42,
)

print(f"C_t samples: {samples.shape}")  # (5, 3)
```

## Testing

### Unit Tests

```bash
cd experiments
PYTHONPATH=$(pwd):$(pwd)/src pytest tests/test_model_loader.py -v
```

22 tests covering:
- Enum values and conversions
- Config validation
- Single mode sampling (each model type)
- Union mode sampling
- Edge cases (missing models, empty texts)

### Integration Test Script

```bash
# On remote GPU
cd diffusion-markets/experiments
PYTHONPATH=$(pwd):$(pwd)/src .venv/bin/python scripts/test_ct_modes.py \
    --data polymarket_backups/pm_suite_derived/gamma_yesno_ready_20k.parquet \
    --max-rows 100 \
    --rlcr-model runs/ar_rlcr_20k_longrun/best
```

Sample output:
```
Mode                 Shape           Mean       Std        Diversity 
-----------------------------------------------------------------
RLCR                 (5, 10)         0.0650     0.0896     0.0000    
AR+Diffusion         (16, 10)        0.4523     0.1234     0.0567    
Union                (21, 10)        0.3012     0.1456     0.0423    
```

## File Structure

```
experiments/backtest/
├── model_loader.py      # NEW: UnifiedModelLoader, individual loaders
├── config.py            # MODIFIED: Added ct_mode, ct_model fields
├── engine.py            # MODIFIED: Uses UnifiedModelLoader
├── ct_loader.py         # EXISTING: Legacy loader (still supported)

experiments/scripts/
├── test_ct_modes.py     # NEW: Compare C_t across models

experiments/tests/
├── test_model_loader.py # NEW: 22 unit tests
```

## Performance Characteristics

| Model | GPU Memory | Time/10 Markets | Samples | Diversity |
|-------|------------|-----------------|---------|-----------|
| RLCR | ~28 GB | ~30s | K × n | Low (point predictions) |
| AR+Diffusion | ~28 GB + 1 GB | ~45s | n_samples × n | Medium |
| Bundle | ~2 GB | ~5s | n_samples × n | High (joint) |
| Legacy | ~1 GB | ~2s | n_samples × n | Medium |

## Known Limitations

1. **Bundle State Dict Compatibility:** Some older bundle checkpoints have incompatible state_dict structure. Need to match checkpoint format with loader.

2. **RLCR Diversity:** RLCR generates point predictions; diversity comes only from temperature sampling, which may not capture full uncertainty.

3. **Memory Requirements:** Running UNION mode with all models requires significant GPU memory (~30+ GB).

4. **AR+Diffusion Checkpoint Format:** Expects `diff_head.pt` in checkpoint directory. If missing, uses untrained diffusion head.

## Future Work

- [ ] Add weighted union (different weights per model)
- [ ] Implement online model selection based on market category
- [ ] Add calibration metrics per model for adaptive selection
- [ ] Support streaming/incremental C_t updates
