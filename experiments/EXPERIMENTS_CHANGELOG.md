# Experiments Changelog

This document tracks changes to the experimental setup, particularly addressing gaps identified in the theory-experiment alignment.

---

## v2: Hierarchical Constraint Arbitrage Bounds (2025-12-28)

### Summary

Major refactoring to properly test **H4 (arbitrage bounds via d(q, C_t))** by implementing:
1. Hierarchical constraint sets (multicalibration + Fréchet)
2. Bootstrap confidence intervals on approachability rate
3. Per-sample hybrid correction analysis
4. Multi-market Fréchet constraint validation

### Gaps Addressed

| Gap ID | Description | Previous State | v2 Fix |
|--------|-------------|----------------|--------|
| **GAP-1** | C_t used only calibration bins | `BlackwellConstraintTracker` tracked (bin) only | `MulticalibrationTracker` tracks (group × bin) for arbitrary group crossings |
| **GAP-2** | No cross-market correlation constraints | No Fréchet bounds implemented for real data | `FrechetConstraintTracker` bundles markets by category and computes Fréchet violations |
| **GAP-3** | C_t not hierarchical | Single constraint type | `HierarchicalConstraintSet` combines multicalib (inner) + Fréchet (outer) |
| **GAP-4** | Approachability rate point estimate only | `compute_approachability_rate()` returned single value | `compute_approachability_rate_with_ci()` provides bootstrap 95% CI and tests H0: α = 0.5 |
| **GAP-5** | No per-sample correction analysis | Could not tell if diffusion corrects AR errors | `classify_corrections()` and `compute_error_correlations()` in `hybrid_analysis.py` |

### Files Created

| File | Purpose |
|------|---------|
| `metrics/hierarchical_constraints.py` | MulticalibrationTracker, FrechetConstraintTracker, HierarchicalConstraintSet, compute_arbitrage_bound_hierarchical() |
| `benchmarks/hybrid_analysis.py` | Per-sample correction classification, error correlation analysis, complexity estimation |
| `scripts/remote_suite_h4_arbitrage_v2.sh` | Comprehensive experiment suite for H4 validation |

### Files Modified

| File | Changes |
|------|---------|
| `metrics/multiscale_approachability.py` | Added `compute_approachability_rate_with_ci()` with bootstrap CIs; updated `compare_constraint_convergence()` to include bootstrap analysis |
| `cli.py` | Added `pm_eval_v2` command with hierarchical constraints; added `multimarket_arb` command for Fréchet analysis |

### New CLI Commands

#### `pm_eval_v2`

Evaluate predictions with hierarchical constraint analysis:

```bash
forecastbench pm_eval_v2 \
  --dataset-path predictions.parquet \
  --hierarchical-constraints \
  --multicalib-groups "topic,volume_q5" \
  --frechet-bundle-col category \
  --bundle-size 3 \
  --approachability-rate \
  --bootstrap-n 1000
```

**New metrics computed:**
- `hierarchical_constraints.distance_to_C`: Max of multicalib and Fréchet distance
- `hierarchical_constraints.d_multicalib`: Worst-group calibration error
- `hierarchical_constraints.d_frechet`: Max Fréchet bound violation
- `approachability_rate.rate`: Decay exponent α
- `approachability_rate.rate_ci_lo/hi`: 95% bootstrap CI
- `approachability_rate.consistent_with_theory`: True if 0.5 ∈ CI

#### `multimarket_arb`

Analyze cross-market Fréchet constraints:

```bash
forecastbench multimarket_arb \
  --dataset-path data.parquet \
  --bundle-col category \
  --bundle-size 3 \
  --constraint-type frechet
```

### Theory Alignment

#### H2: Blackwell Approachability Rate

**Theory claim**: Convergence at rate O(1/√T), i.e., α = 0.5

**v2 validation**:
- `compute_approachability_rate_with_ci()` computes bootstrap CI
- Tests null hypothesis α = 0.5
- Reports whether 0.5 is within 95% CI

#### H3: Diffusion Learns C_t Better

**Theory claim**: Diffusion model reduces distance to constraint set

**v2 validation**:
- Compare `model_distance_to_C` between AR-only and AR+Diffusion
- Hybrid should have smaller distance than AR
- Use hierarchical C_t (multicalib ∩ Fréchet)

#### H4: d(q, C_t) Bounds Arbitrage

**Theory claim**: Distance from market prices to C_t bounds extractable profit

**v2 validation**:
- `compute_arbitrage_bound_hierarchical()` measures:
  - `market_distance_to_C`: Distance of market prices to learned C_t
  - `model_distance_to_C`: Distance of model predictions to C_t
  - `distance_reduction`: How much model improves over market
  - `arbitrage_capture_rate`: Fraction of theoretical arbitrage captured
  - `sharpe_ratio`: Risk-adjusted return from trading on model

### Constraint Set Definition

The hierarchical constraint set C_t is defined as:

```
C_t = C_multicalib ∩ C_frechet

where:
  C_multicalib = {q : |E[Y - q | group, bin(q)]| ≤ ε for all (group, bin)}
  C_frechet = {q : max(0, q_A + q_B - 1) ≤ q_AB ≤ min(q_A, q_B) for all triples}
```

Distance to C_t:
```
d(q, C_t) = max(d_multicalib(q), d_frechet(q))
```

### Experiment Suite: `remote_suite_h4_arbitrage_v2.sh`

| Phase | Description | Duration |
|-------|-------------|----------|
| 1 | Synthetic multi-market validation (Fréchet, chain, independent) | ~30 min |
| 2 | Hierarchical constraint evaluation on predictions | ~1 hour |
| 3 | Multi-market Fréchet arbitrage analysis | ~30 min |
| 4 | Per-sample hybrid correction analysis | ~15 min |
| 5 | Full comparison and summary report | ~15 min |

---

## v1: Initial Experiment Suite (2025-12-24)

### Summary

Initial implementation supporting H1-H4 with:
- Synthetic parity benchmarks
- Polymarket evaluation pipeline
- GRPO/ReMax training
- Basic approachability tracking

### Known Gaps (Fixed in v2)

1. C_t used only calibration bins (not group-conditional)
2. No cross-market correlation constraints
3. Approachability rate without confidence intervals
4. No per-sample correction analysis for hybrid

---

## Migration Notes

### Upgrading from v1 to v2

1. **New dependencies**: None (uses existing numpy, pandas)

2. **Backward compatible**: All v1 commands still work

3. **New experiments**: Run v2 suite with:
   ```bash
   bash scripts/remote_suite_h4_arbitrage_v2.sh
   ```

4. **Identifying v2 runs**: All v2 runs have `h4_v2` prefix

---

*Last updated: 2025-12-28*



