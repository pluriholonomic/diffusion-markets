# Dynamic Ensemble Model for Prediction Market Statistical Arbitrage

**Date**: December 31, 2025

## Your Proposed Model

```
π(t) = Σᵢ aᵢ(t) πᵢ(t)
```

Where:
- `πᵢ(t)` = sub-portfolios that evolve locally linearly: `πᵢ(t) = A₁πᵢ(t-1) + A₂πᵢ(t-2) + ...`
- `aᵢ(t)` = dynamic weights (decision tree or neural net) that account for defaults/resolutions

---

## Answer: Why This Is Theoretically Sound But Empirically Challenging

### ✅ Why It COULD Work

1. **Sub-portfolios capture themes/factors**
   - Markets on related topics (Trump/Harris elections, NBA Finals, crypto prices) are correlated
   - Correlation within NBA theme: r = 0.45 (significant!)
   - These create natural sub-portfolios πᵢ

2. **Dynamic weights handle regime changes**
   - Market behavior shifts over time (we found y_rate dropped from 43% to 26%)
   - aᵢ(t) can learn when to increase/decrease exposure

3. **Resolution as "default" is analogous to bonds**
   - When a market resolves, you get recovery value (0 or 1)
   - Similar to bond default modeling
   - Can be incorporated into dynamics

### ❌ Why It's Hard in Practice

| Issue | Description | Impact |
|-------|-------------|--------|
| **Sparse Data** | Only 5-10 markets per theme with CLOB data | Can't fit VAR models |
| **Non-smooth dynamics** | Prices don't follow AR/VAR; often stale or jump at resolution | VAR assumptions violated |
| **Price ≠ continuous** | Prices are probabilities bounded [0,1] | Creates asymmetric dynamics |
| **Resolution discontinuity** | Price jumps to 0 or 1 at resolution | Breaks mean-reversion |
| **Correlation ≠ outcome dependency** | Price-correlated markets may resolve differently | Spread doesn't converge |

---

## Empirical Results

### Test 1: VAR on Theme Returns
```
Theme       | Train R² | Out-of-Sample Sharpe
------------|----------|---------------------
NBA         | 0.207    | -0.64 (FAILED)
Crypto      | 0.013    | -1.02 (FAILED)
Politics    | 0.002    | N/A
```
**Conclusion**: VAR dynamics don't work on theme-level returns.

### Test 2: Spread Mean-Reversion
```
Pair                          | Win Rate | Total PnL
------------------------------|----------|----------
Biden/Harris nomination spread| 81.2%    | +0.27
```
**Conclusion**: Some spreads work, but limited sample size.

### Test 3: Calibration-Based Ensemble
```
Category | Train Sharpe | Test Win Rate | Test PnL
---------|--------------|---------------|---------
Weather  | 22.64        | 86.1%         | +134.60
Sports   | 25.14        | 69.9%         | +366.94
Crypto   | 9.64         | 73.8%         | +232.80
```
**WARNING**: These Sharpes are impossibly high (34.95 overall). This is NOT real alpha - it's exploiting a systematic calibration bias in the data (YES is overpriced across all categories).

### Test 4: Fréchet Bound Arbitrage
```
Trades: 9
Win Rate: 100%
Avg PnL: 0.23
```
**Conclusion**: True arbitrage exists but sample is tiny (need more data).

---

## Recommendations to Make Your Framework Work

### 1. Replace VAR with Probability Constraints

Instead of:
```
πᵢ(t) = A₁πᵢ(t-1) + A₂πᵢ(t-2) + ...
```

Use:
```
πᵢ(t) = f(Fréchet bounds, outcome correlations, Bayesian updates)
```

**Specific approaches**:
- **Fréchet bounds**: P(A ∩ B) ≤ min(P(A), P(B)) - trade violations
- **Copula models**: Joint probability structure between related markets
- **Information flow**: If market A updates, predict market B should update

### 2. Define Sub-portfolios by Outcome, Not Price

Bad:
```python
# Cluster by price correlation
corr_matrix = prices.corr()
```

Good:
```python
# Cluster by outcome relationship
# Markets with same underlying event
# Or: markets with logical implication
```

### 3. Resolution-Aware Dynamics

```python
def weight_by_resolution(days_to_resolution):
    # Reduce weight as resolution approaches
    # Near resolution = prices are informed = no edge
    if days_to_resolution < 7:
        return 0.0
    elif days_to_resolution < 30:
        return 0.5
    else:
        return 1.0
```

### 4. Dynamic Weights Based on Features

```python
features = [
    'calibration_strength',     # How miscalibrated is this category?
    'volume',                   # Liquid markets = harder to trade
    'time_to_resolution',       # Edge decays near resolution
    'historical_win_rate',      # Past performance
    'regime_indicator'          # Has the market behavior shifted?
]

# Train decision tree
weights = DecisionTreeRegressor().fit(features, returns)
```

### 5. Handle Default (Resolution) Explicitly

```python
class ResolutionAwarePortfolio:
    def update(self, market_id, resolved=False, outcome=None):
        if resolved:
            # Market exits portfolio
            # Realize P&L based on outcome
            # Rebalance remaining positions
            self.positions.pop(market_id)
            self.realized_pnl += self.compute_resolution_pnl(market_id, outcome)
```

---

## Data Requirements

To properly fit your model, you need:

| Data Type | Current Status | Needed |
|-----------|---------------|--------|
| CLOB prices | 200 markets | 10,000+ markets |
| Price horizons | Mixed (~7 days before) | Fixed (30 days before) |
| Outcome relationships | None | Labeled pairs |
| Resolution dates | Available | ✅ |
| Look-ahead protection | Partially fixed | Fully fixed |

---

## Bottom Line

### Your Framework IS Valid

The structure `π(t) = Σᵢ aᵢ(t) πᵢ(t)` is correct for prediction markets. It's essentially:
- A factor model with regime switching
- Dynamic allocation across arbitrage types
- Resolution-aware rebalancing

### But Implementation Needs Modification

1. **Don't use VAR** - prediction markets don't have smooth dynamics
2. **Use probability theory** - Fréchet bounds, conditional probabilities
3. **Model outcomes, not prices** - Prices are noisy; outcomes are ground truth
4. **Weight by information content** - Not all signals are equal

### True Statistical Arbitrage Opportunities in PMs

| Type | Example | Edge |
|------|---------|------|
| **Fréchet violations** | P(A) > P(A ∪ B) | Guaranteed |
| **Same-event repricing** | "Trump wins" vs "Republican wins" | Low risk |
| **Information flow** | One market updates before another | Speed-dependent |
| **Multi-outcome constraints** | State probabilities sum to federal | Structural |

### What We Can't Do

| Type | Why It Fails |
|------|--------------|
| **Traditional pairs trading** | Prices don't converge smoothly |
| **VAR on returns** | Sparse data, non-stationary |
| **Mean-reversion on spreads** | Resolution creates jumps |

---

## Next Steps

1. **Get more CLOB data** - Need 10x more markets with full price history
2. **Label outcome relationships** - Which markets are logically related?
3. **Build Fréchet scanner** - Real-time detection of probability violations
4. **Implement resolution handling** - Exit strategy before final jump
5. **Paper trade** - Validate in real-time before deploying capital
