# Adversarial Analysis: Holes in the Prediction Market Backtest

**Date**: December 31, 2025  
**Purpose**: Critical examination of all assumptions, data, and methodology

---

## Executive Summary: The Strategy Likely Has NO Real Edge

After rigorous adversarial analysis, the claimed performance (Sharpe 0.8-1.5, $70M capacity) is almost certainly overstated due to:

| Issue | Impact | Severity |
|-------|--------|----------|
| Look-ahead bias | Inflates returns 30-50% | **CRITICAL** |
| Regime shift (2024) | 96% of data from unusual period | **CRITICAL** |
| Capacity constraints | Real capacity ~$5M not $70M | **SEVERE** |
| Execution costs | Not properly modeled | **MODERATE** |
| Survivorship bias | Unknown | **UNKNOWN** |

**Honest Performance Estimate:**
| Metric | Claimed | Reality |
|--------|---------|---------|
| Sharpe | 0.8-1.5 | **0.0-0.3** |
| Capacity | $70M/year | **$5M/year** |
| Win Rate | 55-65% | **50-55%** |

---

## Issue 1: Look-Ahead Bias in Entry Prices

### Finding
The `market_prob` prices used for backtesting are sampled **too close to resolution**:

| Timing | Percentage | Price-Outcome Correlation |
|--------|------------|---------------------------|
| < 1 day before | 11.8% | 0.325 |
| < 1 week before | 48.2% | 0.325 |
| 1-4 weeks before | - | 0.279 |
| > 30 days before | 12.6% | 0.274 |

**Critical**: Correlation increases from 0.274 to 0.325 as we get closer to resolution. This is evidence that prices near resolution already incorporate outcome information.

### Impact
- Overall correlation = 0.304 (should be ~0 for fair backtest)
- This inflates apparent returns by 20-40%
- In production, we would enter at prices without this information advantage

### Fix Required
Use prices from a **fixed horizon** (e.g., 30 days) before resolution, or use only the earliest available price for each market.

---

## Issue 2: Massive Regime Shift in 2024-2025

### Finding
96% of the data comes from 2024-2025, and this period has dramatically different characteristics:

| Year | Markets | y_rate (YES) | Change |
|------|---------|--------------|--------|
| 2021 | 1,235 | 45.4% | baseline |
| 2022 | 1,982 | 43.3% | -2.1% |
| 2023 | 1,284 | 38.9% | -4.4% |
| **2024** | 12,016 | **26.1%** | **-12.8%** |
| **2025** | 88,365 | **25.9%** | **-12.8%** |

**Statistical Test**: Trend in y_rate is highly significant (p < 0.0001)

### Impact
- "Short everything" strategy appears highly profitable only because outcomes shifted massively toward NO
- The backtest is essentially testing only on 2024-2025 data
- If 2024-2025 has data quality issues, the entire analysis is invalid
- **Pre-2024 Sharpe ~0.5, Post-2024 Sharpe ~10+ (impossible)**

### Possible Causes
1. **Data artifact**: Collection method changed
2. **Market structure change**: More "longshot" markets created
3. **Selection bias**: Different markets being resolved
4. **Real phenomenon**: Prediction markets became more overconfident

### Fix Required
- Report results separately for pre-2024 and post-2024 periods
- Use only pre-2024 data for conservative estimates
- Investigate the cause of the regime shift

---

## Issue 3: Capacity Is Severely Overstated

### Finding
| Metric | Claimed | Reality |
|--------|---------|---------|
| Annual capacity | $70M | **$4.8M** |
| Monthly tradeable | $5.8M | **$400K** |

### Calculation
```
Monthly volume median: $8.0M
At 5% participation rate: $8M × 5% = $400K/month
Annual: $400K × 12 = $4.8M/year
```

### Volume Distribution
- Min monthly volume: $0.8M
- Median monthly volume: $8.0M
- Max monthly volume: $6,188M (outlier month)

### Impact
- Strategy is not scalable for institutional capital
- At $5M AUM, transaction costs dominate returns
- Cannot run this as a serious hedge fund strategy

---

## Issue 4: Transaction Cost Underestimation

### Assumed
- 2% Polymarket fee
- 1% price impact
- **Total: 3%**

### Reality
| Cost Component | Estimate | Notes |
|----------------|----------|-------|
| Polymarket fee | 2% | Correct |
| Bid-ask spread | 2-5% | **Not modeled** |
| Price impact | 3-10% | **Underestimated** |
| Slippage | 1-3% | **Not modeled** |
| **Total** | **8-20%** | **Much higher** |

### Evidence
- Intraday price volatility: 10.2% average, 39.6% max
- This volatility creates execution uncertainty
- Large orders will move prices significantly

### Impact
- Real transaction costs may be 3-5x higher than modeled
- At 15% round-trip cost, the 30% claimed return becomes 15%
- Most of the "edge" is eaten by execution costs

---

## Issue 5: Survivorship Bias (Unknown Magnitude)

### Problem
The dataset contains only **resolved** markets. We don't know about:
- Markets that were cancelled/voided
- Markets that are still open
- Markets that were delisted before resolution

### Questions
1. How many markets were created but never resolved?
2. Are cancelled markets systematically different?
3. Would we have traded (and lost on) cancelled markets?

### Impact
- **Unknown** - could be significant
- If 10% of markets are cancelled and those are the "bad" ones, our edge is inflated

---

## Issue 6: Category Classification Is Fragile

### Finding
- **44.2% of markets fall into "other" category** (unclassified)
- Classification uses simple regex matching
- Order of regex checks matters (113 markets match both "trump" AND "win")

### Problems
1. Strategy cannot trade "other" category (almost half the market)
2. Category boundaries are arbitrary
3. New question types may not fit existing patterns

### Example Ambiguity
```
"Will Trump win the 2024 election?" 
→ Could be "politics" OR "sports" depending on regex order
```

### Impact
- Strategy only works on 55% of markets
- Reduces effective capacity by ~50%
- Edge may not transfer to future market types

---

## Issue 7: Statistical Multiple Testing

### Problem
We tested 20+ strategy variants and reported the best one:
1. Individual market shorting
2. Category rotation (5 categories)
3. Volume tier arbitrage (5 tiers)
4. Pairs trading (48 pairs)
5. Calibration momentum
6. Cross-category relative value
7. Various combinations

### Correction
- Bonferroni corrected p-value threshold: 0.0025
- Our best strategy p-value: 0.0003
- **Survives correction**, but barely

### Impact
- Some reported performance may be due to chance
- Should use holdout data for final validation

---

## Issue 8: Adverse Selection

### Finding
High-volume markets are **better calibrated**:
| Volume Tier | y_rate | Deviation from 50% |
|-------------|--------|-------------------|
| Q1 (lowest) | 25.0% | 25.0% |
| Q5 (highest) | 29.2% | 20.8% |

### Implication
- Institutional money (high volume) IS smarter
- Our "edge" comes from low-volume retail markets
- We're trading against uninformed retail, not generating alpha

### Risk
- As markets mature, retail leaves and edge disappears
- We may be on the wrong side when informed traders enter

---

## Issue 9: No Exit Strategy

### Problem
The backtest assumes we hold to resolution. But:
1. What if price moves against us significantly?
2. What if we need liquidity before resolution?
3. What if the market gets cancelled?

### Reality
- We cannot exit most positions without huge price impact
- Locked capital for weeks/months
- No stop-loss possible

### Impact
- Tail risk is unmodeled
- Could have catastrophic losses in adversarial scenarios

---

## Issue 10: What an Adversary Would Do

### 1. Front-Running
The category rotation signal is monthly and predictable. An adversary could:
- Observe our rebalancing pattern
- Trade ahead to capture our price impact
- Fade us after we're done

### 2. Pick-Off
When our position is in-the-money near resolution:
- We can't exit (no liquidity)
- Adversary knows our position
- Can trade against us with superior information

### 3. Stale Signal Exploitation
Our 6-month lookback is slow:
- Calibration can shift faster than we detect
- Trade the opposite when our signal is outdated
- Especially around major events (elections, etc.)

### 4. Liquidity Provision
Market-make against us:
- Collect bid-ask spread on our trades
- Hedge directional risk
- Risk-free profit from our flow

---

## Conclusion: Honest Assessment

### What the Strategy Actually Is
This is **NOT** sophisticated statistical arbitrage. It's:
- Betting that markets overprice YES outcomes
- Category rotation based on historical calibration
- Volume filtering for efficiency

### Why It Appeared to Work
1. **Regime shift**: 2024-2025 had extreme NO bias (26% vs 43%)
2. **Look-ahead bias**: Prices sampled near resolution
3. **Data mining**: Best of many tested strategies reported

### Realistic Performance
| Scenario | Sharpe | Capacity |
|----------|--------|----------|
| As claimed | 0.8-1.5 | $70M/yr |
| After execution costs | 0.3-0.6 | $30M/yr |
| After look-ahead fix | 0.1-0.3 | $15M/yr |
| **After all corrections** | **0.0-0.2** | **$5M/yr** |

### Recommendation
**Do not deploy** this strategy without:
1. Fixing look-ahead bias (use 30-day horizon prices)
2. Testing on pre-2024 data only
3. Realistic execution cost modeling
4. Paper trading for 6+ months
5. Starting with <$100K capital

---

*This adversarial analysis was conducted to identify weaknesses before deployment. Many of these issues may have solutions, but they must be addressed before considering live trading.*
