# Statistical Arbitrage Analysis for Prediction Markets

**Date**: December 31, 2025  
**Author**: Automated Analysis

## Executive Summary

This document summarizes a comprehensive analysis of statistical arbitrage opportunities in prediction markets using Polymarket data. The analysis covers 104,963 resolved markets from 2020-2025.

### Key Findings

1. **No true pairs/basket trading alpha exists** - Price correlations between markets do not predict outcome correlations
2. **The edge is calibration arbitrage** - Markets systematically overprice YES outcomes
3. **Regime shift in 2024** - Outcome rates dropped from ~40% to ~26%, inflating backtest results
4. **Conservative Sharpe: 0.8-1.5** - Based on 2022-2023 data (pre-regime shift)

---

## 1. Data Overview

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Resolved Markets | 104,963 |
| Date Range | Nov 2020 - Dec 2025 |
| Markets with CLOB Prices | 24,096 (23%) |
| Markets with Headlines | 20,000 (19%) |

### Outcome Rate by Year

| Year | Markets | y_rate (YES) | Implication |
|------|---------|--------------|-------------|
| 2020 | 9 | 44.4% | Balanced |
| 2021 | 1,235 | 45.4% | Balanced |
| 2022 | 1,982 | 43.3% | Balanced |
| 2023 | 1,284 | 38.9% | Slightly skewed |
| 2024 | 12,016 | 26.1% | **Heavily skewed NO** |
| 2025 | 88,365 | 25.9% | **Heavily skewed NO** |

**Critical Finding**: The 2024-2025 regime shift makes all "short" strategies appear highly profitable. Results from this period should not be extrapolated.

---

## 2. Correlation Analysis

### Price Correlation vs Outcome Correlation

We analyzed 200 markets with full CLOB price history to find correlated pairs.

**Found 48 pairs with |correlation| > 0.3**

Top positively correlated pairs:
- Kamala Harris winning 2024 + Democrat winning DC (r=+0.93)
- Republicans flip Biden state + SF candidate (r=+0.58)

Top negatively correlated pairs:
- Kamala winning + "another candidate wins SF" (r=-0.91)
- DC Democrat win + "another candidate wins SF" (r=-0.82)

### Critical Discovery: Price Correlation ≠ Outcome Correlation

| Correlation Type | Same Outcome | Different Outcome |
|-----------------|--------------|-------------------|
| Positively correlated (r>0) | 45.8% | 54.2% |
| Negatively correlated (r<0) | 54.2% | 45.8% |

**Conclusion**: Traditional pairs trading (long one, short other) does not work because price correlation does not predict outcome correlation.

---

## 3. Pairs Trading Backtest Results

### Traditional Pairs Trading (FAILED)

| Metric | Value |
|--------|-------|
| Total Trades | 3,620 |
| Total PnL | **-$4,904** |
| Win Rate | 9.3% |
| Sharpe | -26.18 |

### Comparison to Naive Strategy

| Strategy | PnL |
|----------|-----|
| Pairs Trading | -$4,904 |
| Naive Short-All | +$1,335 |
| **Edge from Pairs** | **-$6,239** |

**Conclusion**: Pairs trading destroys value compared to simply shorting everything.

---

## 4. What Actually Works: Calibration Arbitrage

### The True Edge

Markets systematically overprice YES outcomes:
- Average market price: 39.5%
- Actual outcome rate: 26.3%
- **Calibration error: -13.2%**

### Category-Level Calibration (2024 data)

| Category | y_rate | Deviation from 50% |
|----------|--------|-------------------|
| Sports | 21.9% | 28.1% |
| Weather | 26.1% | 23.9% |
| Other | 23.5% | 26.5% |
| Crypto | 31.2% | 18.8% |
| Politics | 33.6% | 16.4% |

### Volume-Tier Analysis

High-volume markets are better calibrated:

| Volume Tier | y_rate | Deviation |
|-------------|--------|-----------|
| Q1 (lowest) | 25.0% | 25.0% |
| Q5 (highest) | 29.2% | 20.8% |

---

## 5. Portfolio Strategies Tested

### Strategy A: Monthly Category Rotation
- **Method**: Short the most overpriced category each month
- **2022-2023 Sharpe**: ~0.5 - 1.0
- **2024-2025 Sharpe**: 7-14 (inflated)

### Strategy B: Calibration Momentum
- **Method**: Bet on categories whose calibration is improving
- **Sharpe**: 0.98

### Strategy C: Volume-Tier Arbitrage
- **Method**: Long high-volume, short low-volume markets
- **Sharpe**: 1.27

### Combined Strategy: Informed Category Short
- **Method**: High-volume + lowest y_rate category
- **Conservative Sharpe**: 0.8 - 1.5
- **Capacity**: ~$70M/year

---

## 6. Recommended Strategy

### "Informed Category Short" Portfolio

**Rules**:
1. Focus on high-volume markets only (top 40% by volume)
2. Each month, compute trailing 6-month calibration by category
3. Short the category with lowest y_rate (most overpriced)
4. Position size: proportional to volume, max 5% of market liquidity
5. Hold until market resolution (typically 1-4 weeks)

**Expected Performance (Conservative)**:
| Metric | Value |
|--------|-------|
| Sharpe | 0.8 - 1.5 |
| Win Rate | 55-65% |
| Annual Return | 15-30% |
| Holding Period | 1-4 weeks |
| Capacity | ~$70M/year |

**Risks**:
- Regime shifts can dramatically change performance
- Limited liquidity in individual markets
- Correlation with crypto/politics sentiment

**Advantages**:
- Monthly rebalancing (not HFT)
- Portfolio approach (not single-market)
- Based on observable calibration differences

---

## 7. Data Files Created

| File | Description |
|------|-------------|
| `data/gamma_all_prices_combined.parquet` | 24K markets with actual prices |
| `data/gamma_resolved_with_headlines.parquet` | 104K markets with Exa headlines (20K enriched) |
| `data/clob_market_prices.parquet` | Average prices from CLOB history |
| `data/market_correlations.parquet` | 48 correlated market pairs |
| `data/polymarket/clob_history_full/` | 5,001 market CLOB files (4.2GB) |
| `runs/portfolio_strategy_summary.json` | Strategy configuration |
| `runs/comprehensive_backtest_24k.json` | Backtest results |

---

## 8. Technical Notes

### Data Sources
- **Resolved Markets**: `polymarket_backups/pm_suite_derived/gamma_yesno_resolved.parquet`
- **CLOB History**: Synced from remote GPU box (5,001 markets)
- **Exa Headlines**: 20,000 markets enriched with news headlines

### Validation Methods
- Walk-forward backtesting with 6-12 month lookback
- Year-by-year performance breakdown
- Bootstrap confidence intervals for Sharpe
- T-tests for statistical significance

### Known Limitations
1. Entry prices estimated for 81K markets without CLOB data
2. Transaction costs assumed at 2% + 1% price impact
3. Position sizing capped at 10% of market volume
4. No modeling of market creation/expiration timing

---

## 9. Conclusions

### What Works
- **Calibration arbitrage**: Betting against systematic market overpricing
- **Category rotation**: Some categories more overpriced than others
- **Volume filtering**: High-volume markets better calibrated

### What Doesn't Work
- **Pairs trading**: Price correlations don't predict outcome correlations
- **Mean-reversion on spreads**: Markets resolve independently
- **Basket hedging**: Cannot hedge one market with another

### Honest Assessment
The "edge" in prediction markets is simply that markets systematically overprice YES outcomes. This is not sophisticated statistical arbitrage—it's calibration arbitrage. The strategy is profitable but:
- Highly regime-dependent
- Limited capacity (~$70M/year)
- Subject to structural changes in market behavior

---

*Generated by automated analysis pipeline. Results should be validated before live trading.*
