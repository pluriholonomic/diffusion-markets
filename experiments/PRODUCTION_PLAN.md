# Production Deployment Plan: Prediction Market Trading Strategies

## Executive Summary

Based on backtesting across Polymarket and Kalshi with realistic market impact models, we have identified profitable strategies ready for production testing:

| Platform | Strategy | Sharpe | Win Rate | ES_Sharpe | Recommended |
|----------|----------|--------|----------|-----------|-------------|
| **Polymarket** | Calibration Mean-Reversion | 11.06 | 78% | 10.83 | ✅ Primary |
| **Polymarket** | Statistical Arbitrage | 12.65 | 91% | 66.12 | ✅ Secondary |
| **Kalshi** | Longshot Betting | 4.43 | 15% | 20.27 | ✅ Primary |
| **Kalshi** | Calibration Mean-Reversion | 4.58 | 16% | 19.21 | ✅ Secondary |
| **Kalshi** | Contrarian | 4.23 | 18% | 5.99 | ⚠️ Test |

---

## Phase 1: Paper Trading (Week 1-2)

### 1.1 Infrastructure Setup

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Polymarket │    │    Kalshi    │    │   Database   │       │
│  │     API      │    │     API      │    │  (Postgres)  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └─────────┬─────────┴───────────────────┘                │
│                   │                                              │
│         ┌─────────▼─────────┐                                    │
│         │   Signal Engine   │                                    │
│         │  - Calibration    │                                    │
│         │  - Kelly Sizing   │                                    │
│         │  - Risk Limits    │                                    │
│         └─────────┬─────────┘                                    │
│                   │                                              │
│         ┌─────────▼─────────┐                                    │
│         │ Execution Engine  │                                    │
│         │  - Order Mgmt     │                                    │
│         │  - Fill Tracking  │                                    │
│         │  - Slippage Calc  │                                    │
│         └─────────┬─────────┘                                    │
│                   │                                              │
│         ┌─────────▼─────────┐                                    │
│         │    Monitoring     │                                    │
│         │  - PnL Tracking   │                                    │
│         │  - Alerts         │                                    │
│         │  - Dashboards     │                                    │
│         └───────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Tasks

- [ ] Set up Polymarket API integration (CLOB API)
- [ ] Set up Kalshi API integration (REST API)
- [ ] Create signal generation pipeline
- [ ] Implement paper trading mode (log orders without execution)
- [ ] Set up PostgreSQL for trade logging
- [ ] Create monitoring dashboard (Grafana or simple web UI)

### 1.3 Paper Trading Metrics to Track

- Signal frequency (trades/day)
- Hypothetical fill rate
- Theoretical PnL
- Calibration drift (is learned calibration still valid?)
- Category breakdown

---

## Phase 2: Small Capital Test (Week 3-4)

### 2.1 Capital Allocation

| Platform | Initial Capital | Max Position | Max Daily Loss |
|----------|----------------|--------------|----------------|
| Polymarket | $1,000 | $100 (10%) | $200 (20%) |
| Kalshi | $1,000 | $50 (5%) | $200 (20%) |

### 2.2 Risk Limits

```python
RISK_LIMITS = {
    'polymarket': {
        'max_position_pct': 0.10,      # 10% of bankroll per trade
        'max_daily_loss_pct': 0.20,    # 20% daily stop
        'max_drawdown_pct': 0.30,      # 30% total drawdown stop
        'kelly_fraction': 0.15,         # Conservative Kelly (60% of full)
        'min_edge': 0.05,               # Minimum 5% edge to trade
        'min_liquidity': 1000,          # Minimum $1K liquidity
    },
    'kalshi': {
        'max_position_pct': 0.05,      # 5% per trade (higher variance)
        'max_daily_loss_pct': 0.20,
        'max_drawdown_pct': 0.30,
        'kelly_fraction': 0.10,         # Very conservative for longshots
        'min_edge': 0.10,               # Higher edge requirement
        'min_liquidity': 100,           # Lower liquidity threshold
    },
}
```

### 2.3 Strategy Configuration

```python
STRATEGIES = {
    'polymarket_calibration': {
        'enabled': True,
        'spread_threshold': 0.05,
        'n_bins': 10,
        'recalibrate_days': 7,        # Recalibrate weekly
        'categories': ['politics', 'economics', 'crypto'],  # Focus categories
    },
    'polymarket_stat_arb': {
        'enabled': True,
        'category_min_markets': 10,   # Min markets per category
        'correlation_threshold': 0.3,
    },
    'kalshi_longshot': {
        'enabled': True,
        'price_threshold': 0.10,      # Only trade <10% priced markets
        'min_expected_edge': 0.15,    # Require 15% expected edge
    },
    'kalshi_contrarian': {
        'enabled': False,             # Start disabled, enable after testing
        'extreme_move_threshold': 0.10,
    },
}
```

---

## Phase 3: Scale-Up (Week 5-8)

### 3.1 Scaling Criteria

Move to next capital tier when:
1. ✅ 2 weeks profitable
2. ✅ Sharpe > 2.0 (realized)
3. ✅ Max drawdown < 15%
4. ✅ No critical system failures
5. ✅ Fill rate within 20% of expected

### 3.2 Capital Tiers

| Tier | Capital | Condition to Advance |
|------|---------|---------------------|
| 1 | $2,000 | Paper trading success |
| 2 | $5,000 | Tier 1 criteria met |
| 3 | $10,000 | Tier 2 criteria met |
| 4 | $25,000 | Tier 3 + 4 weeks profitable |
| 5 | $50,000 | Tier 4 + Sharpe > 3.0 |

### 3.3 Capacity Limits

Based on backtest analysis:

| Platform | Strategy | Est. Capacity | Reason |
|----------|----------|---------------|--------|
| Polymarket | Calibration | ~$50K | Liquidity constraints at extremes |
| Polymarket | Stat Arb | ~$20K | Limited category diversity |
| Kalshi | Longshot | ~$30K | Many small markets |
| Kalshi | Calibration | ~$30K | Sports volume constraints |

---

## Phase 4: Production Monitoring

### 4.1 Key Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE TRADING DASHBOARD                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PORTFOLIO SUMMARY                                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐      │
│  │ Total PnL   │ Daily PnL   │ Sharpe(30d) │ Drawdown    │      │
│  │ $X,XXX      │ $XXX        │ X.XX        │ X.X%        │      │
│  └─────────────┴─────────────┴─────────────┴─────────────┘      │
│                                                                  │
│  STRATEGY BREAKDOWN                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Strategy          │ PnL    │ Trades │ Win%  │ Status   │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ PM Calibration    │ $XXX   │ XX     │ XX%   │ 🟢 Active│    │
│  │ PM Stat Arb       │ $XXX   │ XX     │ XX%   │ 🟢 Active│    │
│  │ Kalshi Longshot   │ $XXX   │ XX     │ XX%   │ 🟢 Active│    │
│  │ Kalshi Calibration│ $XXX   │ XX     │ XX%   │ 🟢 Active│    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ALERTS                                                          │
│  ⚠️  Calibration drift detected (PM politics)                   │
│  ✅  All systems operational                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Alert Triggers

| Alert | Condition | Action |
|-------|-----------|--------|
| 🔴 CRITICAL | Daily loss > 25% | Stop all trading |
| 🔴 CRITICAL | API failure > 5 min | Stop trading, notify |
| 🟡 WARNING | Sharpe(7d) < 1.0 | Review strategy |
| 🟡 WARNING | Fill rate < 50% expected | Check liquidity |
| 🟡 WARNING | Calibration drift > 20% | Recalibrate |
| 🟢 INFO | New category opportunity | Log for review |

### 4.3 Automated Safeguards

```python
class RiskManager:
    def check_pre_trade(self, order):
        """Pre-trade risk checks."""
        checks = [
            self.check_position_limit(order),
            self.check_daily_loss_limit(),
            self.check_drawdown_limit(),
            self.check_liquidity(order),
            self.check_concentration(order),
        ]
        return all(checks)
    
    def check_post_trade(self, fill):
        """Post-trade monitoring."""
        self.update_pnl(fill)
        self.check_for_alerts()
        self.log_trade(fill)
```

---

## Implementation Tasks

### Week 1: Foundation
- [ ] Create `trading/` directory structure
- [ ] Implement Polymarket CLOB API client
- [ ] Implement Kalshi REST API client
- [ ] Set up PostgreSQL schema for trades
- [ ] Create base Signal and Order classes

### Week 2: Signal Generation
- [ ] Port calibration strategy to production code
- [ ] Implement Kelly sizing with risk limits
- [ ] Add category filtering
- [ ] Create signal logging

### Week 3: Execution
- [ ] Implement paper trading mode
- [ ] Add order management (submit, cancel, track)
- [ ] Implement fill tracking
- [ ] Calculate realized slippage

### Week 4: Monitoring
- [ ] Create monitoring dashboard
- [ ] Set up alert system (email/Slack)
- [ ] Implement daily PnL reports
- [ ] Add strategy performance tracking

### Week 5-8: Live Trading
- [ ] Deploy to production server
- [ ] Start Tier 1 ($2K)
- [ ] Monitor and iterate
- [ ] Scale based on criteria

---

## Regulatory Considerations

### Polymarket
- Based in crypto/offshore
- No US restrictions (use VPN if needed)
- Deposits via USDC on Polygon

### Kalshi
- CFTC-regulated
- US-compliant
- Bank/wire transfers
- Tax reporting (1099s)

### Tax Implications
- Track all trades for tax purposes
- Prediction markets may be treated as gambling or derivatives
- Consult tax professional for proper treatment

---

## Risk Disclosure

⚠️ **WARNING**: This is a speculative trading system. Risks include:
- Total loss of capital
- Regulatory changes
- API/platform failures
- Model degradation
- Execution slippage
- Adverse selection

**Start small. Scale gradually. Monitor continuously.**

---

## Next Steps

1. **Immediate**: Create `experiments/trading/` directory structure
2. **This week**: Implement API clients for both platforms
3. **Next week**: Paper trading deployment
4. **Week 3**: Small capital test ($1K each platform)

---

*Document created: 2026-01-01*
*Last updated: 2026-01-01*
