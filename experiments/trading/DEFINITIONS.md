# Trading System Definitions

This document defines key terms used consistently across paper trading, backtesting, and optimization.

## Metric Definitions

### Signals, Orders, and Trades

| Term | Paper Trading | Backtesting | Description |
|------|--------------|-------------|-------------|
| **Signal** | Trading opportunity detected by strategy | N/A (implicit) | Raw output from strategy |
| **Order Attempt** | Order placed for a unique market | N/A | Orders that pass risk checks (excludes duplicate markets) |
| **Position/Trade** | Order that resulted in an open position | Each simulated trade | An actual position entry |
| **Total Trades** | `open + wins + losses` | Count of trades executed | Unique positions ever entered |

### Rate Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Fill Rate** | `positions_opened / order_attempts` | % of unique orders that became positions |
| **Win Rate** | `wins / (wins + losses)` | % of closed trades that were profitable |

### P&L Definitions

| Term | Description |
|------|-------------|
| **Unrealized P&L** | Mark-to-market value of open positions |
| **Realized P&L** | P&L from closed/resolved positions |
| **Total P&L** | Unrealized + Realized |

## Current Risk Limits

### Global Risk Limits (`RiskLimits`)
- `max_position_pct`: 10% of bankroll per position
- `max_daily_loss_pct`: 20% daily loss stop
- `max_drawdown_pct`: 30% total drawdown stop
- `kelly_fraction`: 0.25 (Kelly scaling)
- `min_edge`: 5% minimum edge to trade
- `min_liquidity`: $1,000 minimum market liquidity
- `max_concentration`: 30% max exposure per category

### Position Management (`EngineConfig` / `PositionManagerConfig`)
- `profit_take_pct`: +20% take profit threshold
- `stop_loss_pct`: -20% stop loss threshold
- `max_positions`: 50 concurrent positions
- `max_position_per_market`: 1 position per market
- `max_position_size`: $1,000 per position
- `max_hold_time_hours`: 72 hours forced close

### Execution Limits
- `max_orders_per_run`: 20 orders per cycle
- `min_signal_confidence`: 0.3 minimum confidence

## Consistency Notes

1. **Duplicate Filtering**: In paper trading, signals for markets with existing positions are filtered out BEFORE order logging. This ensures `order_attempts` only counts unique market attempts.

2. **Backtesting vs Paper Trading**: Backtesting simulates every trade with 100% fill rate. There's no concept of "order attempts" - each data point that triggers a trade IS a trade.

3. **Threshold Consistency**: `EngineConfig.profit_take_pct/stop_loss_pct` MUST match `PositionManagerConfig.default_profit_take_pct/default_stop_loss_pct` (both currently 20%/20%).
