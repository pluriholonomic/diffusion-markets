"""
Blackwell Arbitrage Backtesting Framework.

This module provides a comprehensive backtesting framework for trading strategies
based on the learned Blackwell constraint set C_t from diffusion models.

Key components:
- engine: Main BacktestEngine orchestration
- strategies: Trading strategies (OnlineMaxArb, StatArbPortfolio, ConditionalGraph)
- metrics: PnL tracking, regret curves, H4 validation
- data: CLOB loading and group registry
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "BacktestConfig":
        from backtest.config import BacktestConfig
        return BacktestConfig
    if name == "StrategyConfig":
        from backtest.config import StrategyConfig
        return StrategyConfig
    if name == "BacktestEngine":
        from backtest.engine import BacktestEngine
        return BacktestEngine
    if name == "BacktestResults":
        from backtest.engine import BacktestResults
        return BacktestResults
    raise AttributeError(f"module 'backtest' has no attribute {name!r}")

__all__ = [
    "BacktestConfig",
    "StrategyConfig",
    "BacktestEngine",
    "BacktestResults",
]

