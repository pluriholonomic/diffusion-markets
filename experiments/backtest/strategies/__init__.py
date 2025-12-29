"""Trading strategies for the Blackwell backtesting framework."""

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "BaseStrategy":
        from backtest.strategies.base import BaseStrategy
        return BaseStrategy
    if name == "StrategyState":
        from backtest.strategies.base import StrategyState
        return StrategyState
    if name == "OnlineMaxArbStrategy":
        from backtest.strategies.online_max_arb import OnlineMaxArbStrategy
        return OnlineMaxArbStrategy
    if name == "StatArbPortfolioStrategy":
        from backtest.strategies.stat_arb import StatArbPortfolioStrategy
        return StatArbPortfolioStrategy
    if name == "ConditionalGraphStrategy":
        from backtest.strategies.conditional_graph import ConditionalGraphStrategy
        return ConditionalGraphStrategy
    if name == "ConfidenceGatedStrategy":
        from backtest.strategies.confidence_gated import ConfidenceGatedStrategy
        return ConfidenceGatedStrategy
    if name == "ConfidenceGatedConfig":
        from backtest.strategies.confidence_gated import ConfidenceGatedConfig
        return ConfidenceGatedConfig
    raise AttributeError(f"module 'backtest.strategies' has no attribute {name!r}")

__all__ = [
    "BaseStrategy",
    "StrategyState",
    "OnlineMaxArbStrategy",
    "StatArbPortfolioStrategy",
    "ConditionalGraphStrategy",
    "ConfidenceGatedStrategy",
    "ConfidenceGatedConfig",
]

