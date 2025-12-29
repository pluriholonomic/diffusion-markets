"""Execution simulation and cost estimation."""

from backtest.execution.cost_model import (
    CostModelConfig,
    ExecutionCostEstimate,
    ExecutionCostModel,
    MarketImpactEstimator,
    SpreadEstimator,
    quick_cost_estimate,
)

# Lazy import for CLOB fetcher (requires py-clob-client)
def __getattr__(name: str):
    if name in ("CLOBFetcher", "OrderBookSnapshot", "fetch_and_cache_clob_data"):
        from backtest.execution.clob_fetcher import (
            CLOBFetcher,
            OrderBookSnapshot,
            fetch_and_cache_clob_data,
        )
        return locals()[name]
    raise AttributeError(f"module 'backtest.execution' has no attribute {name!r}")

__all__ = [
    "CostModelConfig",
    "ExecutionCostEstimate",
    "ExecutionCostModel",
    "MarketImpactEstimator",
    "SpreadEstimator",
    "quick_cost_estimate",
    "CLOBFetcher",
    "OrderBookSnapshot",
    "fetch_and_cache_clob_data",
]

