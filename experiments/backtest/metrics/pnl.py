"""
PnL tracking and risk metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PnLTracker:
    """
    Track PnL over time with per-strategy attribution.
    """

    # Per-strategy cumulative PnL
    strategy_pnl: Dict[str, float] = field(default_factory=dict)
    strategy_trades: Dict[str, int] = field(default_factory=dict)

    # Time series
    daily_pnl: List[float] = field(default_factory=list)
    cumulative_pnl: List[float] = field(default_factory=list)

    # Trade-level
    trade_pnls: List[float] = field(default_factory=list)

    def record_trade(
        self,
        strategy: str,
        pnl: float,
        cost: float = 0.0,
    ) -> None:
        """Record a single trade."""
        net_pnl = pnl - cost

        if strategy not in self.strategy_pnl:
            self.strategy_pnl[strategy] = 0.0
            self.strategy_trades[strategy] = 0

        self.strategy_pnl[strategy] += net_pnl
        self.strategy_trades[strategy] += 1
        self.trade_pnls.append(net_pnl)

    def record_daily(self, pnl: float) -> None:
        """Record end-of-day PnL."""
        self.daily_pnl.append(pnl)
        prev = self.cumulative_pnl[-1] if self.cumulative_pnl else 0.0
        self.cumulative_pnl.append(prev + pnl)

    def get_total_pnl(self) -> float:
        """Get total PnL across all strategies."""
        return sum(self.strategy_pnl.values())

    def get_n_trades(self) -> int:
        """Get total number of trades."""
        return sum(self.strategy_trades.values())

    def get_sharpe_ratio(self, annualization: float = 252.0) -> float:
        """
        Compute Sharpe ratio from daily returns.

        Args:
            annualization: Trading days per year

        Returns:
            Annualized Sharpe ratio
        """
        if len(self.daily_pnl) < 2:
            return 0.0

        returns = np.array(self.daily_pnl)
        mean = np.mean(returns)
        std = np.std(returns)

        if std < 1e-12:
            return 0.0

        return float(mean / std * np.sqrt(annualization))

    def get_max_drawdown(self) -> float:
        """
        Compute maximum drawdown.

        Returns:
            Maximum drawdown (positive number)
        """
        if not self.cumulative_pnl:
            return 0.0

        cumsum = np.array(self.cumulative_pnl)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum

        return float(np.max(drawdown))

    def get_win_rate(self) -> float:
        """Get fraction of winning trades."""
        if not self.trade_pnls:
            return 0.0

        wins = sum(1 for p in self.trade_pnls if p > 0)
        return wins / len(self.trade_pnls)

    def get_profit_factor(self) -> float:
        """
        Compute profit factor (gross profits / gross losses).

        Returns:
            Profit factor (>1 is profitable)
        """
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))

        if gross_loss < 1e-12:
            return float("inf") if gross_profit > 0 else 1.0

        return gross_profit / gross_loss

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_pnl": self.get_total_pnl(),
            "n_trades": self.get_n_trades(),
            "n_days": len(self.daily_pnl),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "max_drawdown": self.get_max_drawdown(),
            "win_rate": self.get_win_rate(),
            "profit_factor": self.get_profit_factor(),
            "strategy_pnl": dict(self.strategy_pnl),
            "strategy_trades": dict(self.strategy_trades),
        }


def compute_risk_metrics(
    daily_pnl: List[float],
    annualization: float = 252.0,
) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics from daily PnL.

    Args:
        daily_pnl: List of daily PnL values
        annualization: Trading days per year

    Returns:
        Dict with risk metrics
    """
    if len(daily_pnl) < 2:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "mean_daily_pnl": 0.0,
            "std_daily_pnl": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    returns = np.array(daily_pnl)
    mean = np.mean(returns)
    std = np.std(returns)

    # Sharpe ratio
    sharpe = mean / std * np.sqrt(annualization) if std > 1e-12 else 0.0

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else std
    sortino = mean / downside_std * np.sqrt(annualization) if downside_std > 1e-12 else 0.0

    # Max drawdown
    cumsum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = float(np.max(drawdown))

    # Calmar ratio (return / max drawdown)
    total_return = float(np.sum(returns))
    calmar = total_return / max_dd if max_dd > 1e-12 else 0.0

    # Higher moments
    skewness = float(np.mean(((returns - mean) / std) ** 3)) if std > 1e-12 else 0.0
    kurtosis = float(np.mean(((returns - mean) / std) ** 4) - 3) if std > 1e-12 else 0.0

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_dd,
        "calmar_ratio": float(calmar),
        "mean_daily_pnl": float(mean),
        "std_daily_pnl": float(std),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "total_return": total_return,
        "n_days": len(daily_pnl),
    }



