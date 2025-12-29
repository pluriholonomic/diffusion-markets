"""
Backtest Engine.

Main orchestration loop for running backtests with the Blackwell
arbitrage framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig, DailyMetrics
from backtest.ct_loader import CtCheckpointLoader, CtCheckpointSpec
from backtest.data.clob_loader import MarketEvent, load_clob_series, load_resolutions
from backtest.data.group_registry import GroupRegistry, build_group_registry
from backtest.market_state import MarketStateManager
from backtest.strategies import (
    BaseStrategy,
    ConditionalGraphStrategy,
    OnlineMaxArbStrategy,
    StatArbPortfolioStrategy,
)
from backtest.strategies.conditional_graph import ConditionalGraphConfig
from backtest.strategies.online_max_arb import OnlineMaxArbConfig
from backtest.strategies.stat_arb import StatArbConfig


@dataclass
class BacktestResults:
    """Results from a backtest run."""

    config: BacktestConfig
    start_date: str
    end_date: str

    # Aggregate metrics
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    n_trades: int = 0
    n_days: int = 0

    # Risk metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # Per-strategy breakdown
    strategy_pnl: Dict[str, float] = field(default_factory=dict)
    strategy_trades: Dict[str, int] = field(default_factory=dict)

    # H4 validation
    h4_correlation: float = 0.0
    h4_distances: List[float] = field(default_factory=list)
    h4_pnls: List[float] = field(default_factory=list)

    # Regret curves
    regret_curves: Dict[str, List[float]] = field(default_factory=dict)

    # Daily breakdown
    daily_metrics: List[DailyMetrics] = field(default_factory=list)

    # C_t validation
    ct_validation_results: List[Dict] = field(default_factory=list)
    ct_sufficient_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_pnl": self.total_pnl,
            "gross_pnl": self.gross_pnl,
            "total_costs": self.total_costs,
            "n_trades": self.n_trades,
            "n_days": self.n_days,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "strategy_pnl": self.strategy_pnl,
            "strategy_trades": self.strategy_trades,
            "h4_correlation": self.h4_correlation,
            "ct_sufficient_rate": self.ct_sufficient_rate,
        }

    def save(self, output_dir: Path) -> None:
        """Save results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        with open(output_dir / "summary.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Daily PnL CSV
        if self.daily_metrics:
            daily_df = pd.DataFrame([
                {"date": m.date, "pnl": m.pnl, "n_trades": m.n_trades}
                for m in self.daily_metrics
            ])
            daily_df.to_csv(output_dir / "daily_pnl.csv", index=False)

        # H4 validation
        if self.h4_distances and self.h4_pnls:
            h4_df = pd.DataFrame({
                "distance_to_ct": self.h4_distances,
                "realized_pnl": self.h4_pnls,
            })
            h4_df.to_csv(output_dir / "h4_validation.csv", index=False)

        # Regret curves
        if self.regret_curves:
            with open(output_dir / "regret_curves.json", "w") as f:
                json.dump(self.regret_curves, f, indent=2)

        # C_t validation results
        if self.ct_validation_results:
            # Save summary
            ct_summary = {
                "n_validations": len(self.ct_validation_results),
                "sufficient_rate": self.ct_sufficient_rate,
                "per_day": [
                    {
                        "date": r.get("date"),
                        "n_samples": r.get("n_samples"),
                        "n_markets": r.get("n_markets"),
                        "sufficient": r.get("sufficient"),
                        "stability_ok": r.get("stability_ok"),
                        "coverage_ok": r.get("coverage_ok"),
                        "split_half_diff": r.get("split_half_diff"),
                    }
                    for r in self.ct_validation_results
                ],
            }
            with open(output_dir / "ct_validation.json", "w") as f:
                json.dump(ct_summary, f, indent=2)


class BacktestEngine:
    """
    Main orchestration engine for backtesting.

    Coordinates:
    - Data loading and iteration
    - C_t checkpoint loading
    - Strategy execution
    - Metrics tracking
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

        # Load data
        self.clob_data = self._load_clob_data()
        self.resolutions = self._load_resolutions()

        # Build group registry
        self.group_registry = self._build_group_registry()

        # Initialize C_t loader
        self.ct_loader = CtCheckpointLoader(CtCheckpointSpec(
            checkpoint_dir=cfg.checkpoint_dir,
            model_type=cfg.model_type,
            embed_dim=cfg.embed_dim,
            bundle_size=cfg.bundle_size,
        ))

        # Initialize market state
        self.market_state = MarketStateManager(
            clob_data=self.clob_data,
            group_registry=self.group_registry,
            resolutions=self.resolutions,
            snapshot_interval_seconds=cfg.snapshot_interval_seconds,
            price_change_threshold=cfg.price_change_threshold,
        )

        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        self._init_strategies()

        # Results tracking
        self._daily_pnls: List[float] = []
        self._h4_distances: List[float] = []
        self._h4_pnls: List[float] = []
        self._cumulative_pnl: float = 0.0

        # C_t validation
        self._ct_validator = None
        self._ct_validation_results: List[Dict] = []
        if cfg.validate_ct:
            from backtest.metrics.ct_validation import CtValidator, CtValidationConfig
            val_cfg = CtValidationConfig(
                sample_counts=(16, 32, 64, 128, 256),
                n_test_samples=cfg.ct_validation_samples,
                coverage_threshold=cfg.ct_min_coverage,
                distance_tolerance=cfg.ct_distance_tolerance,
            )
            self._ct_validator = CtValidator(val_cfg)

    def _load_clob_data(self) -> pd.DataFrame:
        """Load CLOB data for the backtest period."""
        if self.cfg.verbose:
            print(f"Loading CLOB data from {self.cfg.clob_data_path}")

        df = load_clob_series(
            self.cfg.clob_data_path,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
        )

        if self.cfg.verbose:
            print(f"  Loaded {len(df)} CLOB snapshots for {df['market_id'].nunique()} markets")

        return df

    def _load_resolutions(self) -> Optional[pd.DataFrame]:
        """Load resolution data if available."""
        if self.cfg.resolution_data_path is None:
            return None

        if not self.cfg.resolution_data_path.exists():
            if self.cfg.verbose:
                print(f"Resolution data not found: {self.cfg.resolution_data_path}")
            return None

        if self.cfg.verbose:
            print(f"Loading resolutions from {self.cfg.resolution_data_path}")

        return load_resolutions(self.cfg.resolution_data_path)

    def _build_group_registry(self) -> GroupRegistry:
        """Build group registry from CLOB data."""
        # Get unique markets with their metadata
        market_df = self.clob_data.drop_duplicates(subset=["market_id"]).copy()

        return build_group_registry(
            market_df,
            market_id_col="market_id",
            text_cols=list(self.cfg.text_cols),
            category_col=self.cfg.category_col if self.cfg.category_col in market_df.columns else None,
            allowed_topics=self.cfg.allowed_topics,
            allowed_cross_group=not self.cfg.enforce_group_boundaries,
        )

    def _init_strategies(self) -> None:
        """Initialize trading strategies."""
        for name in self.cfg.strategies:
            params = self.cfg.strategy_configs.get(name, {})

            if name == "online_max_arb":
                cfg = OnlineMaxArbConfig(
                    B_max=params.get("B_max", 1.0),
                    transaction_cost=self.cfg.transaction_cost,
                )
                self.strategies[name] = OnlineMaxArbStrategy(cfg)

            elif name == "stat_arb":
                cfg = StatArbConfig(
                    risk_aversion=params.get("risk_aversion", 1.0),
                    max_gross_exposure=params.get("max_gross_exposure", 5.0),
                    max_position_per_market=params.get("max_position", 1.0),
                )
                self.strategies[name] = StatArbPortfolioStrategy(cfg)

            elif name == "conditional_graph":
                cfg = ConditionalGraphConfig(
                    edge_threshold=params.get("edge_threshold", 0.1),
                    flip_threshold=params.get("flip_threshold", 0.4),
                    max_position=params.get("max_position", 1.0),
                )
                self.strategies[name] = ConditionalGraphStrategy(cfg)

            else:
                if self.cfg.verbose:
                    print(f"Unknown strategy: {name}")

        if self.cfg.verbose:
            print(f"Initialized {len(self.strategies)} strategies: {list(self.strategies.keys())}")

    def _date_range(self) -> Iterator[str]:
        """Iterate over dates in the backtest range."""
        start = datetime.strptime(self.cfg.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.cfg.end_date, "%Y-%m-%d")

        current = start
        while current <= end:
            yield current.strftime("%Y-%m-%d")
            current += timedelta(days=1)

    def _load_ct_for_date(self, date: str) -> Tuple[np.ndarray, List[str]]:
        """Load C_t checkpoint and sample for a date."""
        try:
            self.ct_loader.load_for_date(date)
        except FileNotFoundError:
            if self.cfg.verbose:
                print(f"  No checkpoint for {date}, using previous")
            if not self.ct_loader.is_loaded():
                return np.array([]), []

        # Get active markets
        active_markets = self.market_state.get_active_markets()

        # Filter by group if enforcing boundaries
        if self.cfg.enforce_group_boundaries:
            active_markets = self.group_registry.get_tradeable_bundle(active_markets)

        if not active_markets:
            return np.array([]), []

        # For now, create dummy embeddings (in production, load from cache)
        # This is a placeholder - real implementation would load embeddings
        embeddings = {m: np.random.randn(self.cfg.embed_dim).astype(np.float32)
                      for m in active_markets}

        # Sample C_t
        try:
            samples, valid_ids = self.ct_loader.sample_ct_for_markets(
                market_ids=active_markets,
                embeddings=embeddings,
                n_samples=self.cfg.ct_mc_samples,
                seed=hash(date) % (2**31),
            )
            return samples, valid_ids
        except Exception as e:
            if self.cfg.verbose:
                print(f"  Error sampling C_t: {e}")
            return np.array([]), []

    def _refresh_strategies(self, ct_samples: np.ndarray, market_ids: List[str]) -> None:
        """Refresh all strategies with new C_t."""
        for strategy in self.strategies.values():
            strategy.on_ct_refresh(ct_samples, market_ids)

    def _validate_ct_samples(
        self,
        ct_samples: np.ndarray,
        market_ids: List[str],
        date: str,
    ) -> Optional[Dict]:
        """
        Validate that C_t samples are sufficient.

        Runs quick validation tests to check if the number of MC samples
        is enough to accurately represent the learned constraint set.

        Args:
            ct_samples: (mc, k) samples from diffusion model
            market_ids: List of market IDs
            date: Current date for logging

        Returns:
            Validation results dict, or None if validation disabled
        """
        if self._ct_validator is None or ct_samples.size == 0:
            return None

        # Get current market prices for projection test
        q = self.market_state.get_bundle_prices(market_ids)
        valid_mask = ~np.isnan(q)
        if not valid_mask.any():
            return None

        q_valid = q[valid_mask]
        samples_valid = ct_samples[:, valid_mask]

        if samples_valid.shape[0] < 2 or samples_valid.shape[1] < 1:
            return None

        # Run quick validation
        from backtest.metrics.ct_validation import quick_validate_ct
        result = quick_validate_ct(
            samples=samples_valid,
            q=q_valid,
            n_test=min(100, samples_valid.shape[0]),
            seed=hash(date) % (2**31),
        )

        # Add date and market info
        result["date"] = date
        result["market_ids"] = [m for m, v in zip(market_ids, valid_mask) if v]

        # Check sufficiency
        sufficient = result.get("stability_ok", False) and result.get("coverage_ok", False)
        result["sufficient"] = sufficient

        # Warn if insufficient
        if not sufficient and self.cfg.ct_warn_on_insufficient:
            import warnings
            issues = []
            if not result.get("stability_ok", False):
                issues.append(f"split-half diff={result['split_half_diff']:.4f}")
            if not result.get("coverage_ok", False):
                issues.append(f"LOO coverage low")
            warnings.warn(
                f"C_t samples may be insufficient on {date} for {len(result['market_ids'])} markets: "
                f"{'; '.join(issues)}. Consider increasing ct_mc_samples."
            )

        self._ct_validation_results.append(result)
        return result

    def _process_event(self, event: MarketEvent) -> Dict[str, Dict[str, float]]:
        """
        Process a single market event.

        Returns dict of strategy -> {market_id -> position}.
        """
        self.market_state.update(event)

        all_positions: Dict[str, Dict[str, float]] = {}

        if event.type == "price_update":
            # Let strategies react to price changes
            for name, strategy in self.strategies.items():
                positions = strategy.on_price_update(
                    market_id=event.market_id,
                    old_price=event.data["old_price"],
                    new_price=event.data["new_price"],
                    state=self.market_state,
                )
                if positions:
                    all_positions[name] = positions

                    # Track H4 distance if available
                    if hasattr(strategy, "state") and "last_dist_to_ct" in strategy.state.custom:
                        self._h4_distances.append(strategy.state.custom["last_dist_to_ct"])

        elif event.type == "snapshot":
            # Periodic strategy decisions
            for name, strategy in self.strategies.items():
                positions = strategy.on_snapshot(
                    timestamp=event.timestamp,
                    state=self.market_state,
                )
                if positions:
                    all_positions[name] = positions

                    # Track H4 distance if available
                    if hasattr(strategy, "state") and "last_dist_to_ct" in strategy.state.custom:
                        self._h4_distances.append(strategy.state.custom["last_dist_to_ct"])

        elif event.type == "resolution":
            # Update strategies with outcome
            for strategy in self.strategies.values():
                strategy.on_resolution(
                    market_id=event.market_id,
                    outcome=event.data["outcome"],
                )

            # Track H4 PnL
            # This is simplified - would need to match with distances
            if self._h4_distances:
                self._h4_pnls.append(0.0)  # Placeholder

        return all_positions

    def _execute_positions(
        self,
        positions: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Execute position changes across strategies.

        Returns total PnL from trades.
        """
        total_pnl = 0.0

        for strategy_name, market_positions in positions.items():
            for market_id, target_position in market_positions.items():
                # Apply group filter
                if self.cfg.enforce_group_boundaries:
                    valid = self.group_registry.get_tradeable_bundle([market_id])
                    if not valid:
                        continue

                # Apply position limits
                target_position = np.clip(
                    target_position,
                    -self.cfg.max_position_usd,
                    self.cfg.max_position_usd,
                )

                # Execute trade
                trade = self.market_state.open_position(
                    market_id=market_id,
                    size=target_position,
                    strategy=strategy_name,
                    transaction_cost=self.cfg.transaction_cost,
                )

                if trade:
                    total_pnl -= trade.cost  # Costs reduce PnL

        return total_pnl

    def run(self) -> BacktestResults:
        """
        Run the full backtest.

        Returns:
            BacktestResults with all metrics
        """
        if self.cfg.verbose:
            print(f"Starting backtest from {self.cfg.start_date} to {self.cfg.end_date}")

        # Reset state
        self.market_state.reset()
        self._daily_pnls = []
        self._h4_distances = []
        self._h4_pnls = []
        self._cumulative_pnl = 0.0

        results = BacktestResults(
            config=self.cfg,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
        )

        # Main loop over dates
        for date in self._date_range():
            daily_pnl = self._run_day(date)

            # Record daily metrics
            self._daily_pnls.append(daily_pnl)
            self._cumulative_pnl += daily_pnl

            metrics = DailyMetrics(
                date=date,
                pnl=daily_pnl,
                gross_pnl=daily_pnl,  # Simplified
                costs=0.0,  # Would need to track separately
                n_trades=len(self.market_state.trades),
                n_positions=sum(len(p) for p in self.market_state.positions.values()),
                max_drawdown=0.0,  # Would need running calculation
                sharpe_daily=0.0,
            )
            results.daily_metrics.append(metrics)
            results.n_days += 1

            if self.cfg.verbose and results.n_days % 10 == 0:
                print(f"  Day {results.n_days}: cumulative PnL = {self._cumulative_pnl:.4f}")

        # Compile final results
        results = self._compile_results(results)

        # Save results
        if self.cfg.output_dir:
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            results.save(self.cfg.output_dir)

            # Save config
            with open(self.cfg.output_dir / "config.json", "w") as f:
                json.dump({
                    "start_date": self.cfg.start_date,
                    "end_date": self.cfg.end_date,
                    "strategies": list(self.cfg.strategies),
                    "enforce_group_boundaries": self.cfg.enforce_group_boundaries,
                    "allowed_topics": self.cfg.allowed_topics,
                }, f, indent=2)

        if self.cfg.verbose:
            print(f"Backtest complete. Total PnL: {results.total_pnl:.4f}")

        return results

    def _run_day(self, date: str) -> float:
        """Run a single day of the backtest."""
        # Load C_t for this day
        ct_samples, market_ids = self._load_ct_for_date(date)

        if len(ct_samples) > 0 and len(market_ids) > 0:
            # Validate C_t samples if enabled
            if self.cfg.validate_ct:
                self._validate_ct_samples(ct_samples, market_ids, date)

            self._refresh_strategies(ct_samples, market_ids)

        # Process events for this day
        daily_pnl = 0.0

        for event in self.market_state.events_for_date(date):
            positions = self._process_event(event)

            if positions:
                pnl = self._execute_positions(positions)
                daily_pnl += pnl

        return daily_pnl

    def _compile_results(self, results: BacktestResults) -> BacktestResults:
        """Compile final backtest results."""
        # Total PnL
        results.total_pnl = sum(self._daily_pnls)
        results.n_trades = len(self.market_state.trades)

        # Per-strategy breakdown
        for name in self.strategies:
            results.strategy_pnl[name] = self.market_state.get_realized_pnl(name)
            results.strategy_trades[name] = len([
                t for t in self.market_state.trades if t.strategy == name
            ])

        # Risk metrics
        if len(self._daily_pnls) > 1:
            daily_returns = np.array(self._daily_pnls)
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                results.sharpe_ratio = mean_return / std_return * np.sqrt(252)

            # Max drawdown
            cumsum = np.cumsum(daily_returns)
            running_max = np.maximum.accumulate(cumsum)
            drawdown = running_max - cumsum
            results.max_drawdown = float(np.max(drawdown))

        # Win rate
        if results.n_trades > 0:
            wins = sum(1 for t in self.market_state.trades if t.pnl > 0)
            results.win_rate = wins / results.n_trades

        # H4 validation
        results.h4_distances = self._h4_distances
        results.h4_pnls = self._h4_pnls

        if len(self._h4_distances) > 10 and len(self._h4_pnls) > 10:
            min_len = min(len(self._h4_distances), len(self._h4_pnls))
            corr = np.corrcoef(
                self._h4_distances[:min_len],
                self._h4_pnls[:min_len]
            )[0, 1]
            if np.isfinite(corr):
                results.h4_correlation = float(corr)

        # Regret curves
        for name, strategy in self.strategies.items():
            if hasattr(strategy, "get_regret_curve"):
                results.regret_curves[name] = strategy.get_regret_curve()

        # C_t validation results
        results.ct_validation_results = self._ct_validation_results
        if self._ct_validation_results:
            sufficient_count = sum(
                1 for r in self._ct_validation_results if r.get("sufficient", False)
            )
            results.ct_sufficient_rate = sufficient_count / len(self._ct_validation_results)

            if self.cfg.verbose:
                print(f"C_t validation: {results.ct_sufficient_rate:.1%} of days had sufficient samples")

        return results

