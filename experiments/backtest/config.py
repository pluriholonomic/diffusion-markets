"""
Configuration dataclasses for the backtesting framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for a single trading strategy."""

    name: str
    enabled: bool = True

    # Position sizing
    max_position_per_market: float = 1.0  # Max position in a single market
    max_gross_exposure: float = 10.0  # Max total |positions|

    # Strategy-specific parameters
    params: dict = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """
    Master configuration for a backtest run.

    Controls data sources, strategies, execution parameters, and output.
    """

    # Date range
    start_date: str
    end_date: str

    # Data paths
    checkpoint_dir: Path  # Directory with date-named C_t model checkpoints
    clob_data_path: Path  # Path to CLOB parquet file
    resolution_data_path: Optional[Path] = None  # Optional resolution data

    # Model configuration
    model_type: str = "bundle"  # "bundle" or "single"
    embed_dim: int = 4096  # Embedding dimension for the model
    bundle_size: int = 8  # Number of markets per bundle

    # C_t sampling
    ct_mc_samples: int = 64  # Number of MC samples for C_t representation
    ct_refresh_freq: str = "daily"  # When to reload C_t checkpoint

    # C_t generation mode (new multi-model support)
    ct_mode: str = "legacy"  # "single", "union", or "legacy" (backward compat)
    ct_model: str = "legacy"  # "ar_diffusion", "rlcr", "bundle", "legacy"
    
    # Model paths for multi-model C_t generation
    ar_diffusion_checkpoint: Optional[Path] = None  # AR+Diffusion hybrid checkpoint
    rlcr_model: Optional[str] = None  # RLCR model path (HF or local)
    bundle_checkpoint: Optional[Path] = None  # Bundle diffusion checkpoint
    
    # Per-model sampling configuration
    ar_diffusion_samples: int = 16  # Samples from AR+Diffusion
    rlcr_K: int = 5  # Self-consistency samples from RLCR
    bundle_samples: int = 16  # Samples from bundle diffusion
    
    # Which models to include in UNION mode
    union_models: Tuple[str, ...] = ("ar_diffusion", "rlcr", "bundle")
    
    # Embedding model for multi-model loaders
    ct_embed_model: str = "all-MiniLM-L6-v2"
    ct_embed_dim: int = 384

    # C_t validation
    validate_ct: bool = True  # Run C_t sample sufficiency validation
    ct_validation_samples: int = 256  # Extra samples for validation tests
    ct_validation_freq: str = "daily"  # How often to run validation
    ct_min_coverage: float = 0.90  # Min fraction of test samples within tolerance
    ct_distance_tolerance: float = 0.05  # Distance tolerance for "close"
    ct_warn_on_insufficient: bool = True  # Warn if samples appear insufficient

    # Strategy selection
    strategies: Tuple[str, ...] = ("online_max_arb", "stat_arb", "conditional_graph")
    strategy_configs: dict = field(default_factory=dict)  # name -> StrategyConfig.params

    # Blackwell calibration strategy settings (validated parameters)
    blackwell_n_bins: int = 10  # Number of price bins
    blackwell_g_threshold: float = 0.05  # Minimum |g̅| to trade
    blackwell_t_stat_threshold: float = 0.0  # t-stat filter (0=disabled, use g̅ only)
    blackwell_lookback: int = 2000  # Rolling window size (larger = more stable)
    blackwell_recalibrate_freq: int = 100  # Recalibrate every N resolutions
    blackwell_use_risk_parity: bool = True  # Size by inverse max loss
    blackwell_target_max_loss: float = 0.2  # Target max loss per trade
    blackwell_leverage: float = 1.5  # Position multiplier

    # Group robustness
    enforce_group_boundaries: bool = True
    allowed_topics: Optional[List[str]] = None  # None = all topics
    text_cols: Tuple[str, ...] = ("question", "slug", "description")
    category_col: str = "category"

    # Execution simulation
    slippage_bps: float = 5.0  # Slippage in basis points
    transaction_cost: float = 0.002  # Per-trade cost
    max_position_usd: float = 1000.0  # Max position per market in USD
    latency_ms: float = 100.0  # Simulated latency

    # Event generation
    snapshot_interval_seconds: float = 300.0  # 5 minutes
    price_change_threshold: float = 0.01  # Min price change for event

    # Output
    output_dir: Path = Path("runs/backtest")
    save_positions: bool = True
    save_trades: bool = True
    generate_plots: bool = True

    # Misc
    seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.checkpoint_dir, str):
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        if isinstance(self.clob_data_path, str):
            object.__setattr__(self, "clob_data_path", Path(self.clob_data_path))
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.resolution_data_path and isinstance(self.resolution_data_path, str):
            object.__setattr__(self, "resolution_data_path", Path(self.resolution_data_path))
        # Multi-model paths
        if self.ar_diffusion_checkpoint and isinstance(self.ar_diffusion_checkpoint, str):
            object.__setattr__(self, "ar_diffusion_checkpoint", Path(self.ar_diffusion_checkpoint))
        if self.bundle_checkpoint and isinstance(self.bundle_checkpoint, str):
            object.__setattr__(self, "bundle_checkpoint", Path(self.bundle_checkpoint))

    def get_strategy_config(self, name: str) -> StrategyConfig:
        """Get configuration for a specific strategy."""
        params = self.strategy_configs.get(name, {})
        return StrategyConfig(name=name, params=params)


@dataclass
class ExecutionResult:
    """Result of executing a trade."""

    market_id: str
    timestamp: float
    requested_position: float
    executed_position: float
    price: float
    slippage: float
    cost: float
    success: bool
    reason: str = ""


@dataclass
class DailyMetrics:
    """Metrics for a single day of backtesting."""

    date: str
    pnl: float
    gross_pnl: float  # Before costs
    costs: float
    n_trades: int
    n_positions: int
    max_drawdown: float
    sharpe_daily: float

    # Per-strategy breakdown
    strategy_pnl: dict = field(default_factory=dict)

    # H4 validation
    mean_dist_to_ct: float = 0.0
    dist_pnl_correlation: float = 0.0

