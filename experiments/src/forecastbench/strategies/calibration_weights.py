"""
Calibration-Deviation Dynamic Weights for Prediction Market Portfolios

Implements the weight function:
    aᵢ(t) = f(|calibration|, price_deviation, volume)

Where:
    - |calibration| = |y_rate - price_mean| for category
    - price_deviation = how far current price is from fair value
    - volume = trading liquidity

Reference: Dynamic Ensemble Model for PM Stat Arb
    π(t) = Σᵢ aᵢ(t) πᵢ(t)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class CalibrationState:
    """Stores calibration statistics for a category/group."""
    calibration: float  # y_rate - price_mean
    fair_price: float   # Historical outcome rate (what price should be)
    avg_price: float    # Average observed market price
    direction: int      # -1 = short YES (overpriced), +1 = long YES (underpriced)
    n_samples: int      # Number of samples used for estimation
    price_std: float = 0.2  # Standard deviation of prices
    
    @property
    def calib_strength(self) -> float:
        """Absolute calibration deviation."""
        return abs(self.calibration)
    
    @classmethod
    def from_data(cls, outcomes: np.ndarray, prices: np.ndarray) -> "CalibrationState":
        """Compute calibration state from data."""
        y_rate = np.mean(outcomes)
        price_mean = np.mean(prices)
        calibration = y_rate - price_mean
        
        return cls(
            calibration=calibration,
            fair_price=y_rate,
            avg_price=price_mean,
            direction=-1 if calibration < 0 else 1,
            n_samples=len(outcomes),
            price_std=np.std(prices)
        )


@dataclass
class CalibrationWeightConfig:
    """Configuration for calibration-based weight function."""
    # Calibration scaling: weight += |calibration| * calib_scale
    calib_scale: float = 5.0
    
    # Price deviation scaling: weight += price_deviation * price_scale
    price_scale: float = 3.0
    
    # Volume effect: weight *= log(volume/vol_base) / vol_divisor
    vol_base: float = 10000.0
    vol_divisor: float = 2.0
    vol_min: float = 0.5
    vol_max: float = 1.5
    
    # Overall weight bounds
    weight_min: float = 0.1
    weight_max: float = 3.0
    
    # Base weight (ensures minimum allocation)
    base_weight: float = 0.1
    
    # Minimum calibration to trade (filter noise)
    min_calibration: float = 0.02


def compute_calibration_weight(
    calib_state: CalibrationState,
    market_price: float,
    volume: float,
    cfg: CalibrationWeightConfig = CalibrationWeightConfig()
) -> float:
    """
    Compute dynamic weight based on calibration deviation.
    
    Weight increases when:
    1. Category has larger calibration error (more mispriced)
    2. Current price is far from fair value
    3. Volume is higher (can execute)
    
    Args:
        calib_state: Calibration statistics for the category
        market_price: Current market price
        volume: Market volume
        cfg: Weight function configuration
    
    Returns:
        Weight in [cfg.weight_min, cfg.weight_max]
    """
    # Skip if calibration is too small
    if calib_state.calib_strength < cfg.min_calibration:
        return cfg.weight_min
    
    # Component 1: Category-level calibration strength
    calib_component = calib_state.calib_strength * cfg.calib_scale
    
    # Component 2: Price deviation from fair value
    fair = calib_state.fair_price
    if calib_state.direction == -1:  # Short YES (overpriced)
        # Profit potential = price - fair (higher price = more profit when shorting)
        price_deviation = max(0, market_price - fair)
    else:  # Long YES (underpriced)
        # Profit potential = fair - price (lower price = more profit when longing)
        price_deviation = max(0, fair - market_price)
    
    price_component = price_deviation * cfg.price_scale
    
    # Component 3: Volume multiplier (log scale)
    vol_component = np.log1p(volume / cfg.vol_base) / cfg.vol_divisor
    vol_component = np.clip(vol_component, cfg.vol_min, cfg.vol_max)
    
    # Final weight: sum of components times volume multiplier
    weight = (calib_component + price_component + cfg.base_weight) * vol_component
    
    return np.clip(weight, cfg.weight_min, cfg.weight_max)


class CalibrationWeightTracker:
    """
    Tracks calibration states and computes dynamic weights.
    
    Usage:
        tracker = CalibrationWeightTracker()
        
        # Training phase: learn calibration per category
        for category, outcomes, prices in training_data:
            tracker.update_calibration(category, outcomes, prices)
        
        # Trading phase: compute weights
        weight = tracker.get_weight(category, market_price, volume)
    """
    
    def __init__(self, cfg: CalibrationWeightConfig = CalibrationWeightConfig()):
        self.cfg = cfg
        self.calibration_states: Dict[str, CalibrationState] = {}
    
    def update_calibration(
        self, 
        category: str, 
        outcomes: np.ndarray, 
        prices: np.ndarray,
        min_samples: int = 30
    ) -> Optional[CalibrationState]:
        """Update calibration state for a category."""
        if len(outcomes) < min_samples:
            return None
        
        state = CalibrationState.from_data(outcomes, prices)
        self.calibration_states[category] = state
        return state
    
    def get_weight(
        self, 
        category: str, 
        market_price: float, 
        volume: float
    ) -> float:
        """Get weight for a market based on its category and features."""
        if category not in self.calibration_states:
            return self.cfg.weight_min
        
        return compute_calibration_weight(
            self.calibration_states[category],
            market_price,
            volume,
            self.cfg
        )
    
    def get_direction(self, category: str) -> int:
        """Get trading direction for a category. -1 = short YES, +1 = long YES."""
        if category not in self.calibration_states:
            return 0
        return self.calibration_states[category].direction
    
    def get_all_weights(
        self, 
        df: pd.DataFrame,
        category_col: str = "category",
        price_col: str = "market_prob",
        volume_col: str = "volume"
    ) -> np.ndarray:
        """Compute weights for all rows in a DataFrame."""
        weights = []
        for _, row in df.iterrows():
            w = self.get_weight(
                row[category_col], 
                row[price_col], 
                row.get(volume_col, 10000)
            )
            weights.append(w)
        return np.array(weights)
    
    def summary(self) -> pd.DataFrame:
        """Return summary of calibration states."""
        rows = []
        for cat, state in self.calibration_states.items():
            rows.append({
                'category': cat,
                'calibration': state.calibration,
                '|calibration|': state.calib_strength,
                'fair_price': state.fair_price,
                'avg_price': state.avg_price,
                'direction': 'short' if state.direction == -1 else 'long',
                'n_samples': state.n_samples
            })
        return pd.DataFrame(rows).sort_values('|calibration|', ascending=False)


def backtest_calibration_weighted_portfolio(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_col: str = "cat",
    price_col: str = "market_prob",
    outcome_col: str = "y",
    volume_col: str = "volume",
    cfg: CalibrationWeightConfig = CalibrationWeightConfig()
) -> Dict:
    """
    Backtest a calibration-weighted portfolio.
    
    Returns:
        Dictionary with metrics: pnl, weighted_pnl, win_rate, sharpe, etc.
    """
    # Initialize tracker
    tracker = CalibrationWeightTracker(cfg)
    
    # Learn calibration from training data
    for cat in train_df[category_col].unique():
        cat_df = train_df[train_df[category_col] == cat]
        tracker.update_calibration(
            cat,
            cat_df[outcome_col].values,
            cat_df[price_col].values
        )
    
    # Trade on test data
    results = []
    for _, row in test_df.iterrows():
        cat = row[category_col]
        if cat not in tracker.calibration_states:
            continue
        
        price = row[price_col]
        y = row[outcome_col]
        volume = row.get(volume_col, 10000)
        
        direction = tracker.get_direction(cat)
        weight = tracker.get_weight(cat, price, volume)
        
        # Compute PnL
        if direction == -1:  # Short YES
            pnl = price if y == 0 else -(1 - price)
        else:  # Long YES
            pnl = (1 - price) if y == 1 else -price
        
        results.append({
            'category': cat,
            'price': price,
            'weight': weight,
            'pnl': pnl,
            'weighted_pnl': weight * pnl
        })
    
    results_df = pd.DataFrame(results)
    
    # Compute metrics
    n = len(results_df)
    if n == 0:
        return {'error': 'No trades'}
    
    pnl_std = results_df['pnl'].std()
    wpnl_std = results_df['weighted_pnl'].std()
    
    return {
        'n_trades': n,
        'win_rate': (results_df['pnl'] > 0).mean(),
        'total_pnl': results_df['pnl'].sum(),
        'total_weighted_pnl': results_df['weighted_pnl'].sum(),
        'avg_pnl': results_df['pnl'].mean(),
        'avg_weighted_pnl': results_df['weighted_pnl'].mean(),
        'sharpe': results_df['pnl'].mean() / pnl_std * np.sqrt(n) if pnl_std > 0 else 0,
        'weighted_sharpe': results_df['weighted_pnl'].mean() / wpnl_std * np.sqrt(n) if wpnl_std > 0 else 0,
        'avg_weight': results_df['weight'].mean(),
        'calibration_summary': tracker.summary(),
        'results_df': results_df
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load data
    prices_df = pd.read_parquet('../../data/gamma_all_prices_combined.parquet')
    df = prices_df[prices_df['market_prob'].notna()].copy()
    
    # Categorize
    def categorize(q):
        q = str(q).lower()
        if 'bitcoin' in q or 'crypto' in q:
            return 'crypto'
        if 'trump' in q or 'election' in q:
            return 'politics'
        if 'nba' in q or 'nfl' in q:
            return 'sports'
        return 'other'
    
    df['cat'] = df['question'].apply(categorize)
    df['volume'] = pd.to_numeric(df['volumeNum'], errors='coerce').fillna(0)
    
    # Split
    train_size = int(len(df) * 0.6)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Backtest
    results = backtest_calibration_weighted_portfolio(train, test)
    
    print("Calibration Summary:")
    print(results['calibration_summary'])
    
    print(f"\nTest Results:")
    print(f"  N trades: {results['n_trades']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Total PnL: {results['total_pnl']:.2f}")
    print(f"  Weighted PnL: {results['total_weighted_pnl']:.2f}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Weighted Sharpe: {results['weighted_sharpe']:.2f}")
