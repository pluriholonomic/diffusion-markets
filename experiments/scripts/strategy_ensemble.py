"""
Strategy Ensemble with Dynamic Regime-Based Weights

This module combines all strategies (momentum, mean-reversion, dispersion, pairs)
into a unified ensemble that dynamically adjusts weights based on detected regime.

Key Features:
1. Regime detection to determine market state
2. Dynamic weight allocation based on regime
3. Risk parity across strategies
4. Combined position sizing with correlation adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

# Import our strategies
from regime_detector import RegimeDetector, RegimeType, RegimeResult
from momentum_strategies import (
    calibration_momentum_strategy,
    trend_following_strategy,
    event_momentum_strategy,
    cross_category_momentum_strategy,
)
from dispersion_strategy import (
    dispersion_strategy,
    lead_lag_pairs_strategy,
)


@dataclass
class StrategyAllocation:
    """Allocation to a single strategy"""
    name: str
    weight: float
    regime_affinity: Dict[str, float]  # Which regimes this strategy works in
    

@dataclass
class EnsembleResult:
    """Result from ensemble strategy"""
    total_pnl: float
    final_bankroll: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float
    strategy_contributions: Dict[str, float]
    regime_history: List[str]
    pnl_series: np.ndarray


# Define strategy regime affinities
STRATEGY_AFFINITIES = {
    'calibration_mean_reversion': {
        'trending': 0.2,
        'mean_reverting': 1.0,
        'random_walk': 0.5,
    },
    'calibration_momentum': {
        'trending': 1.0,
        'mean_reverting': 0.2,
        'random_walk': 0.4,
    },
    'trend_following': {
        'trending': 1.0,
        'mean_reverting': 0.1,
        'random_walk': 0.3,
    },
    'event_momentum': {
        'trending': 0.8,
        'mean_reverting': 0.4,
        'random_walk': 0.6,
    },
    'cross_category_momentum': {
        'trending': 0.9,
        'mean_reverting': 0.3,
        'random_walk': 0.5,
    },
    'dispersion': {
        'trending': 0.5,
        'mean_reverting': 0.7,
        'random_walk': 0.6,
    },
    'pairs': {
        'trending': 0.4,
        'mean_reverting': 0.9,
        'random_walk': 0.5,
    },
}


class StrategyEnsemble:
    """
    Ensemble that combines multiple strategies with regime-based weighting.
    """
    
    def __init__(
        self,
        strategies: List[str] = None,
        regime_lookback: int = 50,
        rebalance_frequency: int = 20,
        risk_parity: bool = True,
        max_concentration: float = 0.5,  # Max weight on any single strategy
    ):
        if strategies is None:
            strategies = list(STRATEGY_AFFINITIES.keys())
        
        self.strategies = strategies
        self.regime_lookback = regime_lookback
        self.rebalance_frequency = rebalance_frequency
        self.risk_parity = risk_parity
        self.max_concentration = max_concentration
        
        self.regime_detector = RegimeDetector()
        self.current_weights = {s: 1.0 / len(strategies) for s in strategies}
        self.strategy_vols = {s: 0.15 for s in strategies}  # Initial vol estimate
    
    def compute_weights(
        self,
        regime: RegimeResult,
        strategy_returns: Dict[str, List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute strategy weights based on current regime and performance.
        """
        regime_type = regime.regime_type.value
        if regime_type == 'unknown':
            regime_type = 'random_walk'
        
        # Base weights from regime affinity
        raw_weights = {}
        for strategy in self.strategies:
            affinity = STRATEGY_AFFINITIES.get(strategy, {}).get(regime_type, 0.5)
            raw_weights[strategy] = affinity * regime.confidence + (1 - regime.confidence) * 0.5
        
        # Risk parity adjustment
        if self.risk_parity and strategy_returns:
            for strategy in self.strategies:
                returns = strategy_returns.get(strategy, [])
                if len(returns) > 10:
                    vol = np.std(returns) + 1e-8
                    self.strategy_vols[strategy] = vol
            
            # Inverse vol weighting
            total_inv_vol = sum(1 / v for v in self.strategy_vols.values())
            for strategy in self.strategies:
                vol_weight = (1 / self.strategy_vols[strategy]) / total_inv_vol
                # Blend affinity weights with vol weights
                raw_weights[strategy] = 0.7 * raw_weights[strategy] + 0.3 * vol_weight
        
        # Normalize
        total = sum(raw_weights.values())
        weights = {s: w / total for s, w in raw_weights.items()}
        
        # Cap concentration
        for strategy in weights:
            if weights[strategy] > self.max_concentration:
                excess = weights[strategy] - self.max_concentration
                weights[strategy] = self.max_concentration
                # Redistribute excess
                others = [s for s in weights if s != strategy]
                for other in others:
                    weights[other] += excess / len(others)
        
        return weights
    
    def run(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        initial_bankroll: float = 10000.0,
        fee: float = 0.01,
        max_drawdown_stop: float = 0.50,
    ) -> EnsembleResult:
        """
        Run the ensemble strategy.
        """
        price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
        
        # Detect initial regime from training data
        train_prices = train[price_col].values
        regime = self.regime_detector.detect_regime(train_prices)
        
        # Run each individual strategy
        strategy_results = {}
        
        # Mean reversion (existing calibration strategy)
        # We'll approximate by using momentum with negative direction
        strategy_results['calibration_mean_reversion'] = self._run_calibration_mean_reversion(
            train, test, initial_bankroll, fee
        )
        
        # Momentum strategies
        strategy_results['calibration_momentum'] = calibration_momentum_strategy(
            train, test, initial_bankroll=initial_bankroll, fee=fee
        )
        strategy_results['trend_following'] = trend_following_strategy(
            train, test, initial_bankroll=initial_bankroll, fee=fee
        )
        strategy_results['event_momentum'] = event_momentum_strategy(
            train, test, initial_bankroll=initial_bankroll, fee=fee
        )
        strategy_results['cross_category_momentum'] = cross_category_momentum_strategy(
            train, test, initial_bankroll=initial_bankroll, fee=fee
        )
        
        # Dispersion and pairs
        strategy_results['dispersion'] = dispersion_strategy(
            train, test, initial_bankroll=initial_bankroll, fee=fee
        )
        strategy_results['pairs'] = lead_lag_pairs_strategy(
            train, test, initial_bankroll=initial_bankroll, fee=fee
        )
        
        # Filter to strategies we're using
        available_strategies = [s for s in self.strategies if s in strategy_results]
        
        # Compute weights based on regime
        strategy_returns = {}
        for s, result in strategy_results.items():
            if 'pnl_series' in result:
                strategy_returns[s] = list(result['pnl_series'])
        
        weights = self.compute_weights(regime, strategy_returns)
        
        # Combine PnL series with weights
        # Get max length
        max_len = max(
            len(strategy_results[s].get('pnl_series', [])) 
            for s in available_strategies
        )
        
        # Pad shorter series with zeros
        aligned_pnl = {}
        for s in available_strategies:
            pnl = strategy_results[s].get('pnl_series', np.array([]))
            if len(pnl) < max_len:
                pnl = np.concatenate([pnl, np.zeros(max_len - len(pnl))])
            aligned_pnl[s] = pnl
        
        # Weighted combination
        combined_pnl = np.zeros(max_len)
        for s in available_strategies:
            combined_pnl += weights.get(s, 0) * aligned_pnl[s]
        
        # Compute ensemble metrics
        total_pnl = np.sum(combined_pnl)
        final_bankroll = initial_bankroll + total_pnl
        
        # Sharpe
        sharpe = np.mean(combined_pnl) / (np.std(combined_pnl) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        cumsum = np.cumsum(combined_pnl) + initial_bankroll
        peak = np.maximum.accumulate(cumsum)
        drawdowns = (peak - cumsum) / peak
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Strategy contributions
        contributions = {}
        for s in available_strategies:
            contributions[s] = {
                'weight': weights.get(s, 0),
                'pnl': strategy_results[s].get('total_pnl', 0),
                'sharpe': strategy_results[s].get('sharpe', 0),
                'trades': strategy_results[s].get('trades', 0),
            }
        
        # Total trades and wins
        total_trades = sum(strategy_results[s].get('trades', 0) for s in available_strategies)
        total_wins = sum(
            strategy_results[s].get('trades', 0) * strategy_results[s].get('win_rate', 0)
            for s in available_strategies
        )
        
        return EnsembleResult(
            total_pnl=total_pnl,
            final_bankroll=final_bankroll,
            sharpe=sharpe,
            max_drawdown=max_dd,
            trades=total_trades,
            win_rate=total_wins / total_trades if total_trades > 0 else 0,
            strategy_contributions=contributions,
            regime_history=[regime.regime_type.value],
            pnl_series=combined_pnl,
        )
    
    def _run_calibration_mean_reversion(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        initial_bankroll: float,
        fee: float,
    ) -> Dict[str, Any]:
        """
        Run calibration mean reversion (bet AGAINST miscalibration).
        This is the inverse of calibration momentum.
        """
        price_col = 'avg_price' if 'avg_price' in train.columns else 'price'
        outcome_col = 'y' if 'y' in train.columns else 'outcome'
        
        train_copy = train.copy()
        train_copy['_price'] = train_copy[price_col].clip(0.01, 0.99)
        train_copy['_outcome'] = train_copy[outcome_col]
        
        n_bins = 10
        train_copy['price_bin'] = pd.cut(train_copy['_price'], bins=n_bins, labels=False)
        
        # Learn calibration
        calibration = train_copy.groupby('price_bin').agg({
            '_price': 'mean',
            '_outcome': 'mean'
        }).rename(columns={'_price': 'bin_price', '_outcome': 'outcome_rate'})
        calibration['spread'] = calibration['outcome_rate'] - calibration['bin_price']
        
        # Apply to test
        test_copy = test.copy()
        test_copy['_price'] = test_copy[price_col].clip(0.01, 0.99)
        test_copy['_outcome'] = test_copy[outcome_col]
        test_copy['price_bin'] = pd.cut(test_copy['_price'], bins=n_bins, labels=False)
        test_copy = test_copy.merge(
            calibration[['spread']], 
            left_on='price_bin', 
            right_index=True, 
            how='left'
        )
        test_copy['spread'] = test_copy['spread'].fillna(0)
        
        bankroll = initial_bankroll
        peak_bankroll = initial_bankroll
        pnl_list = []
        trades = 0
        wins = 0
        
        kelly_fraction = 0.25
        max_position_pct = 0.10  # Reduced from 0.20
        spread_threshold = 0.03
        max_drawdown_stop = 0.50
        
        for _, row in test_copy.iterrows():
            price = row['_price']
            outcome = row['_outcome']
            spread = row.get('spread', 0)
            
            if abs(spread) < spread_threshold:
                pnl_list.append(0)
                continue
            
            # Check drawdown
            dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            if dd >= max_drawdown_stop:
                pnl_list.append(0)
                continue
            
            trades += 1
            
            # Mean reversion: bet AGAINST the spread
            # If spread > 0 (outcomes > prices), prices should rise → bet YES
            direction = 1 if spread > 0 else -1
            
            edge = abs(spread)
            position_frac = kelly_fraction * min(edge / 0.10, 1.0)
            position_frac = min(position_frac, max_position_pct)
            
            # Use FIXED position size, not compounding (more realistic)
            position = initial_bankroll * position_frac
            
            if direction > 0:
                pnl = position * (outcome - price) - fee * position
            else:
                pnl = position * (price - outcome) - fee * position
            
            pnl_list.append(pnl)
            bankroll += pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            
            if pnl > 0:
                wins += 1
        
        pnl_array = np.array(pnl_list)
        total_pnl = float(np.sum(pnl_array))
        sharpe = float(np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252))
        max_dd = float((peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0)
        
        return {
            'strategy': 'calibration_mean_reversion',
            'total_pnl': total_pnl,
            'final_bankroll': float(bankroll),
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': int(trades),
            'win_rate': float(wins / trades if trades > 0 else 0),
            'pnl_series': pnl_array,
        }


def run_ensemble_backtest(
    data_path: str = 'optimization_cache.parquet',
    train_ratio: float = 0.7,
    initial_bankroll: float = 10000.0,
    output_dir: str = 'runs/ensemble',
) -> EnsembleResult:
    """
    Run ensemble backtest and save results.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Split
    n = len(df)
    train_idx = int(n * train_ratio)
    train = df.iloc[:train_idx].copy()
    test = df.iloc[train_idx:].copy()
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    # Run ensemble
    ensemble = StrategyEnsemble()
    result = ensemble.run(train, test, initial_bankroll=initial_bankroll)
    
    # Print results
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Total PnL: ${result.total_pnl:,.0f}")
    print(f"Final Bankroll: ${result.final_bankroll:,.0f}")
    print(f"Sharpe Ratio: {result.sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print(f"Total Trades: {result.trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    
    print(f"\n{'='*60}")
    print("STRATEGY CONTRIBUTIONS")
    print(f"{'='*60}")
    print(f"{'Strategy':<30} {'Weight':>8} {'PnL':>12} {'Sharpe':>8} {'Trades':>8}")
    print("-" * 70)
    
    for strategy, contrib in sorted(
        result.strategy_contributions.items(),
        key=lambda x: -x[1]['pnl']
    ):
        print(f"{strategy:<30} {contrib['weight']:>7.1%} ${contrib['pnl']:>11,.0f} "
              f"{contrib['sharpe']:>7.2f} {contrib['trades']:>8}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj
    
    results_dict = convert_to_native({
        'total_pnl': result.total_pnl,
        'final_bankroll': result.final_bankroll,
        'sharpe': result.sharpe,
        'max_drawdown': result.max_drawdown,
        'trades': result.trades,
        'win_rate': result.win_rate,
        'strategy_contributions': result.strategy_contributions,
        'regime_history': result.regime_history,
    })
    
    with open(output_path / 'ensemble_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    np.save(output_path / 'ensemble_pnl.npy', result.pnl_series)
    
    print(f"\nResults saved to {output_path}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategy Ensemble')
    parser.add_argument('--data', type=str, default='optimization_cache.parquet')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--bankroll', type=float, default=10000.0)
    parser.add_argument('--output-dir', type=str, default='runs/ensemble')
    args = parser.parse_args()
    
    result = run_ensemble_backtest(
        data_path=args.data,
        train_ratio=args.train_ratio,
        initial_bankroll=args.bankroll,
        output_dir=args.output_dir,
    )
