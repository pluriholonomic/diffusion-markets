"""
Conditional Graph Strategy.

Learns a dependency graph from C_t and trades based on
cross-market signals (e.g., market flips, conditional expectations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np

from backtest.strategies.base import StrategyState

if TYPE_CHECKING:
    from backtest.market_state import MarketStateManager


@dataclass
class ConditionalGraphConfig:
    """Configuration for the conditional graph strategy."""

    # Graph learning
    edge_threshold: float = 0.1  # Min partial correlation for edge
    max_edges_per_node: int = 5  # Sparsity constraint

    # Flip detection
    flip_threshold: float = 0.4  # Min price change for "significant" move
    flip_lookback_seconds: float = 3600.0  # 1 hour

    # Conditional trading
    min_expected_move: float = 0.05  # Min expected move to trade
    bandwidth: float = 0.1  # Kernel bandwidth for conditional estimation
    max_position: float = 1.0

    # Delay before trading (simulate latency)
    trade_delay_seconds: float = 5.0


@dataclass
class GraphEdge:
    """An edge in the dependency graph."""

    source: str
    target: str
    weight: float  # Partial correlation
    direction: float  # Average conditional effect


class ConditionalGraphStrategy:
    """
    Trade based on conditional dependencies in C_t.

    Key use cases:

    A) Market flipping (ε → 1-ε):
       When market A flips, C_t tells us which other markets should move.
       We can trade these secondary moves.

    B) Conditional bets:
       "If market 1 goes to 0.6 YES, then markets 2,3 should go up 25-30%"
       C_t encodes these via E[p_j | p_i = v].

    C) Cross-market arbitrage:
       Find groups of markets where joint prices violate C_t bounds.
    """

    name = "conditional_graph"

    def __init__(self, cfg: ConditionalGraphConfig):
        self.cfg = cfg
        self.state = StrategyState()

        # Dependency graph
        self._adjacency: Optional[np.ndarray] = None  # (k, k) partial correlations
        self._edges: List[GraphEdge] = []

        # Pending trades (with delay)
        self._pending_trades: List[Dict] = []

        # Recent flip detections
        self._recent_flips: List[Dict] = []

    def on_ct_refresh(
        self,
        ct_samples: np.ndarray,
        market_ids: List[str],
    ) -> None:
        """Update C_t and learn the dependency graph."""
        self.state.ct_samples = ct_samples
        self.state.market_ids = list(market_ids)

        # Learn graph from samples
        self._learn_dependency_graph()

    def _learn_dependency_graph(self) -> None:
        """
        Learn a sparse dependency graph from C_t samples.

        Uses partial correlations (from the precision matrix) to find
        direct dependencies between markets.
        """
        if self.state.ct_samples is None or len(self.state.market_ids) < 2:
            self._adjacency = None
            self._edges = []
            return

        samples = self.state.ct_samples
        k = samples.shape[1]

        # Compute sample covariance
        cov = np.cov(samples.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        # Regularized precision matrix
        reg = 1e-4 * np.eye(k)
        try:
            prec = np.linalg.inv(cov + reg)
        except np.linalg.LinAlgError:
            self._adjacency = None
            self._edges = []
            return

        # Partial correlations: ρ_ij = -P_ij / sqrt(P_ii * P_jj)
        diag = np.sqrt(np.diag(prec) + 1e-12)
        partial_corr = -prec / np.outer(diag, diag)
        np.fill_diagonal(partial_corr, 0)

        self._adjacency = partial_corr

        # Build edge list
        self._edges = []
        for i in range(k):
            # Find top edges for this node
            weights = np.abs(partial_corr[i])
            top_idx = np.argsort(weights)[::-1][:self.cfg.max_edges_per_node]

            for j in top_idx:
                if i == j:
                    continue
                w = partial_corr[i, j]
                if abs(w) >= self.cfg.edge_threshold:
                    self._edges.append(GraphEdge(
                        source=self.state.market_ids[i],
                        target=self.state.market_ids[j],
                        weight=w,
                        direction=np.sign(w),
                    ))

    def _estimate_conditional_move(
        self,
        trigger_market: str,
        new_price: float,
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Estimate expected moves in other markets given a trigger move.

        Uses kernel conditional mean estimation on C_t samples:
            E[p_j | p_i = new_price] ≈ Σ_s w_s * p_j^(s)

        where w_s ∝ exp(-|p_i^(s) - new_price|² / 2σ²)
        """
        if self.state.ct_samples is None:
            return {}

        if trigger_market not in self.state.market_ids:
            return {}

        trigger_idx = self.state.market_ids.index(trigger_market)
        samples = self.state.ct_samples  # (mc, k)

        # Kernel weights based on distance to trigger value
        trigger_samples = samples[:, trigger_idx]
        distances = np.abs(trigger_samples - new_price)
        weights = np.exp(-distances**2 / (2 * self.cfg.bandwidth**2))
        weights /= weights.sum() + 1e-12

        # Conditional means
        conditional_means = weights @ samples  # (k,)

        # Expected moves
        expected_moves = {}
        for i, market_id in enumerate(self.state.market_ids):
            if market_id == trigger_market:
                continue
            if market_id not in current_prices:
                continue

            current = current_prices[market_id]
            expected = conditional_means[i]
            move = expected - current

            if abs(move) >= self.cfg.min_expected_move:
                expected_moves[market_id] = move

        return expected_moves

    def on_price_update(
        self,
        market_id: str,
        old_price: float,
        new_price: float,
        state: "MarketStateManager",
    ) -> Optional[Dict[str, float]]:
        """
        React to a significant price change.

        Check for flips and compute conditional moves.
        """
        if self.state.ct_samples is None:
            return None

        # Check for flip
        delta = abs(new_price - old_price)
        is_flip = (old_price < 0.5 and new_price > 0.5) or (
            old_price > 0.5 and new_price < 0.5
        )
        is_significant = delta >= self.cfg.flip_threshold

        if not (is_flip or is_significant):
            return None

        # Record flip
        flip_info = {
            "market_id": market_id,
            "old_price": old_price,
            "new_price": new_price,
            "timestamp": state.current_time,
            "is_flip": is_flip,
        }
        self._recent_flips.append(flip_info)

        # Keep only recent flips
        cutoff = state.current_time - self.cfg.flip_lookback_seconds
        self._recent_flips = [f for f in self._recent_flips if f["timestamp"] > cutoff]

        # Compute expected moves in connected markets
        current_prices = {m: state.current_prices.get(m, float("nan"))
                         for m in self.state.market_ids}
        expected_moves = self._estimate_conditional_move(
            trigger_market=market_id,
            new_price=new_price,
            current_prices=current_prices,
        )

        if not expected_moves:
            return None

        # Filter by graph connectivity
        if self._adjacency is not None:
            trigger_idx = self.state.market_ids.index(market_id)
            connected_markets = set()
            for i, m in enumerate(self.state.market_ids):
                if abs(self._adjacency[trigger_idx, i]) >= self.cfg.edge_threshold:
                    connected_markets.add(m)
            expected_moves = {m: v for m, v in expected_moves.items()
                             if m in connected_markets}

        if not expected_moves:
            return None

        # Convert expected moves to positions
        positions = {}
        for market_id, move in expected_moves.items():
            # Position proportional to expected move
            pos = np.clip(move * 2, -self.cfg.max_position, self.cfg.max_position)
            positions[market_id] = pos

        self.state.custom["last_conditional_signal"] = {
            "trigger": market_id,
            "expected_moves": expected_moves,
            "positions": positions,
        }

        return positions

    def on_snapshot(
        self,
        timestamp: float,
        state: "MarketStateManager",
    ) -> Dict[str, float]:
        """
        Periodic check for arbitrage opportunities.

        Look for markets where current prices are far from C_t
        based on graph structure.
        """
        if self.state.ct_samples is None or self._adjacency is None:
            return {}

        # Get current prices
        current_prices = {}
        for m in self.state.market_ids:
            if m in state.current_prices and m not in state.resolved_markets:
                current_prices[m] = state.current_prices[m]

        if len(current_prices) < 2:
            return {}

        # Find markets with high graph centrality that are mispriced
        positions = {}

        for i, market_id in enumerate(self.state.market_ids):
            if market_id not in current_prices:
                continue

            # Centrality = sum of edge weights
            centrality = np.sum(np.abs(self._adjacency[i]))
            if centrality < self.cfg.edge_threshold:
                continue

            # Check if this market is mispriced relative to neighbors
            neighbors = []
            for j, other_id in enumerate(self.state.market_ids):
                if i == j or other_id not in current_prices:
                    continue
                if abs(self._adjacency[i, j]) >= self.cfg.edge_threshold:
                    neighbors.append((j, other_id, self._adjacency[i, j]))

            if not neighbors:
                continue

            # Expected value based on neighbors (weighted average)
            weighted_sum = 0.0
            weight_total = 0.0
            for j, other_id, edge_weight in neighbors:
                # Use samples to estimate E[p_i | p_j = current_j]
                other_idx = j
                trigger_price = current_prices[other_id]
                samples = self.state.ct_samples

                # Kernel weights
                other_samples = samples[:, other_idx]
                distances = np.abs(other_samples - trigger_price)
                weights = np.exp(-distances**2 / (2 * self.cfg.bandwidth**2))
                weights /= weights.sum() + 1e-12

                expected_i = weights @ samples[:, i]
                weighted_sum += abs(edge_weight) * expected_i
                weight_total += abs(edge_weight)

            if weight_total > 0:
                expected_price = weighted_sum / weight_total
                current = current_prices[market_id]
                mispricing = expected_price - current

                if abs(mispricing) >= self.cfg.min_expected_move:
                    pos = np.clip(
                        mispricing * 2,
                        -self.cfg.max_position,
                        self.cfg.max_position,
                    )
                    positions[market_id] = pos

        self.state.n_steps += 1
        return positions

    def on_resolution(
        self,
        market_id: str,
        outcome: float,
    ) -> None:
        """Track resolutions for learning."""
        self.state.custom.setdefault("resolutions", []).append({
            "market_id": market_id,
            "outcome": outcome,
        })

    def get_graph_summary(self) -> Dict:
        """Return summary of the learned graph."""
        if self._adjacency is None:
            return {"n_nodes": 0, "n_edges": 0}

        n_nodes = len(self.state.market_ids)
        n_edges = len(self._edges)

        # Find strongest edges
        top_edges = sorted(self._edges, key=lambda e: abs(e.weight), reverse=True)[:10]

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "top_edges": [
                {"source": e.source, "target": e.target, "weight": e.weight}
                for e in top_edges
            ],
        }

    def get_diagnostics(self) -> Dict:
        """Return strategy diagnostics."""
        diagnostics = {
            "name": self.name,
            "n_steps": self.state.n_steps,
            "n_markets": len(self.state.market_ids),
            "n_recent_flips": len(self._recent_flips),
        }

        graph_summary = self.get_graph_summary()
        diagnostics["n_edges"] = graph_summary["n_edges"]

        if "last_conditional_signal" in self.state.custom:
            signal = self.state.custom["last_conditional_signal"]
            diagnostics["last_trigger"] = signal.get("trigger")
            diagnostics["last_n_expected_moves"] = len(signal.get("expected_moves", {}))

        return diagnostics


