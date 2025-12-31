"""
Polymarket CLOB Data Fetcher.

Fetches real order book data from Polymarket's CLOB API for
accurate execution cost estimation.

Requires: pip install py-clob-client
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from py_clob_client.client import ClobClient
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False


@dataclass
class OrderBookSnapshot:
    """A snapshot of an order book."""

    token_id: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, size), ...] sorted desc by price
    asks: List[Tuple[float, float]]  # [(price, size), ...] sorted asc by price

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return float("inf")
        return 10000 * self.spread / mid

    def bid_depth(self, levels: int = 5) -> float:
        """Total bid size in top N levels."""
        return sum(size for _, size in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        """Total ask size in top N levels."""
        return sum(size for _, size in self.asks[:levels])

    def slippage_for_size(self, size: float, side: str) -> Tuple[float, float]:
        """
        Calculate average fill price and slippage for a given trade size.

        Args:
            size: Trade size in shares
            side: "buy" or "sell"

        Returns:
            (avg_fill_price, slippage_bps)
        """
        levels = self.asks if side == "buy" else self.bids
        if not levels:
            return float("inf"), float("inf")

        remaining = abs(size)
        total_cost = 0.0

        for price, available in levels:
            fill = min(remaining, available)
            total_cost += fill * price
            remaining -= fill
            if remaining <= 0:
                break

        if remaining > 0:
            # Insufficient liquidity
            return float("inf"), float("inf")

        avg_price = total_cost / abs(size)
        slippage = abs(avg_price - self.mid_price) / self.mid_price * 10000
        return avg_price, slippage

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "timestamp": self.timestamp,
            "bids": self.bids,
            "asks": self.asks,
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
        }


class CLOBFetcher:
    """
    Fetches order book data from Polymarket CLOB API.

    Usage:
        fetcher = CLOBFetcher()
        snapshot = fetcher.get_order_book(token_id)
        print(f"Spread: {snapshot.spread_bps:.1f} bps")
    """

    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        cache_dir: Optional[Path] = None,
        rate_limit_delay: float = 0.1,
    ):
        if not HAS_CLOB_CLIENT:
            raise ImportError(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )

        self.client = ClobClient(host=host)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def get_order_book(
        self,
        token_id: str,
        use_cache: bool = True,
        cache_ttl_seconds: float = 60.0,
    ) -> OrderBookSnapshot:
        """
        Fetch current order book for a token.

        Args:
            token_id: Polymarket token ID
            use_cache: Whether to use cached data
            cache_ttl_seconds: Cache TTL in seconds

        Returns:
            OrderBookSnapshot
        """
        # Check cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{token_id}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        cached = json.load(f)
                    if time.time() - cached["timestamp"] < cache_ttl_seconds:
                        return OrderBookSnapshot(
                            token_id=token_id,
                            timestamp=cached["timestamp"],
                            bids=[(b["price"], b["size"]) for b in cached["bids"]],
                            asks=[(a["price"], a["size"]) for a in cached["asks"]],
                        )
                except (json.JSONDecodeError, KeyError):
                    pass

        # Fetch from API
        self._rate_limit()
        try:
            book = self.client.get_order_book(token_id)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch order book for {token_id}: {e}")

        # Parse response
        bids = []
        asks = []

        for bid in book.get("bids", []):
            price = float(bid.get("price", 0))
            size = float(bid.get("size", 0))
            if price > 0 and size > 0:
                bids.append((price, size))

        for ask in book.get("asks", []):
            price = float(ask.get("price", 0))
            size = float(ask.get("size", 0))
            if price > 0 and size > 0:
                asks.append((price, size))

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])

        snapshot = OrderBookSnapshot(
            token_id=token_id,
            timestamp=time.time(),
            bids=bids,
            asks=asks,
        )

        # Cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{token_id}.json"
            with open(cache_file, "w") as f:
                json.dump({
                    "timestamp": snapshot.timestamp,
                    "bids": [{"price": p, "size": s} for p, s in bids],
                    "asks": [{"price": p, "size": s} for p, s in asks],
                }, f)

        return snapshot

    def get_order_books_batch(
        self,
        token_ids: List[str],
        use_cache: bool = True,
    ) -> Dict[str, OrderBookSnapshot]:
        """
        Fetch order books for multiple tokens.

        Args:
            token_ids: List of token IDs
            use_cache: Whether to use cached data

        Returns:
            Dict mapping token_id -> OrderBookSnapshot
        """
        results = {}
        for token_id in token_ids:
            try:
                results[token_id] = self.get_order_book(token_id, use_cache=use_cache)
            except Exception as e:
                print(f"Warning: Could not fetch order book for {token_id}: {e}")
        return results


def fetch_and_cache_clob_data(
    token_ids: List[str],
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, OrderBookSnapshot]:
    """
    Fetch and cache CLOB data for a list of tokens.

    Args:
        token_ids: List of token IDs to fetch
        output_dir: Directory to cache data
        verbose: Print progress

    Returns:
        Dict of snapshots
    """
    if not HAS_CLOB_CLIENT:
        raise ImportError("py-clob-client not installed")

    fetcher = CLOBFetcher(cache_dir=output_dir)

    if verbose:
        print(f"Fetching order books for {len(token_ids)} tokens...")

    results = {}
    for i, token_id in enumerate(token_ids):
        try:
            results[token_id] = fetcher.get_order_book(token_id, use_cache=False)
            if verbose and (i + 1) % 10 == 0:
                print(f"  Fetched {i + 1}/{len(token_ids)}")
        except Exception as e:
            if verbose:
                print(f"  Error fetching {token_id}: {e}")

    if verbose:
        print(f"Successfully fetched {len(results)} order books")

    return results


# Fallback for when we don't have real CLOB data
def estimate_spread_from_volume(
    volume_usd: float,
    mid_price: float,
) -> float:
    """
    Estimate spread in bps from volume when CLOB data unavailable.

    Based on empirical prediction market observations.
    """
    # Base spread: 1% (100 bps)
    base = 100.0

    # Volume adjustment: higher volume = tighter spread
    # ~$100K volume → ~50 bps
    # ~$10K volume → ~100 bps
    # ~$1K volume → ~200 bps
    if volume_usd > 0:
        volume_factor = np.log10(max(volume_usd, 100)) - 2  # 0 at $100, 2 at $10K
        spread = base * np.exp(-0.3 * volume_factor)
    else:
        spread = base * 2

    # Price extreme adjustment
    price_dist = min(mid_price, 1 - mid_price)
    if price_dist < 0.1:
        spread *= 1 + (0.1 - price_dist) / 0.1

    return max(spread, 20.0)  # Minimum 20 bps



