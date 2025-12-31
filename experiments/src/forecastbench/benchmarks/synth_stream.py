from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

from forecastbench.benchmarks.logical_graphs import LogicalGraphSpec, make_graph_cond, sample_truth_prices
from forecastbench.utils.logits import clip_probs


@dataclass(frozen=True)
class DynamicSegment:
    """
    One regime in a dynamic synthetic stream.

    Each segment uses a fixed constraint structure and its own latent Truth mapping.
    """

    structure: Literal["chain", "star_in", "star_out"]
    length: int
    noise: float = 0.25


@dataclass(frozen=True)
class DynamicLogicalGraphStreamSpec:
    """
    Synthetic stream over m markets with a switching implication-graph constraint family.

    Returns a time-ordered sequence of bundle examples:
      t=0..T-1, each with:
        - context X_t (R^d)
        - true probabilities p_true[t, i] for i in [m]
        - market prices market_prob[t, i] (noisy proxy)
        - realized outcomes y[t, i] ~ Bernoulli(p_true[t, i])

    Notes:
    - This is intentionally lightweight: it reuses the existing `logical_graphs` generator,
      but concatenates multiple segments to make constraints (and thus C_t) time-varying.
    """

    d: int = 16
    m: int = 10
    segments: Tuple[DynamicSegment, ...] = (
        DynamicSegment("chain", 2000, noise=0.25),
        DynamicSegment("star_in", 2000, noise=0.25),
        DynamicSegment("star_out", 2000, noise=0.25),
    )
    seed: int = 0

    # Market microstructure proxy: AR(1) smoothing around truth plus iid noise.
    market_ar: float = 0.85
    market_noise: float = 0.05

    # Optional delayed feedback: event_time = forecast_time + lag.
    # (lag is in synthetic “steps”; downstream code can treat it as seconds if desired.)
    resolution_lag: int = 0


@dataclass(frozen=True)
class SyntheticBundleStream:
    """
    Container for a synthetic bundle stream.
    """

    # shape metadata
    d: int
    m: int
    T: int

    # time indices
    t: np.ndarray  # (T,) int64
    forecast_time: np.ndarray  # (T,) float64
    event_time: np.ndarray  # (T,) float64

    # data
    X: np.ndarray  # (T, d) float32
    cond: np.ndarray  # (T, m, d+m) float32 (for bundle diffusion)
    p_true: np.ndarray  # (T, m) float32
    market_prob: np.ndarray  # (T, m) float32
    y: np.ndarray  # (T, m) int8

    # regime labels
    structure: List[str]  # len T
    segment_id: np.ndarray  # (T,) int64

    def flatten_events(self) -> dict[str, np.ndarray]:
        """
        Flatten bundle stream into per-(t, market) events.

        Returns arrays of length T*m suitable for scalar-forecast baselines (incl RLVR-toy).
        """
        T, m = int(self.T), int(self.m)
        feat = self.cond.reshape(T * m, -1).astype(np.float32)
        out = {
            "t": np.repeat(self.t, m).astype(np.int64),
            "forecast_time": np.repeat(self.forecast_time, m).astype(np.float64),
            "event_time": np.repeat(self.event_time, m).astype(np.float64),
            "segment_id": np.repeat(self.segment_id, m).astype(np.int64),
            "market_idx": np.tile(np.arange(m, dtype=np.int64), T).astype(np.int64),
            "x": feat,
            "p_true": self.p_true.reshape(T * m).astype(np.float32),
            "market_prob": self.market_prob.reshape(T * m).astype(np.float32),
            "y": self.y.reshape(T * m).astype(np.int8),
        }
        return out


def sample_dynamic_logical_graph_stream(spec: DynamicLogicalGraphStreamSpec) -> SyntheticBundleStream:
    if spec.d <= 0:
        raise ValueError("d must be positive")
    if spec.m <= 0:
        raise ValueError("m must be positive")
    if not spec.segments:
        raise ValueError("segments must be non-empty")
    if not (0.0 <= float(spec.market_ar) <= 1.0):
        raise ValueError("market_ar must be in [0,1]")
    if spec.market_noise < 0:
        raise ValueError("market_noise must be >= 0")

    rng = np.random.default_rng(int(spec.seed))

    X_parts: List[np.ndarray] = []
    P_parts: List[np.ndarray] = []
    struct: List[str] = []
    seg_ids: List[np.ndarray] = []

    for j, seg in enumerate(spec.segments):
        if int(seg.length) <= 0:
            raise ValueError("segment lengths must be positive")
        gspec = LogicalGraphSpec(
            d=int(spec.d),
            m=int(spec.m),
            structure=str(seg.structure),
            seed=int(spec.seed) + 10_000 * (j + 1),
            noise=float(seg.noise),
        )
        Xj, Pj = sample_truth_prices(gspec, n=int(seg.length))
        X_parts.append(Xj.astype(np.float32))
        P_parts.append(Pj.astype(np.float32))
        struct.extend([str(seg.structure)] * int(seg.length))
        seg_ids.append(np.full((int(seg.length),), int(j), dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    P = np.concatenate(P_parts, axis=0)
    segment_id = np.concatenate(seg_ids, axis=0)
    T = int(P.shape[0])

    # Market price proxy: AR(1) around p_true + iid noise.
    q = np.empty_like(P, dtype=np.float32)
    prev = None
    a = float(spec.market_ar)
    sig = float(spec.market_noise)
    for t in range(T):
        noise = sig * rng.standard_normal(size=(int(spec.m),)).astype(np.float32)
        base = P[t] if prev is None else (a * prev + (1.0 - a) * P[t])
        qt = clip_probs(base + noise, eps=1e-6)
        q[t] = qt.astype(np.float32)
        prev = q[t]

    y = rng.binomial(n=1, p=P.astype(np.float64)).astype(np.int8)

    # Bundle conditioning
    cond = make_graph_cond(X, int(spec.m)).astype(np.float32)

    t_idx = np.arange(T, dtype=np.int64)
    forecast_time = t_idx.astype(np.float64)
    event_time = (t_idx + int(spec.resolution_lag)).astype(np.float64)

    return SyntheticBundleStream(
        d=int(spec.d),
        m=int(spec.m),
        T=int(T),
        t=t_idx,
        forecast_time=forecast_time,
        event_time=event_time,
        X=X.astype(np.float32),
        cond=cond,
        p_true=P.astype(np.float32),
        market_prob=q.astype(np.float32),
        y=y.astype(np.int8),
        structure=struct,
        segment_id=segment_id,
    )




