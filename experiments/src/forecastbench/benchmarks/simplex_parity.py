from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from forecastbench.benchmarks.parity import sample_rademacher
from forecastbench.utils.logits import alr_to_simplex


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class SimplexParitySpec:
    d: int
    k: int
    n_outcomes: int
    alpha: float
    seed: int = 0
    subsets: Optional[Sequence[Sequence[int]]] = None  # length n_outcomes-1 if provided


@dataclass(frozen=True)
class SimplexParityMarket:
    spec: SimplexParitySpec
    subsets: Tuple[Tuple[int, ...], ...]  # length n_outcomes-1

    @staticmethod
    def create(spec: SimplexParitySpec) -> "SimplexParityMarket":
        if spec.k <= 0 or spec.k > spec.d:
            raise ValueError(f"Invalid k={spec.k} for d={spec.d}")
        if spec.n_outcomes < 2:
            raise ValueError("n_outcomes must be >= 2")
        if not (0.0 < spec.alpha <= 5.0):
            raise ValueError("alpha must be in (0, 5] (logit-space amplitude)")

        m = spec.n_outcomes - 1
        if spec.subsets is not None:
            if len(spec.subsets) != m:
                raise ValueError("subsets must have length n_outcomes-1")
            subsets = tuple(tuple(int(i) for i in s) for s in spec.subsets)
        else:
            rng = _rng(spec.seed)
            subsets = []
            for _ in range(m):
                S = tuple(int(i) for i in rng.choice(spec.d, size=spec.k, replace=False))
                subsets.append(S)
            subsets = tuple(subsets)

        return SimplexParityMarket(spec=spec, subsets=subsets)

    def _chi(self, z: np.ndarray, S: Sequence[int]) -> np.ndarray:
        return np.prod(z[:, list(S)], axis=1).astype(np.int8)

    def alr_true(self, z: np.ndarray) -> np.ndarray:
        """
        Returns ALR coordinates u ∈ R^{n_outcomes-1}:
          u_i(z) = alpha * chi_{S_i}(z).
        Then p(z) = alr_to_simplex(u(z)).
        """
        u = np.zeros((z.shape[0], self.spec.n_outcomes - 1), dtype=np.float32)
        for i, S in enumerate(self.subsets):
            chi = self._chi(z, S).astype(np.float32)
            u[:, i] = self.spec.alpha * chi
        return u

    def p_true(self, z: np.ndarray) -> np.ndarray:
        u = self.alr_true(z)
        return alr_to_simplex(u).astype(np.float32)

    def sample_y(self, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # categorical sample from probabilities p (N, C)
        cs = np.cumsum(p, axis=1)
        r = rng.random(size=(p.shape[0], 1))
        y = (r > cs).sum(axis=1).astype(np.int64)
        return y

    def encode_text(self, z_row: np.ndarray, style: Literal["plain", "natural"] = "natural") -> str:
        if style == "plain":
            bits = " ".join("1" if b == 1 else "0" for b in z_row.tolist())
            return f"Bits: {bits}"
        if style == "natural":
            parts = [f"Bit {i} is {'+1' if int(b) == 1 else '-1'}." for i, b in enumerate(z_row)]
            return " ".join(parts)
        raise ValueError(f"Unknown style: {style}")


def sample_simplex_parity_dataset(spec: SimplexParitySpec, n: int) -> dict:
    mkt = SimplexParityMarket.create(spec)
    rng = _rng(spec.seed + 1_000_000)
    z = sample_rademacher(n=n, d=spec.d, rng=rng)
    p = mkt.p_true(z)
    y = mkt.sample_y(p, rng=rng)
    return {
        "z": z.astype(np.int8),
        "p_true": p.astype(np.float32),
        "y": y.astype(np.int64),
        "subsets": [list(S) for S in mkt.subsets],
    }



