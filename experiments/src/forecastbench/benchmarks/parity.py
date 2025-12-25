from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_rademacher(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    # ±1 with equal probability
    x = rng.integers(0, 2, size=(n, d), dtype=np.int8)
    return (2 * x - 1).astype(np.int8)


@dataclass(frozen=True)
class ParitySpec:
    d: int
    k: int
    alpha: float
    seed: int = 0
    subset: Optional[list[int]] = None  # if provided, overrides random S


@dataclass(frozen=True)
class ParityMarket:
    spec: ParitySpec
    S: tuple[int, ...]  # active parity indices

    @staticmethod
    def create(spec: ParitySpec) -> "ParityMarket":
        if spec.k <= 0 or spec.k > spec.d:
            raise ValueError(f"Invalid k={spec.k} for d={spec.d}")
        if not (0.0 < spec.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        if spec.subset is not None:
            if len(spec.subset) != spec.k:
                raise ValueError("subset must have length k")
            S = tuple(int(i) for i in spec.subset)
        else:
            rng = _rng(spec.seed)
            S = tuple(int(i) for i in rng.choice(spec.d, size=spec.k, replace=False))
        return ParityMarket(spec=spec, S=S)

    def chi(self, z: np.ndarray) -> np.ndarray:
        # chi_S(z) = ∏_{i in S} z_i
        return np.prod(z[:, self.S], axis=1).astype(np.int8)

    def p_true(self, z: np.ndarray) -> np.ndarray:
        chi = self.chi(z).astype(np.float32)
        return 0.5 + 0.5 * self.spec.alpha * chi

    def sample_y(self, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.binomial(n=1, p=p).astype(np.int8)

    def diffusion_analytic(self, z: np.ndarray, rho: float) -> np.ndarray:
        # For parity: (T_rho f)(z) = 1/2 + (alpha/2) rho^k chi_S(z)
        if not (0.0 <= rho <= 1.0):
            raise ValueError("rho must be in [0,1]")
        chi = self.chi(z).astype(np.float32)
        return 0.5 + 0.5 * self.spec.alpha * (rho ** self.spec.k) * chi

    def encode_text(self, z_row: np.ndarray, style: Literal["plain", "natural"] = "natural") -> str:
        if style == "plain":
            bits = " ".join("1" if b == 1 else "0" for b in z_row.tolist())
            return f"Bits: {bits}"
        if style == "natural":
            parts = [f"Bit {i} is {'+1' if int(b) == 1 else '-1'}." for i, b in enumerate(z_row)]
            return " ".join(parts)
        raise ValueError(f"Unknown style: {style}")


def sample_parity_dataset(spec: ParitySpec, n: int) -> dict[str, np.ndarray]:
    mkt = ParityMarket.create(spec)
    rng = _rng(spec.seed + 1_000_000)
    z = sample_rademacher(n=n, d=spec.d, rng=rng)
    p = mkt.p_true(z)
    y = mkt.sample_y(p, rng=rng)
    return {
        "z": z,
        "p_true": p.astype(np.float32),
        "y": y.astype(np.int8),
        "S": np.asarray(mkt.S, dtype=np.int64),
    }


