from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def realized_profit(
    *,
    b: np.ndarray,
    y: np.ndarray,
    price: np.ndarray,
    transaction_cost: float = 0.0,
) -> float:
    """
    Realized linear profit with per-unit transaction cost:
      profit = b · (y - price) - c * ||b||_1

    Shapes:
      b, y, price: (..., k)
    Returns:
      mean profit across leading batch dims (float)
    """
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    if b.shape != y.shape or b.shape != price.shape:
        raise ValueError(f"b/y/price shapes must match; got b={b.shape} y={y.shape} price={price.shape}")
    pnl = np.sum(b * (y - price), axis=-1) - float(transaction_cost) * np.sum(np.abs(b), axis=-1)
    return float(np.mean(pnl))


def profit_range(*, k: int, B: float, transaction_cost: float) -> Tuple[float, float, float]:
    """
    Uniform bound for profit(b) over all b in [-B,B]^k when y-price in [-1,1]^k.
    """
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    B = float(B)
    if B < 0:
        raise ValueError("B must be >= 0")
    c = float(transaction_cost)
    if c < 0:
        raise ValueError("transaction_cost must be >= 0")

    p_max = B * k
    p_min = -B * k - c * B * k
    R = p_max - p_min
    return float(p_min), float(p_max), float(R)


@dataclass
class HedgeState:
    """
    Exponential-weights (Hedge) mixture over a finite expert set.

    This is designed for realized profit payoffs; we scale to losses in [0,1] using a
    conservative per-round profit range bound.
    """

    n_experts: int
    k: int
    B: float = 1.0
    transaction_cost: float = 0.0
    eta: Optional[float] = None

    # state
    w: Optional[np.ndarray] = None  # (N,)
    t: int = 0
    cum_profit_mix: float = 0.0
    cum_profit_expert: Optional[np.ndarray] = None  # (N,)

    def reset(self) -> None:
        self.t = 0
        self.w = np.full((int(self.n_experts),), 1.0 / float(self.n_experts), dtype=np.float64)
        self.cum_profit_mix = 0.0
        self.cum_profit_expert = np.zeros((int(self.n_experts),), dtype=np.float64)

    def _eta(self) -> float:
        if self.eta is not None:
            eta = float(self.eta)
            if eta <= 0:
                raise ValueError("eta must be > 0")
            return eta
        # default: safe constant (caller can override with tuned eta once T is known)
        return 0.5

    def step(self, *, expert_b: np.ndarray, y: np.ndarray, price: np.ndarray) -> dict:
        """
        One online step.

        Args:
          expert_b: (N,k) expert actions
          y: (k,) realized outcomes in {0,1} (or probabilities for synthetic)
          price: (k,) traded price
        """
        if self.w is None or self.cum_profit_expert is None:
            self.reset()

        expert_b = np.asarray(expert_b, dtype=np.float64)
        if expert_b.shape != (int(self.n_experts), int(self.k)):
            raise ValueError(f"expert_b must have shape (N,k)={(self.n_experts, self.k)} but got {expert_b.shape}")

        y = np.asarray(y, dtype=np.float64).reshape(int(self.k))
        price = np.asarray(price, dtype=np.float64).reshape(int(self.k))

        # mixture action
        b_mix = (self.w.reshape(-1, 1) * expert_b).sum(axis=0)

        # realized profits
        prof_ex = np.sum(expert_b * (y - price), axis=1) - float(self.transaction_cost) * np.sum(
            np.abs(expert_b), axis=1
        )
        prof_mix = float(np.sum(b_mix * (y - price)) - float(self.transaction_cost) * np.sum(np.abs(b_mix)))

        self.cum_profit_mix += prof_mix
        self.cum_profit_expert += prof_ex
        self.t += 1

        # scale to losses in [0,1]
        p_min, p_max, R = profit_range(k=int(self.k), B=float(self.B), transaction_cost=float(self.transaction_cost))
        # Note: our computed p_max assumes |b_i|<=B; caller should ensure expert_b obeys this.
        loss_ex = (p_max - prof_ex) / max(R, 1e-12)

        eta = self._eta()
        # update weights: w_i <- w_i * exp(-eta * loss_i)
        # stabilize by subtracting min loss
        m = float(np.min(loss_ex))
        upd = np.exp(-eta * (loss_ex - m))
        w_new = self.w * upd
        s = float(np.sum(w_new))
        if s <= 0 or not np.isfinite(s):
            # fallback: uniform
            w_new = np.full_like(self.w, 1.0 / float(self.n_experts), dtype=np.float64)
        else:
            w_new /= s
        self.w = w_new

        return {
            "t": int(self.t),
            "profit_mix": float(prof_mix),
            "profit_best_expert_so_far": float(np.max(self.cum_profit_expert)),
            "profit_mix_cum": float(self.cum_profit_mix),
        }

    def regret_bound(self, *, T: Optional[int] = None) -> float:
        """
        Standard Hedge regret bound (losses in [0,1]):
          Regret <= log N / eta + eta * T / 8

        Converted back into profit units using the conservative range R.
        """
        if T is None:
            T = int(self.t)
        T = int(T)
        if T <= 0:
            return 0.0
        N = int(self.n_experts)
        eta = float(self._eta())
        _, _, R = profit_range(k=int(self.k), B=float(self.B), transaction_cost=float(self.transaction_cost))
        return float(R * ((np.log(float(N)) / eta) + (eta * float(T) / 8.0)))

    def best_expert_profit(self) -> float:
        if self.cum_profit_expert is None:
            return 0.0
        return float(np.max(self.cum_profit_expert))

    def upper_bound_best_expert_profit(self) -> float:
        """
        Upper bound on best expert's cumulative profit (best in hindsight) using Hedge regret.
        """
        return float(self.cum_profit_mix + self.regret_bound(T=int(self.t)))



