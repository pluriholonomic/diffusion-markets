from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MLPWitnessTraderSpec:
    """
    Online neural witness/trader that outputs bounded positions b in [-B,B]^k.

    This is an empirical arb-search tool (no strong formal guarantees).
    """

    in_dim: int
    out_dim: int
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.0

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    B: float = 1.0
    transaction_cost: float = 0.0

    device: str = "auto"  # cpu|cuda|mps|auto
    seed: int = 0


class MLPWitnessTrader:
    """
    Minimal PyTorch MLP that is trained online to maximize realized linear profit.
    """

    def __init__(self, spec: MLPWitnessTraderSpec):
        self.spec = spec
        self._torch = None
        self._nn = None
        self._device = None
        self._model = None
        self._opt = None
        self._step = 0
        self._lazy_init()

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError(
                "MLPWitnessTrader requires torch. Install forecastbench with extras: `pip install -e .[llm]`"
            ) from e

        self._torch = torch
        self._nn = nn

        device = str(self.spec.device)
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        torch.manual_seed(int(self.spec.seed))

        layers = []
        in_dim = int(self.spec.in_dim)
        h = int(self.spec.hidden_dim)
        depth = int(self.spec.depth)
        drop = float(self.spec.dropout)
        if depth <= 0:
            raise ValueError("depth must be positive")

        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else h, h))
            layers.append(nn.ReLU())
            if drop > 0:
                layers.append(nn.Dropout(p=drop))
        layers.append(nn.Linear(h, int(self.spec.out_dim)))
        self._model = nn.Sequential(*layers).to(device)

        self._opt = torch.optim.AdamW(
            self._model.parameters(), lr=float(self.spec.lr), weight_decay=float(self.spec.weight_decay)
        )

    @property
    def device(self) -> str:
        return str(self._device)

    def act(self, x: np.ndarray) -> np.ndarray:
        """
        Compute bounded action b in [-B,B]^k.

        Args:
          x: (in_dim,) or (batch, in_dim)
        Returns:
          b: (out_dim,) or (batch, out_dim) float32
        """
        self._lazy_init()
        torch = self._torch
        x = np.asarray(x, dtype=np.float32)
        x_t = torch.from_numpy(x).to(self._device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        with torch.no_grad():
            raw = self._model(x_t)
            b = float(self.spec.B) * torch.tanh(raw)
        out = b.detach().cpu().to(torch.float32).numpy()
        return out[0] if x.ndim == 1 else out

    def update(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        price: np.ndarray,
        lr: Optional[float] = None,
    ) -> dict:
        """
        One online update step using realized outcomes.

        Maximizes realized profit:
          profit = sum_j b_j (y_j - price_j) - c * sum_j |b_j|
        via gradient ascent (implemented as minimizing -profit).
        """
        self._lazy_init()
        torch = self._torch

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        price = np.asarray(price, dtype=np.float32)

        if x.ndim == 1:
            x = x[None, :]
        if y.ndim == 1:
            y = y[None, :]
        if price.ndim == 1:
            price = price[None, :]

        if x.shape[0] != y.shape[0] or x.shape[0] != price.shape[0]:
            raise ValueError("x,y,price must have matching batch dimension")
        if y.shape[1] != int(self.spec.out_dim) or price.shape[1] != int(self.spec.out_dim):
            raise ValueError("y and price must have shape (batch, out_dim)")

        x_t = torch.from_numpy(x).to(self._device)
        y_t = torch.from_numpy(y).to(self._device)
        q_t = torch.from_numpy(price).to(self._device)

        if lr is not None:
            for pg in self._opt.param_groups:
                pg["lr"] = float(lr)

        raw = self._model(x_t)
        b = float(self.spec.B) * torch.tanh(raw)
        profit = torch.sum(b * (y_t - q_t), dim=1) - float(self.spec.transaction_cost) * torch.sum(
            torch.abs(b), dim=1
        )
        loss = -torch.mean(profit)

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.spec.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), float(self.spec.grad_clip))
        self._opt.step()

        self._step += 1
        return {
            "step": int(self._step),
            "loss": float(loss.detach().cpu().item()),
            "profit_mean": float(torch.mean(profit).detach().cpu().item()),
            "device": str(self._device),
        }

    def state_dict(self):
        self._lazy_init()
        return {"spec": self.spec, "model": self._model.state_dict(), "step": int(self._step)}

    def load_state_dict(self, payload) -> None:
        self._lazy_init()
        sd = payload.get("model")
        if sd is None:
            raise ValueError("payload missing model state_dict")
        self._model.load_state_dict(sd)
        self._step = int(payload.get("step", 0))



