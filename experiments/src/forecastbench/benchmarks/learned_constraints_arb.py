from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from forecastbench.metrics.online_arb import HedgeState, realized_profit
from forecastbench.utils.convex_hull_projection import ct_projection_features, summarize_ct_samples


def _sign(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    # keep zeros as 0
    return s.astype(np.float64)


def projection_direction_experts(*, direction: np.ndarray, B: float) -> np.ndarray:
    """
    Build a small finite expert family from the projection residual direction.

    direction: (k,) vector (typically normalized residual = q - Π_C(q))

    Experts:
      - full-vector toward-projection:  b = -B * sign(direction)
      - full-vector away-from-projection: b = +B * sign(direction)
      - coordinate-only versions for each j (toward and away)
    """
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    k = int(direction.size)
    if k <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    B = float(B)
    sd = _sign(direction)

    experts: List[np.ndarray] = []
    experts.append((-B) * sd)
    experts.append((+B) * sd)
    for j in range(k):
        e = np.zeros((k,), dtype=np.float64)
        e[j] = -B * sd[j]
        experts.append(e)
        e2 = np.zeros((k,), dtype=np.float64)
        e2[j] = +B * sd[j]
        experts.append(e2)
    return np.stack(experts, axis=0).astype(np.float64)


def default_feature_vector(
    *,
    q: np.ndarray,
    p_hat: np.ndarray,
    direction: np.ndarray,
    dist_l2: float,
) -> np.ndarray:
    """
    Small, purely-numeric feature vector for online traders:
      [q, p_hat, direction, dist_l2]
    """
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    p_hat = np.asarray(p_hat, dtype=np.float32).reshape(-1)
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    if q.shape != p_hat.shape or q.shape != direction.shape:
        raise ValueError("q, p_hat, direction must have matching shapes")
    return np.concatenate([q, p_hat, direction, np.asarray([float(dist_l2)], dtype=np.float32)], axis=0)


@dataclass(frozen=True)
class LearnedCtArbConfig:
    B_trade: float = 1.0
    transaction_cost: float = 0.0
    hedge_eta: Optional[float] = None
    witness_hidden: int = 128
    witness_depth: int = 2
    witness_lr: float = 1e-3
    witness_weight_decay: float = 0.0
    witness_grad_clip: float = 1.0
    seed: int = 0
    max_steps: Optional[int] = None


def run_learnedCt_online_arb(
    *,
    mc_samples: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    cfg: LearnedCtArbConfig,
    p_hat: Optional[np.ndarray] = None,
    sort_key: Optional[np.ndarray] = None,
    enable_neural: bool = True,
) -> Dict:
    """
    Run a lightweight online statistical arbitrage estimation loop on bundle examples.

    Args:
      mc_samples: (mc, T, k) probability samples from the diffusion model.
      q: (T, k) market prices.
      y: (T, k) realized outcomes in {0,1}.
      p_hat: (T, k) point estimate (defaults to mean over mc).
      sort_key: (T,) optional key to reorder time steps (ascending).
    """
    mc_samples = np.asarray(mc_samples, dtype=np.float32)
    if mc_samples.ndim != 3:
        raise ValueError("mc_samples must have shape (mc, T, k)")
    mc, T, k = mc_samples.shape
    q = np.asarray(q, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if q.shape != (T, k) or y.shape != (T, k):
        raise ValueError(f"q/y must have shape (T,k)={(T,k)} but got q={q.shape} y={y.shape}")
    if p_hat is None:
        p_hat = np.mean(mc_samples.astype(np.float64), axis=0).astype(np.float32)
    else:
        p_hat = np.asarray(p_hat, dtype=np.float32)
        if p_hat.shape != (T, k):
            raise ValueError("p_hat must have shape (T,k)")

    order = np.arange(T, dtype=np.int64)
    if sort_key is not None:
        sk = np.asarray(sort_key)
        if sk.shape != (T,):
            raise ValueError("sort_key must have shape (T,)")
        # stable sort; NaNs last
        sk2 = np.where(np.isnan(sk), np.inf, sk.astype(np.float64))
        order = np.argsort(sk2, kind="stable")

    if cfg.max_steps is not None:
        order = order[: int(cfg.max_steps)]

    # Optional neural traders (market and our-quotes versions)
    trader_mkt = None
    trader_ours = None
    if enable_neural:
        try:
            from forecastbench.models.witness_trader import MLPWitnessTrader, MLPWitnessTraderSpec

            in_dim = int(3 * k + 1)
            spec = MLPWitnessTraderSpec(
                in_dim=in_dim,
                out_dim=int(k),
                hidden_dim=int(cfg.witness_hidden),
                depth=int(cfg.witness_depth),
                lr=float(cfg.witness_lr),
                weight_decay=float(cfg.witness_weight_decay),
                grad_clip=float(cfg.witness_grad_clip),
                B=float(cfg.B_trade),
                transaction_cost=float(cfg.transaction_cost),
                seed=int(cfg.seed),
            )
            trader_mkt = MLPWitnessTrader(spec)
            trader_ours = MLPWitnessTrader(spec)
        except Exception:
            trader_mkt = None
            trader_ours = None

    # Hedge baseline (finite experts derived from projection direction)
    hedge_mkt = HedgeState(
        n_experts=int(2 + 2 * k),
        k=int(k),
        B=float(cfg.B_trade),
        transaction_cost=float(cfg.transaction_cost),
        eta=cfg.hedge_eta,
    )
    hedge_ours = HedgeState(
        n_experts=int(2 + 2 * k),
        k=int(k),
        B=float(cfg.B_trade),
        transaction_cost=float(cfg.transaction_cost),
        eta=cfg.hedge_eta,
    )
    hedge_mkt.reset()
    hedge_ours.reset()

    # Baseline sign trader (uses p_hat vs q)
    cum_sign_mkt = 0.0
    cum_sign_ours = 0.0
    cum_neural_mkt = 0.0
    cum_neural_ours = 0.0

    dists: List[float] = []
    ct_summaries: List[dict] = []

    for idx in order.tolist():
        samples_t = mc_samples[:, idx, :].astype(np.float32)
        q_t = q[idx].astype(np.float32)
        y_t = y[idx].astype(np.float32)
        p_t = p_hat[idx].astype(np.float32)

        # C_t summary + projection features (use market vector as the point to project)
        ct_summaries.append(summarize_ct_samples(samples_t))
        proj, feats = ct_projection_features(x=q_t, samples=samples_t, max_iter=200)
        dists.append(float(feats["dist_l2"]))

        # Shared feature vector for online traders
        f = default_feature_vector(q=q_t, p_hat=p_t, direction=feats["direction"], dist_l2=float(feats["dist_l2"]))

        # Finite experts from projection residual direction
        expert_b = projection_direction_experts(direction=feats["direction"], B=float(cfg.B_trade))

        # Baseline sign trading against market and our quote
        b_sign = float(cfg.B_trade) * _sign(p_t - q_t).astype(np.float64)
        cum_sign_mkt += realized_profit(b=b_sign, y=y_t, price=q_t, transaction_cost=float(cfg.transaction_cost))
        cum_sign_ours += realized_profit(b=b_sign, y=y_t, price=p_t, transaction_cost=float(cfg.transaction_cost))

        # Hedge mixtures
        hedge_mkt.step(expert_b=expert_b, y=y_t, price=q_t)
        hedge_ours.step(expert_b=expert_b, y=y_t, price=p_t)

        # Neural witnesses (if available)
        if trader_mkt is not None and trader_ours is not None:
            b_mkt = trader_mkt.act(f).astype(np.float64)
            b_ours = trader_ours.act(f).astype(np.float64)
            cum_neural_mkt += realized_profit(b=b_mkt, y=y_t, price=q_t, transaction_cost=float(cfg.transaction_cost))
            cum_neural_ours += realized_profit(b=b_ours, y=y_t, price=p_t, transaction_cost=float(cfg.transaction_cost))
            trader_mkt.update(x=f, y=y_t, price=q_t)
            trader_ours.update(x=f, y=y_t, price=p_t)

    T_used = int(len(order))
    out: Dict[str, object] = {
        "T": int(T),
        "T_used": int(T_used),
        "k": int(k),
        "mc": int(mc),
        "cfg": {
            "B_trade": float(cfg.B_trade),
            "transaction_cost": float(cfg.transaction_cost),
            "hedge_eta": None if cfg.hedge_eta is None else float(cfg.hedge_eta),
            "witness_hidden": int(cfg.witness_hidden),
            "witness_depth": int(cfg.witness_depth),
            "witness_lr": float(cfg.witness_lr),
            "seed": int(cfg.seed),
        },
        "Ct": {
            "dist_l2_mean": float(np.mean(dists)) if dists else float("nan"),
            "dist_l2_p90": float(np.quantile(dists, 0.9)) if dists else float("nan"),
        },
        "profits": {
            "sign": {
                "cum_vs_market": float(cum_sign_mkt),
                "cum_vs_ours": float(cum_sign_ours),
                "mean_vs_market": float(cum_sign_mkt / max(T_used, 1)),
                "mean_vs_ours": float(cum_sign_ours / max(T_used, 1)),
            },
            "hedge": {
                "cum_mix_vs_market": float(hedge_mkt.cum_profit_mix),
                "cum_mix_vs_ours": float(hedge_ours.cum_profit_mix),
                "best_expert_vs_market": float(hedge_mkt.best_expert_profit()),
                "best_expert_vs_ours": float(hedge_ours.best_expert_profit()),
                "ub_best_expert_vs_market": float(hedge_mkt.upper_bound_best_expert_profit()),
                "ub_best_expert_vs_ours": float(hedge_ours.upper_bound_best_expert_profit()),
                "regret_bound_vs_market": float(hedge_mkt.regret_bound()),
                "regret_bound_vs_ours": float(hedge_ours.regret_bound()),
            },
            "neural": None
            if trader_mkt is None
            else {
                "cum_vs_market": float(cum_neural_mkt),
                "cum_vs_ours": float(cum_neural_ours),
                "mean_vs_market": float(cum_neural_mkt / max(T_used, 1)),
                "mean_vs_ours": float(cum_neural_ours / max(T_used, 1)),
            },
        },
    }

    # Keep sample diagnostics compact (don’t dump full vectors by default).
    out["Ct"]["sample_summary_mean"] = {
        "radius2": float(np.mean([s.get("radius2", 0.0) for s in ct_summaries])) if ct_summaries else float("nan"),
    }
    return out



