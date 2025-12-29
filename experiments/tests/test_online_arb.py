import numpy as np

from forecastbench.metrics.online_arb import HedgeState, realized_profit


def test_realized_profit_scalar():
    b = np.array([[1.0, -1.0]], dtype=np.float64)
    y = np.array([[1.0, 0.0]], dtype=np.float64)
    q = np.array([[0.25, 0.25]], dtype=np.float64)
    # profit = 1*(1-0.25) + (-1)*(0-0.25) = 0.75 + 0.25 = 1.0
    p = realized_profit(b=b, y=y, price=q, transaction_cost=0.0)
    assert abs(p - 1.0) < 1e-9


def test_hedge_prefers_better_expert_and_bounds():
    # Two experts, k=1. Expert0 always takes +1, Expert1 always takes -1.
    # Outcome residual is positive each round, so expert0 is best.
    hs = HedgeState(n_experts=2, k=1, B=1.0, transaction_cost=0.0, eta=1.0)
    hs.reset()

    for _ in range(50):
        expert_b = np.array([[1.0], [-1.0]], dtype=np.float64)
        y = np.array([1.0], dtype=np.float64)
        price = np.array([0.0], dtype=np.float64)
        hs.step(expert_b=expert_b, y=y, price=price)

    assert hs.best_expert_profit() > 0
    assert hs.best_expert_profit() >= hs.cum_profit_mix - 1e-6
    # Regret bound should upper-bound the gap to best expert.
    ub = hs.upper_bound_best_expert_profit()
    assert ub + 1e-6 >= hs.best_expert_profit()



