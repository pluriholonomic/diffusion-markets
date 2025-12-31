import numpy as np

from forecastbench.metrics.trading_sim import KellySimConfig, simulate_kelly_roi


def test_kelly_roi_no_edge_if_p_equals_q():
    p = np.array([0.2, 0.8, 0.5], dtype=np.float64)
    q = p.copy()
    y = np.array([0, 1, 1], dtype=np.float64)
    out = simulate_kelly_roi(p=p, q=q, y=y, cfg=KellySimConfig(initial_bankroll=1.0, frac_cap=0.25))
    # No bets => bankroll unchanged
    assert abs(out["roi"]) < 1e-12
    assert abs(out["final_bankroll"] - 1.0) < 1e-12


def test_kelly_roi_moves_bankroll_when_edge():
    p = np.array([0.9] * 10, dtype=np.float64)
    q = np.array([0.5] * 10, dtype=np.float64)
    y = np.array([1] * 10, dtype=np.float64)
    out = simulate_kelly_roi(p=p, q=q, y=y, cfg=KellySimConfig(initial_bankroll=1.0, frac_cap=0.05))
    assert out["final_bankroll"] > 1.0
    assert out["roi"] > 0.0




