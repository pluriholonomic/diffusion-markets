import numpy as np

from forecastbench.utils.logits import alr_to_simplex, logit_to_prob, prob_to_logit, simplex_to_alr


def test_prob_logit_roundtrip():
    p = np.array([1e-3, 0.2, 0.5, 0.9, 1 - 1e-3], dtype=np.float64)
    u = prob_to_logit(p, eps=1e-9)
    p2 = logit_to_prob(u)
    assert np.max(np.abs(p - p2)) < 1e-10


def test_simplex_alr_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.random(size=(100, 5))
    p = x / x.sum(axis=1, keepdims=True)
    alr = simplex_to_alr(p)
    p2 = alr_to_simplex(alr)
    assert np.max(np.abs(p - p2)) < 1e-10


