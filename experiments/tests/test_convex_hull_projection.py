import numpy as np

from forecastbench.utils.convex_hull_projection import (
    ct_projection_features,
    project_point_to_convex_hull,
    project_to_simplex,
)


def test_project_to_simplex_basic():
    v = np.array([0.2, -0.1, 3.0], dtype=np.float64)
    w = project_to_simplex(v)
    assert w.shape == (3,)
    assert np.all(w >= 0)
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_project_point_to_convex_hull_inside_triangle():
    # Triangle in R^2; point inside.
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    x = np.array([0.2, 0.3], dtype=np.float64)
    res = project_point_to_convex_hull(x, P, max_iter=500, tol=1e-12)
    assert res.proj.shape == (2,)
    assert res.weights.shape == (3,)
    assert np.all(res.weights >= -1e-12)
    assert abs(float(res.weights.sum()) - 1.0) < 1e-8
    assert res.dist_l2 < 1e-6


def test_project_point_to_convex_hull_outside_triangle():
    # Triangle in R^2; point outside near (2,2) should project to (1,1)/? actually hull is right triangle,
    # closest point is (0.5,0.5)? No: the closest point to (2,2) on x+y<=1 is (0.5,0.5).
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    x = np.array([2.0, 2.0], dtype=np.float64)
    res = project_point_to_convex_hull(x, P, max_iter=2000, tol=1e-12)
    assert np.allclose(res.proj, np.array([0.5, 0.5]), atol=1e-2)
    assert res.dist_l2 > 1.0


def test_ct_projection_features_shapes():
    rng = np.random.default_rng(0)
    samples = rng.random((16, 5))
    x = rng.random((5,))
    proj, feats = ct_projection_features(x=x, samples=samples, max_iter=200)
    assert proj.shape == (5,)
    assert "dist_l2" in feats and "direction" in feats and "residual" in feats
    assert feats["direction"].shape == (5,)
    assert feats["residual"].shape == (5,)



