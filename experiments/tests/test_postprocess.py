import numpy as np

from forecastbench.postprocess import fit_group_bin_postprocessor


def test_group_bin_postprocessor_runs_and_outputs_probs():
    rng = np.random.default_rng(0)
    n = 2000
    d = 8
    # Boolean cube ±1
    z = rng.choice([-1, 1], size=(n, d)).astype(np.int8)
    # synthetic q and y
    q = rng.random(size=(n,)).astype(np.float32)
    y = rng.binomial(1, q).astype(np.int8)

    J = (0, 1, 2)
    post = fit_group_bin_postprocessor(z=z, q=q, y=y, J=J, n_bins=10, prior_strength=2.0, clip_eps=1e-4)
    q2 = post.predict(z, q)

    assert q2.shape == q.shape
    assert float(q2.min()) >= 0.0
    assert float(q2.max()) <= 1.0

import numpy as np

from forecastbench.benchmarks.parity import ParitySpec, sample_parity_dataset
from forecastbench.benchmarks.subcubes import worst_group_abs_cond_mean_over_assignments
from forecastbench.postprocess import fit_group_bin_postprocessor


def test_group_bin_postprocess_reduces_parity_group_error():
    # Small k so groups are reasonably frequent; then the wrapper should learn them.
    d = 10
    k = 4
    alpha = 0.8
    n_train = 50_000
    n_test = 20_000
    spec = ParitySpec(d=d, k=k, alpha=alpha, seed=0)
    data = sample_parity_dataset(spec, n=n_train + n_test)
    z = data["z"]
    p_true = data["p_true"]
    y = data["y"]

    # Base forecaster: constant (a stylized "AR L<k" baseline).
    q = np.full_like(p_true, 0.5, dtype=np.float32)

    # Use the true parity bits as the group coordinates (this partitions into 2^k subcubes).
    S = tuple(int(i) for i in data["S"].tolist())
    S_sorted = tuple(sorted(S))

    sl_tr = slice(0, n_train)
    sl_te = slice(n_train, n_train + n_test)

    post = fit_group_bin_postprocessor(
        z=z[sl_tr],
        q=q[sl_tr],
        y=y[sl_tr],
        J=S_sorted,
        n_bins=10,
        prior_strength=2.0,
        clip_eps=1e-4,
    )
    q_post = post.predict(z[sl_te], q[sl_te])

    # Evaluate worst-group conditional mean of the *population* residual on test.
    base_gcal, _ = worst_group_abs_cond_mean_over_assignments(
        z=z[sl_te], residual=(p_true[sl_te] - q[sl_te]), J=S_sorted
    )
    post_gcal, _ = worst_group_abs_cond_mean_over_assignments(
        z=z[sl_te], residual=(p_true[sl_te] - q_post), J=S_sorted
    )

    # Sanity: base should be ~alpha/2; post should be much smaller.
    assert base_gcal > 0.25
    assert post_gcal < 0.08


