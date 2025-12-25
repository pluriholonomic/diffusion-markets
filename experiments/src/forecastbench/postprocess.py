from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from forecastbench.benchmarks.subcubes import keys_for_J
from forecastbench.utils.logits import clip_probs


@dataclass(frozen=True)
class GroupBinPostprocessor:
    """
    A minimal group-conditional binning calibrator.

    Groups are defined by fixing coordinates J on the Boolean cube (subcubes G_{J,a}).
    Within each group we bucket q into n_bins and estimate E[Y|group,bin] with Beta prior smoothing.

    This is intentionally simple and intended for synthetic controls (e.g. parity),
    where the family of groups is exponential in |J| and thus illustrates the sample tax.
    """

    J: Tuple[int, ...]
    n_bins: int
    prior_strength: float
    clip_eps: float
    # tables are indexed by [group_key, bin]
    group_bin_mean: np.ndarray  # float32
    group_bin_count: np.ndarray  # int32
    global_bin_mean: np.ndarray  # float32
    global_bin_count: np.ndarray  # int32

    def predict(self, z: np.ndarray, q: np.ndarray) -> np.ndarray:
        q = clip_probs(q.astype(np.float64), eps=self.clip_eps)
        n = q.shape[0]
        k = len(self.J)

        # binning on q
        bins = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float64)
        b = np.digitize(q, bins, right=True) - 1
        b = np.clip(b, 0, self.n_bins - 1).astype(np.int64)

        # group keys
        g = keys_for_J(z, self.J).astype(np.int64)

        out = np.empty(n, dtype=np.float64)
        prior_means = (b + 0.5) / self.n_bins  # identity-ish prior per bin
        ps = float(self.prior_strength)

        # Vectorized: lookup count/mean; fall back to global bins when group-bin count is 0.
        gb_count = self.group_bin_count[g, b].astype(np.float64)
        gb_mean = self.group_bin_mean[g, b].astype(np.float64)

        # Convert stored means back to sums for smoothing.
        gb_sum = gb_mean * gb_count

        # global fallback
        gl_count = self.global_bin_count[b].astype(np.float64)
        gl_mean = self.global_bin_mean[b].astype(np.float64)
        gl_sum = gl_mean * gl_count

        use_group = gb_count > 0
        denom_group = gb_count + ps
        denom_gl = gl_count + ps

        out[use_group] = (gb_sum[use_group] + ps * prior_means[use_group]) / denom_group[use_group]
        out[~use_group] = (gl_sum[~use_group] + ps * prior_means[~use_group]) / denom_gl[~use_group]

        # If even global bin is empty (rare), fall back to original q.
        empty_gl = (~use_group) & (gl_count == 0)
        out[empty_gl] = q[empty_gl]

        return out.astype(np.float32)


def fit_group_bin_postprocessor(
    *,
    z: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    J: Sequence[int],
    n_bins: int = 20,
    prior_strength: float = 5.0,
    clip_eps: float = 1e-4,
) -> GroupBinPostprocessor:
    """
    Fit a group-conditional binning calibrator on (z,q,y).
    """
    if n_bins <= 1:
        raise ValueError("n_bins must be >= 2")
    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")

    J = tuple(int(i) for i in J)
    k = len(J)
    if k <= 0:
        raise ValueError("J must be non-empty")

    q = clip_probs(q.astype(np.float64), eps=clip_eps)
    y = y.astype(np.float64)

    # binning
    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    b = np.digitize(q, bins, right=True) - 1
    b = np.clip(b, 0, n_bins - 1).astype(np.int64)

    # groups
    g = keys_for_J(z, J).astype(np.int64)
    n_groups = int(1 << k)  # manageable for synthetic k (parity)

    counts = np.zeros((n_groups, n_bins), dtype=np.int32)
    sums = np.zeros((n_groups, n_bins), dtype=np.float64)

    # global bins
    gl_counts = np.zeros((n_bins,), dtype=np.int32)
    gl_sums = np.zeros((n_bins,), dtype=np.float64)

    for gi, bi, yi in zip(g.tolist(), b.tolist(), y.tolist()):
        counts[gi, bi] += 1
        sums[gi, bi] += yi
        gl_counts[bi] += 1
        gl_sums[bi] += yi

    means = np.zeros_like(sums, dtype=np.float32)
    mask = counts > 0
    means[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    gl_means = np.zeros_like(gl_sums, dtype=np.float32)
    gl_mask = gl_counts > 0
    gl_means[gl_mask] = (gl_sums[gl_mask] / gl_counts[gl_mask]).astype(np.float32)

    return GroupBinPostprocessor(
        J=J,
        n_bins=int(n_bins),
        prior_strength=float(prior_strength),
        clip_eps=float(clip_eps),
        group_bin_mean=means.astype(np.float32),
        group_bin_count=counts.astype(np.int32),
        global_bin_mean=gl_means.astype(np.float32),
        global_bin_count=gl_counts.astype(np.int32),
    )

    n_groups = 1 << k
    keys = keys_for_J(z, J).astype(np.int64)
    bins = _bin_index(q, n_bins)
    flat = keys * int(n_bins) + bins

    counts = np.bincount(flat, minlength=n_groups * n_bins).astype(np.float64)
    sums = np.bincount(flat, weights=y, minlength=n_groups * n_bins).astype(np.float64)

    # Prior: shrink towards the bin center (identity-ish mapping when data is scarce).
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    centers_rep = np.tile(centers, n_groups)

    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")

    denom = counts + float(prior_strength)
    num = sums + float(prior_strength) * centers_rep
    table_flat = np.zeros_like(num, dtype=np.float64)
    nonzero = denom > 0
    table_flat[nonzero] = num[nonzero] / denom[nonzero]
    table = table_flat.reshape(n_groups, n_bins)

    return GroupBinPostProcessor(
        J=J,
        n_bins=int(n_bins),
        prior_strength=float(prior_strength),
        clip_eps=float(clip_eps),
        table=table.astype(np.float32),
    )


