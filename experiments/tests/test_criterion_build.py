from __future__ import annotations

import numpy as np

from forecastbench.data.criterion_build import (
    _build_prob_segments_from_points,
    _prob_at_time,
    _prob_time_avg,
    _parse_criterion,
)


def test_parse_criterion() -> None:
    assert _parse_criterion("midpoint") == ("midpoint", None)
    assert _parse_criterion("time-average") == ("time-average", None)
    assert _parse_criterion("before-close-hours-24") == ("before-close-hours-24", 24 * 3600)
    assert _parse_criterion("before-close-days-7") == ("before-close-days-7", 7 * 86400)


def test_segments_match_themis_piecewise_constant() -> None:
    # History points: each point i defines the probability for [t_i, t_{i+1}).
    t = np.asarray([0, 10, 20], dtype=np.int64)
    p = np.asarray([0.2, 0.7, 0.9], dtype=np.float64)  # last value is unused in segment construction

    segs = _build_prob_segments_from_points(t=t, p=p)
    assert segs.shape == (2, 3)
    assert segs[0, 0] == 0 and segs[0, 1] == 10 and abs(segs[0, 2] - 0.2) < 1e-12
    assert segs[1, 0] == 10 and segs[1, 1] == 20 and abs(segs[1, 2] - 0.7) < 1e-12

    # Midpoint of [0,20] is t=10. The segment definition is start <= t < end,
    # so t=10 falls into the second segment and returns 0.7.
    prob, used_ts = _prob_at_time(segs, 10)
    assert abs(prob - 0.7) < 1e-12
    assert used_ts == 10

    # Time-average over [0,20] is (0.2*10 + 0.7*10)/20 = 0.45
    avg = _prob_time_avg(segs, start=0, end=20)
    assert abs(avg - 0.45) < 1e-12




