import numpy as np

from forecastbench.benchmarks.synth_stream import (
    DynamicLogicalGraphStreamSpec,
    DynamicSegment,
    sample_dynamic_logical_graph_stream,
)


def test_dynamic_logical_graph_stream_shapes_and_ranges():
    spec = DynamicLogicalGraphStreamSpec(
        d=8,
        m=5,
        segments=(
            DynamicSegment("chain", 7, noise=0.2),
            DynamicSegment("star_in", 9, noise=0.2),
        ),
        seed=123,
        market_ar=0.7,
        market_noise=0.05,
        resolution_lag=3,
    )
    stream = sample_dynamic_logical_graph_stream(spec)
    assert stream.T == 16
    assert stream.X.shape == (16, 8)
    assert stream.p_true.shape == (16, 5)
    assert stream.market_prob.shape == (16, 5)
    assert stream.y.shape == (16, 5)
    assert stream.cond.shape == (16, 5, 8 + 5)
    assert len(stream.structure) == 16
    assert stream.segment_id.shape == (16,)
    assert stream.forecast_time.shape == (16,)
    assert stream.event_time.shape == (16,)
    assert np.all(stream.event_time >= stream.forecast_time)

    assert np.all((stream.p_true >= 0.0) & (stream.p_true <= 1.0))
    assert np.all((stream.market_prob >= 0.0) & (stream.market_prob <= 1.0))
    assert np.all((stream.y == 0) | (stream.y == 1))

    flat = stream.flatten_events()
    assert flat["x"].shape == (16 * 5, 8 + 5)
    assert flat["y"].shape == (16 * 5,)
    assert flat["market_prob"].shape == (16 * 5,)
    assert np.all((flat["market_prob"] >= 0.0) & (flat["market_prob"] <= 1.0))



