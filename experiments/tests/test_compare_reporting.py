import json
from pathlib import Path

import pandas as pd

from forecastbench.reporting.compare import ModelRunSpec, compare_polymarket_runs


def _write_run(tmp: Path, name: str, pred_col: str, pred_values) -> Path:
    run_dir = tmp / name
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "y": [0.0, 1.0, 1.0],
            "market_prob": [0.2, 0.4, 0.6],
            pred_col: pred_values,
        }
    )
    df.to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "config.json").write_text(json.dumps({"pred_col": pred_col}) + "\n")
    return run_dir


def test_compare_handles_market_prob_as_pred_col(tmp_path: Path):
    # One model is the market baseline: pred_col == market_prob.
    run_market = _write_run(tmp_path, "run_market", "market_prob", [0.2, 0.4, 0.6])
    run_model = _write_run(tmp_path, "run_model", "pred_prob", [0.1, 0.9, 0.7])

    res = compare_polymarket_runs(
        specs=[
            ModelRunSpec(name="market", run_dir=run_market),
            ModelRunSpec(name="model", run_dir=run_model),
        ],
        baseline="market",
        n_boot=10,
        seed=0,
    )
    assert res["n"] == 3
    assert "market" in res["point"]
    assert "model" in res["point"]
    # sanity: model has different metrics than market
    assert res["point"]["model"]["brier"] != res["point"]["market"]["brier"]



