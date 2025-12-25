from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def run_to_latex_table(run_dir: Union[str, Path]) -> str:
    """
    Convert a forecastbench run directory into a simple LaTeX tabular.
    Currently supports runs that write metrics as {"models":[...]}.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    cfg_path = run_dir / "config.json"
    metrics = json.loads(metrics_path.read_text())
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    models: List[Dict[str, Any]] = metrics.get("models", [])
    if not models:
        raise ValueError(f"No models found in {metrics_path}")

    header = "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    header += "Model & Brier & LogLoss & SCE & ECE & Arb \\\\\n\\midrule\n"
    rows = []
    for m in models:
        name = _latex_escape(str(m["name"]))
        rows.append(
            f"{name} & {m['brier']:.4f} & {m['logloss']:.4f} & {m['sce']:.4f} & {m['ece']:.4f} & {m['arb_profit']:.4f} \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(rows) + "\n" + footer


