from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path

    @staticmethod
    def create(run_name: str, base_dir: Optional[Union[str, Path]] = None) -> "RunArtifacts":
        # artifacts.py lives at experiments/src/forecastbench/runner/artifacts.py
        # parents[3] is the experiments/ directory.
        base = (
            Path(base_dir)
            if base_dir is not None
            else Path(__file__).resolve().parents[3] / "runs"
        )
        base.mkdir(parents=True, exist_ok=True)

        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name).strip("_")
        run_dir = base / f"{_utc_timestamp()}_{safe}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        return RunArtifacts(run_dir=run_dir)

    def write_json(self, relpath: str, payload: Any) -> Path:
        path = self.run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def write_text(self, relpath: str, text: str) -> Path:
        path = self.run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def savefig(self, fig, name: str) -> Path:
        path = self.run_dir / "plots" / name
        fig.savefig(path, bbox_inches="tight", dpi=int(os.getenv("FORECASTBENCH_DPI", "160")))
        return path

    def maybe_write_env(self) -> Optional[Path]:
        keys = [
            "CUDA_VISIBLE_DEVICES",
            "TORCH_VERSION",
            "TRANSFORMERS_VERSION",
            "FORECASTBENCH_SEED",
        ]
        env = {k: os.getenv(k) for k in keys if os.getenv(k) is not None}
        if not env:
            return None
        return self.write_json("env.json", env)


