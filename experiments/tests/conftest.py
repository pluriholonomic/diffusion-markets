from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_forecastbench_on_path() -> None:
    """
    Ensure tests run against the repo's local `experiments/src` package, not an
    older site-packages install of `forecastbench`.
    """
    here = Path(__file__).resolve()
    exp_dir = here.parent.parent  # experiments/
    src_dir = exp_dir / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


_ensure_local_forecastbench_on_path()



