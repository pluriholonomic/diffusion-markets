from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_packages_on_path() -> None:
    """
    Ensure tests run against the repo's local packages:
    - experiments/src for forecastbench
    - experiments/ for trading, backtest, etc.
    """
    here = Path(__file__).resolve()
    exp_dir = here.parent.parent  # experiments/
    
    # Add experiments/ for trading, backtest modules
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    
    # Add experiments/src for forecastbench
    src_dir = exp_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_local_packages_on_path()




