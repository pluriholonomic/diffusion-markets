"""
Root conftest.py for pytest.
Sets up import paths before any tests are collected.
"""

import sys
from pathlib import Path

# Add experiments directory to path for trading, backtest imports
exp_dir = Path(__file__).resolve().parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

# Add src directory for forecastbench
src_dir = exp_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
