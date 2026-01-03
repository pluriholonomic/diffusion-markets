"""
Pytest configuration and fixtures for trading tests.
"""

import pytest
import sys
from pathlib import Path

# Add experiments directory to path for absolute imports
experiments_dir = Path(__file__).parent.parent.parent
if str(experiments_dir) not in sys.path:
    sys.path.insert(0, str(experiments_dir))

# Also add parent to ensure trading module is found
parent_dir = experiments_dir
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


@pytest.fixture
def sample_market():
    """Create a sample market for testing."""
    from trading.utils.models import Market, Platform
    
    return Market(
        market_id="test_market_1",
        platform=Platform.POLYMARKET,
        question="Will it rain tomorrow?",
        current_yes_price=0.50,
        volume=10000,
        metadata={
            "category": "weather",
            "end_date": "2026-01-10",
        }
    )


@pytest.fixture
def sample_markets():
    """Create multiple sample markets for testing."""
    from trading.utils.models import Market, Platform
    
    return [
        Market(
            market_id=f"test_market_{i}",
            platform=Platform.POLYMARKET,
            question=f"Test question {i}?",
            current_yes_price=0.3 + 0.05 * i,
            volume=1000 * (i + 1),
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_clob_data():
    """Create mock CLOB data for backtesting."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_records = 1000
    
    data = pd.DataFrame({
        "market_id": [f"MKT_{i % 10}" for i in range(n_records)],
        "timestamp": [1000 + i * 60 for i in range(n_records)],
        "price": np.clip(np.random.normal(0.5, 0.15, n_records), 0.05, 0.95),
        "volume": np.random.randint(100, 10000, n_records),
        "bid": np.clip(np.random.normal(0.48, 0.15, n_records), 0.05, 0.95),
        "ask": np.clip(np.random.normal(0.52, 0.15, n_records), 0.05, 0.95),
    })
    
    return data
