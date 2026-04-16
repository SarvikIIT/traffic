"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_frame():
    """A synthetic 640×480 BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_adj():
    """5-node adjacency matrix (grid)."""
    adj = np.zeros((5, 5), dtype=np.float32)
    for i in range(4):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


@pytest.fixture
def grid_builder():
    from src.graph.builder import TrafficGraphBuilder
    return TrafficGraphBuilder.create_grid_city(rows=3, cols=3)


@pytest.fixture
def tmp_db(tmp_path):
    from src.utils.db import DatabaseManager
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.create_tables()
    return db
