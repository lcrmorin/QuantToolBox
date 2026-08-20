"""Shared pytest fixtures (e.g. reference matrices/portfolios for regression
tests against the original MATLAB outputs)."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)
