"""Tests for quanttoolbox.portfolio.tracking_error."""

import numpy as np
import pytest

from quanttoolbox.portfolio.tracking_error import (
    minimum_te_portfolio,
    te_frontier,
    te_portfolio,
)


@pytest.fixture
def sample_data(rng):
    n = 4
    a = rng.standard_normal((200, n))
    cov = a.T @ a / 200 + 0.01 * np.eye(n)
    mu = np.array([0.08, 0.05, 0.06, 0.03])
    x_benchmark = np.full(n, 0.25)
    return x_benchmark, mu, cov


def test_te_portfolio_sums_to_one(sample_data):
    x_b, mu, cov = sample_data
    result = te_portfolio(x_b, mu, cov, gamma=1.0)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-5)


def test_te_portfolio_zero_gamma_stays_at_benchmark(sample_data):
    x_b, mu, cov = sample_data
    result = te_portfolio(x_b, mu, cov, gamma=0.0)
    # with no return tilt and starting feasible, should stay at (or very near) benchmark
    assert np.allclose(result.weights, x_b, atol=1e-4)
    assert np.isclose(result.tracking_error, 0.0, atol=1e-4)


def test_te_portfolio_higher_gamma_increases_active_return(sample_data):
    x_b, mu, cov = sample_data
    low = te_portfolio(x_b, mu, cov, gamma=0.1)
    high = te_portfolio(x_b, mu, cov, gamma=5.0)
    assert high.active_return >= low.active_return


def test_te_portfolio_higher_gamma_increases_tracking_error(sample_data):
    x_b, mu, cov = sample_data
    low = te_portfolio(x_b, mu, cov, gamma=0.1)
    high = te_portfolio(x_b, mu, cov, gamma=5.0)
    assert high.tracking_error >= low.tracking_error - 1e-6


def test_minimum_te_portfolio_matches_te_portfolio_gamma_zero(sample_data):
    x_b, _, cov = sample_data
    min_te = minimum_te_portfolio(x_b, cov)
    zero_gamma = te_portfolio(x_b, np.zeros(4), cov, gamma=0.0)
    assert np.allclose(min_te.weights, zero_gamma.weights, atol=1e-6)


def test_te_frontier_returns_list(sample_data):
    x_b, mu, cov = sample_data
    results = te_frontier(x_b, mu, cov, gamma_values=np.array([0.0, 1.0, 3.0]))
    assert len(results) == 3
    tes = [r.tracking_error for r in results]
    assert tes[0] <= tes[1] <= tes[2] + 1e-6


def test_te_portfolio_respects_box_bounds(sample_data):
    x_b, mu, cov = sample_data
    result = te_portfolio(x_b, mu, cov, gamma=2.0, lb=0.1, ub=0.4)
    assert np.all(result.weights >= 0.1 - 1e-4)
    assert np.all(result.weights <= 0.4 + 1e-4)
