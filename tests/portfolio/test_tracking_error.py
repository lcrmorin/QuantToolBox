"""Tests for quanttoolbox.portfolio.tracking_error."""

import numpy as np
import pytest

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.black_litterman import black_litterman_moments, implied_risk_premia
from quanttoolbox.portfolio.tracking_error import (
    minimum_te_portfolio,
    te_frontier,
    te_portfolio,
    te_target_portfolio,
)
from quanttoolbox.stats.moments import corr_to_cov


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


def test_te_target_portfolio_matches_matlab_reference():
    # Examples/rpb/test_bl4.m: Black-Litterman posterior mean, sigma-problem
    # target-matching against x0 -- golden values cross-verified against
    # the original MATLAB source (compute_te_portfolio.m) run via Octave.
    sigma = np.array([0.15, 0.20, 0.25, 0.30])
    rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
    cov = corr_to_cov(sigma, rho)

    x0 = np.array([0.40, 0.30, 0.20, 0.10])
    irp = implied_risk_premia(x0, cov, 0.25)
    mu_tilde = 0.03 + irp.pi
    p_matrix = np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float)
    bl = black_litterman_moments(
        mu_tilde, cov, p_matrix, np.array([0.04, -0.01]), np.diag([0.10**2, 0.05**2])
    )

    te_targets = np.array([0.01, 0.02, 0.03])
    results = te_target_portfolio(x0, bl.mu_bar, cov, te_targets, problem="sigma", lb=0.0, ub=1.0)
    expected_weights = np.array(
        [
            [0.3553, 0.3024, 0.2294, 0.1129],
            [0.3107, 0.3048, 0.2589, 0.1257],
            [0.2660, 0.3071, 0.2883, 0.1386],
        ]
    )
    for r, target, expected in zip(results, te_targets, expected_weights, strict=True):
        assert np.isclose(r.tracking_error, target, atol=1e-4)
        assert np.allclose(r.weights, expected, atol=2e-3)
        # every scenario should land on the same information ratio (the
        # BL posterior mean's implied Sharpe ratio doesn't change with gamma)
        assert np.isclose(r.active_return / r.tracking_error, 0.143, atol=1e-3)


def test_te_target_portfolio_matches_frontier_at_recovered_gamma(sample_data):
    x_b, mu, cov = sample_data
    result = te_target_portfolio(x_b, mu, cov, 0.05, problem="sigma")[0]
    direct = te_portfolio(x_b, mu, cov, gamma=result.gamma)
    assert np.allclose(result.weights, direct.weights, atol=1e-4)
