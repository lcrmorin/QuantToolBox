"""Tests for quanttoolbox.portfolio.mean_variance."""

import numpy as np
import pytest

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.mean_variance import (
    mdp_portfolio,
    minvar_portfolio,
    mvo_frontier,
    mvo_portfolio,
    mvo_target_portfolio,
)
from quanttoolbox.stats.moments import corr_to_cov


@pytest.fixture
def sample_data(rng):
    n = 4
    a = rng.standard_normal((200, n))
    cov = a.T @ a / 200 + 0.01 * np.eye(n)
    mu = np.array([0.08, 0.05, 0.06, 0.03])
    return mu, cov


def test_mvo_portfolio_sums_to_one(sample_data):
    mu, cov = sample_data
    result = mvo_portfolio(mu, cov, gamma=1.0)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-5)


def test_mvo_at_zero_gamma_matches_minvar(sample_data):
    mu, cov = sample_data
    mvo_result = mvo_portfolio(mu, cov, gamma=0.0)
    minvar_result = minvar_portfolio(cov)
    assert np.allclose(mvo_result.weights, minvar_result.weights, atol=1e-4)


def test_mvo_higher_gamma_increases_expected_return(sample_data):
    mu, cov = sample_data
    low_gamma = mvo_portfolio(mu, cov, gamma=0.1)
    high_gamma = mvo_portfolio(mu, cov, gamma=5.0)
    assert high_gamma.expected_return >= low_gamma.expected_return


def test_mvo_respects_box_bounds(sample_data):
    mu, cov = sample_data
    result = mvo_portfolio(mu, cov, gamma=1.0, lb=0.1, ub=0.4)
    assert np.all(result.weights >= 0.1 - 1e-4)
    assert np.all(result.weights <= 0.4 + 1e-4)


def test_mvo_frontier_returns_list(sample_data):
    mu, cov = sample_data
    results = mvo_frontier(mu, cov, gamma_values=np.array([0.0, 1.0, 5.0]))
    assert len(results) == 3
    # expected return should be non-decreasing along the frontier
    returns = [r.expected_return for r in results]
    assert returns[0] <= returns[1] <= returns[2]


def test_minvar_has_lowest_volatility_on_frontier(sample_data):
    mu, cov = sample_data
    minvar_result = minvar_portfolio(cov)
    mvo_result = mvo_portfolio(mu, cov, gamma=2.0)
    assert minvar_result.volatility <= mvo_result.volatility + 1e-6


def test_minvar_matches_direct_qp_solution(sample_data):
    _, cov = sample_data
    result = minvar_portfolio(cov)
    # verify against the closed-form unconstrained-but-budget-constrained
    # minimum variance solution: x = Cov^-1 1 / (1' Cov^-1 1)
    ones = np.ones(cov.shape[0])
    inv_cov_ones = np.linalg.solve(cov, ones)
    expected = inv_cov_ones / np.sum(inv_cov_ones)
    if np.all(expected >= -100) and np.all(expected <= 100):  # within default box
        assert np.allclose(result.weights, expected, atol=1e-3)


def test_mdp_portfolio_sums_to_one_and_converges(sample_data):
    _, cov = sample_data
    result = mdp_portfolio(cov)
    assert result.converged
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-4)


def test_mdp_diversification_ratio_at_least_one(sample_data):
    _, cov = sample_data
    result = mdp_portfolio(cov)
    # diversification ratio is always >= 1 by Cauchy-Schwarz
    assert result.diversification_ratio >= 1.0 - 1e-6


def test_mdp_equals_equal_weight_for_identical_assets():
    # if all assets have identical variance and correlation, MDP == equal weight
    n = 3
    cov = np.full((n, n), 0.5) + 0.5 * np.eye(n)
    result = mdp_portfolio(cov)
    assert np.allclose(result.weights, 1.0 / n, atol=1e-2)


@pytest.fixture
def roncalli_data():
    # Roncalli [2013], "Introduction to Risk Parity and Budgeting", the
    # 4-asset example used throughout Examples/rpb/test_mvo{2,3}.m --
    # golden values below are cross-verified against the original MATLAB
    # source (compute_mvo_portfolio.m) run via Octave.
    mu = np.array([0.05, 0.06, 0.08, 0.06])
    sigma = np.array([0.15, 0.20, 0.25, 0.30])
    rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
    cov = corr_to_cov(sigma, rho)
    return mu, cov


def test_mvo_target_portfolio_sigma_matches_matlab_reference(roncalli_data):
    mu, cov = roncalli_data
    targets = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
    results = mvo_target_portfolio(mu, cov, targets, problem="sigma", lb=-100.0, ub=100.0)
    expected_weights = np.array(
        [
            [0.6252, 0.1558, 0.5892, -0.3701],
            [0.5457, -0.1075, 1.2058, -0.6441],
            [0.4784, -0.3307, 1.7285, -0.8762],
            [0.4153, -0.5400, 2.2188, -1.0940],
            [0.3542, -0.7425, 2.6931, -1.3048],
        ]
    )
    for r, target, expected in zip(results, targets, expected_weights, strict=True):
        assert np.isclose(r.volatility, target, atol=1e-4)
        assert np.allclose(r.weights, expected, atol=2e-3)


def test_mvo_target_portfolio_mu_matches_matlab_reference(roncalli_data):
    mu, cov = roncalli_data
    targets = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
    results = mvo_target_portfolio(mu, cov, targets, problem="mu", lb=-100.0, ub=100.0)
    expected_sigma = np.array([0.1202, 0.1344, 0.1654, 0.2058, 0.2510])
    for r, target, sigma in zip(results, targets, expected_sigma, strict=True):
        assert np.isclose(r.expected_return, target, atol=1e-4)
        assert np.isclose(r.volatility, sigma, atol=1e-3)


def test_mvo_target_portfolio_sigma_with_bound_constraints(roncalli_data):
    # test_mvo3.m, x2 (long-only) and x3 (long-only, 40% cap) cases.
    mu, cov = roncalli_data
    targets = np.array([0.15, 0.20])

    long_only = mvo_target_portfolio(mu, cov, targets, problem="sigma", lb=0.0, ub=100.0)
    assert np.allclose(long_only[0].weights, [0.4559, 0.2474, 0.2967, -0.0], atol=2e-3)
    assert np.allclose(long_only[1].weights, [0.2488, 0.0496, 0.7015, -0.0], atol=2e-3)

    capped = mvo_target_portfolio(mu, cov, targets, problem="sigma", lb=0.0, ub=0.40)
    assert np.allclose(capped[0].weights, [0.40, 0.3436, 0.2564, -0.0], atol=2e-3)
    assert np.allclose(capped[1].weights, [0.0613, 0.40, 0.40, 0.1387], atol=2e-3)


def test_mvo_target_portfolio_below_min_is_unreachable(roncalli_data):
    mu, cov = roncalli_data
    # the minimum achievable volatility (gamma=0, unbounded) is ~12%
    result = mvo_target_portfolio(mu, cov, 0.05, problem="sigma", lb=-100.0, ub=100.0)[0]
    assert np.isnan(result.volatility)
    assert np.all(np.isnan(result.weights))


def test_mvo_target_portfolio_matches_frontier_at_recovered_gamma(roncalli_data):
    # internal consistency: bisecting on sigma to gamma_star, then solving
    # mvo_portfolio at gamma_star directly, must reproduce the same point.
    mu, cov = roncalli_data
    result = mvo_target_portfolio(mu, cov, 0.18, problem="sigma", lb=-100.0, ub=100.0)[0]
    direct = mvo_portfolio(mu, cov, gamma=result.gamma, lb=-100.0, ub=100.0)
    assert np.allclose(result.weights, direct.weights, atol=1e-4)
