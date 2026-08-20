"""Tests for quanttoolbox.portfolio.mean_variance."""

import numpy as np
import pytest

from quanttoolbox.portfolio.mean_variance import (
    mdp_portfolio,
    minvar_portfolio,
    mvo_frontier,
    mvo_portfolio,
)


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
