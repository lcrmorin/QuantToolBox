"""Tests for quanttoolbox.portfolio.attribution."""

import numpy as np

from quanttoolbox.portfolio.attribution import beta_pi_alpha


def test_beta_pi_alpha_matches_hand_derived_formula():
    rng = np.random.default_rng(0)
    n = 4
    mu = rng.uniform(0.02, 0.10, n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.01 + np.eye(n) * 0.001
    x = np.array([0.3, 0.2, 0.3, 0.2])
    r = 0.02

    result = beta_pi_alpha(x, mu, sigma, r)

    mu_x = x @ mu
    sigma_x = np.sqrt(x @ sigma @ x)
    beta_x = (sigma @ x) / sigma_x**2
    pi_x = beta_x * (mu_x - r)
    alpha_x = (mu - r) - pi_x

    assert np.isclose(result.mu_x, mu_x)
    assert np.isclose(result.sigma_x, sigma_x)
    assert np.allclose(result.beta_x, beta_x)
    assert np.allclose(result.pi_x, pi_x)
    assert np.allclose(result.alpha_x, alpha_x)


def test_beta_pi_alpha_of_x_against_itself_has_beta_one_and_zero_alpha():
    # x's own beta against itself is always 1, so all of (mu_x - r) is
    # "priced" and x's own residual alpha is exactly 0.
    rng = np.random.default_rng(1)
    n = 3
    mu = rng.uniform(0.02, 0.10, n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.01 + np.eye(n) * 0.001
    x = np.array([0.5, 0.3, 0.2])
    r = 0.01

    result = beta_pi_alpha(x, mu, sigma, r)
    beta_of_x_itself = x @ result.beta_x
    pi_of_x_itself = x @ result.pi_x
    alpha_of_x_itself = x @ result.alpha_x

    assert np.isclose(beta_of_x_itself, 1.0)
    assert np.isclose(pi_of_x_itself, result.mu_x - r)
    assert np.isclose(alpha_of_x_itself, 0.0, atol=1e-10)


def test_beta_pi_alpha_pi_plus_alpha_recovers_excess_return():
    rng = np.random.default_rng(2)
    n = 5
    mu = rng.uniform(0.02, 0.10, n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.01 + np.eye(n) * 0.001
    x = rng.dirichlet(np.ones(n))
    r = 0.015

    result = beta_pi_alpha(x, mu, sigma, r)
    assert np.allclose(result.pi_x + result.alpha_x, mu - r)
