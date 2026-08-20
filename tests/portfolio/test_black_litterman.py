"""Tests for quanttoolbox.portfolio.black_litterman."""

import numpy as np

from quanttoolbox.portfolio.black_litterman import (
    black_litterman_moments,
    implied_risk_premia,
)


def test_implied_risk_premia_basic(rng):
    n = 4
    a = rng.standard_normal((100, n))
    cov = a.T @ a / 100 + 0.01 * np.eye(n)
    x = np.full(n, 0.25)

    result = implied_risk_premia(x, cov, sharpe_ratio=0.4)
    sigma_x = np.sqrt(x @ cov @ x)
    assert np.isclose(result.phi, 0.4 / sigma_x)
    assert np.isclose(result.gamma, sigma_x / 0.4)
    # implied returns should be proportional to Cov @ x
    assert np.allclose(result.pi / (cov @ x), result.pi[0] / (cov @ x)[0], atol=1e-6)


def test_implied_risk_premia_achieves_target_sharpe(rng):
    n = 3
    a = rng.standard_normal((100, n))
    cov = a.T @ a / 100 + 0.01 * np.eye(n)
    x = np.array([0.5, 0.3, 0.2])
    target_sr = 0.35

    result = implied_risk_premia(x, cov, sharpe_ratio=target_sr)
    # x'pi / sigma_x should reproduce the target sharpe ratio
    sigma_x = np.sqrt(x @ cov @ x)
    implied_sr = (x @ result.pi) / sigma_x
    assert np.isclose(implied_sr, target_sr, atol=1e-6)


def test_black_litterman_no_views_returns_prior():
    n = 3
    mu_tilde = np.array([0.05, 0.03, 0.04])
    gamma_matrix = 0.01 * np.eye(n)
    # a view with essentially infinite uncertainty (huge Omega) should not move the posterior
    p = np.array([[1.0, 0.0, 0.0]])
    q = np.array([0.20])  # wildly different from prior
    omega = np.array([[1e10]])

    result = black_litterman_moments(mu_tilde, gamma_matrix, p, q, omega)
    assert np.allclose(result.mu_bar, mu_tilde, atol=1e-4)
    assert np.allclose(result.sigma_bar, gamma_matrix, atol=1e-4)


def test_black_litterman_confident_view_pulls_toward_target():
    mu_tilde = np.array([0.05, 0.03])
    gamma_matrix = np.array([[0.02, 0.005], [0.005, 0.015]])
    p = np.array([[1.0, 0.0]])
    q = np.array([0.15])  # much higher than prior for asset 1
    omega = np.array([[1e-6]])  # very confident view

    result = black_litterman_moments(mu_tilde, gamma_matrix, p, q, omega)
    # posterior mean for asset 1 should move strongly toward the view target
    assert result.mu_bar[0] > mu_tilde[0]
    assert np.isclose(result.mu_bar[0], q[0], atol=0.01)


def test_black_litterman_posterior_variance_shrinks():
    mu_tilde = np.array([0.05, 0.03])
    gamma_matrix = np.array([[0.02, 0.005], [0.005, 0.015]])
    p = np.array([[1.0, 0.0]])
    q = np.array([0.10])
    omega = np.array([[0.001]])  # moderately confident view

    result = black_litterman_moments(mu_tilde, gamma_matrix, p, q, omega)
    # incorporating a view should reduce uncertainty (posterior variance <= prior variance)
    assert result.sigma_bar[0, 0] <= gamma_matrix[0, 0] + 1e-10
