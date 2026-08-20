"""Tests for quanttoolbox.stats.regression.ridge."""

import numpy as np

from quanttoolbox.stats.regression.ridge import ridge, ridge_tau_targeted


def test_ridge_at_zero_lambda_matches_ols(rng):
    n, p = 200, 3
    x = rng.standard_normal((n, p))
    true_beta = np.array([1.0, -1.0, 0.5])
    y = x @ true_beta + rng.standard_normal(n) * 0.01

    beta, df, complexity = ridge(y, x, lambda_=0.0)
    beta_ols = np.linalg.inv(x.T @ x) @ (x.T @ y)
    assert np.allclose(beta, beta_ols, atol=1e-6)
    assert np.isclose(df, p, atol=1e-6)


def test_ridge_shrinks_coefficients_toward_zero(rng):
    n, p = 200, 3
    x = rng.standard_normal((n, p))
    y = x @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.5

    beta_small, _, _ = ridge(y, x, lambda_=0.1)
    beta_large, _, _ = ridge(y, x, lambda_=1000.0)
    assert np.sum(np.abs(beta_large)) < np.sum(np.abs(beta_small))


def test_ridge_multiple_lambdas_shape(rng):
    x = rng.standard_normal((100, 3))
    y = rng.standard_normal(100)
    lambdas = np.array([0.1, 1.0, 10.0])
    beta, df, complexity = ridge(y, x, lambda_=lambdas)
    assert beta.shape == (3, 3)
    assert df.shape == (3,)


def test_ridge_tau_targeted_matches_desired_norm(rng):
    n, p = 200, 3
    x = rng.standard_normal((n, p))
    y = x @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.1

    # Use a wide lambda grid: with the default (faithful to the original
    # MATLAB 0:0.01:100 range), a small tau may be unreachable for some
    # data scales since ridge only shrinks the norm down from the OLS
    # value as lambda grows.
    beta, lam, df, complexity = ridge_tau_targeted(
        y, x, tau=0.5, lambda_search=np.linspace(0, 1000, 20000)
    )
    assert abs(np.sum(beta**2) - 0.5) < 0.05
