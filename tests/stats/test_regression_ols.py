"""Tests for quanttoolbox.stats.regression.ols."""

import numpy as np

from quanttoolbox.stats.regression.ols import (
    center,
    conditional_normal_regression,
    ols,
    pca,
    standardize,
)


def test_ols_recovers_known_coefficients(rng):
    n, p = 500, 3
    x = rng.standard_normal((n, p))
    true_beta = np.array([1.5, -2.0, 0.5])
    y = x @ true_beta + rng.standard_normal(n) * 0.01

    result = ols(y, x)
    assert np.allclose(result.beta, true_beta, atol=0.05)
    assert result.nobs == n
    assert result.nvar == p


def test_ols_matches_lstsq(rng):
    n, p = 200, 4
    x = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    result = ols(y, x)
    beta_ref, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert np.allclose(result.beta, beta_ref, atol=1e-8)


def test_ols_drops_nan_rows(rng):
    x = rng.standard_normal((100, 2))
    y = x @ np.array([1.0, 2.0])
    y[5] = np.nan
    result = ols(y, x)
    assert result.nobs == 100
    assert np.isnan(result.residuals[5])
    assert not np.any(np.isnan(np.delete(result.residuals, 5)))


def test_center_zeroes_mean(rng):
    x = rng.standard_normal((100, 3)) + 5.0
    centered = center(x)
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-8)


def test_standardize_unit_variance(rng):
    x = rng.standard_normal((200, 3)) * 3.0 + 10.0
    std = standardize(x)
    assert np.allclose(std.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(std.std(axis=0, ddof=1), 1.0, atol=1e-8)


def test_conditional_normal_regression_matches_ols(rng):
    n = 5000
    x = rng.standard_normal((n, 1))
    true_beta = 2.0
    y = true_beta * x[:, 0] + rng.standard_normal(n) * 0.5

    data = np.column_stack([y, x[:, 0]])
    mu = data.mean(axis=0)
    sigma = np.cov(data, rowvar=False)

    result = conditional_normal_regression(
        mu_y=mu[0],
        mu_x=np.array([mu[1]]),
        sigma_yy=sigma[0, 0],
        sigma_yx=np.array([sigma[0, 1]]),
        sigma_xx=np.array([[sigma[1, 1]]]),
    )
    assert np.allclose(result.beta, true_beta, atol=0.05)


def test_conditional_normal_regression_leave_one_out(rng):
    sigma = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
    mu = np.array([1.0, 2.0, 3.0])
    result = conditional_normal_regression(mu=mu, sigma=sigma)
    assert result.beta.shape == (3, 3)
    assert result.beta0.shape == (3,)
    assert np.all(result.r_squared >= 0) and np.all(result.r_squared <= 1)


def test_pca_recovers_known_factor_structure(rng):
    n = 5000
    factor = rng.standard_normal(n)
    loadings = np.array([1.0, 0.8, -0.5, 0.3])
    noise = rng.standard_normal((n, 4)) * 0.05
    x = factor[:, None] * loadings[None, :] + noise

    result = pca(x, num_factors=1)
    # first factor should explain the vast majority of variance
    assert result.quality[0] > 0.9


def test_pca_on_correlation_matrix_directly():
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    result = pca(corr)
    assert result.eigenvalues.shape[0] == 3
    assert np.all(np.diff(result.eigenvalues) <= 0)  # descending order
    assert np.isclose(result.cum_quality[-1], 1.0, atol=1e-8)
