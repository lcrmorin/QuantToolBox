"""Tests for quanttoolbox.econometrics.estimation."""

import numpy as np

from quanttoolbox.econometrics.estimation import (
    gmm_estimation,
    ml_estimation,
    ols_estimation,
    wald_test,
)


def test_ols_recovers_known_coefficients(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    result = ols_estimation(y, x)
    assert np.allclose(result.beta, true_beta, atol=0.05)
    assert result.n_obs_valid == n
    assert 0 <= result.r_squared <= 1


def test_ols_matches_closed_form_ols(rng):
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)

    result = ols_estimation(y, x)
    beta_ref, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert np.allclose(result.beta, beta_ref, atol=1e-8)


def test_ols_with_equality_restriction(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.05

    rr = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    r_vec = np.zeros(3)
    result = ols_estimation(y, x, restriction=(rr, r_vec))
    assert np.isclose(result.beta[1], result.beta[2], atol=1e-8)  # restriction exactly enforced
    assert np.allclose(result.beta, true_beta, atol=0.05)


def test_ols_drops_nan_rows(rng):
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0])
    y[5] = np.nan
    result = ols_estimation(y, x)
    assert result.n_obs == n
    assert result.n_obs_valid == n - 1
    assert np.isnan(result.residuals[5])


def test_gmm_recovers_ols_via_moment_conditions(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.3

    def moments(theta):
        u = y - x @ theta
        return x * u[:, None]

    result = gmm_estimation(moments, sv=np.zeros(2))
    assert result.converged
    assert np.allclose(result.theta, true_beta, atol=0.1)
    assert np.all(result.stderr > 0)


def test_gmm_j_test_zero_when_exactly_identified(rng):
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.3

    def moments(theta):
        u = y - x @ theta
        return x * u[:, None]

    result = gmm_estimation(moments, sv=np.zeros(2))
    # exactly identified (2 moments, 2 params) -> J-test is trivially 0
    assert result.j_test == 0.0


def test_ml_recovers_normal_regression_parameters(rng):
    n = 1000
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    true_sigma = 0.5
    y = x @ true_beta + rng.standard_normal(n) * true_sigma

    def logpdf(theta):
        beta, sigma = theta[:2], np.abs(theta[2])
        u = y - x @ beta
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (u / sigma) ** 2

    result = ml_estimation(logpdf, sv=np.array([0.0, 0.0, 1.0]))
    assert result.converged
    assert np.allclose(result.theta, [1.0, 2.0, true_sigma], atol=0.1)


def test_ml_covariance_types_all_run(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.4

    def logpdf(theta):
        beta, sigma = theta[:2], np.abs(theta[2])
        u = y - x @ beta
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (u / sigma) ** 2

    for cov_type in ["hessian", "opg", "hc"]:
        result = ml_estimation(logpdf, sv=np.array([0.0, 0.0, 1.0]), cov=cov_type)
        assert result.cov_type == cov_type
        assert np.all(np.isfinite(result.stderr))


def test_wald_test_fails_to_reject_true_null(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0, 2.0])  # slopes equal
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    result = ols_estimation(y, x)

    def constraint(theta):
        return np.array([theta[1] - theta[2]])

    wr = wald_test(constraint, result.beta, result.vcv, n_obs=n)
    assert wr.chi2_pvalue > 0.05  # should not reject true null


def test_wald_test_rejects_false_null(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0, -5.0])  # slopes very different
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    result = ols_estimation(y, x)

    def constraint(theta):
        return np.array([theta[1] - theta[2]])

    wr = wald_test(constraint, result.beta, result.vcv, n_obs=n)
    assert wr.chi2_pvalue < 0.01  # should strongly reject false null
