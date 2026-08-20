"""Tests for quanttoolbox.stats.regression.quantile."""

import numpy as np

from quanttoolbox.stats.regression.quantile import (
    qr_copula_normal,
    qr_copula_student,
    quantile_regression,
)


def test_quantile_regression_median_close_to_ols(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.3

    beta, u, v = quantile_regression(y, x, tau=0.5)
    assert np.allclose(beta, true_beta, atol=0.2)


def test_quantile_regression_multiple_tau_shape(rng):
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.3

    tau = np.array([0.1, 0.5, 0.9])
    beta, u, v = quantile_regression(y, x, tau=tau)
    assert beta.shape == (2, 3)
    # higher tau should give a higher intercept for symmetric noise
    assert beta[0, 0] < beta[0, 2]


def test_qr_copula_normal_zero_correlation_ignores_u1():
    # rho=0 -> u2 should not depend on u1 at all
    result1 = qr_copula_normal(u1=0.1, rho=0.0, alpha=0.5)
    result2 = qr_copula_normal(u1=0.9, rho=0.0, alpha=0.5)
    assert np.isclose(result1, result2)


def test_qr_copula_normal_rho_one_perfect_dependence():
    # rho=1 -> u2 should equal u1 regardless of alpha
    u1 = 0.3
    result = qr_copula_normal(u1=u1, rho=1.0, alpha=0.7)
    assert np.isclose(result, u1, atol=1e-6)


def test_qr_copula_student_converges_to_normal_at_high_dof():
    u1, rho, alpha = 0.4, 0.5, 0.6
    normal_result = qr_copula_normal(u1, rho, alpha)
    student_result = qr_copula_student(u1, rho, nu=1000, alpha=alpha)
    assert np.isclose(normal_result, student_result, atol=0.01)
