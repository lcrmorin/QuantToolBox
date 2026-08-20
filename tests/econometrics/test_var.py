"""Tests for quanttoolbox.econometrics.var."""

import numpy as np
import pytest

from quanttoolbox.econometrics.var import varx_estimate, varx_estimate_cml, varx_order


@pytest.fixture
def var1_data(rng):
    n = 1000
    k = 2
    phi = np.array([[0.5, 0.1], [0.0, 0.3]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = phi @ y[t - 1] + rng.standard_normal(k) * 0.3
    return y, phi


def test_varx_recovers_var1_coefficients(var1_data):
    y, phi = var1_data
    result = varx_estimate(y, p=1, method="ls")
    assert np.allclose(result.phi, phi, atol=0.1)
    assert result.k == 2
    assert result.l_exog == 0
    assert result.beta is None


def test_varx_residuals_padded_with_nan(var1_data):
    y, _ = var1_data
    result = varx_estimate(y, p=1)
    assert np.all(np.isnan(result.residuals[0]))
    assert not np.any(np.isnan(result.residuals[1:]))


def test_varx_with_exogenous_regressor(rng):
    n = 800
    k = 2
    phi = np.array([[0.4, 0.0], [0.0, 0.4]])
    beta_true = np.array([[1.5], [-0.5]])
    x = rng.standard_normal((n, 1))
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = (
            phi @ y[t - 1] + (beta_true @ x[t : t + 1].T).flatten() + rng.standard_normal(k) * 0.2
        )

    result = varx_estimate(y, x, p=1)
    assert result.l_exog == 1
    assert result.beta is not None
    assert np.allclose(result.beta, beta_true, atol=0.15)


def test_varx_ml_method_includes_cholesky_vech(var1_data):
    y, _ = var1_data
    result_ls = varx_estimate(y, p=1, method="ls")
    result_ml = varx_estimate(y, p=1, method="ml")
    # ML theta includes extra vech(chol(Sigma)) entries appended
    assert result_ml.theta.shape[0] > result_ls.theta.shape[0]


def test_varx_with_restriction_diagonal_only(rng):
    # restrict off-diagonal Phi entries to zero
    n = 800
    k = 2
    phi = np.array([[0.5, 0.0], [0.0, 0.3]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = phi @ y[t - 1] + rng.standard_normal(k) * 0.3

    # theta = vec(Phi) has 4 entries [phi11, phi21, phi12, phi22] (column-major)
    # restrict phi21=phi12=0: free params are [phi11, phi22]
    rr = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    r_vec = np.zeros(4)
    result = varx_estimate(y, p=1, restriction=(rr, r_vec))
    assert np.isclose(result.phi[1, 0], 0.0, atol=1e-8)
    assert np.isclose(result.phi[0, 1], 0.0, atol=1e-8)
    assert np.allclose(np.diag(result.phi), np.diag(phi), atol=0.15)


def test_varx_cml_converges_close_to_ls(var1_data):
    y, phi = var1_data
    result = varx_estimate_cml(y, p=1)
    assert np.allclose(result.phi, phi, atol=0.15)


def test_varx_order_selects_true_lag(var1_data):
    y, _ = var1_data
    result = varx_order(y, x=None, p_max=4)
    assert result.criteria.shape == (5, 7)
    # all 7 criteria should agree the true lag (1) is optimal for this clean DGP
    assert np.all(result.optimal_p == 1)


def test_varx_order_with_explicit_lag_list(var1_data):
    y, _ = var1_data
    result = varx_order(y, x=None, p_max=np.array([0, 1, 2]))
    assert result.criteria.shape == (3, 7)
    assert list(result.p_values) == [0, 1, 2]
