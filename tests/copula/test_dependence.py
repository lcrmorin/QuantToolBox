"""Tests for quanttoolbox.copula.dependence."""

import numpy as np
import pytest

from quanttoolbox.copula.dependence import (
    clayton_rho,
    clayton_tau,
    debye_function,
    dependogram,
    dilog_function,
    frank_rho,
    frank_tau,
    gaussian_rho,
    gaussian_tau,
    gumbel_rho,
    gumbel_tau,
    spearman_rho_numeric,
)
from quanttoolbox.copula.families import clayton_cdf, frank_cdf, gumbel_cdf

# ---------------------------------------------------------------------------
# Debye / dilogarithm special functions
# ---------------------------------------------------------------------------


def test_debye_function_at_zero_is_1_over_k_plus_1():
    assert np.isclose(debye_function(0.0, 1), 0.5)
    assert np.isclose(debye_function(0.0, 2), 1.0 / 3.0)


def test_debye_function_reflection_identity():
    # D_k(-x) = D_k(x) + k*x/(k+1)
    x, k = 1.7, 1
    d_pos = debye_function(x, k)
    d_neg = debye_function(-x, k)
    assert np.isclose(d_neg, d_pos + k * x / (k + 1.0))


def test_debye_function_matches_direct_quadrature():
    from scipy.integrate import quad

    x, k = 2.3, 1
    integral, _ = quad(lambda t: t**k / np.expm1(t), 0.0, x)
    expected = (k / x**k) * integral
    assert np.isclose(debye_function(x, k), expected)


def test_dilog_function_matches_known_identity():
    # integral_0^1 log(t) / (1 - t) dt = -pi^2 / 6
    assert np.isclose(dilog_function(0.0), -(np.pi**2) / 6.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Kendall's tau -- cross-checked against statsmodels
# ---------------------------------------------------------------------------


def test_clayton_tau_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    theta = 2.5
    copula = sm.ClaytonCopula(theta=theta, k_dim=2)
    assert np.isclose(clayton_tau(theta), copula.tau())


def test_frank_tau_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    theta = 3.0
    copula = sm.FrankCopula(theta=theta, k_dim=2)
    assert np.isclose(frank_tau(theta), copula.tau(), atol=1e-6)


def test_gumbel_tau_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    theta = 2.0
    copula = sm.GumbelCopula(theta=theta, k_dim=2)
    assert np.isclose(gumbel_tau(theta), copula.tau())


def test_gaussian_tau_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    rho = 0.6
    corr = np.array([[1.0, rho], [rho, 1.0]])
    copula = sm.GaussianCopula(corr=corr, k_dim=2)
    assert np.isclose(gaussian_tau(rho), copula.tau())


# ---------------------------------------------------------------------------
# Spearman's rho -- closed forms vs. the generic numeric estimator
# ---------------------------------------------------------------------------


def test_spearman_rho_numeric_matches_frank_closed_form():
    theta = 3.0
    numeric = spearman_rho_numeric(lambda u1, u2: frank_cdf(u1, u2, theta))
    closed = frank_rho(theta)
    assert np.isclose(numeric, closed, atol=1e-4)


def test_spearman_rho_numeric_matches_gaussian_closed_form():
    rho = 0.5
    corr = np.array([[1.0, rho], [rho, 1.0]])
    from quanttoolbox.copula.families import gaussian_copula_cdf

    numeric = spearman_rho_numeric(
        lambda u1, u2: gaussian_copula_cdf(np.array([[u1, u2]]), corr)[0]
    )
    closed = gaussian_rho(rho)
    assert np.isclose(numeric, closed, atol=1e-3)


def test_clayton_rho_is_generic_numeric_wrapper():
    theta = 2.0
    expected = spearman_rho_numeric(lambda u1, u2: clayton_cdf(u1, u2, theta))
    assert np.isclose(clayton_rho(theta), expected)


def test_gumbel_rho_is_generic_numeric_wrapper():
    theta = 2.0
    expected = spearman_rho_numeric(lambda u1, u2: gumbel_cdf(u1, u2, theta))
    assert np.isclose(gumbel_rho(theta), expected)


def test_independence_copula_has_zero_tau_and_rho():
    assert np.isclose(gaussian_tau(0.0), 0.0)
    assert np.isclose(gaussian_rho(0.0), 0.0)
    numeric = spearman_rho_numeric(lambda u1, u2: u1 * u2)
    assert np.isclose(numeric, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Dependogram
# ---------------------------------------------------------------------------


def test_dependogram_output_in_open_unit_interval():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 2))
    pseudo = dependogram(data)
    assert pseudo.shape == data.shape
    assert np.all(pseudo > 0.0) and np.all(pseudo < 1.0)


def test_dependogram_matches_rank_over_n_plus_1():
    from scipy.stats import rankdata

    data = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    pseudo = dependogram(data)
    expected = np.column_stack([rankdata(data[:, 0]) / 4.0, rankdata(data[:, 1]) / 4.0])
    assert np.allclose(pseudo, expected)


def test_dependogram_is_invariant_to_monotone_transform():
    rng = np.random.default_rng(1)
    data = rng.lognormal(size=(30, 2))
    pseudo1 = dependogram(data)
    pseudo2 = dependogram(np.log(data))
    assert np.allclose(pseudo1, pseudo2)
