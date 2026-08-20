"""Tests for quanttoolbox.mixtures.jump_diffusion."""

import numpy as np
import pytest

from quanttoolbox.mixtures.gaussian_mixture import mixture_compute_var
from quanttoolbox.mixtures.jump_diffusion import (
    bivariate_lognormal_skewness,
    jump_compute_es,
    jump_compute_rb_var,
    jump_compute_rc_var,
    jump_compute_var,
    jump_pdf_assets,
    jump_simulate,
    jump_skewness,
    jump_to_mixture_params,
    lognormal_moments,
    lognormal_skewness,
)


@pytest.fixture
def jump_params():
    return dict(
        mu_bar=np.array([0.06, 0.05, 0.04]),
        sigma_bar=np.diag([0.02, 0.03, 0.025]) + 0.005,
        mu_tilde=np.array([-0.15, -0.10, -0.08]),
        sigma_tilde=np.diag([0.05, 0.06, 0.05]) + 0.01,
        lambda_=2.0,
        dt=1 / 260,
    )


def test_jump_to_mixture_params_matches_manual_transform(jump_params):
    p = jump_params
    params = jump_to_mixture_params(**p)
    assert np.isclose(params.pi1, 1 - p["lambda_"] * p["dt"])
    assert np.isclose(params.pi2, p["lambda_"] * p["dt"])
    assert np.allclose(params.mu1, p["mu_bar"] * p["dt"])
    assert np.allclose(params.mu2, p["mu_bar"] * p["dt"] + p["mu_tilde"])


def test_jump_compute_var_matches_direct_mixture_call(jump_params):
    p = jump_params
    x = np.array([0.4, 0.3, 0.3])
    alpha = 0.95

    var_jump, var_gaussian = jump_compute_var(x, **p, alpha=alpha)
    params = jump_to_mixture_params(**p)
    var_direct, var_gaussian_direct = mixture_compute_var(x, params, alpha)

    assert np.isclose(var_jump, var_direct)
    assert np.isclose(var_gaussian, var_gaussian_direct)


def test_jump_compute_es_exceeds_var(jump_params):
    p = jump_params
    x = np.array([0.4, 0.3, 0.3])
    es, var, es_g, var_g = jump_compute_es(x, **p, alpha=0.95)
    assert es > var


def test_jump_compute_rc_var_sums_to_total_risk(jump_params):
    p = jump_params
    x = np.array([0.4, 0.3, 0.3])
    result = jump_compute_rc_var(x, **p, alpha=0.95)
    assert np.isclose(np.sum(result.risk_contribution), result.risk, atol=1e-6)


def test_jump_compute_rb_var_equal_budget(jump_params):
    p = jump_params
    result = jump_compute_rb_var(**p, alpha=0.95, b=np.full(3, 1 / 3))
    assert result.converged
    assert np.allclose(result.pct_risk_contribution, 1 / 3, atol=0.02)


def test_jump_pdf_assets_integrates_to_one():
    mu_bar = np.array([0.0])
    sigma_bar = np.array([[1.0]])
    mu_tilde = np.array([-0.5])
    sigma_tilde = np.array([[2.0]])
    lambda_, dt = 5.0, 0.01

    y = np.linspace(-20, 20, 4000)[:, None]
    pdf, _, _ = jump_pdf_assets(y, mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    integral = np.trapezoid(pdf[:, 0], y[:, 0])
    assert np.isclose(integral, 1.0, atol=1e-3)


def test_jump_simulate_regime_frequency_matches_lambda_dt(rng):
    mu_bar = np.array([0.05])
    sigma_bar = np.array([[0.01]])
    mu_tilde = np.array([-0.2])
    sigma_tilde = np.array([[0.05]])
    lambda_, dt = 5.0, 1 / 260

    samples, regime = jump_simulate(
        mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt, 20000, rng=rng
    )
    expected_jump_freq = lambda_ * dt
    observed_jump_freq = np.mean(regime == 2)
    assert np.isclose(observed_jump_freq, expected_jump_freq, atol=0.01)


def test_jump_skewness_zero_jump_intensity_gives_zero_skew():
    # with lambda=0, there's no jump component at all -> pure Gaussian, skew=0
    mu, sigma, gamma1 = jump_skewness(
        mu_bar=0.05, sigma_bar=0.1, mu_tilde=-0.5, sigma_tilde=0.2, lambda_=0.0, dt=1 / 260
    )
    assert np.isclose(gamma1, 0.0, atol=1e-8)


def test_jump_skewness_negative_jump_mean_gives_negative_skew():
    mu, sigma, gamma1 = jump_skewness(
        mu_bar=0.05, sigma_bar=0.1, mu_tilde=-0.5, sigma_tilde=0.2, lambda_=5.0, dt=1 / 260
    )
    assert gamma1 < 0


def test_lognormal_moments_small_sigma_mean_near_exp_mu():
    mu, sigma = np.array([0.5]), np.array([0.01])
    mu_x, sigma_x, gamma1, gamma2 = lognormal_moments(mu, sigma)
    assert np.isclose(mu_x[0], np.exp(0.5), atol=1e-3)


def test_lognormal_skewness_matches_moments_based_computation():
    mu, sigma = np.array([0.2]), np.array([0.3])
    _, _, gamma1_from_moments, _ = lognormal_moments(mu, sigma)
    gamma1_direct = lognormal_skewness(mu, sigma)
    assert np.allclose(gamma1_from_moments, gamma1_direct, atol=1e-6)


def test_lognormal_skewness_positive_and_increasing_in_sigma():
    sigma_low = lognormal_skewness(np.array([0.0]), np.array([0.1]))
    sigma_high = lognormal_skewness(np.array([0.0]), np.array([0.5]))
    assert sigma_low[0] > 0
    assert sigma_high[0] > sigma_low[0]


def test_bivariate_lognormal_skewness_symmetric_case():
    # identical X and Y (same params, rho=1) should give skewness equal to
    # the univariate lognormal skewness of 2X (scaling doesn't affect skewness)
    mu, sigma = 0.1, 0.2
    skew = bivariate_lognormal_skewness(mu, sigma, mu, sigma, rho=1.0)
    univariate_skew = lognormal_skewness(np.array([mu]), np.array([sigma]))[0]
    assert np.isclose(skew, univariate_skew, atol=1e-6)
