"""Tests for quanttoolbox.mixtures.gaussian_mixture."""

import numpy as np
import pytest

from quanttoolbox.mixtures.gaussian_mixture import (
    MixtureParams,
    estimate_em_mixture,
    mixture_compute_es,
    mixture_compute_rb_es,
    mixture_compute_rb_var,
    mixture_compute_rc_es,
    mixture_compute_rc_var,
    mixture_compute_var,
    mixture_moments,
    mixture_pdf_assets,
    mixture_pdf_portfolio,
    mixture_probability_filtering,
    mixture_simulate,
    mixture_skewness,
    mixture_skewness_portfolio,
    mixture_univariate_thresholding,
)


@pytest.fixture
def sample_params():
    mu1 = np.array([0.06, 0.05, 0.04])
    sigma1 = np.diag([0.02, 0.03, 0.025]) + 0.005
    mu2 = np.array([-0.20, -0.15, -0.10])
    sigma2 = np.diag([0.08, 0.10, 0.09]) + 0.02
    return MixtureParams(pi1=0.95, mu1=mu1, sigma1=sigma1, pi2=0.05, mu2=mu2, sigma2=sigma2)


def test_mixture_moments_degenerate_pi1_one():
    # pi1=1 should reduce exactly to the first component's own moments
    mu1, sigma1 = np.array([0.05, 0.03]), np.eye(2) * 0.02
    mu2, sigma2 = np.array([-0.1, -0.1]), np.eye(2) * 0.1
    params = MixtureParams(pi1=1.0, mu1=mu1, sigma1=sigma1, pi2=0.0, mu2=mu2, sigma2=sigma2)
    mu_bar, sigma_bar = mixture_moments(params)
    assert np.allclose(mu_bar, mu1)
    assert np.allclose(sigma_bar, sigma1)


def test_mixture_skewness_symmetric_mixture_is_zero():
    mu, sigma, gamma1 = mixture_skewness(0.5, 1.0, 1.0, 0.5, -1.0, 1.0)
    assert np.isclose(mu, 0.0)
    assert np.isclose(gamma1, 0.0, atol=1e-10)


def test_mixture_skewness_stress_regime_gives_negative_skew():
    # regime 2 is rare (low pi2) and has much lower mean -> negative skew
    _, _, gamma1 = mixture_skewness(0.95, 0.05, 0.1, 0.05, -0.5, 0.2)
    assert gamma1 < 0


def test_mixture_skewness_portfolio_matches_direct_call(sample_params):
    x = np.array([0.4, 0.3, 0.3])
    mu_x, sigma_x, gamma1_x = mixture_skewness_portfolio(x, sample_params)
    p = sample_params
    mu1_x = float(x @ p.mu1)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    mu2_x = float(x @ p.mu2)
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))
    expected = mixture_skewness(p.pi1, mu1_x, sigma1_x, p.pi2, mu2_x, sigma2_x)
    assert np.allclose([mu_x, sigma_x, gamma1_x], expected)


def test_mixture_pdf_assets_integrates_to_one():
    mu1, sigma1 = np.array([0.0]), np.array([[1.0]])
    mu2, sigma2 = np.array([0.0]), np.array([[4.0]])
    params = MixtureParams(pi1=0.5, mu1=mu1, sigma1=sigma1, pi2=0.5, mu2=mu2, sigma2=sigma2)
    y = np.linspace(-20, 20, 4000)[:, None]
    pdf, _, _ = mixture_pdf_assets(y, params)
    integral = np.trapezoid(pdf[:, 0], y[:, 0])
    assert np.isclose(integral, 1.0, atol=1e-3)


def test_mixture_pdf_portfolio_matches_pdf_assets_for_unit_weight():
    mu1, sigma1 = np.array([0.05]), np.array([[0.02]])
    mu2, sigma2 = np.array([-0.1]), np.array([[0.05]])
    params = MixtureParams(pi1=0.9, mu1=mu1, sigma1=sigma1, pi2=0.1, mu2=mu2, sigma2=sigma2)
    x = np.array([1.0])
    y = np.array([0.0, 0.02, -0.05])
    pdf_port, _, _ = mixture_pdf_portfolio(y, x, params)
    pdf_asset, _, _ = mixture_pdf_assets(y[:, None], params)
    assert np.allclose(pdf_port, pdf_asset[:, 0], atol=1e-8)


def test_mixture_simulate_shape_and_regime_frequency(rng):
    params = MixtureParams(
        pi1=0.8,
        mu1=np.array([0.05]),
        sigma1=np.array([[0.01]]),
        pi2=0.2,
        mu2=np.array([-0.15]),
        sigma2=np.array([[0.04]]),
    )
    samples, regime = mixture_simulate(params, n_samples=5000, rng=rng)
    assert samples.shape == (5000, 1)
    assert np.isclose(np.mean(regime == 1), 0.8, atol=0.03)


def test_mixture_probability_filtering_favors_correct_regime():
    mu1, sigma1 = np.array([0.05]), np.array([[0.01]])
    mu2, sigma2 = np.array([-0.5]), np.array([[0.01]])
    params = MixtureParams(pi1=0.5, mu1=mu1, sigma1=sigma1, pi2=0.5, mu2=mu2, sigma2=sigma2)

    pi1_normal, _ = mixture_probability_filtering(np.array([0.05]), params)
    pi1_stress, _ = mixture_probability_filtering(np.array([-0.5]), params)
    assert pi1_normal > 0.9
    assert pi1_stress < 0.1


def test_mixture_univariate_thresholding_symmetric_case():
    # symmetric case around 0 -> thresholds should be symmetric
    y_minus, y_plus = mixture_univariate_thresholding(0.9, 0.0, 0.1, 0.1, 0.0, 0.3, 0.5)
    assert np.isclose(y_minus, -y_plus, atol=1e-6)


def test_mixture_compute_var_exceeds_gaussian_only(sample_params):
    x = np.array([0.4, 0.3, 0.3])
    var_mixture, var_gaussian = mixture_compute_var(x, sample_params, alpha=0.95)
    assert var_mixture > var_gaussian


def test_mixture_compute_es_exceeds_var(sample_params):
    x = np.array([0.4, 0.3, 0.3])
    es_mixture, var_mixture, es_gaussian, var_gaussian = mixture_compute_es(
        x, sample_params, alpha=0.95
    )
    assert es_mixture > var_mixture
    assert es_gaussian > var_gaussian


def test_mixture_compute_rc_var_sums_to_total_risk(sample_params):
    x = np.array([0.4, 0.3, 0.3])
    result = mixture_compute_rc_var(x, sample_params, alpha=0.95)
    assert np.isclose(np.sum(result.risk_contribution), result.risk, atol=1e-6)
    assert np.isclose(np.sum(result.pct_risk_contribution), 1.0, atol=1e-6)


def test_mixture_compute_rc_es_sums_to_total_risk(sample_params):
    x = np.array([0.4, 0.3, 0.3])
    result = mixture_compute_rc_es(x, sample_params, alpha=0.95)
    assert np.isclose(np.sum(result.risk_contribution), result.risk, atol=1e-6)


def test_mixture_compute_rb_var_equal_budget_gives_equal_contributions(sample_params):
    result = mixture_compute_rb_var(sample_params, alpha=0.95, b=np.full(3, 1 / 3))
    assert result.converged
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-4)
    assert np.allclose(result.pct_risk_contribution, 1 / 3, atol=0.01)


def test_mixture_compute_rb_es_equal_budget_gives_equal_contributions(sample_params):
    result = mixture_compute_rb_es(sample_params, alpha=0.95, b=np.full(3, 1 / 3))
    assert result.converged
    assert np.allclose(result.pct_risk_contribution, 1 / 3, atol=0.01)


def test_estimate_em_mixture_recovers_known_parameters(rng):
    true_params = MixtureParams(
        pi1=0.8,
        mu1=np.array([0.05]),
        sigma1=np.array([[0.005]]),
        pi2=0.2,
        mu2=np.array([-0.30]),
        sigma2=np.array([[0.02]]),
    )
    samples, _ = mixture_simulate(true_params, n_samples=10000, rng=rng)

    init = MixtureParams(
        pi1=0.5,
        mu1=np.array([0.0]),
        sigma1=np.array([[0.01]]),
        pi2=0.5,
        mu2=np.array([-0.2]),
        sigma2=np.array([[0.01]]),
    )
    result = estimate_em_mixture(samples, init, max_iters=5000, tol=1e-8)

    assert result.converged
    assert np.isclose(result.params.pi1, 0.8, atol=0.03)
    assert np.isclose(result.params.mu1[0], 0.05, atol=0.02)
    assert np.isclose(result.params.mu2[0], -0.30, atol=0.05)


def test_estimate_em_mixture_fixed_weights_mode(rng):
    true_params = MixtureParams(
        pi1=0.7,
        mu1=np.array([0.05]),
        sigma1=np.array([[0.005]]),
        pi2=0.3,
        mu2=np.array([-0.20]),
        sigma2=np.array([[0.015]]),
    )
    samples, _ = mixture_simulate(true_params, n_samples=5000, rng=rng)

    init = MixtureParams(
        pi1=0.7,
        mu1=np.array([0.0]),
        sigma1=np.array([[0.01]]),
        pi2=0.3,
        mu2=np.array([-0.15]),
        sigma2=np.array([[0.01]]),
    )
    result = estimate_em_mixture(samples, init, estimate_mixing_weights=False, max_iters=2000)
    # weights should stay fixed at their initial values
    assert result.params.pi1 == 0.7
    assert result.params.pi2 == 0.3
