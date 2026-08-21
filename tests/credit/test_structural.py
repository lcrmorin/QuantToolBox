"""Tests for quanttoolbox.credit.structural."""

import numpy as np
from scipy.stats import norm

from quanttoolbox.credit.structural import (
    b0_extended_merton_model,
    black_scholes,
    e0_extended_merton_model,
    merton_jump_climate_model,
    merton_jump_model,
    pd_black_cox_model,
    pd_extended_merton_model,
    pd_merton_model,
    reinders_credit_model,
)


def test_black_scholes_put_call_parity():
    s0, k, sigma, t, b, r = 100.0, 95.0, 0.2, 1.0, 0.03, 0.03
    result = black_scholes(s0, k, sigma, t, b, r)

    lhs = result.call - result.put
    rhs = s0 * np.exp((b - r) * t) - k * np.exp(-r * t)
    assert np.isclose(lhs, rhs)


def test_black_scholes_matches_hand_computation():
    s0, k, sigma, t, b, r = 100.0, 90.0, 0.25, 0.5, 0.02, 0.02
    d1 = (np.log(s0 / k) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    expected_call = s0 * np.exp((b - r) * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    result = black_scholes(s0, k, sigma, t, b, r)
    assert np.isclose(result.call, expected_call)


def test_pd_merton_model_recovers_true_asset_value_and_volatility():
    a0_true, sigma_a_true, d, r, t = 150.0, 0.25, 100.0, 0.03, 1.0

    d1 = (np.log(a0_true / d) + (r + 0.5 * sigma_a_true**2) * t) / (sigma_a_true * np.sqrt(t))
    d2 = d1 - sigma_a_true * np.sqrt(t)
    e0 = a0_true * norm.cdf(d1) - d * np.exp(-r * t) * norm.cdf(d2)
    sigma_e = sigma_a_true * a0_true * norm.cdf(d1) / e0

    result = pd_merton_model(e0, sigma_e, d, mu_a=0.08, r=r, t=t, tau=1.0)

    assert np.isclose(result.a0, a0_true, rtol=1e-4)
    assert np.isclose(result.sigma_a, sigma_a_true, rtol=1e-4)
    assert np.isclose(result.s_tau + result.pd_tau, 1.0)


def test_pd_merton_model_vectorizes_over_scenarios():
    a0_true, sigma_a_true, d, r, t = 150.0, 0.25, 100.0, 0.03, 1.0
    d1 = (np.log(a0_true / d) + (r + 0.5 * sigma_a_true**2) * t) / (sigma_a_true * np.sqrt(t))
    d2 = d1 - sigma_a_true * np.sqrt(t)
    e0 = a0_true * norm.cdf(d1) - d * np.exp(-r * t) * norm.cdf(d2)
    sigma_e = sigma_a_true * a0_true * norm.cdf(d1) / e0

    result = pd_merton_model(
        np.array([e0, e0 * 1.1]), np.array([sigma_e, sigma_e * 0.9]), d, 0.08, r, t, 1.0
    )
    assert result.a0.shape == (2,)
    assert np.isclose(result.a0[0], a0_true, rtol=1e-4)


def test_extended_merton_balance_sheet_identity():
    a0, d, r, mu_a, delta0, mu_delta = 150.0, 100.0, 0.03, 0.08, 0.02, 0.01
    sigma_a, sigma_delta, rho, t = 0.25, 0.05, 0.3, 1.0

    b0 = b0_extended_merton_model(a0, d, r, mu_a, delta0, mu_delta, sigma_a, sigma_delta, rho, t)
    e0 = e0_extended_merton_model(a0, d, r, mu_a, delta0, mu_delta, sigma_a, sigma_delta, rho, t)

    delta0_prime = (
        delta0
        + 0.5 * mu_delta * t
        + 0.5 * rho * sigma_a * sigma_delta * t
        - (1 / 6) * sigma_delta**2 * t**2
    )
    expected_total = a0 * np.exp(-delta0_prime * t)

    assert np.isclose(e0 + b0, expected_total)


def test_pd_extended_merton_model_is_a_valid_probability():
    a0, d, r, mu_a, delta0, mu_delta = 150.0, 100.0, 0.03, 0.08, 0.02, 0.01
    sigma_a, sigma_delta, rho, t = 0.25, 0.05, 0.3, 1.0

    pd = pd_extended_merton_model(a0, d, r, mu_a, delta0, mu_delta, sigma_a, sigma_delta, rho, t)
    assert 0.0 <= pd <= 1.0


def test_pd_extended_merton_model_increases_with_debt_level():
    a0, r, mu_a, delta0, mu_delta = 150.0, 0.03, 0.08, 0.02, 0.01
    sigma_a, sigma_delta, rho, t = 0.25, 0.05, 0.3, 1.0

    pd_low = pd_extended_merton_model(
        a0, 80.0, r, mu_a, delta0, mu_delta, sigma_a, sigma_delta, rho, t
    )
    pd_high = pd_extended_merton_model(
        a0, 130.0, r, mu_a, delta0, mu_delta, sigma_a, sigma_delta, rho, t
    )
    assert pd_high > pd_low


def test_pd_black_cox_model_survival_and_pd_sum_to_one():
    result = pd_black_cox_model(a0=150.0, mu_a=0.08, sigma_a=0.25, b=80.0, tau=2.0)
    assert np.isclose(result.s_tau + result.pd_tau, 1.0)
    assert 0.0 <= result.pd_tau <= 1.0


def test_pd_black_cox_model_higher_barrier_gives_higher_default_probability():
    low_barrier = pd_black_cox_model(a0=150.0, mu_a=0.05, sigma_a=0.25, b=60.0, tau=2.0)
    high_barrier = pd_black_cox_model(a0=150.0, mu_a=0.05, sigma_a=0.25, b=100.0, tau=2.0)
    assert high_barrier.pd_tau > low_barrier.pd_tau


def test_merton_jump_model_reduces_to_black_scholes_when_lambda_is_zero():
    s0, k, sigma, t, b, r = 100.0, 95.0, 0.2, 1.0, 0.03, 0.03
    bs = black_scholes(s0, k, sigma, t, b, r)
    mj = merton_jump_model(s0, k, sigma, t, b, r, lambda_=0.0, mu_z=0.1, sigma_z=0.3)

    assert np.isclose(mj.call, bs.call)
    assert np.isclose(mj.put, bs.put)


def test_merton_jump_model_jump_intensity_changes_option_price():
    s0, k, sigma, t, b, r = 100.0, 95.0, 0.2, 1.0, 0.03, 0.03
    no_jump = merton_jump_model(s0, k, sigma, t, b, r, lambda_=0.0, mu_z=0.0, sigma_z=0.3)
    with_jump = merton_jump_model(s0, k, sigma, t, b, r, lambda_=1.0, mu_z=0.0, sigma_z=0.3)

    # adding jump risk (extra variance) should raise both option prices
    assert with_jump.call > no_jump.call
    assert with_jump.put > no_jump.put


def test_merton_jump_climate_model_reduces_to_plain_merton_when_lambda_is_zero():
    a0, d, sigma_a, t, r = 150.0, 100.0, 0.25, 1.0, 0.03
    mjc = merton_jump_climate_model(a0, d, sigma_a, t, r, lambda_=0.0, mu_z=0.0, sigma_z=0.3)

    d1 = (np.log(a0 / d) + (r + 0.5 * sigma_a**2) * t) / (sigma_a * np.sqrt(t))
    d2 = d1 - sigma_a * np.sqrt(t)
    e0_expected = a0 * norm.cdf(d1) - d * np.exp(-r * t) * norm.cdf(d2)
    b0_expected = a0 * norm.cdf(-d1) + d * np.exp(-r * t) * norm.cdf(d2)

    assert np.isclose(mjc.e0, e0_expected)
    assert np.isclose(mjc.b0, b0_expected)


def test_merton_jump_climate_model_balance_sheet_identity():
    a0, d, sigma_a, t, r = 150.0, 100.0, 0.25, 1.0, 0.03
    mjc = merton_jump_climate_model(a0, d, sigma_a, t, r, lambda_=0.5, mu_z=-0.1, sigma_z=0.2)
    assert np.isclose(mjc.e0 + mjc.b0, a0)


def test_reinders_credit_model_loss_matches_value_difference():
    xi, a0, d, r, sigma_a, t, omega_e, omega_d = 0.1, 150.0, 100.0, 0.03, 0.25, 1.0, 1.0, 1.0
    result = reinders_credit_model(xi, a0, d, r, sigma_a, t, omega_e, omega_d)

    expected_loss = omega_e * (result.mv_e_t0 - result.mv_e_t) + omega_d * (
        result.mv_d_t0 - result.mv_d_t
    )
    assert np.isclose(result.loss, expected_loss)


def test_reinders_credit_model_first_derivative_matches_finite_difference():
    a0, d, r, sigma_a, t, omega_e, omega_d = 150.0, 100.0, 0.03, 0.25, 1.0, 1.0, 0.5
    xi0 = 0.2
    h = 1e-6

    loss_plus = reinders_credit_model(xi0 + h, a0, d, r, sigma_a, t, omega_e, omega_d).loss
    loss_minus = reinders_credit_model(xi0 - h, a0, d, r, sigma_a, t, omega_e, omega_d).loss
    numeric_d_loss = (loss_plus - loss_minus) / (2 * h)

    result = reinders_credit_model(xi0, a0, d, r, sigma_a, t, omega_e, omega_d)
    assert np.isclose(result.d_loss, numeric_d_loss, rtol=1e-4)


def test_reinders_credit_model_second_derivative_matches_finite_difference():
    a0, d, r, sigma_a, t, omega_e, omega_d = 150.0, 100.0, 0.03, 0.25, 1.0, 1.0, 0.5
    xi0 = 0.2
    h = 1e-4

    d_loss_plus = reinders_credit_model(xi0 + h, a0, d, r, sigma_a, t, omega_e, omega_d).d_loss
    d_loss_minus = reinders_credit_model(xi0 - h, a0, d, r, sigma_a, t, omega_e, omega_d).d_loss
    numeric_d2_loss = (d_loss_plus - d_loss_minus) / (2 * h)

    result = reinders_credit_model(xi0, a0, d, r, sigma_a, t, omega_e, omega_d)
    assert np.isclose(result.d2_loss, numeric_d2_loss, rtol=1e-3)


def test_reinders_credit_model_zero_shock_gives_zero_loss():
    result = reinders_credit_model(0.0, 150.0, 100.0, 0.03, 0.25, 1.0, 1.0, 1.0)
    assert np.isclose(result.loss, 0.0, atol=1e-8)
