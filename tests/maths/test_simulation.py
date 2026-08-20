"""Tests for quanttoolbox.maths.simulation."""

import numpy as np

from quanttoolbox.maths.simulation import (
    algebraic_riccati_equation,
    compute_ewma,
    lyapunov_equation,
    momentum_ewma,
    simulate_gbm,
    simulate_gbm2,
    simulate_multi_gbm,
    volatility_target,
)


def test_simulate_gbm_matches_analytical_lognormal_moments():
    rng = np.random.default_rng(0)
    x0, mu, sigma = 100.0, 0.05, 0.2
    t = np.array([1.0])
    n_paths = 100_000

    paths = simulate_gbm(x0, mu, sigma, t, n_paths, rng=rng)
    final = paths[-1, :]

    expected_mean = x0 * np.exp(mu * 1.0)
    expected_std = x0 * np.exp(mu * 1.0) * np.sqrt(np.exp(sigma**2 * 1.0) - 1)
    assert np.isclose(final.mean(), expected_mean, rtol=0.02)
    assert np.isclose(final.std(), expected_std, rtol=0.05)


def test_simulate_gbm_shape():
    rng = np.random.default_rng(0)
    t = np.array([0.25, 0.5, 0.75, 1.0])
    paths = simulate_gbm(100.0, 0.05, 0.2, t, n_paths=50, rng=rng)
    assert paths.shape == (4, 50)


def test_simulate_gbm2_reproduces_target_correlation():
    rng = np.random.default_rng(0)
    t = np.array([1.0])
    n_paths = 100_000
    rho = 0.7

    x1, x2 = simulate_gbm2(100, 100, 0.05, 0.05, 0.2, 0.2, rho, t, n_paths, rng=rng)
    r1, r2 = x1[-1] / 100 - 1, x2[-1] / 100 - 1
    assert np.isclose(np.corrcoef(r1, r2)[0, 1], rho, atol=0.03)


def test_simulate_multi_gbm_reproduces_correlation_matrix():
    rng = np.random.default_rng(0)
    n_assets = 3
    x0 = np.full(n_assets, 100.0)
    mu = np.full(n_assets, 0.05)
    sigma = np.full(n_assets, 0.2)
    rho = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    t = np.array([1.0])

    paths = simulate_multi_gbm(x0, mu, sigma, rho, t, n_paths=80_000, rng=rng)
    returns = paths[-1] / 100 - 1
    corr = np.corrcoef(returns)
    assert np.allclose(corr, rho, atol=0.05)


def test_simulate_multi_gbm_shape():
    rng = np.random.default_rng(0)
    x0 = np.array([100.0, 50.0])
    mu = np.array([0.05, 0.03])
    sigma = np.array([0.2, 0.15])
    rho = np.eye(2)
    t = np.array([0.5, 1.0])

    paths = simulate_multi_gbm(x0, mu, sigma, rho, t, n_paths=100, rng=rng)
    assert paths.shape == (2, 2, 100)


def test_compute_ewma_shapes():
    rng = np.random.default_rng(0)
    n = 200
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    mu_t, sigma_t = compute_ewma(prices[:, None], lambda_mu=5.0)
    assert mu_t.shape == (n, 1)
    assert sigma_t.shape == (n, 1)
    assert np.all(sigma_t[1:] >= 0)


def test_compute_ewma_annualized_vol_close_to_true(rng):
    n = 2000
    daily_vol = 0.01
    returns = rng.standard_normal(n) * daily_vol
    prices = 100 * np.cumprod(1 + returns)
    _, sigma_t = compute_ewma(prices[:, None], lambda_mu=5.0, lambda_sigma=5.0)

    expected_annual_vol = daily_vol * np.sqrt(260)
    assert np.isclose(np.nanmean(sigma_t[100:]), expected_annual_vol, rtol=0.15)


def test_volatility_target_leverage_within_bounds(rng):
    n = 500
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    result = volatility_target(
        prices[:, None], lambda_=5.0, vol_target=0.10, min_leverage=0.0, max_leverage=2.0
    )
    valid_leverage = result.leverage_t[~np.isnan(result.leverage_t)]
    assert np.all(valid_leverage >= 0.0)
    assert np.all(valid_leverage <= 2.0)


def test_volatility_target_output_shape(rng):
    n = 300
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    result = volatility_target(prices[:, None], lambda_=5.0, vol_target=0.10)
    assert result.y_t.shape == (n, 1)
    assert result.sigma_t.shape == (n, 1)
    assert result.leverage_t.shape == (n, 1)


def test_algebraic_riccati_equation_satisfies_defining_equation():
    a = np.array([[0.0, 1.0], [-2.0, -3.0]])
    b = np.eye(2)
    c = np.eye(2)

    x = algebraic_riccati_equation(a, b, c)
    residual = a.T @ x + x @ a - x @ b @ x + c
    assert np.allclose(residual, 0.0, atol=1e-8)


def test_lyapunov_equation_satisfies_defining_equation():
    a = np.array([[0.0, 1.0], [-2.0, -3.0]])
    c = np.eye(2)

    x = lyapunov_equation(a, c)
    residual = a @ x + x @ a.T - c
    assert np.allclose(residual, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# momentum_ewma
# ---------------------------------------------------------------------------


def test_momentum_ewma_shapes(rng):
    n = 300
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    result = momentum_ewma(prices[:, None], alpha=5.0, lambda_mu=5.0, lambda_sigma=5.0)

    assert result.v_t.shape == (n, 1)
    assert result.v_tilde_t.shape == (n, 1)
    assert result.g_t.shape == (n, 1)
    assert result.g_small_t.shape == (n, 1)
    assert result.e_t.shape == (n, 1)


def test_momentum_ewma_v_tilde_equals_g_times_g_small():
    # algebraic identity: v_tilde_t = exp(cumsum(Rtilde_G + Rtilde_g)) * 100
    #                                = g_t * g_small_t / 100
    # (since g_t and g_small_t are each already scaled by 100)
    rng = np.random.default_rng(1)
    n = 200
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    result = momentum_ewma(prices[:, None], alpha=3.0, lambda_mu=4.0, lambda_sigma=4.0)

    reconstructed = result.g_t * result.g_small_t / 100
    assert np.allclose(result.v_tilde_t, reconstructed, rtol=1e-8)


def test_momentum_ewma_exposure_matches_lagged_ewma_drift():
    rng = np.random.default_rng(2)
    n = 200
    alpha, lambda_mu = 5.0, 5.0
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    result = momentum_ewma(
        prices[:, None], alpha=alpha, lambda_mu=lambda_mu, lambda_sigma=lambda_mu
    )

    from quanttoolbox.maths.simulation import compute_ewma

    mu_t, _ = compute_ewma(prices[:, None], lambda_mu, lambda_mu)
    # e_t[k] = alpha * mu_t[k] * multiplier (multiplier=1 default), for k>=1
    assert np.allclose(result.e_t[1:], alpha * mu_t[1:], atol=1e-8)
    assert np.all(result.e_t[0] == 0)


def test_momentum_ewma_realized_wealth_starts_near_100(rng):
    n = 100
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    result = momentum_ewma(prices[:, None], alpha=5.0, lambda_mu=5.0, lambda_sigma=5.0)
    # first period has zero lagged exposure -> no return applied -> wealth stays at 100
    assert np.isclose(result.v_t[0, 0], 100.0)
    assert np.isclose(result.v_tilde_t[0, 0], 100.0)


def test_momentum_ewma_multiplier_scales_exposure(rng):
    n = 200
    prices = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    r1 = momentum_ewma(prices[:, None], alpha=5.0, lambda_mu=5.0, lambda_sigma=5.0, multiplier=1.0)
    r2 = momentum_ewma(prices[:, None], alpha=5.0, lambda_mu=5.0, lambda_sigma=5.0, multiplier=2.0)
    assert np.allclose(r2.e_t, 2.0 * r1.e_t, atol=1e-8)
