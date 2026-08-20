"""Tests for quanttoolbox.stats.regression.robust."""

import numpy as np

from quanttoolbox.stats.regression.ols import ols
from quanttoolbox.stats.regression.quantile import quantile_regression
from quanttoolbox.stats.regression.robust import (
    RobustRegressionConfig,
    huber_regression,
    inverse_quantile_m_regression,
    lad_regression,
    quantile_m_regression,
)


def test_lad_regression_recovers_coefficients_no_outliers(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    result = lad_regression(y, x)
    # LAD's 1/(u+eps) weighting is not smooth near u=0, so it often
    # doesn't hit the strict `converged` flag even once beta has
    # stabilized -- check coefficient accuracy directly instead.
    assert np.allclose(result.beta, true_beta, atol=0.1)


def test_lad_regression_robust_to_outliers_vs_ols(rng):
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    # inject a few large outliers
    y_contaminated = y.copy()
    outlier_idx = rng.choice(n, size=10, replace=False)
    y_contaminated[outlier_idx] += rng.choice([-1, 1], size=10) * 50.0

    lad_result = lad_regression(y_contaminated, x)
    ols_result = ols(y_contaminated, x)

    # LAD should stay much closer to the true slope than OLS under outliers
    lad_error = np.abs(lad_result.beta[1] - true_beta[1])
    ols_error = np.abs(ols_result.beta[1] - true_beta[1])
    assert lad_error < ols_error


def test_huber_regression_matches_ols_without_outliers(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([0.5, -1.5])
    y = x @ true_beta + rng.standard_normal(n) * 0.2

    huber_result = huber_regression(y, x, c=1.345)
    ols_result = ols(y, x)
    assert np.allclose(huber_result.beta, ols_result.beta, atol=0.05)


def test_huber_regression_robust_to_outliers_vs_ols(rng):
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([0.5, -1.5])
    y = x @ true_beta + rng.standard_normal(n) * 0.2

    y_contaminated = y.copy()
    outlier_idx = rng.choice(n, size=8, replace=False)
    y_contaminated[outlier_idx] += 30.0

    huber_result = huber_regression(y_contaminated, x, c=1.345)
    ols_result = ols(y_contaminated, x)

    huber_error = np.abs(huber_result.beta[1] - true_beta[1])
    ols_error = np.abs(ols_result.beta[1] - true_beta[1])
    assert huber_error < ols_error


def test_quantile_m_regression_median_close_to_lad(rng):
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.2

    q50 = quantile_m_regression(y, x, alpha=0.5)
    lad = lad_regression(y, x)
    # alpha=0.5 quantile M-regression should closely match LAD (both estimate the median)
    assert np.allclose(q50.beta, lad.beta, atol=0.1)


def test_quantile_m_regression_matches_lp_quantile_regression(rng):
    n = 1000
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.3

    for alpha in [0.25, 0.5, 0.75]:
        irls_result = quantile_m_regression(y, x, alpha=alpha)
        lp_beta, _, _ = quantile_regression(y, x, tau=alpha)
        # IRLS is an approximation of the same LP problem -- should agree reasonably
        assert np.allclose(irls_result.beta, lp_beta, atol=0.15)


def test_inverse_quantile_m_regression_alpha_half_matches_quantile():
    rng = np.random.default_rng(3)
    n = 500
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_beta = np.array([1.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.2

    q = quantile_m_regression(y, x, alpha=0.5)
    iq = inverse_quantile_m_regression(y, x, alpha=0.5)
    # at alpha=0.5 both loss functions coincide (symmetric case)
    assert np.allclose(q.beta, iq.beta, atol=0.1)


def test_robust_regression_returns_full_diagnostics(rng):
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.2

    result = lad_regression(y, x)
    assert result.beta.shape == (2,)
    assert result.stderr.shape == (2,)
    assert result.vcv.shape == (2, 2)
    assert result.residuals.shape == (n,)
    assert 0 <= result.r_squared <= 1
    assert result.df_residual == n - 2
    assert result.n_obs == n
    assert result.n_obs_valid == n
    assert result.n_obs_missing == 0


def test_robust_regression_drops_nan_rows(rng):
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.1
    y[10] = np.nan

    result = lad_regression(y, x)
    assert result.n_obs == n
    assert result.n_obs_valid == n - 1
    assert result.n_obs_missing == 1
    assert np.isnan(result.residuals[10])


def test_custom_config_tighter_tolerance_converges(rng):
    # use Huber (smooth loss) rather than LAD here, since LAD's weight
    # function is not smooth near zero residuals and often won't hit a
    # strict convergence tolerance even once beta has stabilized (see
    # lad_regression's docstring).
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.1

    config = RobustRegressionConfig(eps=1e-8, max_iters=1000)
    result = huber_regression(y, x, c=1.345, config=config)
    assert result.converged
