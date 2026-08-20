"""Tests for quanttoolbox.stats.moments."""

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from quanttoolbox.stats.moments import (
    active_share,
    active_share_upper_bound,
    asynchronous_cov,
    corr_to_cov,
    corrx,
    cov_to_corr,
    herfindahl_index,
    kurtosis,
    mean_absolute_difference,
    pearson_correlation,
    rolling_correlation,
    rolling_volatility,
    skewness,
    weekly_cov,
)


def test_skewness_matches_scipy(rng):
    x = rng.standard_normal((1000, 3))
    ours = skewness(x)
    ref = scipy_skew(x, axis=0, bias=True)
    assert np.allclose(ours, ref, atol=1e-8)


def test_kurtosis_matches_scipy_raw_convention(rng):
    x = rng.standard_normal((1000, 3))
    ours = kurtosis(x)
    ref = scipy_kurtosis(x, axis=0, bias=True, fisher=False)  # raw (Pearson) kurtosis
    assert np.allclose(ours, ref, atol=1e-8)


def test_herfindahl_index_equal_weights():
    x = np.full(10, 0.1)
    h, n, n_b = herfindahl_index(x)
    assert np.isclose(h, 0.1)
    assert np.isclose(n, 10.0)
    assert n_b is None


def test_herfindahl_index_concentrated():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    h, n, _ = herfindahl_index(x)
    assert np.isclose(h, 1.0)
    assert np.isclose(n, 1.0)


def test_mean_absolute_difference_uniform_pair():
    x = np.array([0.0, 10.0])
    m = mean_absolute_difference(x)
    # |0-0|+|0-10|+|10-0|+|10-10| = 20, /4 = 5
    assert np.isclose(m[0], 5.0)


def test_cov_corr_roundtrip(rng):
    a = rng.standard_normal((5, 5))
    cov = a @ a.T + 5 * np.eye(5)  # guaranteed PD
    sigma, rho = cov_to_corr(cov)
    cov_back = corr_to_cov(sigma, rho)
    assert np.allclose(cov_back, cov)
    assert np.allclose(np.diag(rho), 1.0)


def test_corrx_matches_numpy(rng):
    x = rng.standard_normal((500, 4))
    rho, sigma_matrix = corrx(x)
    assert np.allclose(rho, np.corrcoef(x, rowvar=False), atol=1e-8)
    assert np.allclose(sigma_matrix, np.cov(x, rowvar=False), atol=1e-8)


def test_pearson_correlation_matching_columns():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1000, 3))
    y = x * 2 + rng.standard_normal((1000, 3)) * 0.01  # nearly perfectly correlated
    rho = pearson_correlation(x, y)
    assert np.all(rho > 0.99)


def test_active_share_identical_portfolios():
    x = np.array([0.2, 0.3, 0.5])
    assert active_share(x, x) == 0.0


def test_active_share_fully_disjoint():
    x = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(active_share(x, b), 1.0)


def test_active_share_upper_bound_feasible():
    b = np.array([0.5, 0.3, 0.2])
    as_max, x_max, retcode = active_share_upper_bound(b, x_minus=0.0, x_plus=1.0)
    assert retcode == 1
    assert np.isclose(x_max.sum(), 1.0)
    assert as_max >= active_share(b, b)  # at least as much as doing nothing


def test_asynchronous_cov_diagonal_matches_plain_cov(rng):
    x = rng.standard_normal((500, 3))
    cov1, cov2 = asynchronous_cov(x)
    assert np.allclose(np.diag(cov1), np.diag(cov2), atol=1e-8)
    assert np.allclose(cov1, np.cov(x, rowvar=False), atol=1e-8)


def test_weekly_cov_without_daily_matches_plain_cov(rng):
    x = rng.standard_normal((200, 3))
    vcv, rho, sigma = weekly_cov(x)
    assert np.allclose(vcv, np.cov(x, rowvar=False), atol=1e-8)
    assert rho is None


def test_weekly_cov_with_daily_uses_daily_vol(rng):
    x_weekly = rng.standard_normal((100, 2))
    x_daily = rng.standard_normal((500, 2)) * 0.1  # much lower daily vol scale
    vcv, rho, sigma_d = weekly_cov(x_weekly, x_daily)
    assert np.allclose(np.sqrt(np.diag(vcv)), sigma_d, atol=1e-8)


def test_rolling_volatility_shape_and_nan_prefix(rng):
    x = rng.standard_normal((100, 2))
    sigma = rolling_volatility(x, n_lags=20)
    assert sigma.shape == (100, 2)
    assert np.all(np.isnan(sigma[:20]))
    assert not np.any(np.isnan(sigma[20:]))


def test_rolling_correlation_perfectly_correlated(rng):
    x = rng.standard_normal((100, 1))
    y = x * 3.0
    rho, sigma_x, sigma_y = rolling_correlation(x, y, n_lags=20)
    assert np.allclose(rho[20:], 1.0, atol=1e-6)
