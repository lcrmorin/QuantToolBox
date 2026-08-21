"""Sample moments, dispersion, covariance/correlation, and portfolio-overlap measures.

Ported from QuantToolBox/stats/{skewness_coefficient,kurtosis_coefficient,
herfindahl_index,mean_absolute_difference,cov2cor,cor2cov,corrx,
pearson_correlation,active_share,active_share_upper_bound,asynchronous_cov,
weekly_cov,rolling_correlation,rolling_volatility}.m

Translation notes:

- Every "column vector output for each column of x" pattern (MATLAB loops
  over ``cols(x)``) is replaced by native NumPy/pandas vectorized
  operations (``axis=0`` reductions) instead of an explicit Python loop.
- ``packr`` (drop rows containing NaN before computing a statistic) is
  replaced by pandas' NaN-aware reductions (``.mean()``, ``.std()``, ...
  skip NaN by default) or explicit ``~np.isnan(...)`` masks where a
  NumPy-only implementation is used.
- Excess-kurtosis convention: like the original, ``kurtosis`` here returns
  the *raw* fourth standardized moment (3.0 for a normal distribution),
  not scipy's default excess-kurtosis convention (which subtracts 3).
"""

from __future__ import annotations

import numpy as np


def skewness(x: np.ndarray) -> np.ndarray:
    """Sample skewness of each column of x. Original: stats/skewness_coefficient.m"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    out = np.full(x.shape[1], np.nan)
    for i in range(x.shape[1]):
        y = x[:, i]
        y = y[~np.isnan(y)]
        yc = y - y.mean()
        m2 = np.mean(yc**2)
        m3 = np.mean(yc**3)
        out[i] = m3 / m2**1.5
    return out


def kurtosis(x: np.ndarray) -> np.ndarray:
    """Sample kurtosis (raw, not excess) of each column of x.
    Original: stats/kurtosis_coefficient.m
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    out = np.full(x.shape[1], np.nan)
    for i in range(x.shape[1]):
        y = x[:, i]
        y = y[~np.isnan(y)]
        yc = y - y.mean()
        m2 = np.mean(yc**2)
        m4 = np.mean(yc**4)
        out[i] = m4 / m2**2
    return out


def herfindahl_index(
    x: np.ndarray, b: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Herfindahl index H = sum(x^2), its inverse N = 1/H (effective number of
    positions), and optionally the same for a benchmark b.

    Original: stats/herfindahl_index.m
    """
    x = np.asarray(x, dtype=float)
    h = np.sum(x**2, axis=0)
    n = 1.0 / h
    if b is None:
        return h, n, None
    b = np.asarray(b, dtype=float)
    h_b = np.sum(b**2, axis=0)
    n_b = h_b / h
    return h, n, n_b


def mean_absolute_difference(x: np.ndarray) -> np.ndarray:
    """Gini-style mean absolute pairwise difference of each column of x.

    Original: stats/mean_absolute_difference.m
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    out = np.full(x.shape[1], np.nan)
    for i in range(x.shape[1]):
        y = x[:, i]
        y = y[~np.isnan(y)]
        n = y.shape[0]
        dx = y[:, None] - y[None, :]
        out[i] = np.sum(np.abs(dx)) / n**2
    return out


def cov_to_corr(cov_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a covariance matrix into (std devs, correlation matrix).

    Original: stats/cov2cor.m
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    sigma = np.sqrt(np.diag(cov_matrix))
    rho = cov_matrix / sigma[:, None] / sigma[None, :]
    return sigma, rho


def corr_to_cov(sigma: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Recombine standard deviations and a correlation matrix into a covariance matrix.

    Original: stats/cor2cov.m
    """
    sigma = np.asarray(sigma, dtype=float).flatten()
    rho = np.asarray(rho, dtype=float)
    n = sigma.shape[0]
    if rho.shape != (n, n):
        raise ValueError("corr_to_cov: dimensions do not match")
    return rho * sigma[:, None] * sigma[None, :]


def corrx(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample correlation matrix and covariance matrix of x (columns = variables,
    rows dropped if they contain NaN).

    Original: stats/corrx.m
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    n = x.shape[0]
    xc = x - x.mean(axis=0)
    sigma_matrix = (xc.T @ xc) / (n - 1)
    sigma = np.sqrt(np.diag(sigma_matrix))
    rho = sigma_matrix / sigma[:, None] / sigma[None, :]
    return rho, sigma_matrix


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise (or cross) Pearson correlation between x and y.

    If x and y have the same number of columns, returns the correlation of
    each matching pair of columns. Otherwise returns the full nX x nY
    cross-correlation matrix.

    Original: stats/pearson_correlation.m
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
    if y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T

    n_x, n_y = x.shape[1], y.shape[1]
    d_x = x - x.mean(axis=0)
    sigma_x = np.sqrt(np.mean(d_x**2, axis=0))

    if n_x == n_y:
        d_y = y - y.mean(axis=0)
        sigma_y = np.sqrt(np.mean(d_y**2, axis=0))
        return np.mean(d_x * d_y, axis=0) / (sigma_x * sigma_y)

    rho = np.zeros((n_x, n_y))
    for i in range(n_y):
        yi = y[:, i]
        d_y = yi - yi.mean()
        sigma_y = np.sqrt(np.mean(d_y**2))
        rho[:, i] = np.mean(d_x * d_y[:, None], axis=0) / (sigma_x * sigma_y)
    return rho


def active_share(x: np.ndarray, b: np.ndarray) -> float:
    """Portfolio active share: 0.5 * sum(|weight - benchmark weight|).

    Original: stats/active_share.m
    """
    return 0.5 * float(np.sum(np.abs(np.asarray(x) - np.asarray(b))))


def active_share_upper_bound(
    b: np.ndarray, x_minus: float | np.ndarray = 0.0, x_plus: float | np.ndarray = 1.0
) -> tuple[float, np.ndarray | None, int]:
    """Compute the maximum achievable active share given per-position weight
    bounds [x_minus, x_plus] that must still sum to 1.

    Greedy algorithm: iteratively push each position to whichever bound
    (x_minus or x_plus) is farthest from the benchmark weight, subject to
    the running total staying feasible.

    Original: stats/active_share_upper_bound.m

    Returns
    -------
    as_max : the maximum active share achieved.
    x_max : the weight vector achieving it (None if infeasible).
    retcode : 1 if converged exactly to a total weight of 1, else 0/-1.
    """
    b = np.asarray(b, dtype=float).flatten()
    n = b.shape[0]
    x_minus = (
        np.full(n, x_minus, dtype=float)
        if np.isscalar(x_minus)
        else np.asarray(x_minus, dtype=float)
    )
    x_plus = (
        np.full(n, x_plus, dtype=float) if np.isscalar(x_plus) else np.asarray(x_plus, dtype=float)
    )

    x_max = x_minus.copy()
    s = x_max.sum()
    if s > 1.0:
        return np.nan, None, -1

    y_minus = np.abs(b - x_minus)
    y_plus = np.abs(x_plus - b)
    y = np.concatenate([y_minus, y_plus])
    order = np.zeros(n, dtype=int)

    for it in range(n):
        if s == 1:
            break
        idx = int(np.nanargmax(y))
        if idx < n:  # "minus" candidate
            y[idx] = np.nan
            y[idx + n] = np.nan
            order[it] = idx
        else:  # "plus" candidate
            idx -= n
            ds = 1 - s + x_minus[idx]
            dx = min(ds, x_plus[idx])
            x_max[idx] = dx
            y[idx] = np.nan
            y[idx + n] = np.nan
            order[it] = idx
        s = x_max.sum()

    retcode = 1
    if not np.isclose(s, 1.0):
        retcode = 0
        for it in range(n - 1, -1, -1):
            idx = order[it]
            ds = 1 - s + x_max[idx]
            dx = min(ds, x_plus[idx])
            x_max[idx] = dx
            s = x_max.sum()
            if np.isclose(s, 1.0):
                break

    as_max = active_share(x_max, b)
    return as_max, x_max, retcode


def asynchronous_cov(x: np.ndarray, method: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Newey-West-style covariance correction for asynchronously-timed data
    (e.g. assets trading in different time zones), using a lag-1
    autocovariance adjustment.

    method=0: half-window correction (default). method=1: full-window.

    Original: stats/asynchronous_cov.m
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    r, c = x.shape

    xc = x - x.mean(axis=0)
    cov1 = (xc.T @ xc) / (r - 1)

    xc_lag = xc[:-1]
    xc_cur = xc[1:]
    cross = (xc_cur.T @ xc_lag) / (r - 1) + (xc_lag.T @ xc_cur) / (r - 1)

    weight = 1.0 if method == 1 else 0.5
    cov2 = cov1 + weight * cross

    sigma2 = np.sqrt(np.diag(cov2))
    cor2 = cov2 / sigma2[:, None] / sigma2[None, :]
    np.fill_diagonal(cor2, 1.0)

    sigma1 = np.sqrt(np.diag(cov1))
    cov2 = cor2 * sigma1[:, None] * sigma1[None, :]
    return cov1, cov2


def weekly_cov(
    x_weekly: np.ndarray, x_daily: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Covariance matrix using weekly correlations but daily-data volatilities
    (useful when daily vol estimates are more reliable than weekly ones but
    weekly correlations are less affected by asynchronous trading).

    Original: stats/weekly_cov.m
    """
    x = np.asarray(x_weekly, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    r, c = x.shape
    xc = x - x.mean(axis=0)
    vcv = (xc.T @ xc) / (r - 1)

    if x_daily is None:
        return vcv, None, None

    sigma_w = np.sqrt(np.diag(vcv))
    rho = vcv / sigma_w[:, None] / sigma_w[None, :]

    xd = np.asarray(x_daily, dtype=float)
    xd = xd[~np.isnan(xd).any(axis=1)]
    sigma_d = xd.std(axis=0, ddof=1)

    vcv_out = rho * sigma_d[:, None] * sigma_d[None, :]
    return vcv_out, rho, sigma_d


def rolling_volatility(x: np.ndarray, n_lags: int, method: int = 0) -> np.ndarray:
    """Rolling (trailing n_lags-window) volatility of each column of x.

    method=1 treats x as prices and converts to simple returns first.

    Original: stats/rolling_volatility.m
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    if method == 1:
        x = x[1:] / x[:-1] - 1.0

    r, c = x.shape
    sigma = np.full((r, c), np.nan)
    for i in range(n_lags, r):
        for j in range(c):
            window = x[i - n_lags : i + 1, j]
            window = window[~np.isnan(window)]
            if window.shape[0] >= 0.5 * n_lags:
                sigma[i, j] = window.std(ddof=1)
    return sigma


def rolling_correlation(
    x: np.ndarray, y: np.ndarray, n_lags: int, method: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling (trailing n_lags-window) correlation, and each series' rolling
    volatility, column-by-column.

    method=1 treats both x and y as prices and converts to returns up front.
    method=2 converts to returns within each window (useful for illiquid
    series where a fixed global return series would drop too many rows).

    Original: stats/rolling_correlation.m
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape != y.shape:
        raise ValueError("rolling_correlation: x and y do not match")

    r, c = x.shape
    rho = np.full((r, c), np.nan)
    sigma_x = np.full((r, c), np.nan)
    sigma_y = np.full((r, c), np.nan)

    if method == 1:
        x = x[1:] / x[:-1] - 1.0
        y = y[1:] / y[:-1] - 1.0
        r -= 1

    for i in range(n_lags, r):
        for j in range(c):
            wx = x[i - n_lags : i + 1, j]
            wy = y[i - n_lags : i + 1, j]
            valid = ~np.isnan(wx) & ~np.isnan(wy)
            wx, wy = wx[valid], wy[valid]

            if method == 2:
                wx = wx[1:] / wx[:-1] - 1.0
                wy = wy[1:] / wy[:-1] - 1.0

            min_obs = 2 if method == 2 else 0.5 * n_lags
            if wx.shape[0] >= min_obs:
                sigma_x[i, j] = wx.std(ddof=1)
                sigma_y[i, j] = wy.std(ddof=1)
                rho[i, j] = np.corrcoef(wx, wy)[0, 1]

    return rho, sigma_x, sigma_y
