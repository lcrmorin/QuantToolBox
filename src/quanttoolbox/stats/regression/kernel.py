"""Nonparametric kernel density estimation and local-polynomial kernel regression.

Ported from QuantToolbox/stats/{regKernelDensity,regKernelMean,
regKernelQuantile,regKernelPayoff}.m

Translation notes:

- ``regKernelDensity`` is a manual Gaussian-kernel density/CDF estimator
  using ``scipy.stats.norm`` as the kernel -- this is equivalent to
  ``scipy.stats.gaussian_kde`` but the original's Silverman-type bandwidth
  rule (``1.364 * std * n^-0.2``) differs slightly from scipy's default
  bandwidth factor, so it's reproduced explicitly here rather than
  delegating to ``gaussian_kde`` (which would silently change bandwidths).
- ``regKernelMean``/``regKernelPayoff`` are local polynomial regression
  (Nadaraya-Watson generalized to include local polynomial terms), solved
  via a small per-evaluation-point weighted least squares -- there's no
  single scipy/sklearn function that matches this directly, so it's
  ported as a direct weighted-least-squares loop.
- ``regKernelQuantile`` depends on ``quantile_regression``
  (``scipy.optimize.linprog``-based); see
  ``quanttoolbox.stats.regression.quantile``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from quanttoolbox.stats.regression.quantile import quantile_regression


def _silverman_bandwidth(y: np.ndarray) -> float:
    n = y.shape[0]
    return 1.364 * y.std(ddof=1) * n ** (-0.2)


def kernel_density(
    data: np.ndarray, x: np.ndarray, h: float | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-kernel density (and CDF) estimate of each column of `data`,
    evaluated at points `x`.

    Original: stats/regKernelDensity.m

    Returns
    -------
    density : (n_x, n_cols) estimated density values.
    cdf : (n_x, n_cols) estimated CDF values.
    bandwidths : (n_cols,) bandwidth used per column (auto-computed if h is None).
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[0] == 1:
        data = data.T
    n_cols = data.shape[1]

    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1 and x.shape[1] != n_cols:
        x = x.T
    if x.shape[1] == 1 and n_cols > 1:
        x = np.tile(x, (1, n_cols))

    bandwidths = np.zeros(n_cols) if h is None else np.broadcast_to(h, (n_cols,)).astype(float)

    n_x = x.shape[0]
    density = np.zeros((n_x, n_cols))
    cdf = np.zeros((n_x, n_cols))

    for col in range(n_cols):
        y = data[:, col]
        y = y[~np.isnan(y)]
        bw = bandwidths[col]
        if bw == 0:
            bw = _silverman_bandwidth(y)
            bandwidths[col] = bw

        for i in range(n_x):
            u = (x[i, col] - y) / bw
            density[i, col] = np.mean(norm.pdf(u)) / bw
            cdf[i, col] = np.mean(norm.cdf(u))

    return density, cdf, bandwidths


def kernel_mean_regression(
    y: np.ndarray, x: np.ndarray, z: np.ndarray, order: int = 1, h: float | None = None
) -> np.ndarray:
    """Local polynomial kernel regression of y on x, evaluated at points z.

    order=1 is standard local-linear (Nadaraya-Watson-generalized)
    regression; higher orders fit a local polynomial of that degree.

    Original: stats/regKernelMean.m
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    n_x = x.shape[0]

    if h is None:
        h = 1.364 * x.std(ddof=1) * n_x ** (-0.20)

    z = np.atleast_1d(np.asarray(z, dtype=float))
    n_z = z.shape[0]
    m = np.zeros(n_z)

    design = np.ones((n_x, order + 1))
    for i in range(n_z):
        dz = x - z[i]
        w = norm.pdf(dz / h)
        for k in range(1, order + 1):
            design[:, k] = dz**k
        weighted_design = design * w[:, None]
        beta = np.linalg.solve(weighted_design.T @ design, weighted_design.T @ y)
        m[i] = beta[0]

    return m


def kernel_quantile_regression(
    y: np.ndarray,
    x: np.ndarray,
    tau: float,
    z: np.ndarray,
    order: int = 1,
    h: float = 1.0,
) -> np.ndarray:
    """Local polynomial kernel quantile regression of y on x at quantile tau,
    evaluated at points z.

    Original: stats/regKernelQuantile.m
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    n_x = x.shape[0]

    h1 = 1.364 * x.std(ddof=1) * n_x ** (-0.20)
    h2 = (tau * (1 - tau) * norm.pdf(norm.ppf(tau)) ** (-2)) ** 0.20
    bandwidth = h * h1 * h2

    z = np.atleast_1d(np.asarray(z, dtype=float))
    n_z = z.shape[0]
    q = np.zeros(n_z)

    const = np.ones(n_x)
    for i in range(n_z):
        dz = z[i] - x
        w = norm.pdf(dz / bandwidth)
        design = np.column_stack([const] + [dz**k for k in range(1, order + 1)])
        beta, _, _ = quantile_regression(y, design, tau, weights=w)
        q[i] = beta[0]

    return q


def kernel_payoff_regression(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | tuple[float, float] | None = None,
    n_points: int = 100,
    order: int = 1,
    h: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around kernel_mean_regression: builds an
    evaluation grid `z` automatically (from data range, or from a
    (min, max) pair) if one isn't supplied directly.

    Original: stats/regKernelPayoff.m
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]

    if z is None:
        rg_min, rg_max = x.min(), x.max()
        z = np.linspace(rg_min, rg_max, n_points)
    elif np.asarray(z).shape[0] == 2:
        rg_min, rg_max = z
        z = np.linspace(rg_min, rg_max, n_points)
    else:
        z = np.asarray(z, dtype=float)

    q = kernel_mean_regression(y, x, z, order=order, h=h)
    return z, q
