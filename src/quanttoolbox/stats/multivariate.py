"""Bivariate normal and Student-t probabilities with element-wise broadcasting.

Ported from HSF toolbox ``stats/{cdfbvn,pdfbvn,cdfbvt}.m``. The CDF functions
accept explicit ``x``, ``y``, ``rho``, and (for Student-t) ``nu`` arguments.
CDF evaluation delegates to SciPy rather than reproducing the original
Genz integration routines.

The CDF wrappers clip ``rho`` just inside (-1, 1) to avoid singular
correlation matrices, including values rounded to +/-1 during intermediate
calculations. This approximates the degenerate boundary distribution.
For general multivariate normal probabilities, see
``stats.distributions.mvn_cdf`` and ``mvn_pdf``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, multivariate_t


def bvn_cdf(x: np.ndarray | float, y: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray:
    """Bivariate standard normal CDF P(X <= x, Y <= y), for (X, Y) with unit
    variances and correlation `rho`. Broadcasts element-wise over
    `x`/`y`/`rho`.

    Original: stats/cdfbvn.m (via genz/bvnu.m)
    """
    x_arr, y_arr, rho_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(rho, dtype=float)
    )
    rho_arr = np.clip(rho_arr, -1.0 + 1e-8, 1.0 - 1e-8)
    out = np.empty(x_arr.shape, dtype=float)
    for idx in np.ndindex(x_arr.shape):
        cov = np.array([[1.0, rho_arr[idx]], [rho_arr[idx], 1.0]])
        out[idx] = multivariate_normal.cdf([x_arr[idx], y_arr[idx]], mean=[0.0, 0.0], cov=cov)
    return out[()] if out.shape == () else out


def bvn_pdf(
    x1: np.ndarray | float,
    x2: np.ndarray | float,
    mu1: np.ndarray | float,
    mu2: np.ndarray | float,
    sigma1: np.ndarray | float,
    sigma2: np.ndarray | float,
    rho: np.ndarray | float,
) -> np.ndarray:
    """Bivariate normal PDF at (x1, x2), for (X1, X2) ~ N((mu1, mu2), Sigma)
    with std devs (sigma1, sigma2) and correlation rho. Closed-form, so
    (unlike `bvn_cdf`) evaluated directly rather than via `scipy`.

    Original: stats/pdfbvn.m
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    rho = np.asarray(rho, dtype=float)

    w = 1.0 - rho**2
    z1 = (x1 - mu1) / sigma1
    z2 = (x2 - mu2) / sigma2
    q = z1**2 - 2.0 * rho * z1 * z2 + z2**2
    return np.exp(-0.5 * q / w) / (2.0 * np.pi * sigma1 * sigma2 * np.sqrt(w))


def bvt_cdf(
    x: np.ndarray | float, y: np.ndarray | float, rho: np.ndarray | float, nu: np.ndarray | float
) -> np.ndarray:
    """Bivariate Student-t CDF P(X <= x, Y <= y), for (X, Y) with unit scale,
    correlation `rho`, and `nu` degrees of freedom. Broadcasts element-wise
    over `x`/`y`/`rho`/`nu`.

    Original: stats/cdfbvt.m (via MATLAB's mvtcdf)
    """
    x_arr, y_arr, rho_arr, nu_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(rho, dtype=float),
        np.asarray(nu, dtype=float),
    )
    rho_arr = np.clip(rho_arr, -1.0 + 1e-8, 1.0 - 1e-8)
    out = np.empty(x_arr.shape, dtype=float)
    for idx in np.ndindex(x_arr.shape):
        shape = np.array([[1.0, rho_arr[idx]], [rho_arr[idx], 1.0]])
        out[idx] = multivariate_t.cdf(
            [x_arr[idx], y_arr[idx]], loc=[0.0, 0.0], shape=shape, df=float(nu_arr[idx])
        )
    return out[()] if out.shape == () else out
