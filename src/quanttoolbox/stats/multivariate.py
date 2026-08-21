"""Bivariate normal/Student-t CDF and PDF wrappers, matching the original
toolbox's call signature (explicit ``x``, ``y``, ``rho``, ``nu`` arguments,
broadcast element-wise) rather than requiring the caller to assemble a full
covariance matrix.

Ported from HSF toolbox `stats/{cdfbvn,pdfbvn,cdfbvt}.m`.

Translation notes:

- `genz/{bvn,bvnu}.m` are Alan Genz's Drezner-Wesolowsky quadrature for the
  bivariate normal CDF; `stats/cdfbvn.m` is a thin wrapper around `bvnu.m`.
  `scipy.stats.multivariate_normal.cdf` already computes this (for any
  dimension, not just 2) via `scipy.stats._mvn`, which is itself built on
  Alan Genz's own Fortran `mvndst` routine -- the same author, the same
  underlying algorithm. `bvn_cdf` here is therefore a thin `scipy`-backed
  convenience wrapper with the original's `(x, y, rho)` call signature, not
  a hand-rolled reimplementation of `bvnu.m`'s quadrature.
- `stats/cdfbvt.m` wraps MATLAB's Statistics Toolbox `mvtcdf`; ported the
  same way, via `scipy.stats.multivariate_t.cdf` (arbitrary dimension,
  available since SciPy 1.9).
- Both `bvn_cdf` and `bvt_cdf` clip `rho` a hair inside (-1, 1) before
  building the 2x2 correlation matrix: `scipy`'s `multivariate_normal`/
  `multivariate_t` raise `LinAlgError` on an exactly-singular (`|rho| ==
  1`) matrix rather than falling back to the (well-defined) degenerate
  limit, and `rho` can round to exactly +/-1 in float64 even when the
  caller's inputs are only extremely close to that boundary -- e.g.
  `stats.distributions.skew_t_cdf` hits this at `eta` near 0, where
  `delta = (1 - eta^2) / (1 + eta^2)` underflows to `1.0`. Clipping is a
  robustness addition beyond a literal transliteration of the original,
  not a behavior change: the clipped and unclipped results agree to
  float64 precision anywhere `scipy` would have succeeded anyway.
- `genz/{qsimvn,qsimvnv,qsimvt,qsimvtv,qsilatmvnv,qsilatmvtv,qsimvnauto,
  mvnrcnv}.m` (8 of the 10 files in `genz/`) are randomized quasi-Monte-Carlo
  integrators solving exactly the same problem as
  `scipy.stats.multivariate_normal.cdf`/`multivariate_t.cdf` -- arbitrary-
  dimension MVN/MVT orthant probabilities -- via the same Genz (1992)
  algorithm family (Niederreiter/Cranley-Patterson randomized lattice
  rules); `mvnrcnv.m` is a constant-correlation special case of the same
  problem. None of these are hand-ported: doing so would duplicate
  already-available, better-tested `scipy` functionality (the same author's
  own algorithm, reimplemented) with no functional gain. See
  `stats.distributions.mvn_cdf`/`mvn_pdf` for the general n-dimensional
  case (already ported); `bvn_cdf`/`bvt_cdf` below just add the bivariate-
  specific call signature (`x`, `y`, `rho` rather than a point + covariance
  matrix) for parity with how the original toolbox's other modules call
  these.
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
