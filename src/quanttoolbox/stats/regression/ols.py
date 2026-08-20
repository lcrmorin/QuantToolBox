"""Ordinary least squares, centering/standardization, conditional-normal
regression, and principal component analysis.

Ported from QuantToolbox/stats/{regOLS,regCenter,regStandardize,regCND,
regPCA}.m

Translation notes:

- ``regOLS`` uses ``numpy.linalg.lstsq`` (QR-based) internally for
  numerical stability, rather than MATLAB's ``inv(x'*x) * x'*y`` normal
  equations, but still returns the explicit ``inv(x'x)`` covariance
  factor since downstream code needs it for standard errors.
- ``regPCA`` decomposes a *correlation* matrix (following the original),
  using ``numpy.linalg.eigh`` (the matrix is symmetric by construction)
  instead of MATLAB's ``eig``; eigenvalues/vectors are then sorted
  descending to match the original's convention.
- The MATLAB ``global Print_Results`` diagnostic-printing branch in
  ``regPCA`` is dropped -- callers should print/inspect the returned
  ``PCAResult`` fields themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OLSResult:
    beta: np.ndarray
    stderr: np.ndarray
    vcv: np.ndarray
    residuals: np.ndarray
    nobs: int
    nvar: int
    sigma: float


def ols(y: np.ndarray, x: np.ndarray) -> OLSResult:
    """Ordinary least squares regression, dropping rows with missing y or x.

    Original: stats/regOLS.m
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    nobs = y.shape[0]
    nvar = x.shape[1]

    valid = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
    y_valid, x_valid = y[valid], x[valid]

    xx = x_valid.T @ x_valid
    inv_xx = np.linalg.inv(xx)
    beta = inv_xx @ (x_valid.T @ y_valid)

    u = y_valid - x_valid @ beta
    sigma = u.std(ddof=1)
    vcv = sigma**2 * inv_xx
    stderr = np.sqrt(np.diag(vcv))

    residuals = np.full(nobs, np.nan)
    residuals[valid] = u

    return OLSResult(
        beta=beta, stderr=stderr, vcv=vcv, residuals=residuals, nobs=nobs, nvar=nvar, sigma=sigma
    )


def center(x: np.ndarray) -> np.ndarray:
    """Center each column of x by its (NaN-dropped) mean.

    Original: stats/regCenter.m
    """
    x = np.asarray(x, dtype=float)
    valid = x[~np.isnan(x).any(axis=1)] if x.ndim > 1 else x[~np.isnan(x)]
    return x - valid.mean(axis=0)


def standardize(x: np.ndarray) -> np.ndarray:
    """Center and scale each column of x to unit variance (using the
    NaN-dropped mean/std).

    Original: stats/regStandardize.m
    """
    x = np.asarray(x, dtype=float)
    valid = x[~np.isnan(x).any(axis=1)] if x.ndim > 1 else x[~np.isnan(x)]
    mean = valid.mean(axis=0)
    std = valid.std(axis=0, ddof=1)
    return (x - mean) / std


@dataclass
class ConditionalNormalResult:
    beta0: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray
    r_squared: np.ndarray


def conditional_normal_regression(
    mu_y: np.ndarray | None = None,
    mu_x: np.ndarray | None = None,
    sigma_yy: float | None = None,
    sigma_yx: np.ndarray | None = None,
    sigma_xx: np.ndarray | None = None,
    *,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> ConditionalNormalResult:
    """Linear regression coefficients implied by a joint normal distribution's
    moments (rather than estimated from a data sample).

    Two calling conventions, matching the two MATLAB entry points:

    - ``conditional_normal_regression(mu_y, mu_x, sigma_yy, sigma_yx, sigma_xx)``:
      regress a single y on x given the joint (y, x) moments.
    - ``conditional_normal_regression(mu=mu, sigma=sigma)``: leave-one-out
      regression of each variable in ``mu``/``sigma`` on all the others.

    Original: stats/regCND.m
    """
    if mu is not None and sigma is not None:
        return _conditional_normal_leave_one_out(mu, sigma)

    if mu_y is None or mu_x is None or sigma_yy is None or sigma_yx is None or sigma_xx is None:
        raise ValueError(
            "conditional_normal_regression: provide either (mu, sigma) or "
            "all of (mu_y, mu_x, sigma_yy, sigma_yx, sigma_xx)"
        )

    sigma_xy = sigma_yx.T
    sigma_xx_inv = np.linalg.inv(sigma_xx)
    beta0 = mu_y - sigma_yx @ sigma_xx_inv @ mu_x
    beta = sigma_xx_inv @ sigma_xy
    sigma2 = sigma_yy - sigma_yx @ sigma_xx_inv @ sigma_xy
    r_squared = 1 - sigma2 / sigma_yy
    sigma_out = np.sqrt(sigma2)
    return ConditionalNormalResult(beta0=beta0, beta=beta, sigma=sigma_out, r_squared=r_squared)


def _conditional_normal_leave_one_out(mu: np.ndarray, sigma: np.ndarray) -> ConditionalNormalResult:
    mu = np.asarray(mu, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.shape[0]

    beta0 = np.zeros(n)
    beta = np.full((n, n), np.nan)
    sigma2 = np.zeros(n)
    r_squared = np.zeros(n)

    for i in range(n):
        idx_x = [j for j in range(n) if j != i]
        mu_y, mu_x = mu[i], mu[idx_x]
        sigma_yy = sigma[i, i]
        sigma_yx = sigma[i, idx_x]
        sigma_xy = sigma[idx_x, i]
        sigma_xx = sigma[np.ix_(idx_x, idx_x)]

        sigma_xx_inv = np.linalg.inv(sigma_xx)
        beta0[i] = mu_y - sigma_yx @ sigma_xx_inv @ mu_x
        beta[idx_x, i] = sigma_xx_inv @ sigma_xy
        sigma2[i] = sigma_yy - sigma_yx @ sigma_xx_inv @ sigma_xy
        r_squared[i] = 1 - sigma2[i] / sigma_yy

    return ConditionalNormalResult(
        beta0=beta0, beta=beta.T, sigma=np.sqrt(sigma2), r_squared=r_squared
    )


@dataclass
class PCAResult:
    loadings: np.ndarray  # eigenvectors, columns = factors
    eigenvalues: np.ndarray
    quality: np.ndarray  # share of total variance per factor
    cum_quality: np.ndarray
    saturation: np.ndarray  # loadings scaled by sqrt(eigenvalue)
    variable_quality: np.ndarray  # squared saturation
    variable_contribution: np.ndarray  # variable_quality / eigenvalue


def pca(x: np.ndarray, num_factors: int | None = None, normalize: bool = False) -> PCAResult:
    """Principal component analysis on a data matrix (standardized first) or
    directly on a pre-computed correlation matrix.

    Original: stats/regPCA.m
    """
    x = np.asarray(x, dtype=float)
    if x.shape[0] == x.shape[1]:
        corr = x
    else:
        x_std = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        corr = (x_std.T @ x_std) / x_std.shape[0]

    n = corr.shape[0]
    if num_factors is None or num_factors == 0:
        num_factors = n

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:num_factors]
    eigenvectors = eigenvectors[:, order][:, :num_factors]

    quality = eigenvalues / np.sum(np.diag(corr))
    cum_quality = np.cumsum(quality)

    saturation = eigenvectors * np.sqrt(eigenvalues)
    variable_quality = saturation**2
    variable_contribution = variable_quality / eigenvalues

    if normalize:
        variable_quality = variable_quality / variable_quality.sum(axis=1, keepdims=True)
        variable_contribution = variable_contribution / variable_contribution.sum(
            axis=0, keepdims=True
        )

    return PCAResult(
        loadings=eigenvectors,
        eigenvalues=eigenvalues,
        quality=quality,
        cum_quality=cum_quality,
        saturation=saturation,
        variable_quality=variable_quality,
        variable_contribution=variable_contribution,
    )
