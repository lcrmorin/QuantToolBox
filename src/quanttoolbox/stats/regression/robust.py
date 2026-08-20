"""Robust M-estimator regression via Iteratively Reweighted Least Squares (IRLS).

Ported from QuantToolbox/ects/{robust_regression,robust_huber_regression,
robust_lad_regression,robust_quantile_regression,
robust_inverse_quantile_regression}.m

Translation notes:

- All four specific estimators (Huber, LAD, quantile, inverse-quantile)
  are thin wrappers supplying a (rho, rho_prime) loss-function pair to one
  shared IRLS core (``robust_regression``), exactly mirroring the
  original's structure.
- ``quantile_m_regression``/``inverse_quantile_m_regression`` here solve
  the quantile-loss problem via IRLS (a smooth M-estimation
  approximation), which is a *different algorithm* from
  ``quanttoolbox.stats.regression.quantile.quantile_regression`` (which
  solves the exact Koenker-Bassett LP formulation). IRLS is faster and
  matches the original MATLAB toolbox's approach here, but the LP version
  is the numerically exact one if precision matters more than speed.
- MATLAB's ``global ROBUST_eps`` convergence tolerance and ``global
  Print_Results`` are replaced by the ``RobustRegressionConfig`` dataclass
  (the original never actually set a value for ``ROBUST_eps`` in the
  files reviewed, so a conventional 1e-6 default is used here -- pass
  ``RobustRegressionConfig(eps=...)`` to match a specific original run).
- The original's ``rho_prime`` numerical-gradient fallback
  (``numerical_gradient(rho, beta)`` when only ``rho`` is supplied) is
  reproduced with a simple central-difference approximation.
- MATLAB's ``cdftc``/``cdffc`` (upper-tail Student-t / F complementary
  CDFs) map directly to ``scipy.stats.t.sf`` / ``scipy.stats.f.sf``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist


@dataclass(frozen=True)
class RobustRegressionConfig:
    """Replaces the ROBUST_eps global. See module docstring for the default's provenance."""

    eps: float = 1e-6
    max_iters: int = 500


@dataclass
class RobustRegressionResult:
    beta: np.ndarray
    stderr: np.ndarray
    vcv: np.ndarray
    residuals: np.ndarray
    converged: bool
    n_iters: int
    weights: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    r_squared: float
    r_squared_adj: float
    r_squared_centered: float
    r_squared_centered_adj: float
    sigma2: float
    sigma: float
    rss: float
    tss: float
    tss_centered: float
    ess: float
    ess_centered: float
    f_stat: float
    f_pvalue: float
    df_residual: int
    n_obs: int
    n_obs_valid: int
    n_obs_missing: int


def robust_regression(
    y: np.ndarray,
    x: np.ndarray,
    rho: Callable[[np.ndarray], np.ndarray],
    rho_prime: Callable[[np.ndarray], np.ndarray] | None = None,
    config: RobustRegressionConfig | None = None,
) -> RobustRegressionResult:
    """Generic M-estimation regression via IRLS, for an arbitrary loss
    function rho (with derivative rho_prime, numerically approximated if
    not supplied).

    Original: ects/robust_regression.m
    """
    if config is None:
        config = RobustRegressionConfig()

    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n_obs = y.shape[0]
    n_params = x.shape[1]

    if rho_prime is None:
        h = 1e-6

        def rho_prime(u: np.ndarray, _rho=rho, _h=h) -> np.ndarray:
            return (_rho(u + _h) - _rho(u - _h)) / (2 * _h)

    valid = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
    valid_idx = np.where(valid)[0]
    n_missing = int(np.sum(~valid))
    y_v, x_v = y[valid], x[valid]
    n_valid = y_v.shape[0]
    n_valid_params = n_params

    beta = np.zeros(n_params)
    new_beta = np.linalg.solve(x_v.T @ x_v, x_v.T @ y_v)
    u = y_v - x_v @ new_beta
    w = rho_prime(u) / (u + 1e-10)
    xw = x_v * w[:, None]

    converged = False
    diff = np.max(np.abs(new_beta - beta))
    n_iter = 1
    while diff > config.eps:
        new_beta = np.linalg.solve(xw.T @ x_v, xw.T @ y_v)
        u = y_v - x_v @ new_beta
        w = rho_prime(u) / (u + 1e-5)
        xw = x_v * w[:, None]
        diff = np.max(np.abs(new_beta - beta))
        beta = new_beta
        n_iter += 1
        if n_iter > config.max_iters:
            break
    else:
        converged = True

    xx_w = xw.T @ x_v
    inv_xx_w = np.linalg.inv(xx_w)

    rss = float(np.sum(u**2))
    df_y = n_valid - 1
    df_x = n_valid_params - 1
    df_u = df_y - df_x

    sigma2 = rss / df_u
    sigma = np.sqrt(sigma2)
    vcv = sigma2 * inv_xx_w
    stderr = np.diag(vcv)
    stderr = np.where(stderr < 0, np.nan, stderr)
    stderr = np.sqrt(stderr)

    t_stat = beta / np.where(stderr == 0, np.nan, stderr)
    p_value = 2 * t_dist.sf(np.abs(t_stat), df_u)

    tss = float(np.sum(y_v**2))
    ess = tss - rss
    r_squared = 1 - rss / tss
    r_squared_adj = 1 - (rss / df_u) / (tss / df_y)

    yc = y_v - y_v.mean()
    tss_c = float(np.sum(yc**2))
    ess_c = tss_c - rss
    r_squared_c = 1 - rss / tss_c
    r_squared_c_adj = 1 - (rss / df_u) / (tss_c / df_y)

    f_stat = (r_squared_c / df_x) / ((1 - r_squared_c) / df_u)
    f_pvalue = f_dist.sf(f_stat, df_x, df_u)

    residuals = np.full(n_obs, np.nan)
    residuals[valid_idx] = u

    return RobustRegressionResult(
        beta=beta,
        stderr=stderr,
        vcv=vcv,
        residuals=residuals,
        converged=converged,
        n_iters=n_iter,
        weights=w,
        t_stat=t_stat,
        p_value=p_value,
        r_squared=r_squared,
        r_squared_adj=r_squared_adj,
        r_squared_centered=r_squared_c,
        r_squared_centered_adj=r_squared_c_adj,
        sigma2=sigma2,
        sigma=sigma,
        rss=rss,
        tss=tss,
        tss_centered=tss_c,
        ess=ess,
        ess_centered=ess_c,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        df_residual=df_u,
        n_obs=n_obs,
        n_obs_valid=n_valid,
        n_obs_missing=n_missing,
    )


def huber_regression(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 1.345,
    config: RobustRegressionConfig | None = None,
) -> RobustRegressionResult:
    """Huber M-estimator regression: quadratic loss for |residual| < c,
    linear loss beyond. c=1.345 is the conventional choice giving ~95%
    efficiency under normality.

    Original: ects/robust_huber_regression.m
    """
    if config is None:
        config = RobustRegressionConfig()

    def rho(u: np.ndarray) -> np.ndarray:
        return (u**2) * (np.abs(u) < c) + c * np.abs(u) * (np.abs(u) >= c)

    def rho_prime(u: np.ndarray) -> np.ndarray:
        return 2 * u * (np.abs(u) < c) + c * np.sign(u) * (np.abs(u) >= c)

    return robust_regression(y, x, rho, rho_prime, config)


def lad_regression(
    y: np.ndarray, x: np.ndarray, config: RobustRegressionConfig | None = None
) -> RobustRegressionResult:
    """Least Absolute Deviations (median) regression.

    Note: LAD's weight function 1/(u+eps) is not smooth near u=0, so the
    IRLS loop typically doesn't settle below a tight ``eps`` tolerance --
    it will often run to ``max_iters`` even after beta has already
    stabilized at the optimum. Check that the coefficients look stable
    across the last few iterations rather than relying solely on
    ``result.converged`` for this estimator.

    Original: ects/robust_lad_regression.m
    """
    if config is None:
        config = RobustRegressionConfig()

    def rho(u: np.ndarray) -> np.ndarray:
        return np.abs(u)

    def rho_prime(u: np.ndarray) -> np.ndarray:
        return np.sign(u)

    return robust_regression(y, x, rho, rho_prime, config)


def quantile_m_regression(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    config: RobustRegressionConfig | None = None,
) -> RobustRegressionResult:
    """Quantile regression at level alpha via IRLS M-estimation (see module
    docstring for how this differs from the exact LP-based
    ``quantile.quantile_regression``).

    Original: ects/robust_quantile_regression.m
    """
    if config is None:
        config = RobustRegressionConfig()

    def rho(u: np.ndarray) -> np.ndarray:
        return u * (alpha - (u < 0))

    def rho_prime(u: np.ndarray) -> np.ndarray:
        return alpha - (u < 0).astype(float)

    return robust_regression(y, x, rho, rho_prime, config)


def inverse_quantile_m_regression(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    config: RobustRegressionConfig | None = None,
) -> RobustRegressionResult:
    """ "Inverse" quantile regression (loss weighted by which side of zero
    the residual falls on, complementary to quantile_m_regression).

    Original: ects/robust_inverse_quantile_regression.m
    """
    if config is None:
        config = RobustRegressionConfig()

    def rho(u: np.ndarray) -> np.ndarray:
        return u * ((u > 0).astype(float) - alpha)

    def rho_prime(u: np.ndarray) -> np.ndarray:
        return (u > 0).astype(float) - alpha

    return robust_regression(y, x, rho, rho_prime, config)
