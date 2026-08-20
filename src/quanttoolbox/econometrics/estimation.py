"""Constrained OLS, GMM, and Maximum Likelihood estimation with linear
parameter restrictions.

Ported from QuantToolbox/ects/{ols_estimation,ols_constrained_estimation,
gmm_estimation,gmm_constrained_estimation,ml_estimation,
ml_constrained_estimation}.m

Translation notes:

- All three estimators share the same "linear restriction" pattern: the
  full parameter vector theta = RR @ gamma + r for a free parameter gamma,
  where RR/r default to the identity/zero (i.e. no restriction). This is
  ported as-is via an optional ``restriction=(RR, r)`` argument.
- The unconstrained convenience wrappers (ols_estimation.m,
  gmm_estimation.m, ml_estimation.m -- each just a nargin-dispatch stub
  calling the "_constrained_" version with default arguments) are not
  ported as separate functions; call the functions below with
  ``restriction=None`` for the same effect.
- GMM/ML optimization: MATLAB's ``fminunc``/``fmincon`` (with a
  trust-region-vs-quasi-Newton branch depending on whether analytical
  gradients/Hessians were supplied) is replaced throughout by
  ``scipy.optimize.minimize`` (BFGS by default, Newton-CG if an
  analytical Hessian is supplied) -- both are standard local optimizers
  for smooth objectives and the choice of exact algorithm doesn't change
  what's being estimated.
- GMM's iteratively-updated efficient weighting matrix uses a
  Bartlett-kernel (Newey-West-style) HAC covariance of the moments,
  ported directly (``_bartlett_covariance``).
- ML's three covariance estimator options (Hessian-based, OPG, and
  heteroskedasticity-consistent "sandwich") are all ported, selected via
  ``cov="hessian"|"opg"|"hc"``.
- MATLAB's ``global Print_Results``/``GMM_*``/``ML_*`` blocks are replaced
  by ``quanttoolbox.config.EstimationConfig``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, f, t

from quanttoolbox.config import EstimationConfig


def _default_restriction(n_params: int) -> tuple[np.ndarray, np.ndarray]:
    return np.eye(n_params), np.zeros(n_params)


@dataclass
class OLSEstimationResult:
    beta: np.ndarray
    stderr: np.ndarray
    vcv: np.ndarray
    residuals: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    r_squared: float
    r_squared_adj: float
    r_squared_centered: float
    r_squared_centered_adj: float
    sigma: float
    sigma2: float
    f_stat: float
    f_pvalue: float
    df_residual: int
    n_obs: int
    n_obs_valid: int


def ols_estimation(
    y: np.ndarray,
    x: np.ndarray,
    restriction: tuple[np.ndarray, np.ndarray] | None = None,
    weights: np.ndarray | None = None,
) -> OLSEstimationResult:
    """Weighted, linearly-restricted OLS: beta = RR @ gamma + r, estimating
    the free parameter gamma by weighted least squares.

    Original: ects/{ols_estimation,ols_constrained_estimation}.m
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n_obs = y.shape[0]
    n_params = x.shape[1]

    rr, r = _default_restriction(n_params) if restriction is None else restriction
    rr = np.asarray(rr, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    n_free = rr.shape[1]

    w = np.ones(n_obs) if weights is None else np.asarray(weights, dtype=float).flatten()

    valid = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
    valid_idx = np.where(valid)[0]
    y_v, x_v, w_v = y[valid], x[valid], w[valid]
    n_valid = y_v.shape[0]

    wx = w_v[:, None] * x_v
    xwx = x_v.T @ wx
    xx = rr.T @ xwx @ rr
    rxw = (wx @ rr).T
    xy = rxw @ (y_v - x_v @ r)

    inv_xx = np.linalg.inv(xx)
    gamma = inv_xx @ xy
    beta = rr @ gamma + r

    u = y_v - x_v @ beta
    rss = np.sum(w_v * u**2)

    df_y = n_valid - 1
    df_x = n_free - 1
    df_u = df_y - df_x

    sigma2 = rss / df_u
    sigma = np.sqrt(sigma2)
    vcv = sigma2 * (rr @ inv_xx @ rr.T)
    stderr = np.sqrt(np.diag(vcv))

    t_stat = beta / np.where(stderr == 0, np.nan, stderr)
    p_value = 2 * t.sf(np.abs(t_stat), df_u)

    tss = np.sum(w_v * y_v**2)
    r_squared = 1 - rss / tss
    r_squared_adj = 1 - (rss / df_u) / (tss / df_y)

    y_bar = np.sum(w_v * y_v) / np.sum(w_v)
    yc = y_v - y_bar
    tss_c = np.sum(w_v * yc**2)
    r_squared_c = 1 - rss / tss_c
    r_squared_c_adj = 1 - (rss / df_u) / (tss_c / df_y)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = (r_squared_c / df_x) / ((1 - r_squared_c) / df_u)
    f_pvalue = f.sf(f_stat, df_x, df_u)

    residuals = np.full(n_obs, np.nan)
    residuals[valid_idx] = u

    return OLSEstimationResult(
        beta=beta,
        stderr=stderr,
        vcv=vcv,
        residuals=residuals,
        t_stat=t_stat,
        p_value=p_value,
        r_squared=r_squared,
        r_squared_adj=r_squared_adj,
        r_squared_centered=r_squared_c,
        r_squared_centered_adj=r_squared_c_adj,
        sigma=sigma,
        sigma2=sigma2,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        df_residual=df_u,
        n_obs=n_obs,
        n_obs_valid=n_valid,
    )


def _bartlett_covariance(h: np.ndarray, n_lags: int) -> np.ndarray:
    """Bartlett-kernel (Newey-West-style) HAC covariance of moment
    conditions h (n_obs, n_moments)."""
    n = h.shape[0]
    vcv = (h.T @ h) / n
    w = vcv.copy()
    for lag in range(1, n_lags + 1):
        lagged_h = np.vstack([np.full((lag, h.shape[1]), np.nan), h[:-lag]])
        valid = ~np.isnan(lagged_h).any(axis=1)
        cov_lag = (h[valid].T @ lagged_h[valid]) / n
        w = w + (1 - lag / (n_lags + 1)) * (cov_lag + cov_lag.T)
    return w


@dataclass
class GMMEstimationResult:
    theta: np.ndarray
    stderr: np.ndarray
    vcv: np.ndarray
    q_min: float
    jacobian: np.ndarray
    j_test: float
    j_test_pvalue: float
    df: int
    n_obs: int
    n_obs_valid: int
    converged: bool
    n_iters: int


def gmm_estimation(
    moments_fn: Callable[[np.ndarray], np.ndarray],
    sv: np.ndarray,
    restriction: tuple[np.ndarray, np.ndarray] | None = None,
    weights: np.ndarray | float = 1.0,
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    weight_matrix: np.ndarray | None = None,
    n_lags: int = 0,
    config: EstimationConfig | None = None,
) -> GMMEstimationResult:
    """Iterated (two-step-and-beyond) efficient GMM estimation:
    theta = RR @ gamma + r, minimizing g(theta)' @ inv(W) @ g(theta) where
    g is the sample average of moments_fn(theta), re-estimating the
    efficient weighting matrix W (Bartlett/HAC, or a fixed weight_matrix if
    given) each iteration until convergence.

    Original: ects/{gmm_estimation,gmm_constrained_estimation}.m
    """
    if config is None:
        config = EstimationConfig()

    sv = np.asarray(sv, dtype=float).flatten()
    n_params = sv.shape[0]
    rr, r = _default_restriction(n_params) if restriction is None else restriction
    rr = np.asarray(rr, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    n_free = rr.shape[1]

    theta0 = rr @ sv + r
    h0 = np.atleast_2d(moments_fn(theta0))
    if h0.shape[0] == 1 and h0.shape[1] != 1:
        h0 = h0.reshape(-1, h0.shape[1]) if h0.ndim == 2 else h0
    m = h0.shape[1]
    n_obs = h0.shape[0]
    valid0 = ~np.isnan(h0).any(axis=1)
    n_valid = int(np.sum(valid0))
    df = n_valid - n_free

    if m < n_free:
        nan_p = np.full(n_free, np.nan)
        return GMMEstimationResult(
            theta=nan_p,
            stderr=nan_p,
            vcv=np.full((n_free, n_free), np.nan),
            q_min=np.nan,
            jacobian=np.full((m, n_free), np.nan),
            j_test=np.nan,
            j_test_pvalue=np.nan,
            df=df,
            n_obs=n_obs,
            n_obs_valid=n_valid,
            converged=False,
            n_iters=0,
        )

    inv_w = np.eye(m) if weight_matrix is None else np.asarray(weight_matrix, dtype=float)
    w = (
        np.broadcast_to(weights, (n_obs,))
        if np.isscalar(weights)
        else np.asarray(weights, dtype=float)
    )

    def _avg_moments(theta: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(moments_fn(theta))
        h = w[:, None] * h
        valid = ~np.isnan(h).any(axis=1)
        return h[valid].mean(axis=0)

    def _objective(gamma: np.ndarray) -> float:
        theta = rr @ gamma + r
        g = _avg_moments(theta)
        return float(g @ inv_w @ g)

    gamma = sv.copy()
    converged = False
    n_iter = 0
    for n_iter in range(1, config.max_iters + 1):  # noqa: B007 (used after loop)
        result = minimize(_objective, gamma, method="BFGS")
        gamma_new = result.x
        if np.max(np.abs(gamma_new - gamma)) < config.tol:
            gamma = gamma_new
            converged = True
            break
        gamma = gamma_new

        if weight_matrix is None:
            theta = rr @ gamma + r
            h = np.atleast_2d(moments_fn(theta))
            h = w[:, None] * h
            valid = ~np.isnan(h).any(axis=1)
            inv_w = np.linalg.pinv(_bartlett_covariance(h[valid], n_lags))

    theta = rr @ gamma + r
    g_final = _avg_moments(theta)
    q_min = float(g_final @ inv_w @ g_final)

    if jacobian_fn is not None:
        d = jacobian_fn(theta)
    else:
        d = _numerical_jacobian(_avg_moments, theta)
    d = d @ rr

    try:
        vcv_gamma = np.linalg.inv(d.T @ inv_w @ d)
    except np.linalg.LinAlgError:
        vcv_gamma = np.full((n_free, n_free), np.nan)

    vcv = rr @ vcv_gamma @ rr.T / n_valid
    stderr = np.sqrt(np.diag(vcv))

    if m > n_free:
        j_test = n_valid * q_min
        j_test_pvalue = float(chi2.sf(j_test, m - n_free))
    else:
        j_test, j_test_pvalue = 0.0, 0.0

    return GMMEstimationResult(
        theta=theta,
        stderr=stderr,
        vcv=vcv,
        q_min=q_min,
        jacobian=d,
        j_test=j_test,
        j_test_pvalue=j_test_pvalue,
        df=df,
        n_obs=n_obs,
        n_obs_valid=n_valid,
        converged=converged,
        n_iters=n_iter,
    )


def _numerical_jacobian(
    fn: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    f0 = np.atleast_1d(fn(theta))
    jac = np.zeros((f0.shape[0], theta.shape[0]))
    for i in range(theta.shape[0]):
        step = np.zeros_like(theta)
        step[i] = h
        jac[:, i] = (np.atleast_1d(fn(theta + step)) - np.atleast_1d(fn(theta - step))) / (2 * h)
    return jac


def _numerical_hessian(
    fn: Callable[[np.ndarray], float], theta: np.ndarray, h: float = 1e-4
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            step_i, step_j = np.zeros(n), np.zeros(n)
            step_i[i], step_j[j] = h, h
            f_pp = fn(theta + step_i + step_j)
            f_pm = fn(theta + step_i - step_j)
            f_mp = fn(theta - step_i + step_j)
            f_mm = fn(theta - step_i - step_j)
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
    return hess


@dataclass
class MLEstimationResult:
    theta: np.ndarray
    stderr: np.ndarray
    vcv: np.ndarray
    log_l: np.ndarray
    sum_log_l: float
    df: int
    n_obs: int
    n_obs_valid: int
    cov_type: str
    converged: bool


def ml_estimation(
    logpdf_fn: Callable[[np.ndarray], np.ndarray],
    sv: np.ndarray,
    restriction: tuple[np.ndarray, np.ndarray] | None = None,
    weights: np.ndarray | float = 1.0,
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    hessian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    cov: str = "hessian",
    config: EstimationConfig | None = None,
) -> MLEstimationResult:
    """Maximum likelihood estimation: theta = RR @ gamma + r, maximizing the
    sum of logpdf_fn(theta) (a per-observation log-density) over the free
    parameter gamma.

    cov="hessian" (default): asymptotic covariance from the (numerical or
    analytical) Hessian of the log-likelihood. cov="opg": outer-product-
    of-gradients estimator. cov="hc": heteroskedasticity-consistent
    "sandwich" estimator (Hessian and OPG combined).

    Original: ects/{ml_estimation,ml_constrained_estimation}.m
    """
    if config is None:
        config = EstimationConfig()

    sv = np.asarray(sv, dtype=float).flatten()
    n_params = sv.shape[0]
    rr, r = _default_restriction(n_params) if restriction is None else restriction
    rr = np.asarray(rr, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    n_free = rr.shape[1]

    logl0 = np.atleast_1d(logpdf_fn(rr @ sv + r))
    n_obs = logl0.shape[0]
    w = (
        np.broadcast_to(weights, (n_obs,))
        if np.isscalar(weights)
        else np.asarray(weights, dtype=float)
    )

    def _neg_sum_logl(gamma: np.ndarray) -> float:
        theta = rr @ gamma + r
        logl = np.atleast_1d(logpdf_fn(theta))
        wl = w * logl
        return float(-np.sum(wl[~np.isnan(wl)]))

    jac_for_opt = None
    if jacobian_fn is not None:

        def jac_for_opt(gamma: np.ndarray) -> np.ndarray:
            theta = rr @ gamma + r
            j = np.atleast_2d(jacobian_fn(theta)) @ rr
            j = w[:, None] * j
            return -np.sum(j[~np.isnan(j).any(axis=1)], axis=0)

    result = minimize(_neg_sum_logl, sv, method="BFGS", jac=jac_for_opt)
    gamma = result.x
    theta = rr @ gamma + r

    logl = np.atleast_1d(logpdf_fn(theta))
    valid = ~np.isnan(logl)
    n_valid = int(np.sum(valid))
    df = n_valid - n_free
    sum_logl = float(np.sum(logl[valid]))

    def _neg_sum_logl_full(t: np.ndarray) -> float:
        ll = np.atleast_1d(logpdf_fn(t))
        wl = w * ll
        return float(-np.sum(wl[~np.isnan(wl)]))

    if cov in ("opg", "hc"):
        g = jacobian_fn(theta) if jacobian_fn is not None else _numerical_jacobian(logpdf_fn, theta)
        g = w[:, None] * g
        g_valid = g[valid] @ rr
        gg_valid = g_valid.T @ g_valid

    if cov == "opg":
        try:
            vcv_free = np.linalg.inv(gg_valid)
        except np.linalg.LinAlgError:
            vcv_free = np.full((n_free, n_free), np.nan)
        cov_type = "opg"
    else:
        h = (
            hessian_fn(theta)
            if hessian_fn is not None
            else -_numerical_hessian(_neg_sum_logl_full, theta)
        )
        h_valid = rr.T @ h @ rr
        try:
            inv_h_valid = np.linalg.inv(h_valid)
        except np.linalg.LinAlgError:
            inv_h_valid = np.full((n_free, n_free), np.nan)

        if cov == "hc":
            vcv_free = inv_h_valid @ gg_valid @ inv_h_valid
            cov_type = "hc"
        else:
            vcv_free = -inv_h_valid
            cov_type = "hessian"

    vcv = rr @ vcv_free @ rr.T
    stderr = np.sqrt(np.diag(vcv))

    residuals_full = np.full(n_obs, np.nan)
    residuals_full[valid] = logl[valid]

    return MLEstimationResult(
        theta=theta,
        stderr=stderr,
        vcv=vcv,
        log_l=residuals_full,
        sum_log_l=sum_logl,
        df=df,
        n_obs=n_obs,
        n_obs_valid=n_valid,
        cov_type=cov_type,
        converged=result.success,
    )


@dataclass
class WaldTestResult:
    chi2_stat: float
    chi2_pvalue: float
    f_stat: float
    f_pvalue: float


def wald_test(
    constraint_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    vcv: np.ndarray,
    n_obs: int,
) -> WaldTestResult:
    """Wald test of the (possibly nonlinear) hypothesis constraint_fn(theta) == 0.

    Original: ects/wald_test.m
    """
    theta = np.asarray(theta, dtype=float).flatten()
    vcv = np.asarray(vcv, dtype=float)
    c0 = np.atleast_1d(constraint_fn(theta))
    g = c0.shape[0]

    jac = _numerical_jacobian(constraint_fn, theta)

    chi2_stat = float(c0 @ np.linalg.inv(jac @ vcv @ jac.T) @ c0)
    chi2_pvalue = float(chi2.sf(chi2_stat, g))

    f_stat = chi2_stat / g
    f_pvalue = float(f.sf(f_stat, g, n_obs))

    return WaldTestResult(
        chi2_stat=chi2_stat, chi2_pvalue=chi2_pvalue, f_stat=f_stat, f_pvalue=f_pvalue
    )
