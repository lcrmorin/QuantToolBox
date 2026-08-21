"""VAR/VARX estimation (with exogenous regressors and linear parameter
restrictions) and lag-order selection.

Ported from QuantToolBox/ects/{varx_cls,varx_cml,varx_ls,varx_ml,
varx_order,var_constrained_estimation_onestep,
varx_constrained_estimation_onestep}.m

Translation notes:

- ``var_constrained_estimation_onestep.m`` and
  ``varx_constrained_estimation_onestep.m`` are near-identical (the
  latter just returns a few extra result fields); only the superset
  (``varx_estimate`` here) is ported, and ``var_*`` convenience wrappers
  become calls to ``varx_estimate`` with no exogenous regressors.
- ``varx_ls``/``varx_ml`` (thin dispatchers to the one-step estimator with
  a fixed identity starting covariance) and ``varx_cls`` (one-step LS with
  a user-supplied restriction) are not ported as separate functions --
  call ``varx_estimate(..., method="ls"|"ml")`` directly.
- ``varx_cml`` (concentrated/iterated ML: re-estimate Sigma from the
  residuals and re-run one-step ML until Sigma stabilizes) is ported as
  ``varx_estimate_cml``.
- The estimator itself uses the standard vec/GLS formulation:
  vec(theta) = (R'(Z Z' âŠ— SigmaÂ¹)R)Â¹ R'(Z âŠ— SigmaÂ¹) vec(Y), built on the
  ``vec``/``vech``/``duplication_matrix``/``elimination_matrix``/
  ``commutation_matrix`` helpers from ``quanttoolbox.linalg.special_matrices``
  ported earlier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanttoolbox.linalg.special_matrices import (
    commutation_matrix,
    duplication_matrix,
    elimination_matrix,
    reshapec,
    vech,
)


@dataclass
class VARXResult:
    theta: np.ndarray  # stacked coefficient vector (+ Cholesky vech if ML)
    stderr: np.ndarray | None
    vcv: np.ndarray | None
    log_l: float
    phi: np.ndarray  # (K, K*p) autoregressive coefficient matrices
    beta: np.ndarray | None  # (K, L) exogenous-regressor coefficients (None if L=0)
    residuals: np.ndarray  # (n_obs, K), NaN-padded for the first p rows
    sigma: np.ndarray  # (K, K) residual covariance
    n_obs: int
    n_obs_valid: int
    n_obs_usable: int
    n_params: int
    n_params_free: int
    df: int
    n_lags: int
    k: int
    l_exog: int


def varx_estimate(
    y: np.ndarray,
    x: np.ndarray | None = None,
    p: int = 1,
    restriction: tuple[np.ndarray, np.ndarray] | None = None,
    sigma: np.ndarray | None = None,
    method: str = "ls",
    compute_cov: bool = True,
) -> VARXResult:
    """One-step VARX(p) estimation: y_t = Phi_1 y_{t-1} + ... + Phi_p y_{t-p}
    + beta @ x_t + u_t, with optional linear restrictions on the stacked
    coefficient vector theta = RR @ gamma + r, via GLS using the given
    (or identity) residual covariance Sigma.

    method="ls" (default): residual covariance normalized by
    (n_usable - K*p - L) degrees of freedom. method="ml": normalized by
    n_usable (and the returned theta additionally includes the vech of
    the Cholesky factor of Sigma, matching the original's convention).

    Original: ects/{var_constrained_estimation_onestep,
    varx_constrained_estimation_onestep}.m
    """
    y = np.asarray(y, dtype=float)
    n_obs, k = y.shape

    if x is None:
        x_arr = np.zeros((n_obs, 1))
        n_exog = 0
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n_exog = x_arr.shape[1]

    n_params = k * (k * p + n_exog)
    rr, r = (np.eye(n_params), np.zeros(n_params)) if restriction is None else restriction
    rr = np.asarray(rr, dtype=float)
    r = np.asarray(r, dtype=float).flatten()

    if sigma is None:
        sigma_mat = np.eye(k)
    else:
        sigma_mat = np.asarray(sigma, dtype=float)

    valid = ~np.isnan(y).any(axis=1) & ~np.isnan(x_arr).any(axis=1)
    valid_idx = np.where(valid)[0]
    y_v = y[valid]
    x_v = x_arr[valid]
    n_valid = y_v.shape[0]
    n_usable = n_valid - p

    z = np.zeros((p * k, n_valid))
    for i in range(1, p + 1):
        w = np.full((n_valid, k), np.nan)
        w[i:] = y_v[:-i]
        z[(i - 1) * k : i * k, :] = w.T

    if n_exog > 0:
        z = np.vstack([z, x_v.T])

    y_stacked = y_v[p:].T
    z = z[:, p:]
    vec_y = y_stacked.flatten(order="F")

    inv_sigma = np.linalg.pinv(sigma_mat)
    y_c = np.kron(z.T, np.eye(k)) @ r
    y_c = vec_y - y_c

    w1 = z @ z.T
    w2 = np.kron(w1, inv_sigma)
    w3 = np.kron(z, inv_sigma)
    w4 = rr.T @ w2 @ rr
    w5 = np.linalg.pinv(w4)
    theta_c = w5 @ rr.T @ w3 @ y_c
    theta = rr @ theta_c + r

    bb = reshapec(theta, k, k * p + n_exog)
    phi = bb[:, : k * p]
    beta = bb[:, k * p :] if n_exog > 0 else None

    u = y_stacked - bb @ z

    residuals = np.full((n_obs, k), np.nan)
    residuals[valid_idx[p:]] = u.T

    sigma_hat = u @ u.T
    if method == "ml":
        sigma_hat = sigma_hat / n_usable
    else:
        sigma_hat = sigma_hat / (n_usable - k * p - n_exog)

    log_l = float(
        -0.5 * n_usable * np.log(np.linalg.det(sigma_hat))
        - 0.5 * n_usable * k * (np.log(2 * np.pi) + 1)
    )

    n_params_free = rr.shape[1]
    n_params_total = theta.shape[0]

    if method == "ml":
        chol = np.linalg.cholesky(sigma_hat)
        p_star = vech(chol, method=2)
        theta = np.concatenate([theta, p_star])
        n_params_total = theta.shape[0]

    if not compute_cov:
        return VARXResult(
            theta=theta,
            stderr=None,
            vcv=None,
            log_l=log_l,
            phi=phi,
            beta=beta,
            residuals=residuals,
            sigma=sigma_hat,
            n_obs=n_obs,
            n_obs_valid=n_valid,
            n_obs_usable=n_usable,
            n_params=n_params_total,
            n_params_free=n_params_free,
            df=n_usable - n_params_free,
            n_lags=p,
            k=k,
            l_exog=n_exog,
        )

    inv_sigma = np.linalg.pinv(sigma_hat)
    w2 = np.kron(w1, inv_sigma)
    w4 = rr.T @ w2 @ rr
    w5 = np.linalg.pinv(w4)
    vcv1 = rr @ w5 @ rr.T
    n_param1 = vcv1.shape[0]

    if method == "ml":
        d_k = duplication_matrix(k)
        l_k = elimination_matrix(k)
        k_kk = commutation_matrix(k, k)

        chol = np.linalg.cholesky(sigma_hat)
        h_mat = l_k @ (np.eye(k**2) + k_kk) @ np.kron(chol, np.eye(k)) @ l_k.T
        h_mat = np.linalg.pinv(h_mat)

        d_star = np.linalg.pinv(d_k.T @ d_k) @ d_k.T
        vcv2 = 2 * d_star @ np.kron(sigma_hat, sigma_hat) @ d_star.T
        vcv2 = vcv2 / n_usable
        vcv2 = h_mat @ vcv2 @ h_mat.T
        n_param2 = vcv2.shape[0]

        vcv = np.zeros((n_param1 + n_param2, n_param1 + n_param2))
        vcv[:n_param1, :n_param1] = vcv1
        vcv[n_param1:, n_param1:] = vcv2
    else:
        vcv = vcv1

    stderr = np.sqrt(np.diag(vcv))

    return VARXResult(
        theta=theta,
        stderr=stderr,
        vcv=vcv,
        log_l=log_l,
        phi=phi,
        beta=beta,
        residuals=residuals,
        sigma=sigma_hat,
        n_obs=n_obs,
        n_obs_valid=n_valid,
        n_obs_usable=n_usable,
        n_params=n_params_total,
        n_params_free=n_params_free,
        df=n_usable - n_params_free,
        n_lags=p,
        k=k,
        l_exog=n_exog,
    )


def varx_estimate_cml(
    y: np.ndarray,
    x: np.ndarray | None = None,
    p: int = 1,
    restriction: tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-8,
    max_iters: int = 100,
) -> VARXResult:
    """Concentrated (iterated) maximum likelihood VARX estimation: alternate
    one-step ML estimation and re-estimating Sigma from the residuals
    until det(Sigma) stabilizes.

    Original: ects/varx_cml.m
    """
    y_arr = np.asarray(y, dtype=float)
    k = y_arr.shape[1]
    sigma = np.eye(k)

    for _ in range(max_iters):
        result = varx_estimate(y, x, p, restriction, sigma, method="ml", compute_cov=False)
        new_sigma = result.sigma
        if abs(np.linalg.det(sigma) - np.linalg.det(new_sigma)) < tol:
            sigma = new_sigma
            break
        sigma = new_sigma

    return varx_estimate(y, x, p, restriction, sigma, method="ml", compute_cov=True)


@dataclass
class VARXOrderResult:
    p_values: np.ndarray
    criteria: np.ndarray  # (n_lags, 7): BIC, AICa, AICc, SIC, FPE, AIC, HQ
    optimal_p: np.ndarray  # (7,) optimal lag per criterion


def varx_order(y: np.ndarray, x: np.ndarray | None, p_max: int | np.ndarray) -> VARXOrderResult:
    """Select the VARX lag order by evaluating BIC, AICa (alpha=3), AICc,
    SIC, FPE, AIC, and HQ information criteria across p=0..p_max (or the
    explicit lag values in p_max, if given as an array).

    Original: ects/varx_order.m
    """
    if np.isscalar(p_max):
        p_values = np.arange(0, int(p_max) + 1)
    else:
        p_values = np.atleast_1d(p_max)
    n_lags = p_values.shape[0]
    criteria = np.zeros((n_lags, 7))

    for i, p in enumerate(p_values):
        result = varx_estimate(y, x, int(p), method="ml", compute_cov=False)
        t = result.n_obs_usable
        k = result.k
        log_det_sigma = np.log(np.linalg.det(result.sigma))

        criteria[i, 0] = log_det_sigma + p * k**2 * np.log(t) / t  # BIC
        criteria[i, 1] = log_det_sigma + 3 * p * k**2 / t  # AICa
        criteria[i, 2] = log_det_sigma + (1 + p * k**2 / t) / (1 - (p * k**2 - 2) / t)  # AICc
        criteria[i, 3] = log_det_sigma + k * np.log(1 + 2 * (p * k**2 + 1) / t)  # SIC
        criteria[i, 4] = log_det_sigma + k * np.log((t + p * k + 1) / (t - p * k - 1))  # FPE
        criteria[i, 5] = log_det_sigma + 2 * p * k**2 / t  # AIC
        criteria[i, 6] = log_det_sigma + 2 * p * k**2 * np.log(np.log(t)) / t  # HQ

    optimal_idx = np.argmin(criteria, axis=0)
    optimal_p = p_values[optimal_idx]

    return VARXOrderResult(p_values=p_values, criteria=criteria, optimal_p=optimal_p)
