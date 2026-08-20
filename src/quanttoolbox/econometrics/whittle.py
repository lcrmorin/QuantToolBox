"""Whittle (frequency-domain) maximum likelihood estimation.

Ported from QuantToolbox/ects/{whittle_estimation,
whittle_constrained_estimation,whittle_local_level,
whittle_local_linear_trend}.m and QuantToolbox/maths/{periodogram,pdgm}.m
(identical duplicates, both covered by ``periodogram`` here rather than
in ``maths.py``).

Translation notes:

- Whittle estimation fits a parametric spectral density function
  sdf(lambda, theta) to the sample periodogram by maximizing the Whittle
  approximate log-likelihood, via the same linearly-restricted
  optimization pattern as ``econometrics.estimation`` (theta = RR @ gamma
  + r).
- MATLAB's "method of scoring" custom Newton-Raphson-with-information-
  matrix optimizer (used only when analytical gradients are supplied) is
  not ported as a separate code path; ``scipy.optimize.minimize`` (BFGS,
  using the analytical gradient if supplied) is used throughout instead
  -- both converge to the same MLE for this smooth, well-behaved
  objective, and BFGS is the standard choice.
- ``whittle_estimation.m``'s nargin-dispatch wrapper is not ported
  separately; call ``whittle_estimation`` directly with
  ``restriction=None`` for the unconstrained case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quanttoolbox.config import EstimationConfig
from quanttoolbox.econometrics.estimation import _numerical_hessian, _numerical_jacobian


def periodogram(y: np.ndarray, scaling: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw periodogram of y via FFT.

    scaling=False (default): I(lambda) = |FFT(y)|^2 / n.
    scaling=True: additionally divided by 2*pi (spectral-density scaling).

    Original: maths/{periodogram,pdgm}.m

    Returns (lambda, fft_coeffs, periodogram_values).
    """
    y = np.asarray(y, dtype=float).flatten()
    n = y.shape[0]
    fft_coeffs = np.fft.fft(y)
    intensity = np.abs(fft_coeffs) ** 2 / n
    lam = 2 * np.pi * np.arange(n) / n
    if scaling:
        intensity = intensity / (2 * np.pi)
    return lam, fft_coeffs, intensity


@dataclass
class WhittleEstimationResult:
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


def whittle_estimation(
    y: np.ndarray,
    sdf_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sv: np.ndarray,
    restriction: tuple[np.ndarray, np.ndarray] | None = None,
    weights: np.ndarray | float = 1.0,
    sdf_jacobian_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    cov: str = "hessian",
    config: EstimationConfig | None = None,
) -> WhittleEstimationResult:
    """Whittle (frequency-domain) MLE: fit a parametric spectral density
    sdf_fn(lambda, theta) to the periodogram of y by maximizing the
    Whittle log-likelihood sum(-log(2*pi) - 0.5*log(f) - 0.5*I/f), with
    theta = RR @ gamma + r.

    Original: ects/{whittle_estimation,whittle_constrained_estimation}.m
    """
    if config is None:
        config = EstimationConfig()

    sv = np.asarray(sv, dtype=float).flatten()
    n_params = sv.shape[0]
    rr, r = (np.eye(n_params), np.zeros(n_params)) if restriction is None else restriction
    rr = np.asarray(rr, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    n_free = rr.shape[1]

    lam, _, intensity = periodogram(y, scaling=True)
    n_obs = intensity.shape[0]
    w = (
        np.broadcast_to(weights, (n_obs,))
        if np.isscalar(weights)
        else np.asarray(weights, dtype=float)
    )

    def _logpdf(theta: np.ndarray) -> np.ndarray:
        f_sdf = sdf_fn(lam, theta)
        return -np.log(2 * np.pi) - 0.5 * np.log(f_sdf) - 0.5 * (intensity / f_sdf)

    def _neg_sum_logl(gamma: np.ndarray) -> float:
        theta = rr @ gamma + r
        ll = w * _logpdf(theta)
        return float(-np.sum(ll[~np.isnan(ll)]))

    jac_for_opt = None
    if sdf_jacobian_fn is not None:

        def jac_for_opt(gamma: np.ndarray) -> np.ndarray:
            theta = rr @ gamma + r
            f_sdf = sdf_fn(lam, theta)
            f_sdf = np.where(f_sdf == 0, 0.01, f_sdf)
            j = sdf_jacobian_fn(lam, theta)
            j = w[:, None] * j
            inv_f = 1.0 / f_sdf
            grad_terms = 0.5 * (intensity * inv_f - 1.0)[:, None] * (j * inv_f[:, None])
            grad_terms = w[:, None] * grad_terms
            return -rr.T @ np.sum(grad_terms[~np.isnan(grad_terms).any(axis=1)], axis=0)

    result = minimize(_neg_sum_logl, sv, method="BFGS", jac=jac_for_opt)
    gamma = result.x
    theta = rr @ gamma + r

    logl = _logpdf(theta)
    valid = ~np.isnan(logl)
    n_valid = int(np.sum(valid))
    df = n_valid - n_free
    sum_logl = float(np.sum(logl[valid]))

    if cov in ("opg", "hc"):
        if sdf_jacobian_fn is not None:
            f_sdf = sdf_fn(lam, theta)
            f_sdf = np.where(f_sdf == 0, 0.01, f_sdf)
            j = sdf_jacobian_fn(lam, theta)
            inv_f = 1.0 / f_sdf
            g = 0.5 * (intensity * inv_f - 1.0)[:, None] * (j * inv_f[:, None])
        else:
            g = _numerical_jacobian(_logpdf, theta)
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
        h = -_numerical_hessian(_neg_sum_logl, gamma)
        h_valid = h  # already in the gamma (restricted) parameterization
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

    return WhittleEstimationResult(
        theta=theta,
        stderr=stderr,
        vcv=vcv,
        log_l=logl,
        sum_log_l=sum_logl,
        df=df,
        n_obs=n_obs,
        n_obs_valid=n_valid,
        cov_type=cov_type,
        converged=result.success,
    )


def _local_level_sdf(lam: np.ndarray, theta: np.ndarray) -> np.ndarray:
    sigma1, sigma2 = abs(theta[0]), abs(theta[1])
    return (2 * (1 - np.cos(lam)) * sigma1**2 + sigma2**2) / (2 * np.pi)


def _local_level_sdf_jacobian(lam: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # NOTE: the original MATLAB Jacobian (local_level_sdf_jacobian.m)
    # differentiates the *unscaled* spectral density while the SDF itself
    # (local_level_sdf.m) is scaled by 1/(2*pi) -- an inconsistency that
    # makes the analytical gradient off by a factor of 2*pi relative to
    # the numerical gradient (verified directly: this fix brings them to
    # within 1e-5 agreement, whereas the unscaled version was ~6.28x too
    # large). The 1/(2*pi) factor below corrects this.
    sigma1, sigma2 = abs(theta[0]), abs(theta[1])
    j1 = 4 * (1 - np.cos(lam)) * sigma1 / (2 * np.pi)
    j2 = 2 * sigma2 * np.ones_like(lam) / (2 * np.pi)
    return np.column_stack([j1, j2])


def whittle_local_level(y: np.ndarray, sv: np.ndarray, **kwargs) -> WhittleEstimationResult:
    """Fit a local-level (random-walk-plus-noise) model
    y_t = mu_t + eps_t, mu_t = mu_{t-1} + eta_t via Whittle estimation of
    (sigma_epsilon, sigma_eta) on the first-differenced series.

    Original: ects/whittle_local_level.m
    """
    sv = np.asarray(sv, dtype=float).flatten()
    if sv.shape[0] != 2:
        raise ValueError("whittle_local_level: sv must have length 2")

    y = np.asarray(y, dtype=float).flatten()
    dy = y[1:] - y[:-1]
    dy = dy[~np.isnan(dy)]

    return whittle_estimation(
        dy, _local_level_sdf, sv, sdf_jacobian_fn=_local_level_sdf_jacobian, **kwargs
    )


def _local_linear_trend_sdf(lam: np.ndarray, theta: np.ndarray) -> np.ndarray:
    sigma1, sigma2, sigma3 = abs(theta[0]), abs(theta[1]), abs(theta[2])
    w = 2 * (1 - np.cos(lam))
    return (w**2 * sigma1**2 + w * sigma2**2 + sigma3**2) / (2 * np.pi)


def _local_linear_trend_sdf_jacobian(lam: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # NOTE: same 1/(2*pi) correction as _local_level_sdf_jacobian -- see
    # that function's docstring/comment for the derivation.
    sigma1, sigma2, sigma3 = abs(theta[0]), abs(theta[1]), abs(theta[2])
    w = 2 * (1 - np.cos(lam))
    j1 = 2 * w**2 * sigma1 / (2 * np.pi)
    j2 = 2 * w * sigma2 / (2 * np.pi)
    j3 = 2 * sigma3 * np.ones_like(lam) / (2 * np.pi)
    return np.column_stack([j1, j2, j3])


def whittle_local_linear_trend(y: np.ndarray, sv: np.ndarray, **kwargs) -> WhittleEstimationResult:
    """Fit a local-linear-trend model via Whittle estimation of
    (sigma_epsilon, sigma_eta, sigma_zeta) on the twice-differenced series.

    Original: ects/whittle_local_linear_trend.m
    """
    sv = np.asarray(sv, dtype=float).flatten()
    if sv.shape[0] != 3:
        raise ValueError("whittle_local_linear_trend: sv must have length 3")

    y = np.asarray(y, dtype=float).flatten()
    dy = y[1:] - y[:-1]
    dy2 = dy[1:] - dy[:-1]
    dy2 = dy2[~np.isnan(dy2)]

    return whittle_estimation(
        dy2, _local_linear_trend_sdf, sv, sdf_jacobian_fn=_local_linear_trend_sdf_jacobian, **kwargs
    )
