"""Quantile regression (via linear programming) and quantile-regression copulas.

Ported from QuantToolbox/stats/{quantile_regression,qrCopulaNormal,
qrCopulaStudent}.m

Translation notes:

- MATLAB's ``linprog`` (Optimization Toolbox, interior-point) maps
  directly onto ``scipy.optimize.linprog`` (also supports an
  interior-point method) -- both are standard-library-adjacent linear
  programming solvers, so no extra dependency is needed.
- The original solves each quantile tau independently in a loop; this is
  preserved here (rather than vectorizing across tau) since each is an
  independent LP.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from quanttoolbox.stats.distributions import normal_cdf, normal_ppf, student_t_cdf, student_t_ppf


def quantile_regression(
    y: np.ndarray, x: np.ndarray, tau: float | np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear quantile regression at one or more quantile levels tau, solved
    via linear programming (Koenker & Bassett's LP formulation).

    Original: stats/quantile_regression.m

    Returns
    -------
    beta : (n_vars, n_tau) coefficients (or (n_vars,) if scalar tau).
    u : (n_obs, n_tau) positive residual parts.
    v : (n_obs, n_tau) negative residual parts.
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n, m = x.shape
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
    p = tau_arr.shape[0]

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    y_w = y * w
    x_w = x * w[:, None]

    a_eq = np.hstack([x_w, np.eye(n), -np.eye(n)])
    b_eq = y_w
    bounds = [(None, None)] * m + [(0, None)] * n + [(0, None)] * n

    beta = np.full((m, p), np.nan)
    u = np.full((n, p), np.nan)
    v = np.full((n, p), np.nan)

    for i, t in enumerate(tau_arr):
        c = np.concatenate([np.zeros(m), t * np.ones(n), (1 - t) * np.ones(n)])
        result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if result.success:
            z = result.x
            beta[:, i] = z[:m]
            u[:, i] = z[m : m + n]
            v[:, i] = z[m + n :]

    if p == 1:
        return beta[:, 0], u[:, 0], v[:, 0]
    return beta, u, v


def qr_copula_normal(
    u1: float | np.ndarray, rho: float, alpha: float | np.ndarray
) -> float | np.ndarray:
    """Conditional quantile u2 = Q(alpha | u1) implied by a bivariate Gaussian
    copula with correlation rho.

    Original: stats/qrCopulaNormal.m
    """
    return normal_cdf(rho * normal_ppf(u1) + np.sqrt(1 - rho**2) * normal_ppf(alpha))


def qr_copula_student(
    u1: float | np.ndarray, rho: float, nu: float, alpha: float | np.ndarray
) -> float | np.ndarray:
    """Conditional quantile u2 = Q(alpha | u1) implied by a bivariate Student-t
    copula with correlation rho and nu degrees of freedom.

    Original: stats/qrCopulaStudent.m
    """
    t1 = student_t_ppf(u1, nu)
    t2 = rho * t1 + np.sqrt((1 - rho**2) * (nu + t1**2) / (1 + nu)) * student_t_ppf(alpha, nu + 1)
    return student_t_cdf(t2, nu)
