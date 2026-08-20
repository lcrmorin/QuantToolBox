"""Lasso and elastic-net regression via coordinate descent / ADMM.

Ported from QuantToolbox/stats/{regLassoCCD,regLassoADMM,regLassoADMM2,
regLasso,regElasticNet,selectLasso}.m

Translation notes:

- The original MATLAB toolbox has *two* Lasso parameterizations that are
  easy to conflate:
    1. **Penalized form** (``regLassoCCD.m``, ``regLassoADMM.m``):
       minimize ||y - X*beta||^2 + lambda * ||beta||_1. This is what
       coordinate descent and ADMM solve directly, and what
       ``scikit-learn``'s own ``Lasso`` (also coordinate-descent-based)
       solves too.
    2. **Constrained/budget form** (``regLasso.m``, ``selectLasso.m``,
       via ``quadprog``): minimize ||y - X*beta||^2 subject to
       ||beta||_1 <= tau.
  These are Lagrangian duals of each other but need different solvers.
  This module ports the penalized form directly (``lasso_ccd``,
  ``lasso_admm``) using plain NumPy coordinate descent -- the same
  algorithm scikit-learn uses internally, so no extra dependency is
  needed. ``lasso_tau_constrained`` recovers the budget-form behavior of
  ``regLasso.m``/``selectLasso.m`` by bisecting on lambda until the
  resulting L1 norm matches the target tau, rather than porting the
  original's ``quadprog``-based QP formulation (MATLAB Optimization
  Toolbox has no free-standing Python equivalent worth adding as a
  dependency just for this).
- ``regElasticNet.m``'s alpha/lambda parameterization
  (``lambda*(1-alpha)*L2 + lambda*alpha*L1``) is preserved in
  ``elastic_net_ccd`` for call-site compatibility, rather than switching
  to scikit-learn's ``l1_ratio`` naming.
- MATLAB's ``global ADMM_*`` settings blocks are replaced by the
  ``quanttoolbox.config.ADMMConfig`` dataclass.
"""

from __future__ import annotations

import numpy as np

from quanttoolbox.config import ADMMConfig


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise soft-thresholding operator: sign(v) * max(|v| - threshold, 0).

    Original: optim/soft_thresholding.m (used by regLassoADMM.m)
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def lasso_ccd(
    y: np.ndarray,
    x: np.ndarray,
    lambda_: float,
    beta_init: np.ndarray | None = None,
    n_iters: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Lasso via cyclical coordinate descent (penalized form).

    Original: stats/regLassoCCD.m
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n, p = x.shape

    beta = (
        np.linalg.inv(x.T @ x) @ (x.T @ y)
        if beta_init is None or np.asarray(beta_init).shape[0] != p
        else np.asarray(beta_init, dtype=float).copy()
    )

    beta_path = np.zeros((p, n_iters))
    for it in range(n_iters):
        beta_path[:, it] = beta
        for j in range(p):
            x_j = x[:, j]
            x_minus_j = x.copy()
            x_minus_j[:, j] = 0.0
            v = x_j @ (y - x_minus_j @ beta)
            denom = x_j @ x_j
            if lambda_ > 0:
                beta[j] = np.sign(v) * max(abs(v) - lambda_, 0.0) / denom
            else:
                beta[j] = v / denom

    return beta, beta_path.T


def lasso_admm(
    y: np.ndarray,
    x: np.ndarray,
    lambda_: float,
    beta_init: np.ndarray | None = None,
    config: ADMMConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Lasso via ADMM (penalized form) with a fixed step-size (varphi).

    Note: the original MATLAB also supported an adaptive-varphi mode
    (``ADMM_varphi_mtd == 2``, residual-balancing per Boyd et al. 2011);
    that branch is not ported here -- only the fixed-varphi path, which
    is what the default config uses anyway. Add adaptive step-size
    updates here if convergence speed on your problem needs it.

    Original: stats/regLassoADMM.m

    Returns
    -------
    beta : final coefficient estimate.
    beta_path : (n_iters_run, n_vars) iterate history.
    converged : whether the convergence tolerance was reached.
    n_iters_run : number of iterations actually performed.
    """
    if config is None:
        config = ADMMConfig()

    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n, p = x.shape

    xx = x.T @ x
    xy = x.T @ y
    identity = np.eye(p)

    beta = (
        np.linalg.inv(xx) @ xy
        if beta_init is None or np.asarray(beta_init).shape[0] != p
        else np.asarray(beta_init, dtype=float).copy()
    )

    varphi = config.varphi
    beta_prev = beta.copy()
    beta_bar = beta.copy()
    beta_bar_prev = beta_bar.copy()
    u = np.zeros(p)

    beta_path = np.full((config.max_iters, p), np.nan)
    converged = False
    n_iters_run = config.max_iters

    for it in range(config.max_iters):
        beta_path[it] = beta

        v = beta_bar - u
        beta = np.linalg.inv(xx + varphi * identity) @ (xy + varphi * v)

        v = beta + u
        beta_bar = soft_threshold(v, lambda_ / varphi)

        r = beta - beta_bar
        u = u + r

        cvg1 = np.sum((beta - beta_prev) ** 2)
        cvg2 = np.sum(r**2)
        cvg3 = np.sum((beta_bar - beta_bar_prev) ** 2)
        cvg = max(cvg1, cvg2, cvg3)
        if cvg <= config.tol:
            converged = True
            n_iters_run = it + 1
            break

        beta_prev = beta.copy()
        beta_bar_prev = beta_bar.copy()

    return beta, beta_path[:n_iters_run], converged, n_iters_run


def lasso_tau_constrained(
    y: np.ndarray,
    x: np.ndarray,
    tau: float,
    lambda_search: np.ndarray | None = None,
    n_iters: int = 200,
) -> tuple[np.ndarray, float, int]:
    """Lasso in budget-constrained form: minimize ||y - X*beta||^2 subject to
    ||beta||_1 <= tau, found via bisection on the penalized-form lambda.

    Reproduces the tau-based interface of stats/regLasso.m and
    stats/selectLasso.m without needing a QP solver.

    Returns
    -------
    beta : coefficients whose L1 norm is closest to tau.
    lambda_matched : the lambda value that achieved it.
    df : number of non-zero coefficients.
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)

    beta_ols = np.linalg.inv(x.T @ x) @ (x.T @ y)
    if np.sum(np.abs(beta_ols)) <= tau:
        # unconstrained OLS already satisfies the budget
        df = int(np.sum(np.abs(beta_ols) >= 1e-10))
        return beta_ols, 0.0, df

    lo, hi = 0.0, np.max(np.abs(x.T @ y)) * 2  # hi large enough to drive beta to 0
    beta = beta_ols
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        beta, _ = lasso_ccd(y, x, mid, beta_init=beta.copy(), n_iters=50)
        l1_norm = np.sum(np.abs(beta))
        if l1_norm > tau:
            lo = mid
        else:
            hi = mid
        if abs(l1_norm - tau) < 1e-6:
            break

    df = int(np.sum(np.abs(beta) >= 1e-10))
    return beta, mid, df


def elastic_net_ccd(
    y: np.ndarray,
    x: np.ndarray,
    lambda_: float,
    alpha: float,
    beta_init: np.ndarray | None = None,
    n_iters: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic net via cyclical coordinate descent:
    minimize ||y - X*beta||^2 + lambda*(1-alpha)*||beta||_2^2 + lambda*alpha*||beta||_1.

    Original: stats/regElasticNet.m (ported as CCD rather than the
    original's quadprog QP formulation -- see module docstring.)
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n, p = x.shape

    beta = (
        np.linalg.inv(x.T @ x) @ (x.T @ y)
        if beta_init is None or np.asarray(beta_init).shape[0] != p
        else np.asarray(beta_init, dtype=float).copy()
    )

    l1_penalty = lambda_ * alpha
    l2_penalty = lambda_ * (1 - alpha)

    beta_path = np.zeros((p, n_iters))
    for it in range(n_iters):
        beta_path[:, it] = beta
        for j in range(p):
            x_j = x[:, j]
            x_minus_j = x.copy()
            x_minus_j[:, j] = 0.0
            v = x_j @ (y - x_minus_j @ beta)
            denom = x_j @ x_j + l2_penalty
            beta[j] = np.sign(v) * max(abs(v) - l1_penalty, 0.0) / denom

    return beta, beta_path.T
