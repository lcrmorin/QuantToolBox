"""Ridge regression: fixed-lambda and tau-targeted (L2-budget) variants.

Ported from QuantToolBox/stats/{regRidge,regRidge2}.m

Translation notes:

- Ridge regression's closed-form solution is a simple 3-line NumPy
  expression, so no external library (not even scikit-learn) is needed
  here -- ``numpy.linalg`` is sufficient and keeps this dependency-free.
- ``ridge_tau_targeted`` (the original ``regRidge2.m``) reproduces the
  original's grid-search approach (scan a dense lambda grid, pick the
  closest match for the desired L2-norm budget tau) rather than solving
  for lambda analytically, to preserve exact behavioral parity.
"""

from __future__ import annotations

import numpy as np


def ridge(
    y: np.ndarray, x: np.ndarray, lambda_: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge regression at one or more penalty values lambda.

    Original: stats/regRidge.m

    Returns
    -------
    beta : (n_lambda, n_vars) array of coefficients (or (n_vars,) if scalar lambda_).
    df : effective degrees of freedom at each lambda.
    complexity : 1 / df.
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    lambdas = np.atleast_1d(np.asarray(lambda_, dtype=float))

    xx = x.T @ x
    xy = x.T @ y
    m = x.shape[1]
    identity = np.eye(m)

    n_lambda = lambdas.shape[0]
    beta = np.zeros((m, n_lambda))
    df = np.zeros(n_lambda)
    for i, lam in enumerate(lambdas):
        xx_inv = np.linalg.inv(xx + lam * identity)
        beta[:, i] = xx_inv @ xy
        df[i] = np.sum(np.diag(xx_inv @ xx))

    complexity = 1.0 / df
    if n_lambda == 1:
        return beta[:, 0], df[0], complexity[0]
    return beta.T, df, complexity


def ridge_tau_targeted(
    y: np.ndarray,
    x: np.ndarray,
    tau: np.ndarray | float,
    lambda_search: np.ndarray | None = None,
    relative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ridge regression targeting a desired sum-of-squared-coefficients
    budget tau, found by grid search over a candidate lambda range.

    Original: stats/regRidge2.m

    Returns
    -------
    beta : (n_tau, n_vars) array of coefficients (or (n_vars,) if scalar tau).
    lambda_out : the lambda achieving the closest match to each tau.
    df : effective degrees of freedom at each match.
    complexity : 1 / df.
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)

    xx = x.T @ x
    xy = x.T @ y
    m = x.shape[1]
    identity = np.eye(m)

    all_lambda = np.arange(0, 100, 0.01) if lambda_search is None else np.atleast_1d(lambda_search)

    if relative:
        beta_ols = np.linalg.inv(xx) @ xy
        tau_ols = np.sum(beta_ols**2)
    else:
        tau_ols = 1.0

    n_lambda = all_lambda.shape[0]
    all_beta = np.zeros((m, n_lambda))
    all_df = np.zeros(n_lambda)
    for i, lam in enumerate(all_lambda):
        xx_inv = np.linalg.inv(xx + lam * identity)
        all_beta[:, i] = xx_inv @ xy
        all_df[i] = np.sum(np.diag(xx_inv @ xx))

    all_tau = np.sum(all_beta**2, axis=0) / tau_ols

    tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
    n_tau = tau_arr.shape[0]
    beta = np.zeros((m, n_tau))
    df = np.zeros(n_tau)
    lambda_out = np.zeros(n_tau)
    for i, t in enumerate(tau_arr):
        idx = int(np.argmin(np.abs(all_tau - t)))
        beta[:, i] = all_beta[:, idx]
        df[i] = all_df[idx]
        lambda_out[i] = all_lambda[idx]

    complexity = 1.0 / df
    if n_tau == 1:
        return beta[:, 0], lambda_out[0], df[0], complexity[0]
    return beta.T, lambda_out, df, complexity
