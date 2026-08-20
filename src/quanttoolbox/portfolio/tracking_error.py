"""Tracking-error-optimized portfolio construction relative to a benchmark.

Ported from QuantToolbox/rpb/{compute_te_portfolio,
compute_minimum_te_portfolio,compute_te_portfolio_mixed_norm}.m.

Consolidation notes:

- All three original functions solve the same underlying QP (minimize
  active variance minus a return-tilt term) via ``quadprog``; here they
  route through ``quanttoolbox.optim.quadprog.solve_qp``.
  ``te_portfolio_mixed_norm`` folds naturally into ``te_portfolio`` as
  optional ``ridge_penalty``/``lasso_penalty`` arguments (already
  supported directly by ``solve_qp``) rather than being a separate
  function.
- ``minimum_te_portfolio`` is ``te_portfolio`` with gamma=0 (pure
  tracking-error minimization, no return tilt).
- ``te_frontier`` covers the "gamma-problem" mode (evaluate at a list of
  gamma values); the "mu-problem"/"sigma-problem" target-matching modes
  are not ported, matching the same scope decision as
  ``risk_budgeting.risk_budgeting_frontier`` / ``mean_variance.mvo_frontier``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanttoolbox.optim.quadprog import solve_qp


@dataclass
class TrackingErrorResult:
    weights: np.ndarray
    active_return: float
    tracking_error: float


def te_portfolio(
    x_benchmark: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    gamma: float = 1.0,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    ridge_penalty: tuple[np.ndarray | float, np.ndarray] | None = None,
    lasso_penalty: tuple[np.ndarray | float, np.ndarray] | None = None,
) -> TrackingErrorResult:
    """Tracking-error-optimal portfolio: minimize 0.5*(x-x_b)'Cov*(x-x_b) -
    gamma*mu'x, i.e. trade off tracking error against active return.

    gamma=0 gives pure tracking-error minimization (no return tilt).

    Original: rpb/compute_te_portfolio.m (gamma-problem branch), also
    covers compute_te_portfolio_mixed_norm.m via ridge_penalty/lasso_penalty.
    """
    x_benchmark = np.asarray(x_benchmark, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    r_lin = gamma * mu + cov_matrix @ x_benchmark
    x = solve_qp(
        cov_matrix,
        r_lin,
        a_eq=a_eq,
        b_eq=b_eq,
        c_ineq=c_ineq,
        d_ineq=d_ineq,
        lb=lb,
        ub=ub,
        ridge_penalty=ridge_penalty,
        lasso_penalty=lasso_penalty,
        default_budget_constraint=a_eq is None,
    )

    active = x - x_benchmark
    return TrackingErrorResult(
        weights=x,
        active_return=float(active @ mu),
        tracking_error=float(np.sqrt(active @ cov_matrix @ active)),
    )


def te_frontier(
    x_benchmark: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    gamma_values: np.ndarray,
    **kwargs,
) -> list[TrackingErrorResult]:
    """Evaluate the tracking-error frontier at each risk-aversion value in
    gamma_values (the "gamma-problem" mode of compute_te_portfolio.m).
    """
    return [
        te_portfolio(x_benchmark, mu, cov_matrix, gamma=float(g), **kwargs)
        for g in np.atleast_1d(gamma_values)
    ]


def minimum_te_portfolio(
    x_benchmark: np.ndarray,
    cov_matrix: np.ndarray,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
) -> TrackingErrorResult:
    """Minimum-tracking-error portfolio (te_portfolio with gamma=0, no
    return tilt).

    Original: rpb/compute_minimum_te_portfolio.m
    """
    x_benchmark = np.asarray(x_benchmark, dtype=float).flatten()
    n = x_benchmark.shape[0]
    return te_portfolio(
        x_benchmark,
        np.zeros(n),
        cov_matrix,
        gamma=0.0,
        a_eq=a_eq,
        b_eq=b_eq,
        c_ineq=c_ineq,
        d_ineq=d_ineq,
        lb=lb,
        ub=ub,
    )
