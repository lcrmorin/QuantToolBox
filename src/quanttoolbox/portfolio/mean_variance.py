"""Mean-variance, minimum-variance, and most-diversified portfolio construction.

Ported from QuantToolbox/rpb/{compute_mvo_portfolio,compute_minvar_portfolio,
compute_mdp_portfolio,compute_mdp_objective_function}.m and
QuantToolbox/mloapa/{compute_MDP_ADMM,compute_MinVar_ADMM1,
compute_MinVar_ADMM2}.m.

Consolidation notes:

- ``mvo_portfolio``/``minvar_portfolio`` route through
  ``quanttoolbox.optim.quadprog.solve_qp`` (see that module's docstring for
  why this eliminates the original's separate quadprog_lasso/ridge/etc.
  variable-splitting machinery); ``minvar_portfolio`` is literally
  ``mvo_portfolio`` with mu=0.
- ``mvo_frontier`` evaluates at a list of risk-aversion (gamma) values,
  covering the original's "gamma-problem" mode. The "mu-problem"/
  "sigma-problem" target-matching bisection modes are not ported here
  (same scope decision as ``risk_budgeting.risk_budgeting_frontier`` --
  see that module's docstring); build target-matching on top of
  ``mvo_frontier`` with ``quanttoolbox.optim.bisection`` if needed.
- ``mdp_portfolio`` (most diversified portfolio) has a genuinely nonlinear
  objective (log(portfolio vol) - log(weighted-avg individual vol)), so it
  uses ``scipy.optimize.minimize`` (SLSQP) rather than ``solve_qp`` --
  matching the original's use of ``fmincon`` rather than ``quadprog``.
  ``mloapa/compute_MDP_ADMM.m`` (an alternative ADMM-based MDP solver) is
  not separately ported since SLSQP solves the same small nonlinear
  program directly and reliably.
- ``mloapa/compute_MinVar_ADMM{1,2}.m`` are alternative ADMM solvers for
  the same minimum-variance QP that ``solve_qp`` already solves directly;
  not separately ported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from quanttoolbox.optim.quadprog import solve_qp


@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float


def mvo_portfolio(
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
) -> PortfolioResult:
    """Mean-variance optimal portfolio: maximize gamma*mu'x - 0.5*x'Cov*x
    subject to a budget constraint (sum(x)=1 by default) and any other
    given constraints.

    gamma=0 gives the minimum-variance portfolio; larger gamma weights
    expected return more heavily relative to risk.

    Original: rpb/compute_mvo_portfolio.m (gamma-problem branch)
    """
    mu = np.asarray(mu, dtype=float).flatten()
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    x = solve_qp(
        cov_matrix,
        gamma * mu,
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
    return PortfolioResult(
        weights=x, expected_return=float(x @ mu), volatility=float(np.sqrt(x @ cov_matrix @ x))
    )


def mvo_frontier(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    gamma_values: np.ndarray,
    **kwargs,
) -> list[PortfolioResult]:
    """Evaluate the mean-variance frontier at each risk-aversion value in
    gamma_values (the "gamma-problem" mode of compute_mvo_portfolio.m).

    See module docstring for target-matching modes not ported here.
    """
    return [
        mvo_portfolio(mu, cov_matrix, gamma=float(g), **kwargs) for g in np.atleast_1d(gamma_values)
    ]


def minvar_portfolio(
    cov_matrix: np.ndarray,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
) -> PortfolioResult:
    """Minimum-variance portfolio (mean-variance optimal with gamma=0).

    Original: rpb/compute_minvar_portfolio.m
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = cov_matrix.shape[0]
    return mvo_portfolio(
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


@dataclass
class MDPResult:
    weights: np.ndarray
    volatility: float
    diversification_ratio: float
    converged: bool


def mdp_portfolio(
    cov_matrix: np.ndarray,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    x0: np.ndarray | None = None,
) -> MDPResult:
    """Most Diversified Portfolio: maximize the diversification ratio
    (weighted-average individual volatility / portfolio volatility), i.e.
    minimize log(portfolio vol) - log(weighted-average individual vol).

    Original: rpb/{compute_mdp_portfolio,compute_mdp_objective_function}.m
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = cov_matrix.shape[0]
    sigma = np.sqrt(np.diag(cov_matrix))

    a_eq = np.ones((1, n)) if a_eq is None else np.asarray(a_eq, dtype=float)
    b_eq = np.array([1.0]) if b_eq is None else np.asarray(b_eq, dtype=float)
    lb_arr = (
        np.full(n, -100.0)
        if lb is None
        else np.full(n, lb)
        if np.isscalar(lb)
        else np.asarray(lb, dtype=float)
    )
    ub_arr = (
        np.full(n, 100.0)
        if ub is None
        else np.full(n, ub)
        if np.isscalar(ub)
        else np.asarray(ub, dtype=float)
    )
    x0_arr = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float)

    def objective(x: np.ndarray) -> float:
        port_vol = np.sqrt(x @ cov_matrix @ x)
        weighted_avg_vol = x @ sigma
        return float(np.log(port_vol) - np.log(weighted_avg_vol))

    constraints = [LinearConstraint(a_eq, b_eq, b_eq)]
    if c_ineq is not None:
        constraints.append(
            LinearConstraint(
                np.asarray(c_ineq, dtype=float), -np.inf, np.asarray(d_ineq, dtype=float)
            )
        )

    result = minimize(
        objective,
        x0_arr,
        method="SLSQP",
        bounds=list(zip(lb_arr, ub_arr, strict=True)),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    x = result.x
    x = np.where(np.abs(x) < 1e-10, 0.0, x)
    sigma_x = float(np.sqrt(x @ cov_matrix @ x))
    dr_x = float((x @ sigma) / sigma_x)

    return MDPResult(
        weights=x, volatility=sigma_x, diversification_ratio=dr_x, converged=result.success
    )
