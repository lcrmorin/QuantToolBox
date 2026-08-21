"""Mean-variance, minimum-variance, and most-diversified portfolio construction.

Ported from QuantToolBox/rpb/{compute_mvo_portfolio,compute_minvar_portfolio,
compute_mdp_portfolio,compute_mdp_objective_function}.m and
QuantToolBox/mloapa/{compute_MDP_ADMM,compute_MinVar_ADMM1,
compute_MinVar_ADMM2}.m.

Consolidation notes:

- ``mvo_portfolio``/``minvar_portfolio`` route through
  ``quanttoolbox.optim.quadprog.solve_qp`` (see that module's docstring for
  why this eliminates the original's separate quadprog_lasso/ridge/etc.
  variable-splitting machinery); ``minvar_portfolio`` is literally
  ``mvo_portfolio`` with mu=0.
- ``mvo_frontier`` evaluates at a list of risk-aversion (gamma) values,
  covering the original's "gamma-problem" mode (``problem=0``).
  ``mvo_target_portfolio`` covers the "mu-problem"/"sigma-problem"
  target-matching modes (``problem=1``/``problem=2``): for each target
  expected-return or target volatility, it bisects on gamma (via
  ``quanttoolbox.optim.bisection.bisection``) until ``mvo_portfolio``'s
  achieved return/volatility hits that target, exactly mirroring
  ``compute_mvo_portfolio.m``'s three-branch structure (boundary cases at
  gamma=0/gamma=100, bisection in between over gamma in [0, 10] by
  default -- both bounds preserved as separate, independently
  configurable parameters, matching a real quirk in the original: the
  *achievability* check against the "infinite risk aversion" case uses
  gamma=100, but the interior bisection search is only ever bracketed to
  [0, 10], so a target only reachable at a gamma between 10 and 100 will
  come back as unreachable (NaN), exactly as the MATLAB source does. See
  ``docs/matlab_bugs_found.md`` for a related, genuine bug this surfaced
  in one of the original's own example scripts.)
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

from quanttoolbox.optim.bisection import bisection
from quanttoolbox.optim.quadprog import solve_qp


@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float


@dataclass
class MVOTargetResult(PortfolioResult):
    gamma: float


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

    See ``mvo_target_portfolio`` for the mu-problem/sigma-problem
    target-matching modes.
    """
    return [
        mvo_portfolio(mu, cov_matrix, gamma=float(g), **kwargs) for g in np.atleast_1d(gamma_values)
    ]


def mvo_target_portfolio(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    targets: np.ndarray | float,
    problem: str = "sigma",
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    gamma_bracket: tuple[float, float] = (0.0, 10.0),
    gamma_max: float = 100.0,
) -> list[MVOTargetResult]:
    """Target-matching mean-variance portfolios: for each value in
    ``targets``, find the risk-aversion gamma whose ``mvo_portfolio``
    solution achieves that target expected return (``problem="mu"``) or
    that target volatility (``problem="sigma"``), via bisection on gamma.

    Original: rpb/compute_mvo_portfolio.m (mu-problem/problem=1 and
    sigma-problem/problem=2 branches), via
    compute_mvo_portfolio_return.m/compute_mvo_portfolio_volatility.m as
    the bisection objective.

    For each target, first the gamma=0 and gamma=``gamma_max`` solutions
    are used to bracket what's achievable:

    - ``problem="sigma"``: a target below the gamma=0 volatility is
      unreachable (NaN weights); a target at or above the gamma=gamma_max
      volatility returns that portfolio directly; otherwise gamma is
      bisected within ``gamma_bracket``.
    - ``problem="mu"``: a target at or below the gamma=0 return returns
      that portfolio directly; a target above the gamma=gamma_max return
      is unreachable (NaN weights); a target exactly at the gamma=gamma_max
      return returns that portfolio directly; otherwise gamma is bisected
      within ``gamma_bracket``.

    Note the original's own quirk, preserved here: the boundary checks use
    ``gamma_max`` (100 by default) but the bisection search itself is only
    ever bracketed to ``gamma_bracket`` ((0, 10) by default) -- so a target
    only reachable at a gamma between the two is misreported as
    unreachable. See the module docstring and
    ``docs/matlab_bugs_found.md`` for more.
    """
    if problem not in ("mu", "sigma"):
        raise ValueError('problem must be "mu" or "sigma"')

    mu = np.asarray(mu, dtype=float).flatten()
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    kwargs = dict(a_eq=a_eq, b_eq=b_eq, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb, ub=ub)

    r_min = mvo_portfolio(mu, cov_matrix, gamma=0.0, **kwargs)
    r_max = mvo_portfolio(mu, cov_matrix, gamma=gamma_max, **kwargs)

    def nan_result() -> MVOTargetResult:
        return MVOTargetResult(
            weights=np.full_like(mu, np.nan),
            expected_return=np.nan,
            volatility=np.nan,
            gamma=np.nan,
        )

    def as_target_result(r: PortfolioResult, gamma: float) -> MVOTargetResult:
        return MVOTargetResult(
            weights=r.weights,
            expected_return=r.expected_return,
            volatility=r.volatility,
            gamma=gamma,
        )

    def achieved(gamma: float) -> float:
        r = mvo_portfolio(mu, cov_matrix, gamma=float(gamma), **kwargs)
        return r.expected_return if problem == "mu" else r.volatility

    results = []
    for target in np.atleast_1d(np.asarray(targets, dtype=float)):
        target = float(target)

        if problem == "sigma":
            if target < r_min.volatility:
                results.append(nan_result())
                continue
            if target == r_min.volatility:
                results.append(as_target_result(r_min, 0.0))
                continue
            if target >= r_max.volatility:
                results.append(as_target_result(r_max, np.inf))
                continue
        else:  # mu-problem
            if target <= r_min.expected_return:
                results.append(as_target_result(r_min, 0.0))
                continue
            if target > r_max.expected_return:
                results.append(nan_result())
                continue
            if target == r_max.expected_return:
                results.append(as_target_result(r_max, np.inf))
                continue

        def objective(gamma: float, target: float = target) -> float:
            return achieved(gamma) - target

        gamma_star = bisection(objective, gamma_bracket[0], gamma_bracket[1])
        if np.isnan(gamma_star):
            results.append(nan_result())
        else:
            results.append(
                as_target_result(
                    mvo_portfolio(mu, cov_matrix, gamma=float(gamma_star), **kwargs), gamma_star
                )
            )

    return results


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
