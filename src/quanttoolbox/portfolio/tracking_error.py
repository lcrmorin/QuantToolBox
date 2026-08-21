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
  gamma values, ``problem=0``). ``te_target_portfolio`` covers the
  "mu-problem"/"sigma-problem" target-matching modes (``problem=1``/
  ``problem=2``), mirroring ``compute_te_portfolio.m`` exactly the same
  way ``mean_variance.mvo_target_portfolio`` mirrors
  ``compute_mvo_portfolio.m`` -- see that function's docstring for the
  bisection-bracket/boundary-check quirk both share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanttoolbox.optim.bisection import bisection
from quanttoolbox.optim.quadprog import solve_qp


@dataclass
class TrackingErrorResult:
    weights: np.ndarray
    active_return: float
    tracking_error: float


@dataclass
class TETargetResult(TrackingErrorResult):
    gamma: float


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


def te_target_portfolio(
    x_benchmark: np.ndarray,
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
) -> list[TETargetResult]:
    """Target-matching tracking-error portfolios: for each value in
    ``targets``, find the risk-aversion gamma whose ``te_portfolio``
    solution achieves that target active return (``problem="mu"``) or
    that target tracking error (``problem="sigma"``), via bisection on
    gamma.

    Original: rpb/compute_te_portfolio.m (mu-problem/problem=1 and
    sigma-problem/problem=2 branches), via
    compute_te_portfolio_return.m/compute_te_portfolio_volatility.m as the
    bisection objective.

    Same boundary-case and bisection-bracket structure as
    ``mean_variance.mvo_target_portfolio`` -- see that function's
    docstring for the full description of the achievability checks and
    the gamma_max/gamma_bracket quirk they share.
    """
    if problem not in ("mu", "sigma"):
        raise ValueError('problem must be "mu" or "sigma"')

    x_benchmark = np.asarray(x_benchmark, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    kwargs = dict(a_eq=a_eq, b_eq=b_eq, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb, ub=ub)

    r_min = te_portfolio(x_benchmark, mu, cov_matrix, gamma=0.0, **kwargs)
    r_max = te_portfolio(x_benchmark, mu, cov_matrix, gamma=gamma_max, **kwargs)

    def nan_result() -> TETargetResult:
        return TETargetResult(
            weights=np.full_like(x_benchmark, np.nan),
            active_return=np.nan,
            tracking_error=np.nan,
            gamma=np.nan,
        )

    def as_target_result(r: TrackingErrorResult, gamma: float) -> TETargetResult:
        return TETargetResult(
            weights=r.weights,
            active_return=r.active_return,
            tracking_error=r.tracking_error,
            gamma=gamma,
        )

    def achieved(gamma: float) -> float:
        r = te_portfolio(x_benchmark, mu, cov_matrix, gamma=float(gamma), **kwargs)
        return r.active_return if problem == "mu" else r.tracking_error

    results = []
    for target in np.atleast_1d(np.asarray(targets, dtype=float)):
        target = float(target)

        if problem == "sigma":
            if target < r_min.tracking_error:
                results.append(nan_result())
                continue
            if target == r_min.tracking_error:
                results.append(as_target_result(r_min, 0.0))
                continue
            if target >= r_max.tracking_error:
                results.append(as_target_result(r_max, np.inf))
                continue
        else:  # mu-problem
            if target <= r_min.active_return:
                results.append(as_target_result(r_min, 0.0))
                continue
            if target > r_max.active_return:
                results.append(nan_result())
                continue
            if target == r_max.active_return:
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
                    te_portfolio(x_benchmark, mu, cov_matrix, gamma=float(gamma_star), **kwargs),
                    gamma_star,
                )
            )

    return results


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
