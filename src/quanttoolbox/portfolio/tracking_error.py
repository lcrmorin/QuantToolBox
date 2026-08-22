"""Tracking-error-optimized portfolio construction relative to a benchmark.

Ported from QuantToolBox/rpb/{compute_te_portfolio,
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
from scipy.optimize import linprog, minimize

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


@dataclass
class MixedNormTEMetrics:
    active_share: float
    md: float
    dts: float
    r_as: float
    r_md: float
    r_dts: float
    ci: float
    reduction: float
    r_mix: float


@dataclass
class MixedNormTEResult:
    weights_slsqp: np.ndarray
    weights_lp: np.ndarray
    metrics_slsqp: MixedNormTEMetrics
    metrics_lp: MixedNormTEMetrics
    metrics_benchmark: MixedNormTEMetrics
    max_abs_diff: float


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


def solve_mixed_norm_te_portfolio(
    x_benchmark: np.ndarray,
    carbon_intensity: np.ndarray,
    md: np.ndarray,
    dts: np.ndarray,
    sector: np.ndarray,
    reduction: float,
    varphi_as: float,
    varphi_md: float,
    varphi_dts: float,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    x0: np.ndarray | None = None,
) -> MixedNormTEResult:
    """L1 mixed active-share/MD/DTS bond tracking-error objective under a
    budget constraint and a carbon-intensity-reduction target, solved two
    ways: direct nonsmooth minimization (SLSQP) and an epigraph LP
    reformulation (``linprog``).

    Minimizes ``R_Mix(w) = varphi_as*R_AS(w) + varphi_md*R_MD(w) +
    varphi_dts*R_DTS(w)`` where ``R_AS(w) = 0.5*||w-x_benchmark||_1``,
    ``R_MD(w) = ||C_MD@(w-x_benchmark)||_1`` and ``R_DTS(w) =
    ||C_DTS@(w-x_benchmark)||_1`` for the per-sector aggregation matrices
    ``C_MD``/``C_DTS`` built from ``sector``, subject to
    ``sum(w) == 1`` and ``carbon_intensity@w <= (1-reduction) *
    carbon_intensity@x_benchmark``. Because ``C_MD``/``C_DTS`` are not
    diagonal, these L1 terms are not expressible via ``solve_qp``'s
    elementwise ``lasso_penalty`` and need this dedicated two-path
    implementation instead.

    Not ported from the MATLAB HSF toolbox -- no .m file in hfs-archive
    implements this. Promoted from HSF-Notebooks chapter 11b/16f.
    """
    x_benchmark = np.asarray(x_benchmark, dtype=float).flatten()
    carbon_intensity = np.asarray(carbon_intensity, dtype=float).flatten()
    md = np.asarray(md, dtype=float).flatten()
    dts = np.asarray(dts, dtype=float).flatten()
    sector = np.asarray(sector)
    n = len(x_benchmark)
    sectors = np.unique(sector)
    n_sectors = len(sectors)
    s_ji = (sector[None, :] == sectors[:, None]).astype(float)  # (n_sectors, n)
    md_star = (s_ji * md) @ x_benchmark  # per-sector MD@benchmark
    dts_star = (s_ji * dts) @ x_benchmark
    ci_b = carbon_intensity @ x_benchmark

    def r_as(w: np.ndarray) -> float:
        return 0.5 * np.abs(w - x_benchmark).sum()

    def r_md(w: np.ndarray) -> float:
        return np.abs((s_ji * md) @ (w - x_benchmark)).sum()

    def r_dts(w: np.ndarray) -> float:
        return np.abs((s_ji * dts) @ (w - x_benchmark)).sum()

    def r_mix(w: np.ndarray) -> float:
        return varphi_as * r_as(w) + varphi_md * r_md(w) + varphi_dts * r_dts(w)

    if lb is None:
        lb_arr = np.zeros(n)
    elif np.isscalar(lb):
        lb_arr = np.full(n, float(lb))
    else:
        lb_arr = np.asarray(lb, dtype=float)
    if ub is None:
        ub_arr = np.ones(n)
    elif np.isscalar(ub):
        ub_arr = np.full(n, float(ub))
    else:
        ub_arr = np.asarray(ub, dtype=float)
    x0_arr = x_benchmark.copy() if x0 is None else np.asarray(x0, dtype=float).flatten()

    # -- direct nonsmooth minimization --
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: (1 - reduction) * ci_b - carbon_intensity @ w},
    ]
    res = minimize(
        r_mix,
        x0_arr,
        method="SLSQP",
        bounds=list(zip(lb_arr, ub_arr, strict=True)),
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    w_slsqp = res.x

    # -- epigraph LP reformulation --
    # variables: [w (n), tau_w (n), tau_md (n_sectors), tau_dts (n_sectors)]
    c = np.concatenate(
        [
            np.zeros(n),
            0.5 * varphi_as * np.ones(n),
            varphi_md * np.ones(n_sectors),
            varphi_dts * np.ones(n_sectors),
        ]
    )
    i_n, i_ns = np.eye(n), np.eye(n_sectors)
    z1, z2, z3 = (
        np.zeros((n, n_sectors)),
        np.zeros((n_sectors, n)),
        np.zeros((n_sectors, n_sectors)),
    )
    c_md = s_ji * md
    c_dts = s_ji * dts
    a_ub = np.block(
        [
            [i_n, -i_n, z1, z1],
            [-i_n, -i_n, z1, z1],
            [c_md, z2, -i_ns, z3],
            [-c_md, z2, -i_ns, z3],
            [c_dts, z2, z3, -i_ns],
            [-c_dts, z2, z3, -i_ns],
        ]
    )
    b_ub_lp = np.concatenate([x_benchmark, -x_benchmark, md_star, -md_star, dts_star, -dts_star])
    a_ub = np.vstack([a_ub, np.concatenate([carbon_intensity, np.zeros(n + 2 * n_sectors)])])
    b_ub_lp = np.concatenate([b_ub_lp, [(1 - reduction) * ci_b]])
    a_eq = np.concatenate([np.ones(n), np.zeros(n + 2 * n_sectors)]).reshape(1, -1)
    b_eq = np.array([1.0])
    bounds_lp = list(zip(lb_arr, ub_arr, strict=True)) + [(0, 1e5)] * (n + 2 * n_sectors)

    res_lp = linprog(
        c, A_ub=a_ub, b_ub=b_ub_lp, A_eq=a_eq, b_eq=b_eq, bounds=bounds_lp, method="highs"
    )
    w_lp = res_lp.x[:n]

    def metrics(w: np.ndarray) -> MixedNormTEMetrics:
        ci_w = w @ carbon_intensity
        return MixedNormTEMetrics(
            active_share=r_as(w),
            md=float(w @ md),
            dts=float(w @ dts),
            r_as=r_as(w),
            r_md=r_md(w),
            r_dts=r_dts(w),
            ci=float(ci_w),
            reduction=float(1 - ci_w / ci_b),
            r_mix=r_mix(w),
        )

    return MixedNormTEResult(
        weights_slsqp=w_slsqp,
        weights_lp=w_lp,
        metrics_slsqp=metrics(w_slsqp),
        metrics_lp=metrics(w_lp),
        metrics_benchmark=metrics(x_benchmark),
        max_abs_diff=float(np.max(np.abs(w_slsqp - w_lp))),
    )


def _prune_redundant_rows(
    a: np.ndarray, b: np.ndarray, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Drop equality-constraint rows that are linearly dependent on
    earlier ones -- SLSQP can fail to converge cleanly on rank-deficient
    equality constraints.

    Original: HSF-Notebooks chapter 11h's local ``_prune_redundant_rows``
    helper.
    """
    keep: list[int] = []
    rows: list[np.ndarray] = []
    for i in range(a.shape[0]):
        cand = np.vstack(rows + [a[i]]) if rows else a[i : i + 1]
        if np.linalg.matrix_rank(cand, tol=tol) > len(rows):
            rows.append(a[i])
            keep.append(i)
    return a[keep], b[keep]


def _minimum_te_portfolio_nearest(
    x_benchmark: np.ndarray,
    cov_matrix: np.ndarray,
    a_eq: np.ndarray | None,
    b_eq: np.ndarray | None,
    c_ineq: np.ndarray | None,
    d_ineq: np.ndarray | None,
    lb: np.ndarray | float | None,
    ub: np.ndarray | float | None,
) -> TrackingErrorResult:
    """SLSQP fallback for ``minimum_te_portfolio(..., on_infeasible="nearest")``
    -- replicates HSF-Notebooks chapter 11h's local ``min_te_portfolio``
    (SLSQP settles at the nearest feasible point on an infeasible input
    instead of raising).
    """
    n = x_benchmark.shape[0]

    if a_eq is None:
        a_eq_arr = np.ones((1, n))
        b_eq_arr = np.array([1.0])
    else:
        a_eq_arr = np.asarray(a_eq, dtype=float)
        b_eq_arr = np.asarray(b_eq, dtype=float).flatten()
    a_eq_arr, b_eq_arr = _prune_redundant_rows(a_eq_arr, b_eq_arr)

    if lb is None:
        lb_arr = np.zeros(n)
    elif np.isscalar(lb):
        lb_arr = np.full(n, float(lb))
    else:
        lb_arr = np.asarray(lb, dtype=float)
    if ub is None:
        ub_arr = np.ones(n)
    elif np.isscalar(ub):
        ub_arr = np.full(n, float(ub))
    else:
        ub_arr = np.asarray(ub, dtype=float)

    def obj(w: np.ndarray) -> float:
        d = w - x_benchmark
        return float(0.5 * d @ cov_matrix @ d)

    def obj_grad(w: np.ndarray) -> np.ndarray:
        return cov_matrix @ (w - x_benchmark)

    constraints = [{"type": "eq", "fun": lambda w, a=a_eq_arr, c=b_eq_arr: a @ w - c}]
    if c_ineq is not None:
        c_ineq_arr = np.asarray(c_ineq, dtype=float)
        d_ineq_arr = np.asarray(d_ineq, dtype=float).flatten()
        constraints.append({"type": "ineq", "fun": lambda w, a=c_ineq_arr, c=d_ineq_arr: c - a @ w})

    res = minimize(
        obj,
        x_benchmark.copy(),
        jac=obj_grad,
        method="SLSQP",
        bounds=list(zip(lb_arr, ub_arr, strict=True)),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    x = res.x
    x = x * (np.abs(x) >= 1e-10)

    active = x - x_benchmark
    return TrackingErrorResult(
        weights=x,
        active_return=0.0,
        tracking_error=float(np.sqrt(max(active @ cov_matrix @ active, 0.0))),
    )


def minimum_te_portfolio(
    x_benchmark: np.ndarray,
    cov_matrix: np.ndarray,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    on_infeasible: str = "raise",
) -> TrackingErrorResult:
    """Minimum-tracking-error portfolio (te_portfolio with gamma=0, no
    return tilt).

    Original: rpb/compute_minimum_te_portfolio.m

    on_infeasible : "raise" (default) lets ``solve_qp``'s ``RuntimeError``
        on infeasibility/non-convergence propagate unchanged -- existing
        behavior. "nearest" catches that ``RuntimeError`` and instead
        solves the same tracking-error QP via ``scipy.optimize.minimize``
        (SLSQP), starting from the benchmark weights and using the same
        equality/inequality/box-constraint handling (including pruning
        rank-deficient equality rows, and defaulting missing lb/ub to
        [0, 1]) as HSF-Notebooks chapter 11h's local ``min_te_portfolio``,
        returning the "nearest" near-feasible point SLSQP settles at
        instead of raising. Not ported from the MATLAB HSF toolbox -- no
        .m file in hfs-archive implements this fallback. Promoted from
        HSF-Notebooks chapter 11h.
    """
    if on_infeasible not in ("raise", "nearest"):
        raise ValueError('on_infeasible must be "raise" or "nearest"')

    x_benchmark = np.asarray(x_benchmark, dtype=float).flatten()
    n = x_benchmark.shape[0]
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    try:
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
    except RuntimeError:
        if on_infeasible != "nearest":
            raise
        return _minimum_te_portfolio_nearest(
            x_benchmark, cov_matrix, a_eq, b_eq, c_ineq, d_ineq, lb, ub
        )
