"""Risk budgeting / risk parity portfolio construction.

Ported from QuantToolbox/rpb/{compute_risk_contribution,compute_rc_sd,
compute_rc_vol,compute_rb_sd,compute_rb_sd_ccd,compute_rb_sd_newton,
lagrange_rb_sd,compute_erc_portfolio}.m and
QuantToolbox/crb/compute_rb_sd_bc_admm1*.m (+ ~40 near-duplicate ADMM
variants), and QuantToolbox/mloapa/compute_ERC_{ADMM,CCD}.m.

Consolidation notes -- this is the single largest simplification in the
whole port:

The original has ~75 files across rpb/ and crb/ implementing risk
budgeting under different combinations of {solver algorithm} x
{constraint type}. The solver algorithms (CCD, Newton, ADMM+Newton,
ADMM+CCD, ADMM+QP, ADMM+fmincon, bisection-on-lambda) are all just
different numerical routes to the *same* mathematical solution -- they
don't change what problem is being solved, only how. This module ports:

- ``risk_contribution`` -- the risk-decomposition arithmetic shared by
  every variant (consolidates compute_risk_contribution/compute_rc_sd/
  compute_rc_vol).
- ``solve_unconstrained`` -- CCD (default, Roncalli's cyclical coordinate
  descent -- the standard reference algorithm) or Newton, for the
  classic budget-constrained-only (sum(x)=1) risk budgeting problem.
- ``solve_box_constrained`` -- ADMM (Newton-based x-update, closed-form
  box-projection z-update, bisection on the budget-constraint Lagrange
  multiplier lambda) for box-constrained [x_minus, x_plus] risk budgeting.
  This one function replaces the entire ~40-file family of
  crb/compute_rb_sd_bc_admm{1,2,3,4}(_lambda).m and
  crb/compute_rb_sd_bc_ccd(_lambda).m variants, which differ only in
  which inner solver (Newton/CCD/QP/fmincon) handles the ADMM x-update or
  whether lambda is searched via bisection vs. passed directly -- all
  converge to the same box-constrained solution.
- ``solve_constrained`` -- the same ADMM structure, generalized to any
  combination of extra linear equality/inequality constraints and box
  bounds (the z-update projects onto their intersection via
  ``optim.proximal.proximal_linear_constraints``'s Dykstra algorithm
  instead of a closed-form box clip). Consolidates the
  crb/compute_rb_sd_constrained_admm*.m family. ``solve_box_constrained``
  is now a thin wrapper around this for the box-only case.
- ``solve_unconstrained_var``/``solve_unconstrained_es`` --
  Value-at-Risk / Expected-Shortfall risk budgeting under a Gaussian
  assumption, by mapping the confidence level alpha to the equivalent
  standard-deviation multiplier c and reusing ``solve_unconstrained``
  directly (consolidates compute_rb_var.m/compute_rb_es.m/
  compute_rc_var.m/compute_rc_es.m -- each just a 5-line wrapper in the
  original, computing a different scalar multiplier).
- ``erc_portfolio`` -- Equal Risk Contribution, the b=1/n special case
  (consolidates compute_erc_portfolio and mloapa/compute_ERC_{ADMM,CCD}.m,
  which are just the unconstrained solver called with equal budgets).
- ``risk_budgeting_target`` -- bisects the risk-aversion parameter c to
  hit a target active return or active volatility (relative to an
  optional benchmark), covering the "mu-problem"/"sigma-problem" modes of
  compute_risk_parity_portfolio.m/compute_risk_parity_portfolio_bounds.m.
  Works with any of the three solvers above via a ``solver=`` argument
  ("unconstrained"/"box"/"constrained"), consolidating both original
  functions (and their four small _return/_volatility objective-function
  helper files) into one ~80-line implementation built on
  ``quanttoolbox.optim.bisection`` -- versus the originals' ~550 lines
  combined.

MATLAB's ``global RB_CCD_*``/``RB_Newton_*``/``RB_ADMM_*`` blocks are
replaced by ``quanttoolbox.config.{CCDConfig,NewtonConfig,ADMMConfig}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from quanttoolbox.config import ADMMConfig, BisectionConfig, CCDConfig, NewtonConfig, ProximalConfig
from quanttoolbox.optim.bisection import bisection
from quanttoolbox.optim.proximal import proximal_bounds, proximal_linear_constraints


@dataclass
class RiskContribution:
    risk: float
    marginal_risk: np.ndarray
    risk_contribution: np.ndarray
    pct_risk_contribution: np.ndarray


def risk_contribution(
    x: np.ndarray, cov_matrix: np.ndarray, mu: np.ndarray | float = 0.0, c: float = 1.0
) -> RiskContribution:
    """Decompose portfolio risk = -x'mu + c*sqrt(x'Cov*x) into each asset's
    marginal and total risk contribution.

    With mu=0, c=1 this is the pure volatility risk measure (consolidates
    compute_risk_contribution.m / compute_rc_vol.m). With mu != 0 it's the
    standard-deviation-based measure net of expected return
    (compute_rc_sd.m).

    Original: rpb/{compute_risk_contribution,compute_rc_sd,compute_rc_vol}.m
    """
    x = np.asarray(x, dtype=float).flatten()
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = x.shape[0]
    mu = np.zeros(n) if np.isscalar(mu) and mu == 0 else np.asarray(mu, dtype=float).flatten()

    sigma = np.sqrt(x @ cov_matrix @ x)
    risk = -x @ mu + c * sigma
    mr = -mu + c * (cov_matrix @ x) / sigma
    rc = x * mr
    prc = rc / np.sum(rc)

    return RiskContribution(
        risk=risk, marginal_risk=mr, risk_contribution=rc, pct_risk_contribution=prc
    )


@dataclass
class RiskBudgetingResult:
    weights: np.ndarray
    risk: float
    marginal_risk: np.ndarray
    risk_contribution: np.ndarray
    pct_risk_contribution: np.ndarray
    converged: bool
    n_iters: int


def _solve_ccd(
    x0: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    c: float,
    b: np.ndarray,
    config: CCDConfig,
) -> tuple[np.ndarray, bool, int]:
    """Roncalli's cyclical coordinate descent for unconstrained risk budgeting."""
    n = cov_matrix.shape[0]
    x = x0.copy()
    var = np.diag(cov_matrix)
    sigma_x = np.sqrt(x @ cov_matrix @ x)
    sx = cov_matrix @ x
    converged = False

    n_iter = 1
    while n_iter < config.max_iters:
        x_prev_outer = x.copy()
        for i in range(n):
            alpha = c * var[i]
            beta = c * (sx[i] - x[i] * var[i]) - mu[i] * sigma_x
            gamma = -b[i] * sigma_x

            x_tilde = (-beta + np.sqrt(beta**2 - 4 * alpha * gamma)) / (2 * alpha)
            if config.correction:
                x_tilde = min(x_tilde, config.x_max)

            xi = x[i]
            sx = sx - cov_matrix[:, i] * xi + cov_matrix[:, i] * x_tilde
            sigma_x = sigma_x**2 - 2 * xi * cov_matrix[i, :] @ x + xi**2 * var[i]
            x[i] = x_tilde
            sigma_x = np.sqrt(sigma_x + 2 * x_tilde * cov_matrix[i, :] @ x - x_tilde**2 * var[i])

        cvg = np.sum((x / np.sum(x) - x_prev_outer / np.sum(x_prev_outer)) ** 2)
        if cvg <= config.tol:
            converged = True
            break
        n_iter += 1

    return x / np.sum(x), converged, n_iter


def _solve_newton(
    x0: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    c: float,
    b: np.ndarray,
    config: NewtonConfig,
    lambda_: float = 1.0,
    varphi: float = 0.0,
    v_x: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, int]:
    """Newton's method for risk budgeting's Lagrangian stationary conditions.

    varphi/v_x (both default to "off") add an optional ADMM proximal
    penalty term varphi*(x - v_x); used internally by
    ``solve_box_constrained``'s x-update step.
    """
    n = cov_matrix.shape[0]
    x = x0.copy()
    converged = False
    eta = 1.0
    v_x = np.zeros(n) if v_x is None else v_x

    n_iter = 1
    while n_iter < config.max_iters:
        if config.correction:
            x = np.minimum(x, config.x_max)
        # b/x requires x > 0 (risk budgeting weights are positive by
        # construction near the solution); without this floor, Newton can
        # diverge to negative x for extreme c/lambda combinations, since
        # nothing in the original unconstrained update prevents it.
        x = np.maximum(x, 1e-8)

        sigma = np.sqrt(x @ cov_matrix @ x)
        cx = cov_matrix @ x
        mr = -mu + c * cx / sigma
        grad = mr - lambda_ * b / x + varphi * (x - v_x)
        hess = (
            c * (cov_matrix * sigma - np.outer(cx, cx) / sigma) / sigma**2
            + lambda_ * np.diag(b / x**2)
            + varphi * np.eye(n)
        )
        dx = np.linalg.solve(hess, grad)
        x_new = x - eta * dx

        cvg = max(np.max(np.abs(x_new - x)), np.max(np.abs(grad)))
        x = x_new
        if cvg <= config.tol:
            converged = True
            break
        n_iter += 1

    return x, converged, n_iter


def solve_unconstrained(
    cov_matrix: np.ndarray,
    b: np.ndarray | None = None,
    mu: np.ndarray | float = 0.0,
    c: float = 1.0,
    x0: np.ndarray | None = None,
    method: str = "ccd",
    config: CCDConfig | NewtonConfig | None = None,
) -> RiskBudgetingResult:
    """Solve the classic (budget-constraint-only, sum(x)=1) risk budgeting
    problem: find weights x such that each asset's risk contribution
    matches its target budget b.

    method="ccd" (default): Roncalli's cyclical coordinate descent.
    method="newton": Newton's method on the Lagrangian stationary conditions.

    Original: rpb/{compute_rb_sd,compute_rb_sd_ccd,compute_rb_sd_newton}.m
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = cov_matrix.shape[0]
    mu_arr = np.zeros(n) if np.isscalar(mu) and mu == 0 else np.asarray(mu, dtype=float).flatten()
    b_arr = np.full(n, 1.0 / n) if b is None else np.asarray(b, dtype=float).flatten()
    b_arr = b_arr / np.sum(b_arr)
    x0_arr = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float).flatten()

    if method == "ccd":
        cfg = config if isinstance(config, CCDConfig) else CCDConfig()
        x, converged, n_iters = _solve_ccd(x0_arr, mu_arr, cov_matrix, c, b_arr, cfg)
    elif method == "newton":
        cfg = config if isinstance(config, NewtonConfig) else NewtonConfig()
        x, converged, n_iters = _solve_newton(x0_arr, mu_arr, cov_matrix, c, b_arr, cfg)
        x = x / np.sum(x)
    else:
        raise ValueError(f"solve_unconstrained: unknown method '{method}' (use 'ccd' or 'newton')")

    rc = risk_contribution(x, cov_matrix, mu_arr, c)
    return RiskBudgetingResult(
        weights=x,
        risk=rc.risk,
        marginal_risk=rc.marginal_risk,
        risk_contribution=rc.risk_contribution,
        pct_risk_contribution=rc.pct_risk_contribution,
        converged=converged,
        n_iters=n_iters,
    )


def _auto_bracket_and_solve(
    budget_gap: Callable[[np.ndarray], np.ndarray],
    lambda_guess: float = 1.0,
    max_expansions: int = 12,
) -> float:
    """Find a lambda bracket that brackets a sign change in budget_gap
    (sum(x) - 1), expanding geometrically outward from lambda_guess if the
    initial (0.5x, 2x) window doesn't bracket a root, then bisect.

    The original MATLAB toolbox's box-constrained case (compute_rb_sd_bc_*)
    always used a fixed (0.5*lambda, 2*lambda) window since the box-only
    problem's budget-crossing lambda stays close to the unconstrained
    lambda=1 case; general linear constraints can shift this crossing
    further, so this widens the search rather than failing silently.
    """
    lo, hi = 0.5 * lambda_guess, 2.0 * lambda_guess
    y_lo, y_hi = float(budget_gap(np.array(lo))), float(budget_gap(np.array(hi)))

    expansions = 0
    while y_lo * y_hi > 0 and expansions < max_expansions:
        lo, hi = lo / 2.0, hi * 2.0
        y_lo, y_hi = float(budget_gap(np.array(lo))), float(budget_gap(np.array(hi)))
        expansions += 1

    if y_lo * y_hi > 0:
        return float("nan")

    return float(bisection(budget_gap, lo, hi))


def solve_constrained(
    cov_matrix: np.ndarray,
    b: np.ndarray | None = None,
    mu: np.ndarray | float = 0.0,
    c: float = 1.0,
    x0: np.ndarray | None = None,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    x_minus: np.ndarray | float | None = None,
    x_plus: np.ndarray | float | None = None,
    lambda_bracket: tuple[float, float] | None = None,
    admm_config: ADMMConfig | None = None,
    newton_config: NewtonConfig | None = None,
    proximal_config: ProximalConfig | None = None,
) -> RiskBudgetingResult:
    """Solve risk budgeting subject to any combination of extra linear
    equality constraints (a_eq @ x == b_eq), inequality constraints
    (c_ineq @ x <= d_ineq), and box bounds [x_minus, x_plus] -- in addition
    to the implicit sum(x)=1 budget constraint (enforced, as in
    ``solve_box_constrained``, via the outer bisection on lambda, not via
    a_eq). Pass None for any constraint set you don't need.

    Same ADMM structure as ``solve_box_constrained``, generalized so the
    z-update projects onto the *intersection* of whichever constraint sets
    are given (via ``optim.proximal.proximal_linear_constraints``'s
    Dykstra alternating-projection algorithm) instead of just a box.
    ``solve_box_constrained`` is now a thin wrapper around this function
    for the box-only case (using the cheaper closed-form box projection
    directly, no Dykstra iteration needed, when no other constraints are
    given).

    Original: crb/compute_rb_sd_constrained_admm1*.m (+ equivalent solver
    variants, same consolidation as ``solve_box_constrained`` -- see
    module docstring)

    Note: Dykstra's alternating projection (used here whenever a_eq or
    c_ineq is given alongside box bounds) can, for constraint sets that
    meet at a sharp corner, exit on a temporary numerical plateau before
    reaching the true intersection point -- see the caveat in
    ``optim.proximal.proximal_linear_constraints``'s docstring. If a
    result looks suspicious, verify constraint satisfaction directly.
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = cov_matrix.shape[0]
    mu_arr = np.zeros(n) if np.isscalar(mu) and mu == 0 else np.asarray(mu, dtype=float).flatten()
    b_arr = np.full(n, 1.0 / n) if b is None else np.asarray(b, dtype=float).flatten()
    b_arr = b_arr / np.sum(b_arr)
    x0_arr = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float).flatten()

    admm_cfg = admm_config or ADMMConfig()
    newton_cfg = newton_config or NewtonConfig()
    prox_cfg = proximal_config or ProximalConfig()

    has_extra_constraints = a_eq is not None or c_ineq is not None
    box_only = not has_extra_constraints and (x_minus is not None or x_plus is not None)

    def _project(v: np.ndarray) -> np.ndarray:
        if has_extra_constraints:
            x_out, _ = proximal_linear_constraints(
                v,
                a_eq=a_eq,
                b_eq=b_eq,
                c_ineq=c_ineq,
                d_ineq=d_ineq,
                lb=x_minus,
                ub=x_plus,
                config=prox_cfg,
            )
            return x_out
        if box_only:
            return proximal_bounds(v, x_minus, x_plus)
        return v  # no constraints beyond the implicit budget constraint

    def _admm_at_lambda(lambda_: float) -> tuple[np.ndarray, bool, int]:
        x = x0_arr.copy()
        z = x.copy()
        z0 = z.copy()
        u = np.zeros(n)
        varphi = admm_cfg.varphi
        converged = False

        n_iter = 1
        while n_iter < admm_cfg.max_iters:
            v_x = z - u
            x, _, _ = _solve_newton(
                x, mu_arr, cov_matrix, c, b_arr, newton_cfg, lambda_, varphi, v_x
            )

            v_z = x + u
            z = _project(v_z)

            r = x - z
            s = varphi * (z - z0)
            u = u + r

            cvg = max(np.sum((x - x0_arr) ** 2), np.sum((x - z) ** 2), np.sum((z - z0) ** 2))
            if cvg <= admm_cfg.tol:
                converged = True
                break

            if admm_cfg.varphi_method == 2:
                primal_error = np.sum(r * r)
                dual_error = np.sum(s * s)
                if primal_error > admm_cfg.tau_primal * dual_error:
                    varphi = varphi * admm_cfg.tau_primal
                    u = u / admm_cfg.tau_primal
                elif dual_error > admm_cfg.tau_dual * primal_error:
                    varphi = varphi / admm_cfg.tau_dual
                    u = u * admm_cfg.tau_dual

            x0_arr[:] = x
            z0 = z.copy()
            n_iter += 1

        return x, converged, n_iter

    def _budget_gap(lambda_: np.ndarray) -> np.ndarray:
        lam = float(lambda_)
        x, _, _ = _admm_at_lambda(lam)
        return np.array(np.sum(x) - 1.0)

    if lambda_bracket is None:
        lambda_star = _auto_bracket_and_solve(_budget_gap)
    else:
        lambda_star = bisection(_budget_gap, lambda_bracket[0], lambda_bracket[1])

    if np.isnan(lambda_star):
        n_arr = np.full(n, np.nan)
        return RiskBudgetingResult(
            weights=n_arr,
            risk=np.nan,
            marginal_risk=n_arr,
            risk_contribution=n_arr,
            pct_risk_contribution=n_arr,
            converged=False,
            n_iters=0,
        )

    x, converged, n_iters = _admm_at_lambda(float(lambda_star))
    rc = risk_contribution(x, cov_matrix, mu_arr, c)
    return RiskBudgetingResult(
        weights=x,
        risk=rc.risk,
        marginal_risk=rc.marginal_risk,
        risk_contribution=rc.risk_contribution,
        pct_risk_contribution=rc.pct_risk_contribution,
        converged=converged,
        n_iters=n_iters,
    )


def solve_box_constrained(
    cov_matrix: np.ndarray,
    x_minus: np.ndarray | float,
    x_plus: np.ndarray | float,
    b: np.ndarray | None = None,
    mu: np.ndarray | float = 0.0,
    c: float = 1.0,
    x0: np.ndarray | None = None,
    lambda_bracket: tuple[float, float] | None = None,
    admm_config: ADMMConfig | None = None,
    newton_config: NewtonConfig | None = None,
) -> RiskBudgetingResult:
    """Solve risk budgeting subject to box constraints x_minus <= x <= x_plus
    (in addition to the sum(x)=1 budget constraint). Thin wrapper around
    ``solve_constrained`` for the box-only case (uses the cheaper
    closed-form box projection directly, no Dykstra iteration needed).

    Original: crb/compute_rb_sd_bc_admm1*.m (+ ~40 equivalent solver
    variants -- see module docstring)
    """
    return solve_constrained(
        cov_matrix,
        b=b,
        mu=mu,
        c=c,
        x0=x0,
        x_minus=x_minus,
        x_plus=x_plus,
        lambda_bracket=lambda_bracket,
        admm_config=admm_config,
        newton_config=newton_config,
    )


def erc_portfolio(
    cov_matrix: np.ndarray, x0: np.ndarray | None = None, method: str = "ccd"
) -> RiskBudgetingResult:
    """Equal Risk Contribution portfolio: risk budgeting with all budgets
    equal (b = 1/n), pure volatility risk measure.

    Original: rpb/compute_erc_portfolio.m (+ mloapa/compute_ERC_{ADMM,CCD}.m,
    equivalent alternative solvers for the same problem)
    """
    n = np.asarray(cov_matrix).shape[0]
    return solve_unconstrained(
        cov_matrix, b=np.full(n, 1.0 / n), mu=0.0, c=1.0, x0=x0, method=method
    )


def risk_budgeting_frontier(
    cov_matrix: np.ndarray,
    b: np.ndarray | None,
    mu: np.ndarray,
    c_values: np.ndarray,
    x0: np.ndarray | None = None,
) -> list[RiskBudgetingResult]:
    """Evaluate the risk budgeting solution at each risk-aversion value in
    c_values (the "gamma-problem" mode of compute_risk_parity_portfolio.m).

    See module docstring for target-matching modes not ported here.

    Original: rpb/compute_risk_parity_portfolio.m (gamma-problem branch)
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    return [
        solve_unconstrained(cov_matrix, b=b, mu=mu, c=float(c), x0=x0, method="ccd")
        for c in np.atleast_1d(c_values)
    ]


def _var_multiplier(alpha: float) -> float:
    """Gaussian VaR risk-measure multiplier c = Phi^-1(alpha)."""
    return float(norm.ppf(alpha))


def _es_multiplier(alpha: float) -> float:
    """Gaussian ES (expected shortfall) risk-measure multiplier
    c = phi(Phi^-1(alpha)) / (1 - alpha)."""
    z = norm.ppf(alpha)
    return float(norm.pdf(z) / (1 - alpha))


def solve_unconstrained_var(
    cov_matrix: np.ndarray,
    alpha: float,
    b: np.ndarray | None = None,
    mu: np.ndarray | float = 0.0,
    x0: np.ndarray | None = None,
    method: str = "ccd",
    config: CCDConfig | NewtonConfig | None = None,
) -> RiskBudgetingResult:
    """Risk budgeting under a (Gaussian) Value-at-Risk risk measure at
    confidence level alpha, rather than plain volatility -- solved by
    mapping alpha to the equivalent standard-deviation multiplier c and
    reusing ``solve_unconstrained`` directly (VaR is just a rescaled
    standard-deviation measure under normality).

    Original: rpb/{compute_rb_var,compute_rc_vol}.m (VaR variant)
    """
    return solve_unconstrained(
        cov_matrix, b=b, mu=mu, c=_var_multiplier(alpha), x0=x0, method=method, config=config
    )


def solve_unconstrained_es(
    cov_matrix: np.ndarray,
    alpha: float,
    b: np.ndarray | None = None,
    mu: np.ndarray | float = 0.0,
    x0: np.ndarray | None = None,
    method: str = "ccd",
    config: CCDConfig | NewtonConfig | None = None,
) -> RiskBudgetingResult:
    """Risk budgeting under a (Gaussian) Expected Shortfall risk measure at
    confidence level alpha. See ``solve_unconstrained_var``.

    Original: rpb/{compute_rb_es,compute_rc_es}.m
    """
    return solve_unconstrained(
        cov_matrix, b=b, mu=mu, c=_es_multiplier(alpha), x0=x0, method=method, config=config
    )


def risk_contribution_var(
    x: np.ndarray, cov_matrix: np.ndarray, mu: np.ndarray | float, alpha: float
) -> RiskContribution:
    """Risk contribution decomposition under the (Gaussian) VaR risk measure.

    Original: rpb/compute_rc_var.m
    """
    return risk_contribution(x, cov_matrix, mu, c=_var_multiplier(alpha))


def risk_contribution_es(
    x: np.ndarray, cov_matrix: np.ndarray, mu: np.ndarray | float, alpha: float
) -> RiskContribution:
    """Risk contribution decomposition under the (Gaussian) ES risk measure.

    Original: rpb/compute_rc_es.m
    """
    return risk_contribution(x, cov_matrix, mu, c=_es_multiplier(alpha))


@dataclass
class RiskBudgetingTargetResult:
    weights: np.ndarray
    active_return: float
    active_volatility: float
    c: float
    converged: bool


def _solve_rb_at_c(
    cov_matrix: np.ndarray,
    b: np.ndarray | None,
    mu: np.ndarray,
    c: float,
    x0: np.ndarray | None,
    x_benchmark: np.ndarray,
    solver: str,
    solver_kwargs: dict,
) -> tuple[np.ndarray, float, float, bool]:
    """Shared helper: solve risk budgeting at a fixed c, return
    (weights, active_return, active_volatility, converged) relative to
    x_benchmark."""
    if solver == "unconstrained":
        result = solve_unconstrained(cov_matrix, b=b, mu=mu, c=c, x0=x0, **solver_kwargs)
    elif solver == "box":
        result = solve_box_constrained(cov_matrix, b=b, mu=mu, c=c, x0=x0, **solver_kwargs)
    elif solver == "constrained":
        result = solve_constrained(cov_matrix, b=b, mu=mu, c=c, x0=x0, **solver_kwargs)
    else:
        raise ValueError(
            f"_solve_rb_at_c: unknown solver '{solver}' (use 'unconstrained', 'box', or 'constrained')"
        )

    x = result.weights
    active = x - x_benchmark
    mu_x = float(active @ mu)
    sigma_x = float(np.sqrt(active @ cov_matrix @ active))
    return x, mu_x, sigma_x, result.converged


def risk_budgeting_target(
    cov_matrix: np.ndarray,
    mu: np.ndarray,
    target: float,
    target_type: str = "return",
    b: np.ndarray | None = None,
    x_benchmark: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    c_min: float = 1.0,
    c_max: float = 100.0,
    solver: str = "unconstrained",
    bisection_config: BisectionConfig | None = None,
    **solver_kwargs,
) -> RiskBudgetingTargetResult:
    """Find the risk budgeting portfolio (relative to an optional benchmark
    x_benchmark) whose active return or active volatility matches `target`,
    by bisecting the risk-aversion parameter c between c_min and c_max.

    target_type="return" (default): bisect to hit a target active return
    (the "mu-problem" mode). target_type="volatility": bisect to hit a
    target active volatility (the "sigma-problem" mode).

    solver="unconstrained" (default), "box", or "constrained" selects
    which underlying risk-budgeting solver to use at each trial c;
    solver_kwargs are forwarded to it (e.g. x_minus/x_plus for "box",
    a_eq/b_eq/c_ineq/d_ineq/x_minus/x_plus for "constrained").

    Returns a result with weights all-NaN and converged=False if `target`
    lies outside the achievable range [c_min, c_max] maps to.

    Original: rpb/{compute_risk_parity_portfolio,
    compute_risk_parity_portfolio_bounds}.m ("mu-problem"/"sigma-problem"
    branches, consolidated with their _return/_volatility objective
    helper files -- see module docstring)
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = cov_matrix.shape[0]
    mu = np.asarray(mu, dtype=float).flatten()
    x_benchmark_arr = (
        np.zeros(n) if x_benchmark is None else np.asarray(x_benchmark, dtype=float).flatten()
    )
    bisect_cfg = bisection_config or BisectionConfig()

    _, mu_min, sigma_min, _ = _solve_rb_at_c(
        cov_matrix, b, mu, c_min, x0, x_benchmark_arr, solver, solver_kwargs
    )
    _, mu_max, sigma_max, _ = _solve_rb_at_c(
        cov_matrix, b, mu, c_max, x0, x_benchmark_arr, solver, solver_kwargs
    )

    if target_type == "volatility":
        lo, hi = min(sigma_min, sigma_max), max(sigma_min, sigma_max)

        def objective(c: np.ndarray) -> np.ndarray:
            _, _, sigma_x, _ = _solve_rb_at_c(
                cov_matrix, b, mu, float(c), x0, x_benchmark_arr, solver, solver_kwargs
            )
            return np.array(sigma_x - target)

    elif target_type == "return":
        lo, hi = min(mu_min, mu_max), max(mu_min, mu_max)

        def objective(c: np.ndarray) -> np.ndarray:
            _, mu_x, _, _ = _solve_rb_at_c(
                cov_matrix, b, mu, float(c), x0, x_benchmark_arr, solver, solver_kwargs
            )
            return np.array(mu_x - target)

    else:
        raise ValueError(f"risk_budgeting_target: unknown target_type '{target_type}'")

    if target < lo or target > hi:
        nan_arr = np.full(n, np.nan)
        return RiskBudgetingTargetResult(
            weights=nan_arr,
            active_return=np.nan,
            active_volatility=np.nan,
            c=np.nan,
            converged=False,
        )

    c_star = bisection(objective, c_min, c_max, bisect_cfg)
    if np.isnan(c_star):
        nan_arr = np.full(n, np.nan)
        return RiskBudgetingTargetResult(
            weights=nan_arr,
            active_return=np.nan,
            active_volatility=np.nan,
            c=np.nan,
            converged=False,
        )

    x, mu_x, sigma_x, converged = _solve_rb_at_c(
        cov_matrix, b, mu, float(c_star), x0, x_benchmark_arr, solver, solver_kwargs
    )
    return RiskBudgetingTargetResult(
        weights=x,
        active_return=mu_x,
        active_volatility=sigma_x,
        c=float(c_star),
        converged=converged,
    )


def risk_budgeting_target_frontier(
    cov_matrix: np.ndarray,
    mu: np.ndarray,
    targets: np.ndarray,
    target_type: str = "return",
    **kwargs,
) -> list[RiskBudgetingTargetResult]:
    """Evaluate ``risk_budgeting_target`` at each value in `targets`.

    Original: rpb/{compute_risk_parity_portfolio,
    compute_risk_parity_portfolio_bounds}.m (looping over multiple targets)
    """
    return [
        risk_budgeting_target(cov_matrix, mu, float(t), target_type=target_type, **kwargs)
        for t in np.atleast_1d(targets)
    ]
