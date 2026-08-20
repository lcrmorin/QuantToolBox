"""Quadratic programming: a general QP solver plus ridge/lasso/turnover
penalized variants, and the closed-form QP-on-a-hyperplane solution.

Ported from QuantToolbox/optim/{quadprog_bc_ccd,quadprog_lasso,
quadprog_ridge,quadprog_turnover,quadprog_mixed_norm,
quadprog_mixed2_norm,qp_hyperplane}.m

Translation notes -- this is the single biggest architectural
simplification in the whole port:

MATLAB's Optimization Toolbox ``quadprog`` cannot express an L1 penalty
term directly, so the original toolbox works around this with a
"variable-splitting" trick: introduce two extra non-negative variable
blocks per L1 term (x = x+ - x-), turning each lasso/mixed-norm/turnover
problem into a *bigger*, purely-quadratic QP that quadprog can solve. This
produces four near-duplicate ~100-line MATLAB functions
(quadprog_lasso/ridge/mixed_norm/mixed2_norm), each hand-building the
block matrices for a different combination of penalties.

``cvxpy`` supports L1/L2 norm terms and turnover (L1-ball) constraints
*natively* in its objective/constraint DSL, so none of that
variable-splitting machinery is needed here. All four MATLAB functions
above collapse into one function, ``solve_qp``, parameterized by which
penalty terms and constraints are supplied:

    solve_qp(q, r)                                    # plain QP  (replaces quadprog_ridge with no penalty)
    solve_qp(q, r, ridge_penalty=(gamma, x_target))    # replaces quadprog_ridge
    solve_qp(q, r, lasso_penalty=(gamma, x_target))    # replaces quadprog_lasso
    solve_qp(q, r, ridge_penalty=..., lasso_penalty=...)  # replaces quadprog_mixed_norm
    solve_qp(q, r, turnover=(x0, tau))                 # replaces the turnover-constrained
                                                        # branch of quadprog_turnover

(``quadprog_mixed2_norm``'s "two separate lasso/ridge penalties toward two
different targets" case is representable by simply adding the two ridge
penalties together algebraically, since ridge penalties toward different
targets combine into a single quadratic form -- not separately exposed as
a distinct code path here.)

The objective throughout is ``0.5 * x'Qx - R'x`` (MATLAB's convention,
note the *minus* sign on the linear term), matching every one of the
original functions' calling convention.

``quadprog_bc_ccd`` (box-constrained QP via cyclical coordinate descent)
is preserved as a separate, dependency-free NumPy implementation since
it's a genuinely different (iterative, matrix-free-friendly) algorithm
worth keeping available alongside the cvxpy path.

``qp_hyperplane`` (closed-form QP with a single equality constraint) has
an exact closed-form Lagrangian solution and needs no solver at all.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_qp(
    q: np.ndarray,
    r: np.ndarray,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    ridge_penalty: tuple[np.ndarray | float, np.ndarray] | None = None,
    lasso_penalty: tuple[np.ndarray | float, np.ndarray] | None = None,
    turnover: tuple[np.ndarray, float] | None = None,
    default_budget_constraint: bool = False,
) -> np.ndarray:
    """Solve min 0.5*x'Qx - R'x subject to the given constraints and
    optional ridge/lasso penalty terms and turnover budget.

    Original: optim/{quadprog_lasso,quadprog_ridge,quadprog_turnover,
    quadprog_mixed_norm,quadprog_mixed2_norm}.m (consolidated -- see
    module docstring)

    Parameters
    ----------
    q, r : QP objective 0.5*x'Qx - R'x.
    a_eq, b_eq : equality constraints A_eq @ x == B_eq.
    c_ineq, d_ineq : inequality constraints C_ineq @ x <= D_ineq.
    lb, ub : box bounds.
    ridge_penalty : (gamma, x_target) adds gamma * ||x - x_target||_2^2 to
        the objective. gamma may be a scalar, a vector (diagonal), or a
        full matrix.
    lasso_penalty : (gamma, x_target) adds gamma' * |x - x_target| to the
        objective (elementwise L1, gamma may be scalar or vector).
    turnover : (x0, tau) constrains ||x - x0||_1 <= tau.
    default_budget_constraint : if True and a_eq/b_eq are not given, adds
        the default sum(x) == 1 budget constraint (matches the original's
        ``isscalar(A) && A==0`` sentinel convention).
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    n = q.shape[0]

    x = cp.Variable(n)
    objective = 0.5 * cp.quad_form(x, cp.psd_wrap(q)) - r @ x

    if ridge_penalty is not None:
        gamma, x_target = ridge_penalty
        gamma = np.asarray(gamma, dtype=float)
        x_target = np.asarray(x_target, dtype=float).flatten()
        if gamma.ndim == 0:
            objective = objective + gamma.item() * cp.sum_squares(x - x_target)
        elif gamma.ndim == 1:
            objective = objective + cp.sum(cp.multiply(gamma, cp.square(x - x_target)))
        else:
            objective = objective + cp.quad_form(x - x_target, cp.psd_wrap(gamma))

    if lasso_penalty is not None:
        gamma, x_target = lasso_penalty
        gamma = np.asarray(gamma, dtype=float)
        x_target = np.asarray(x_target, dtype=float).flatten()
        if gamma.ndim == 0:
            objective = objective + gamma.item() * cp.norm1(x - x_target)
        else:
            objective = objective + cp.sum(cp.multiply(gamma, cp.abs(x - x_target)))

    constraints = []
    if a_eq is not None:
        constraints.append(
            np.asarray(a_eq, dtype=float) @ x == np.asarray(b_eq, dtype=float).flatten()
        )
    elif default_budget_constraint:
        constraints.append(cp.sum(x) == 1)

    if c_ineq is not None:
        constraints.append(
            np.asarray(c_ineq, dtype=float) @ x <= np.asarray(d_ineq, dtype=float).flatten()
        )

    if lb is not None:
        constraints.append(x >= lb)
    if ub is not None:
        constraints.append(x <= ub)

    if turnover is not None:
        x0, tau = turnover
        constraints.append(cp.norm1(x - np.asarray(x0, dtype=float).flatten()) <= tau)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solve_qp: solver did not converge (status={problem.status})")

    if x.value is None:
        raise RuntimeError("solve_qp: solver returned no solution")

    return x.value


def qp_bc_ccd(
    q: np.ndarray,
    r: np.ndarray,
    x_minus: np.ndarray | float | None = None,
    x_plus: np.ndarray | float | None = None,
    x_init: np.ndarray | None = None,
    n_iters: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Box-constrained QP (min 0.5*x'Qx - R'x s.t. x_minus <= x <= x_plus)
    via cyclical coordinate descent -- a lightweight, dependency-free
    alternative to solve_qp for this specific (box-only) case.

    Original: optim/quadprog_bc_ccd.m
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    n = q.shape[0]

    x = (
        np.asarray(x_init, dtype=float).copy()
        if x_init is not None and np.asarray(x_init).shape[0] == n
        else np.linalg.solve(q, r)
    )

    truncate = x_minus is not None or x_plus is not None
    if x_minus is not None:
        x_minus = np.full(n, x_minus) if np.isscalar(x_minus) else np.asarray(x_minus, dtype=float)
    if x_plus is not None:
        x_plus = np.full(n, x_plus) if np.isscalar(x_plus) else np.asarray(x_plus, dtype=float)

    x_path = np.zeros((n_iters, n))
    for it in range(n_iters):
        x_path[it] = x
        for i in range(n):
            new_x = x.copy()
            new_x[i] = 0.0
            new_x[i] = (r[i] - 0.5 * new_x @ q[:, i] - 0.5 * q[i, :] @ new_x) / q[i, i]
            if truncate:
                new_x[i] = min(max(x_minus[i], new_x[i]), x_plus[i])
            x = new_x

    return x, x_path


def qp_hyperplane(
    q: np.ndarray, r: np.ndarray, a: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """Closed-form solution of min 0.5*x'Qx - R'x s.t. a'x = b (single
    equality constraint), via the Lagrangian stationary conditions.

    Original: optim/qp_hyperplane.m

    Returns (x, lagrange_multiplier).
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float).flatten()
    a = np.asarray(a, dtype=float).flatten()

    inv_q = np.linalg.inv(q)
    lagrange = (b - a @ inv_q @ r) / (a @ inv_q @ a)
    x = inv_q @ (r + lagrange * a)
    return x, lagrange
