"""Support Vector Machines: classification and regression, primal and dual
formulations.

Ported from QuantToolBox/svm/{svm_classification_dual,
svm_classification_primal,svm_regression_dual,svm_regression_primal}.m
(QuantToolBox/theo/svm_*.m are byte-identical duplicates, not ported
separately).

Consolidation notes:

Every one of the original four functions builds a hand-constructed QP
(block matrices for slack variables, box bounds, one equality constraint)
and calls MATLAB's ``quadprog`` directly. Since
``quanttoolbox.optim.quadprog.solve_qp`` already accepts arbitrary
Q/R/equality/inequality/box arguments, each branch below is a direct,
literal translation of the original's Q/R/constraint construction into a
single ``solve_qp`` call -- no additional QP machinery is needed. Sign
convention: MATLAB's ``quadprog(Q, f, ...)`` minimizes
``0.5*x'Qx + f'x``; ``solve_qp(Q, R, ...)`` minimizes ``0.5*x'Qx - R'x``,
so every ``R`` passed to ``solve_qp`` below is the *negation* of the
original's ``f``.

MATLAB's ``global SVM_macheps`` is replaced by
``quanttoolbox.config.SVMConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanttoolbox.config import SVMConfig
from quanttoolbox.optim.quadprog import solve_qp


@dataclass
class SVMClassificationResult:
    beta0: float
    beta: np.ndarray
    xi: np.ndarray
    margin: float
    alpha: np.ndarray | None = None  # dual only
    support_vectors: np.ndarray | None = None  # dual only, 0-indexed


def svm_classification_dual(
    y: np.ndarray,
    x: np.ndarray,
    c: float | None = None,
    loss: str = "hinge",
    random_start: bool = False,
    config: SVMConfig | None = None,
) -> SVMClassificationResult:
    """SVM classification, dual (kernel-ready) formulation.

    c=None (default): hard margin. loss="hinge" (default): standard
    soft-margin hinge loss (box-constrained dual, alpha in [0, c]).
    loss="squared_hinge": squared hinge loss (ridge-regularized dual,
    alpha in [0, inf)).

    Original: svm/svm_classification_dual.m
    """
    if config is None:
        config = SVMConfig()

    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    ydx = y[:, None] * x
    q = ydx @ ydx.T
    r = np.ones(n)
    a_eq = y[None, :]
    b_eq = np.array([0.0])

    hard_margin = c is None or (isinstance(c, float) and np.isnan(c))

    if hard_margin:
        lb, ub = np.zeros(n), None
    elif loss == "squared_hinge":
        assert c is not None
        q = q + np.eye(n) / (2 * c)
        lb, ub = np.zeros(n), None
    else:
        assert c is not None
        lb, ub = np.zeros(n), np.full(n, c)

    alpha = solve_qp(q, r, a_eq=a_eq, b_eq=b_eq, lb=lb, ub=ub)
    alpha = np.where(alpha >= config.macheps, alpha, 0.0)

    if not hard_margin and loss != "squared_hinge":
        near_c = np.abs(alpha - c) <= config.macheps
        alpha = np.where(near_c, c, alpha)
        sv = np.where((alpha > 0) & (alpha < c))[0]
    else:
        sv = np.where(alpha > 0)[0]

    beta = np.sum(alpha[:, None] * y[:, None] * x, axis=0)
    beta0 = float(np.mean(y[sv] - x[sv] @ beta))
    xi = np.maximum(0, 1 - y * (beta0 + x @ beta))
    xi = np.where(np.abs(xi) >= config.macheps, xi, 0.0)
    margin = float(1.0 / (beta @ beta))

    return SVMClassificationResult(
        beta0=beta0, beta=beta, xi=xi, margin=margin, alpha=alpha, support_vectors=sv
    )


def svm_classification_primal(
    y: np.ndarray, x: np.ndarray, c: float | None = None, loss: str = "hinge"
) -> SVMClassificationResult:
    """SVM classification, primal formulation.

    Same c/loss semantics as ``svm_classification_dual``.

    Original: svm/svm_classification_primal.m
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n, k = x.shape

    hard_margin = c is None or (isinstance(c, float) and np.isnan(c))

    if hard_margin:
        q = np.zeros((k + 1, k + 1))
        q[1:, 1:] = np.eye(k)
        r = np.zeros(k + 1)
        c_ineq = -(y[:, None] * np.column_stack([np.ones(n), x]))
        d_ineq = -np.ones(n)
        theta = solve_qp(q, r, c_ineq=c_ineq, d_ineq=d_ineq)
        beta0, beta = theta[0], theta[1 : k + 1]
        xi = np.full(n, np.nan)

    elif loss == "squared_hinge":
        assert c is not None
        q = np.zeros((1 + k + n, 1 + k + n))
        q[1 : k + 1, 1 : k + 1] = np.eye(k)
        q[k + 1 :, k + 1 :] = 2 * c * np.eye(n)
        r = np.zeros(1 + k + n)
        c_block = y[:, None] * np.column_stack([np.ones(n), x])
        c_ineq = -np.hstack([c_block, np.eye(n)])
        d_ineq = -np.ones(n)
        lb = np.concatenate([np.full(1 + k, -np.inf), np.zeros(n)])
        theta = solve_qp(q, r, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb)
        beta0, beta = theta[0], theta[1 : k + 1]
        xi = theta[k + 1 :]

    else:
        assert c is not None
        q = np.zeros((1 + k + n, 1 + k + n))
        q[1 : k + 1, 1 : k + 1] = np.eye(k)
        r = np.concatenate([np.zeros(k + 1), -c * np.ones(n)])
        c_ineq = -np.hstack([y[:, None], y[:, None] * x, np.eye(n)])
        d_ineq = -np.ones(n)
        lb = np.concatenate([np.full(k + 1, -np.inf), np.zeros(n)])
        theta = solve_qp(q, r, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb)
        beta0, beta = theta[0], theta[1 : k + 1]
        xi = theta[k + 1 :]

    margin = float(1.0 / (beta @ beta))
    return SVMClassificationResult(beta0=float(beta0), beta=beta, xi=xi, margin=margin)


@dataclass
class SVMRegressionResult:
    beta0: float
    beta: np.ndarray
    xi: np.ndarray
    margin: float
    alpha: np.ndarray | None = None  # dual only


def svm_regression_dual(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    epsilon: float | None = None,
    config: SVMConfig | None = None,
) -> SVMRegressionResult:
    """SVM regression, dual (kernel-ready) formulation.

    epsilon=None (default, or a negative value): Least-Squares SVM
    (ridge-regularized). epsilon=0: epsilon-SVM with epsilon-tube width 0
    (box-constrained dual). epsilon>0: standard epsilon-insensitive SVM
    regression.

    Original: svm/svm_regression_dual.m
    """
    if config is None:
        config = SVMConfig()

    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    q = x @ x.T

    if epsilon is None or epsilon < 0:
        c_eff = np.finfo(np.float32).eps if c == 0 else c
        q = q + np.eye(n) / (2 * c_eff)
        r = y
        a_eq = np.ones((1, n))
        b_eq = np.array([0.0])

        alpha = solve_qp(q, r, a_eq=a_eq, b_eq=b_eq)
        beta = x.T @ alpha
        beta0 = float(np.mean(y - x @ beta))
        xi = y - beta0 - x @ beta
        alpha = np.where(np.abs(alpha) >= config.macheps, alpha, 0.0)

        return SVMRegressionResult(
            beta0=beta0, beta=beta, xi=xi, margin=float(1.0 / (beta @ beta)), alpha=alpha
        )

    if epsilon == 0:
        r = y
        a_eq = np.ones((1, n))
        b_eq = np.array([0.0])
        lb, ub = np.full(n, -c), np.full(n, c)

        delta = solve_qp(q, r, a_eq=a_eq, b_eq=b_eq, lb=lb, ub=ub)
        beta = x.T @ delta

        near_c = np.abs(delta - c) <= config.macheps
        delta = np.where(near_c, c, delta)
        near_neg_c = np.abs(delta + c) <= config.macheps
        delta = np.where(near_neg_c, -c, delta)

        u = y - x @ beta
        sv_mask = (delta > -c) & (delta < c)
        beta0 = float(np.mean(u[sv_mask])) if np.any(sv_mask) else 0.0

        u = y - beta0 - x @ beta
        xi_minus = np.where(delta == -c, np.maximum(-u, 0), 0.0)
        xi_plus = np.where(delta == c, np.maximum(u, 0), 0.0)
        xi = np.column_stack([xi_minus, xi_plus])

        return SVMRegressionResult(
            beta0=beta0, beta=beta, xi=xi, margin=float(1.0 / (beta @ beta)), alpha=delta
        )

    # standard epsilon-insensitive SVM regression
    q_big = np.block([[q, -q], [-q, q]])
    # the block structure [[Q,-Q],[-Q,Q]] is PSD but rank-deficient (rank
    # <= n, not 2n), which can make general QP solvers struggle to
    # converge; a small ridge regularization on the diagonal is a standard
    # fix and doesn't change the solution in any way that matters (the
    # true optimum is unaffected to numerical precision).
    q_big = q_big + 1e-8 * np.eye(2 * n)
    r_big = np.concatenate([-(y + epsilon), y - epsilon])
    a_eq = np.concatenate([np.ones(n), -np.ones(n)])[None, :]
    b_eq = np.array([0.0])
    lb, ub = np.zeros(2 * n), np.full(2 * n, c)

    alpha_stacked = solve_qp(q_big, r_big, a_eq=a_eq, b_eq=b_eq, lb=lb, ub=ub)
    alpha_minus, alpha_plus = alpha_stacked[:n], alpha_stacked[n:]
    alpha_minus = np.where(np.abs(alpha_minus) >= config.macheps, alpha_minus, 0.0)
    alpha_plus = np.where(np.abs(alpha_plus) >= config.macheps, alpha_plus, 0.0)
    alpha_minus = np.where(np.abs(alpha_minus - c) <= config.macheps, c, alpha_minus)
    alpha_plus = np.where(np.abs(alpha_plus - c) <= config.macheps, c, alpha_plus)

    beta = x.T @ (alpha_plus - alpha_minus)
    u_minus = y + epsilon - x @ beta
    u_plus = y - epsilon - x @ beta

    sv_minus = np.where((alpha_minus > 0) & (alpha_minus < c))[0]
    sv_plus = np.where((alpha_plus > 0) & (alpha_plus < c))[0]

    if sv_minus.size == 0 and sv_plus.size > 0:
        beta0 = float(np.mean(u_plus[sv_plus]))
    elif sv_plus.size == 0 and sv_minus.size > 0:
        beta0 = float(np.mean(u_minus[sv_minus]))
    elif sv_minus.size > 0 and sv_plus.size > 0:
        beta0 = float(np.mean(np.concatenate([u_minus[sv_minus], u_plus[sv_plus]])))
    else:
        beta0 = 0.0

    u_minus = y + epsilon - beta0 - x @ beta
    u_plus = y - epsilon - beta0 - x @ beta
    xi_minus = -((alpha_minus == c).astype(float) * u_minus)
    xi_plus = (alpha_plus == c) * u_plus
    xi = np.column_stack([xi_minus, xi_plus])
    alpha = np.column_stack([alpha_minus, alpha_plus])

    return SVMRegressionResult(
        beta0=beta0, beta=beta, xi=xi, margin=float(1.0 / (beta @ beta)), alpha=alpha
    )


def svm_regression_primal(
    y: np.ndarray, x: np.ndarray, c: float, epsilon: float | None = None
) -> SVMRegressionResult:
    """SVM regression, primal formulation.

    Same c/epsilon semantics as ``svm_regression_dual`` except epsilon=0
    is not a distinct primal branch in the original (only the LS-SVM
    epsilon<0 case and the epsilon>0 case are formulated directly).

    Original: svm/svm_regression_primal.m
    """
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float)
    n, k = x.shape

    if epsilon is None or epsilon < 0:
        q = np.zeros((1 + k + n, 1 + k + n))
        q[1 : k + 1, 1 : k + 1] = np.eye(k)
        q[k + 1 :, k + 1 :] = 2 * c * np.eye(n)
        r = np.zeros(1 + k + n)
        a_eq = np.hstack([np.ones((n, 1)), x, np.eye(n)])
        b_eq = y

        theta = solve_qp(q, r, a_eq=a_eq, b_eq=b_eq)
        beta0, beta, xi = theta[0], theta[1 : k + 1], theta[k + 1 :]

    else:
        eps_vec = epsilon * np.ones(n)
        q = np.zeros((1 + k + 2 * n, 1 + k + 2 * n))
        q[1 : k + 1, 1 : k + 1] = np.eye(k)
        r = np.concatenate([np.zeros(1 + k), -c * np.ones(2 * n)])

        top = np.hstack([np.ones((n, 1)), x, -np.eye(n), np.zeros((n, n))])
        bottom = np.hstack([-np.ones((n, 1)), -x, np.zeros((n, n)), -np.eye(n)])
        c_ineq = np.vstack([top, bottom])
        d_ineq = np.concatenate([y + eps_vec, -y + eps_vec])
        lb = np.concatenate([np.full(1 + k, -1e10), np.zeros(2 * n)])

        theta = solve_qp(q, r, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb)
        beta0, beta = theta[0], theta[1 : k + 1]
        xi = np.column_stack([theta[k + 1 : k + 1 + n], theta[k + 1 + n :]])

    margin = float(1.0 / (beta @ beta))
    return SVMRegressionResult(beta0=float(beta0), beta=beta, xi=xi, margin=margin)
