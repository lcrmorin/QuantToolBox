"""Minimize mixed L1 penalties under linear constraints using SLSQP.

The objective is ``sum_k weight_k * ||operator_k @ (x - x0)||_1``.
An operator of ``None`` denotes the identity. The solver accepts equality
constraints, optional inequality constraints, and box bounds.

Adapted from HSF-Notebooks chapter 11h's
``solve_bond_l1_duration_neutral_smooth``, corresponding to the ``fmincon``
branch in ``HSF/11. Portfolio Optimization/chap11_net_zero5.m``. The
notebook's budget, duration, and carbon constraints are generalized to
array arguments; the SLSQP algorithm and default tolerances are retained.
This direct nonsmooth solve remains separate from the LP/QP formulation.

``L1MinimizeResult`` includes solver success and diagnostic information
alongside the solution so callers can detect non-convergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class L1MinimizeResult:
    """Result of `minimize_l1_objective`."""

    x: np.ndarray
    objective: float
    success: bool
    message: str


def minimize_l1_objective(
    x0: np.ndarray,
    weights: Sequence[float],
    operators: Sequence[np.ndarray | None],
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    lb: np.ndarray | float = 0.0,
    ub: np.ndarray | float = 1.0,
    x_init: np.ndarray | None = None,
) -> L1MinimizeResult:
    """Minimize a sum of weighted L1-norm penalty terms, each measuring a
    (possibly linearly-transformed) deviation from a reference point `x0`,
    subject to linear equality/inequality constraints and box bounds, via
    direct nonsmooth SLSQP minimization.

    Original: HSF/11. Portfolio Optimization/chap11_net_zero5.m's
    `fmincon(R_Mix, ...)` branch; promoted from HSF-Notebooks chapter 11h's
    `solve_bond_l1_duration_neutral_smooth` (generalized -- see module
    docstring).

    Solves::

        minimize_x   sum_k weights[k] * ||operators[k] @ (x - x0)||_1
        subject to   a_eq @ x == b_eq
                     a_ub @ x <= b_ub          (if given)
                     lb <= x <= ub

    where `operators[k] is None` denotes the identity map for that term
    (i.e. an elementwise L1 penalty on `x - x0` directly).

    Parameters
    ----------
    x0 : array_like, shape (n,)
        Reference point each penalty term measures deviation from (e.g. a
        benchmark/baseline allocation).
    weights : sequence of float, length K
        Scalar weight for each of the K penalty terms.
    operators : sequence of (array_like of shape (m_k, n)) or None, length K
        Linear operator for each penalty term; `None` means the identity
        (so that term reduces to `weights[k] * ||x - x0||_1`).
    a_eq, b_eq : array_like, shape (p, n) and (p,)
        Linear equality constraints `a_eq @ x == b_eq`.
    a_ub, b_ub : array_like, shape (q, n) and (q,), optional
        Linear inequality constraints `a_ub @ x <= b_ub`.
    lb, ub : array_like or float, default 0.0 / 1.0
        Box bounds on `x`, scalar (broadcast to every component) or
        per-component.
    x_init : array_like, shape (n,), optional
        Initial guess handed to SLSQP. Defaults to `x0` (matching the
        notebook's own default, a warm start at the reference point).

    Returns
    -------
    L1MinimizeResult
        `x`: the SLSQP solution. `objective`: the objective value at `x`.
        `success`: whether SLSQP reported convergence. `message`: SLSQP's
        termination message.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.shape[0]

    if len(weights) != len(operators):
        raise ValueError("minimize_l1_objective: weights and operators must have the same length")
    ops: list[np.ndarray | None] = [
        None if op is None else np.asarray(op, dtype=float) for op in operators
    ]

    a_eq = np.asarray(a_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float).flatten()

    def objective(x: np.ndarray) -> float:
        total = 0.0
        for weight, op in zip(weights, ops, strict=True):
            delta = x - x0 if op is None else op @ (x - x0)
            total += weight * np.abs(delta).sum()
        return total

    constraints: list[dict] = [{"type": "eq", "fun": lambda x: a_eq @ x - b_eq}]
    if a_ub is not None:
        a_ub_arr = np.asarray(a_ub, dtype=float)
        b_ub_arr = np.asarray(b_ub, dtype=float).flatten()
        constraints.append({"type": "ineq", "fun": lambda x: b_ub_arr - a_ub_arr @ x})

    lb_arr = np.full(n, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.full(n, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    x_init_arr = x0.copy() if x_init is None else np.asarray(x_init, dtype=float)

    res = minimize(
        objective,
        x_init_arr,
        method="SLSQP",
        bounds=list(zip(lb_arr, ub_arr, strict=True)),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    return L1MinimizeResult(
        x=res.x,
        objective=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
    )
