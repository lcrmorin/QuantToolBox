"""Direct nonsmooth SLSQP minimization of a mixed L1-norm objective under
linear constraints.

Turns out to have a MATLAB antecedent in `hfs-archive` after all, despite
no filename in the archive matching `solve_bond_l1_duration_neutral_smooth`:
`HSF/11. Portfolio Optimization/chap11_net_zero5.m`'s
`fmincon(R_Mix, x0, C, D, A, B, lb, ub, [], options_fmincon)` call -- same
`b`/`CI`/`MD`/`DTS`/`Sector` data, the same `R_AS`/`R_DTS`/`R_Mix` L1
formulas, the same budget/duration-neutrality/carbon-intensity-reduction
constraints and R-target glide path, and even the same solver tolerances
(`MaxIterations=1000`, `OptimalityTolerance=1e-15`, matching the notebook's
and this module's `maxiter=1000`/`ftol=1e-15`). Not discoverable by
filename search since the MATLAB script is a chapter run-script (named
after the chapter/section, like its neighbors `chap11_net_zero{1..18}.m`),
not after the function it happens to define inline -- so it isn't tracked
in `docs/migration_map.md`'s `hsf/` toolbox-library-folder mapping either.
Promoted here from HSF-Notebooks chapter 11h's own
`solve_bond_l1_duration_neutral_smooth` (the notebook's own Python
translation, itself apparently written independently of the MATLAB script
rather than as a direct port, but implementing the identical algorithm),
generalized to plain array arguments and kept as an independent
SLSQP-based algorithm, deliberately not routed through
quanttoolbox.optim.quadprog.solve_qp -- see docstring of the notebook's
LP-based `solve_bond_l1_duration_neutral` counterpart (chap11_net_zero5.m's
`linprog` branch) for why.

Translation notes:

- The notebook's `solve_bond_l1_duration_neutral_smooth` minimizes
  ``varphi_DTS * ||C_DTS @ (w - b)||_1 + 0.5 * ||w - b||_1`` subject to a
  budget constraint, a duration-neutrality constraint, and a carbon-
  intensity-reduction inequality, all specific to that chapter's bond
  problem. The two L1 terms differ only in their linear operator
  (``C_DTS`` vs. the identity) and scalar weight (``varphi_DTS`` vs. a
  fixed 0.5) -- so here that is generalized directly into its natural
  generic shape: an arbitrary list of ``(weight, operator)`` pairs summed
  into ``sum_k weight_k * ||operator_k @ (x - x0)||_1``, with generic
  linear equality/inequality constraints and box bounds standing in for
  the notebook's budget/duration/carbon-intensity rows. The underlying
  algorithm (direct SLSQP minimization of the nonsmooth L1 objective, via
  scipy's ``method="SLSQP"``, same tolerances/options) is unchanged.
- ``operator=None`` in a penalty term denotes the identity map (matches
  the notebook's ``R_AS`` term, an elementwise L1 penalty with no linear
  transform).
- The notebook's function returns only ``res.x``, silently accepting
  whatever SLSQP converges to. `minimize_l1_objective` instead returns a
  `L1MinimizeResult` surfacing `success`/`message` alongside `x`, so a
  non-convergent solve is visible to the caller rather than swallowed --
  a pure interface addition, not a change to the algorithm itself.
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
