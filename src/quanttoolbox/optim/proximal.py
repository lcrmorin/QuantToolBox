"""Proximal operators for L1/L2/Linf norms, box/equality/inequality
constraints, and combined ("Dykstra alternating projection") constraint sets.

Ported from QuantToolBox/optim/{proximal_L1,proximal_L2,proximal_Linfinity,
proximal_max,proximal_bounds,proximal_equality,proximal_inequality,
proximal_linear_constraints,proximal_turnover,soft_thresholding}.m

Translation notes:

- ``proximal_L1`` and ``soft_thresholding`` (2-argument form) are
  algebraically identical in the original; only ``soft_thresholding`` is
  kept here (as the more common name), with ``proximal_l1`` as an alias.
- ``proximal_bounds``' original had two branches: a trivial closed-form
  clip (``Proximal_Algorithm == 1``) and a redundant ``quadprog`` call for
  the same box projection (``Proximal_Algorithm == 2``, which solves the
  *exact same problem* -- projecting onto a box has a closed-form
  solution, so a QP solver is never actually needed). Only the closed-form
  clip is ported.
- ``proximal_equality``'s Dykstra-loop branch was commented out in the
  original in favor of the closed-form ``pinv`` solution (which is exact
  for a single equality-constraint projection); that closed form is what's
  ported here.
- ``proximal_inequality``/``proximal_linear_constraints``/
  ``proximal_turnover``'s Dykstra alternating-projection loops (combining
  multiple constraint sets) are preserved, since combined constraint
  projection generally has no closed form.
- MATLAB's ``global Proximal_MaxIters`` is replaced by
  ``quanttoolbox.config.ProximalConfig``.
"""

from __future__ import annotations

import numpy as np

from quanttoolbox.config import ProximalConfig


def soft_thresholding(
    v: np.ndarray, lambda_minus: float, lambda_plus: float | None = None
) -> np.ndarray:
    """Soft-thresholding / proximal operator of the L1 norm.

    One-argument-lambda form: symmetric soft threshold, sign(v) * max(|v|-lambda, 0).
    Two-argument form: asymmetric threshold with separate positive/negative shrinkage.

    Original: optim/soft_thresholding.m (also optim/proximal_L1.m, identical)
    """
    v = np.asarray(v, dtype=float)
    if lambda_plus is None:
        lam = lambda_minus
        return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)
    positive_part = np.maximum(v - lambda_plus, 0.0)
    negative_part = np.maximum(-(v + lambda_minus), 0.0)
    return positive_part - negative_part


proximal_l1 = soft_thresholding  # alias, matches optim/proximal_L1.m


def proximal_l2(v: np.ndarray, lambda_: float) -> np.ndarray:
    """Proximal operator of the (scaled) L2 norm: shrinks v toward the
    origin by at most lambda_ in Euclidean length.

    Original: optim/proximal_L2.m
    """
    v = np.asarray(v, dtype=float)
    norm_v = float(np.linalg.norm(v))
    return (1 - lambda_ / max(lambda_, norm_v)) * v


def proximal_max(v: np.ndarray, lambda_: float) -> np.ndarray:
    """Proximal operator of lambda * max(v): caps the largest entries of v
    so that the total amount "shaved off" sums to lambda_ (used inside
    proximal_Linfinity / projection_L1's simplex-style water-filling step).

    Original: optim/proximal_max.m
    """
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    sorted_v = np.sort(v)[::-1]
    c = (np.cumsum(sorted_v) - lambda_) / np.arange(1, n + 1)
    mask = sorted_v > c
    rk = np.where(mask)[0][-1]  # last index where condition holds
    s = c[rk]
    return np.minimum(v, s)


def proximal_linfinity(v: np.ndarray, lambda_: float) -> np.ndarray:
    """Proximal operator of lambda * ||v||_infinity.

    Original: optim/proximal_Linfinity.m
    """
    v = np.asarray(v, dtype=float)
    return np.sign(v) * proximal_max(np.abs(v), lambda_)


def proximal_bounds(v: np.ndarray, lb: np.ndarray | float, ub: np.ndarray | float) -> np.ndarray:
    """Projection onto a box [lb, ub] (closed-form clip).

    Original: optim/proximal_bounds.m
    """
    v = np.asarray(v, dtype=float)
    return np.clip(v, lb, ub)


def proximal_equality(v: np.ndarray, a_eq: np.ndarray, b_eq: np.ndarray) -> tuple[np.ndarray, int]:
    """Projection onto the affine subspace {x : A_eq @ x == b_eq}.

    Original: optim/proximal_equality.m

    Returns (x, retcode) with retcode always 0 (kept for API parity with
    proximal_inequality/proximal_linear_constraints, which can fail).
    """
    v = np.asarray(v, dtype=float)
    a_eq = np.asarray(a_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float).flatten()
    x = v - np.linalg.pinv(a_eq) @ (a_eq @ v - b_eq)
    return x, 0


def proximal_inequality(
    v: np.ndarray, c_ineq: np.ndarray, d_ineq: np.ndarray, config: ProximalConfig | None = None
) -> tuple[np.ndarray, int]:
    """Projection onto a polyhedron {x : C_ineq @ x <= D_ineq} via Dykstra's
    alternating projection algorithm (cycling through each row/half-space).

    Original: optim/proximal_inequality.m

    Returns (x, retcode); retcode is -1 if max_iters was reached without
    convergence, 0 otherwise.
    """
    if config is None:
        config = ProximalConfig()

    v = np.asarray(v, dtype=float)
    c_ineq = np.asarray(c_ineq, dtype=float)
    d_ineq = np.asarray(d_ineq, dtype=float).flatten()
    n = v.shape[0]
    n_ineq = c_ineq.shape[0]

    x = v.copy()
    z = np.zeros((n, n_ineq))
    retcode = 0

    for _it in range(config.max_iters):
        x_old = x.copy()
        for i in range(n_ineq):
            vi = x + z[:, i]
            c = c_ineq[i, :]
            d = d_ineq[i]
            u = np.dot(c, vi) - d
            x_new = vi - u * (u >= 0) * c / np.dot(c, c)
            z[:, i] = x + z[:, i] - x_new
            x = x_new
        if np.allclose(x_old, x):
            break
    else:
        retcode = -1

    return x, retcode


def proximal_linear_constraints(
    v: np.ndarray,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    c_ineq: np.ndarray | None = None,
    d_ineq: np.ndarray | None = None,
    lb: np.ndarray | float | None = None,
    ub: np.ndarray | float | None = None,
    config: ProximalConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Projection onto the intersection of an affine subspace, a polyhedron,
    and a box, via Dykstra's alternating projection algorithm. Pass None
    for any constraint set to omit it.

    Original: optim/proximal_linear_constraints.m

    Note: like the original, the exact-equality convergence check
    (``x1 == x4`` in MATLAB, ``np.allclose`` here) can occasionally exit
    early during a temporary plateau in the iterate sequence, before
    reaching the true intersection point -- this is most likely for
    inputs sitting near a "corner" where the constraint sets meet
    tangentially. If a result looks suspicious, verify constraint
    satisfaction directly and re-run with a larger ``max_iters`` and a
    perturbed starting point if needed.
    """
    if config is None:
        config = ProximalConfig()

    v = np.asarray(v, dtype=float)
    n = v.shape[0]

    x1 = v.copy()
    delta1 = np.zeros(n)
    delta2 = np.zeros(n)
    delta3 = np.zeros(n)
    retcode = 0

    for _it in range(config.max_iters):
        if a_eq is not None:
            x2, _ = proximal_equality(x1 + delta1, a_eq, b_eq)
            delta1 = x1 + delta1 - x2
        else:
            x2 = x1

        if c_ineq is not None:
            x3, _ = proximal_inequality(x2 + delta2, c_ineq, d_ineq, config)
            delta2 = x2 + delta2 - x3
        else:
            x3 = x2

        if lb is not None:
            x4 = proximal_bounds(x3 + delta3, lb, ub)
            delta3 = x3 + delta3 - x4
        else:
            x4 = x3

        if np.allclose(x1, x4):
            x1 = x4
            break
        x1 = x4
    else:
        retcode = -1

    return x1, retcode


def proximal_turnover(
    v: np.ndarray,
    a_eq: np.ndarray | None,
    b_eq: np.ndarray | None,
    c_ineq: np.ndarray | None,
    d_ineq: np.ndarray | None,
    lb: np.ndarray | float | None,
    ub: np.ndarray | float | None,
    x0: np.ndarray,
    tau: float | None,
    config: ProximalConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Projection onto the intersection of an affine subspace, a polyhedron,
    a box, and a turnover constraint (||x - x0||_1 <= tau), via Dykstra's
    alternating projection algorithm.

    Original: optim/proximal_turnover.m
    """
    if config is None:
        config = ProximalConfig()

    from quanttoolbox.optim.projection import projection_l1

    v = np.asarray(v, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n = v.shape[0]

    x1 = v.copy()
    delta1 = np.zeros(n)
    delta2 = np.zeros(n)
    delta3 = np.zeros(n)
    delta4 = np.zeros(n)
    retcode = 0

    for _it in range(config.max_iters):
        if a_eq is not None:
            x2, _ = proximal_equality(x1 + delta1, a_eq, b_eq)
            delta1 = x1 + delta1 - x2
        else:
            x2 = x1

        if c_ineq is not None:
            x3, _ = proximal_inequality(x2 + delta2, c_ineq, d_ineq, config)
            delta2 = x2 + delta2 - x3
        else:
            x3 = x2

        if lb is not None:
            x4 = proximal_bounds(x3 + delta3, lb, ub)
            delta3 = x3 + delta3 - x4
        else:
            x4 = x3

        if tau is not None:
            x5 = projection_l1((x4 - x0) + delta4, tau) + x0
            delta4 = x4 + delta4 - x5
        else:
            x5 = x4

        if np.allclose(x1, x5):
            x1 = x5
            break
        x1 = x5
    else:
        retcode = -1

    return x1, retcode
