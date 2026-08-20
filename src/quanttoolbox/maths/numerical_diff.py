"""Numerical gradient, Jacobian, and Hessian, with an adaptive
magnitude-scaled step size.

Ported from QuantToolbox/maths/{numerical_gradient,numerical_hessian,
numerical_jacobian,sign_operator}.m

Translation notes:

- ``numerical_gradient.m`` handles three cases based on output shape
  (scalar-output gradient, vector-output-matching-input gradient, and
  general Jacobian) via a single dispatch; here this is split into
  ``numerical_gradient`` (scalar-valued fun) and ``numerical_jacobian``
  (vector-valued fun) for clarity, matching how they're actually called
  elsewhere in this package.
- The step size dh is scaled per-parameter by
  ``max(|x0_i|, 0.01) * sign(x0_i)`` (falling back to a fixed direction
  when x0_i == 0), exactly as in the original, so the step is
  proportional to each parameter's magnitude rather than a fixed absolute
  value.
- method="forward" (default) or "central" difference, matching the
  original's method=1/2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _step_size(x0: np.ndarray, dh: float) -> np.ndarray:
    ax0 = np.abs(x0)
    dax0 = np.where(x0 != 0, x0 / np.where(ax0 == 0, 1, ax0), 1.0)
    dx = np.maximum(ax0, 1e-2)
    return dh * dx * dax0


def numerical_gradient(
    fun: Callable[[np.ndarray], float], x0: np.ndarray, dh: float = 1e-8, method: str = "forward"
) -> np.ndarray:
    """Numerical gradient of a scalar-valued function fun at x0 (each
    parameter perturbed one at a time, holding the others fixed).

    Original: maths/numerical_gradient.m (scalar-output case)
    """
    x0 = np.asarray(x0, dtype=float).flatten()
    p = x0.shape[0]
    f0 = float(fun(x0))

    step = _step_size(x0, dh)
    x1, x2 = x0 - step, x0 + step
    dx1, dx2 = x0 - x1, x2 - x0

    f1 = np.zeros(p)
    f2 = np.zeros(p)
    for i in range(p):
        xi2 = x0.copy()
        xi2[i] = x2[i]
        f2[i] = float(fun(xi2))
        if method == "central":
            xi1 = x0.copy()
            xi1[i] = x1[i]
            f1[i] = float(fun(xi1))

    if method == "central":
        dx = dx1 + dx2
        return (f2 - f1) / dx
    return (f2 - f0) / dx2


def numerical_jacobian(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    dh: float = 1e-8,
    method: str = "forward",
) -> np.ndarray:
    """Numerical Jacobian of a vector-valued function fun at x0: fun(x0) has
    shape (n,), x0 has shape (p,), result has shape (n, p).

    Original: maths/{numerical_gradient,numerical_jacobian}.m (Jacobian case)
    """
    x0 = np.asarray(x0, dtype=float).flatten()
    p = x0.shape[0]
    f0 = np.atleast_1d(fun(x0))
    n = f0.shape[0]

    step = _step_size(x0, dh)
    x1, x2 = x0 - step, x0 + step
    dx1, dx2 = x0 - x1, x2 - x0

    f1 = np.zeros((n, p))
    f2 = np.zeros((n, p))
    for i in range(p):
        xi2 = x0.copy()
        xi2[i] = x2[i]
        f2[:, i] = np.atleast_1d(fun(xi2))
        if method == "central":
            xi1 = x0.copy()
            xi1[i] = x1[i]
            f1[:, i] = np.atleast_1d(fun(xi1))

    if method == "central":
        dx = dx1 + dx2
        return (f2 - f1) / dx[None, :]
    return (f2 - f0[:, None]) / dx2[None, :]


def numerical_hessian(
    fun: Callable[[np.ndarray], float], x0: np.ndarray, dh: float = 6e-5
) -> np.ndarray:
    """Numerical Hessian of a scalar-valued function fun at x0, via
    second-order finite differences.

    Original: maths/numerical_hessian.m
    """
    x0 = np.asarray(x0, dtype=float).flatten()
    p = x0.shape[0]
    f0 = float(fun(x0))

    step = _step_size(x0, dh)
    x1 = x0 + step
    dx = x1 - x0
    e = np.diag(dx)

    f1 = np.array([fun(x0 + e[:, i]) for i in range(p)])

    f2 = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            f2[i, j] = fun(x0 + e[:, i] + e[:, j])
            if i != j:
                f2[j, i] = f2[i, j]

    return ((f2 - f1[:, None]) - f1[None, :] + f0) / np.outer(dx, dx)


def sign_operator(x: np.ndarray) -> np.ndarray:
    """Sign function: 1 if x>0, -1 if x<0, 0 if x==0 (equivalent to
    numpy.sign, provided here for direct call-site compatibility with the
    original).

    Original: maths/sign_operator.m
    """
    x = np.asarray(x)
    return (x > 0).astype(float) - (x < 0).astype(float)
