"""Scalar (or elementwise-vectorized) bisection root-finding, and linear
constraint explicit<->implicit (null-space) parametrization conversion.

Ported from QuantToolbox/optim/{bisection,bisection2,explicit2implicit,
implicit2explicit}.m

Translation notes:

- Both bisection variants are vectorized in the original (operating on
  arrays ``a``/``b`` elementwise, not just scalars) -- preserved here via
  plain NumPy elementwise operations.
- ``bisection2`` carries an auxiliary state variable ``z`` through the
  function evaluations (useful when ``fhandle`` also needs to return some
  side computation to warm-start the next evaluation); the Python
  signature keeps the same ``(y, z) = fhandle(x, z)`` calling convention.
- ``explicit2implicit``/``implicit2explicit`` convert between an explicit
  linear-constraint representation (``C @ x = c``) and an implicit
  null-space parametrization (``x = R @ r`` for free parameter r) --
  MATLAB's ``null(...)`` maps to ``scipy.linalg.null_space``.
- MATLAB's ``global BISECTION_Tol`` is replaced by
  ``quanttoolbox.config.BisectionConfig``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import linalg

from quanttoolbox.config import BisectionConfig


def bisection(
    fhandle: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray | float,
    b: np.ndarray | float,
    config: BisectionConfig | None = None,
) -> np.ndarray | float:
    """Find the root of fhandle within bracket [a, b] via bisection
    (elementwise, if a/b are arrays -- each element is bracketed and
    solved independently).

    Original: optim/bisection.m
    """
    if config is None:
        config = BisectionConfig()

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scalar_input = a.ndim == 0

    ya = fhandle(a)
    yb = fhandle(b)

    if np.any(ya * yb > 0):
        result = np.full(a.shape, np.nan)
        return float(result) if scalar_input else result

    if np.all(ya == 0):
        return float(a) if scalar_input else a.copy()
    if np.all(yb == 0):
        return float(b) if scalar_input else b.copy()

    increasing = ya >= 0  # if ya < 0, function increases a->b; else decreases

    c = (a + b) / 2
    for _ in range(config.max_iters):
        if np.max(np.abs(a - b)) <= config.tol:
            break
        c = (a + b) / 2
        yc = fhandle(c)
        e = np.where(increasing, yc > 0, yc < 0)
        a = np.where(e, c, a)
        b = np.where(e, b, c)

    c = (a + b) / 2
    return float(c) if scalar_input else c


def bisection2(
    fhandle: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    a: np.ndarray | float,
    b: np.ndarray | float,
    z0: np.ndarray,
    config: BisectionConfig | None = None,
) -> tuple[np.ndarray | float, np.ndarray]:
    """Bisection root-finding where fhandle also threads an auxiliary state
    z through each evaluation: (y, z) = fhandle(x, z).

    Original: optim/bisection2.m
    """
    if config is None:
        config = BisectionConfig()

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scalar_input = a.ndim == 0

    ya, za = fhandle(a, z0)
    yb, zb = fhandle(b, z0)
    zc = (za + zb) / 2

    if np.any(ya * yb > 0):
        result = np.full(a.shape, np.nan)
        return (float(result) if scalar_input else result), zc

    if np.all(ya == 0):
        return (float(a) if scalar_input else a.copy()), za
    if np.all(yb == 0):
        return (float(b) if scalar_input else b.copy()), zb

    increasing = ya >= 0

    c = (a + b) / 2
    for _ in range(config.max_iters):
        if np.max(np.abs(a - b)) <= config.tol:
            break
        c = (a + b) / 2
        yc, zc = fhandle(c, zc)
        e = np.where(increasing, yc > 0, yc < 0)
        a = np.where(e, c, a)
        b = np.where(e, b, c)

    c = (a + b) / 2
    return (float(c) if scalar_input else c), zc


def explicit_to_implicit(cc: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an explicit linear-constraint system (CC @ x = c) into an
    implicit null-space parametrization x = RR @ r + r0, returning (RR, r0).

    Original: optim/explicit2implicit.m
    """
    cc = np.asarray(cc, dtype=float)
    c = np.asarray(c, dtype=float).flatten()

    if cc.shape[0] != c.shape[0] or cc.shape[0] >= cc.shape[1]:
        raise ValueError("explicit_to_implicit: wrong size of CC and c")

    rr = linalg.null_space(cc)
    mx = np.max(np.abs(rr), axis=0)
    mx = mx + (mx == 0)
    rr = rr / mx

    r = np.linalg.pinv(cc.T @ cc) @ cc.T @ c
    return rr, r


def implicit_to_explicit(rr: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an implicit null-space parametrization (x = RR @ r) back into
    an explicit linear-constraint system (CC @ x = c).

    Original: optim/implicit2explicit.m
    """
    rr = np.asarray(rr, dtype=float)
    r = np.asarray(r, dtype=float).flatten()

    if rr.shape[0] != r.shape[0] or rr.shape[0] <= rr.shape[1]:
        raise ValueError("implicit_to_explicit: wrong size of RR and r")

    cc = linalg.null_space(rr.T).T
    mx = np.max(np.abs(cc), axis=1)
    mx = mx + (mx == 0)
    cc = cc / mx[:, None]

    c = cc @ r
    return cc, c
