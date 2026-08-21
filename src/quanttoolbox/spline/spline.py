"""Cubic smoothing splines: fit, evaluate (value/derivative/integral), and
invert.

Ported from QuantToolBox/spline/{csspline,dspline,fspline,intspline,
invspline}.m (+ band.m/bandrv.m/bandsolpd.m/rotater.m -- see note below).

Consolidation notes:

- The original implements its own banded Cholesky solver
  (band.m/bandrv.m/bandsolpd.m, ported from a GAUSS procedure) purely as
  an efficiency trick for solving the smoothing-spline system, which has
  half-bandwidth 2 by construction. This is replaced here with
  ``scipy.linalg.cho_factor``/``cho_solve`` on the (small, dense) system
  matrix directly -- standard, well-tested LAPACK-backed routines, with
  no need to hand-roll banded storage/factorization. ``rotater.m`` (a
  GAUSS row-rotation idiom used only to build the banded matrices from
  column vectors) is likewise not ported; the same tridiagonal/banded
  matrices are built directly with ``numpy.diag`` instead.
- ``fspline.m`` (evaluate spline value only) is redundant with
  ``dspline.m``'s k=0 case; only ``evaluate_spline`` (covering all of
  value/derivative/integral via a ``order=`` argument, i.e. ``dspline``)
  is ported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class SplineCoefficients:
    """Cubic spline in piecewise form: on [x[i], x[i+1]),
    s(x) = c0[i] + c1[i]*z + c2[i]*z^2/2 + c3[i]*z^3/6, z = x - x[i]."""

    x: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    c3: np.ndarray


def fit_smoothing_spline(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, p: float = 0.5
) -> SplineCoefficients:
    """Fit a cubic smoothing spline: minimizes
    p * integral(s''(t)^2 dt) + (1-p) * sum(w_i * (y_i - s(x_i))^2).

    p=1: pure interpolating spline (no smoothing). p=0: linear least-squares
    fit (maximum smoothing). x must be strictly increasing.

    Original: spline/csspline.m
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    n = x.shape[0]
    w_arr = np.ones(n) if w is None else np.asarray(w, dtype=float).flatten()
    w_arr = w_arr / w_arr.mean()

    t = 1 - p
    n1, n2 = n - 1, n - 2

    dx = x[1:] - x[:-1]
    if np.any(dx <= 0):
        raise ValueError("fit_smoothing_spline: x must be strictly increasing")

    dy = (y[1:] - y[:-1]) / dx
    dxi = 1.0 / dx

    # R: (n2, n2) tridiagonal "integrated basis" matrix
    r = np.zeros((n2, n2))
    r[np.arange(n2), np.arange(n2)] = 2 * (dx[:n2] + dx[1 : n2 + 1])
    if n2 > 1:
        off = dx[1:n2]
        r[np.arange(n2 - 1), np.arange(1, n2)] = off
        r[np.arange(1, n2), np.arange(n2 - 1)] = off
    r = r / 6

    # Q: (n2, n) second-difference operator (note: n columns, matching the
    # n original data points -- not n1 as a naive column count might
    # suggest; verified against the original's rotater-based construction,
    # which builds an n2 x n matrix before transposing).
    q = np.zeros((n2, n))
    q[np.arange(n2), np.arange(n2)] = dxi[:n2]
    q[np.arange(n2), np.arange(1, n2 + 1)] = -dxi[:n2] - dxi[1 : n2 + 1]
    q[np.arange(n2), np.arange(2, n2 + 2)] = dxi[1 : n2 + 1]

    qw = t * (q @ np.diag(1.0 / w_arr) @ q.T) + p * r
    qy = dy[1:n1] - dy[:n2]

    try:
        chol = cho_factor(qw)
        u = cho_solve(chol, qy)
    except np.linalg.LinAlgError:
        chol = cho_factor(r)
        u = cho_solve(chol, qy)

    c2 = np.concatenate([[0.0], p * u, [0.0]])
    u_step2 = (np.concatenate([u, [0.0]]) - np.concatenate([[0.0], u])) / dx  # length n1
    c0 = y - (t / w_arr) * (np.concatenate([u_step2, [0.0]]) - np.concatenate([[0.0], u_step2]))
    c3 = np.concatenate([(c2[1:] - c2[:n1]) / dx, [0.0]])

    c1_part1 = (c0[1:] - c0[:n1]) / dx - dx * (c2[:n1] / 3 + c2[1:] / 6)
    c1_last = c1_part1[n1 - 1] + dx[n1 - 1] * (c2[n1 - 1] + c2[n1]) / 2
    c1 = np.concatenate([c1_part1, [c1_last]])

    return SplineCoefficients(x=x, c0=c0, c1=c1, c2=c2, c3=c3)


def evaluate_spline(spline: SplineCoefficients, x: np.ndarray, order: int = 0) -> np.ndarray:
    """Evaluate a spline, its derivatives, or its integral at points x.

    order=-1: integral (from the spline's first knot). order=0 (default):
    value. order=1/2/3: first/second/third derivative.

    Original: spline/dspline.m (also covers spline/fspline.m's order=0 case)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    c = spline
    # MATLAB's `1 + sum(x > knots[2:])` gives a 1-indexed position; since
    # Python arrays are 0-indexed, the same bracketing search needs no "+1".
    idx = np.sum(x[None, :] > c.x[1:, None], axis=0)
    idx = np.clip(idx, 0, c.x.shape[0] - 1)
    z = x - c.x[idx]
    pos = (z >= 0).astype(float)

    c0, c1, c2, c3 = c.c0[idx], c.c1[idx], c.c2[idx], c.c3[idx]

    if order == -1:
        return z * (c0 + z * (c1 / 2 + pos * z * (c2 / 6 + z * c3 / 24)))
    elif order == 0:
        return c0 + z * (c1 + pos * z * (c2 / 2 + z * c3 / 6))
    elif order == 1:
        return c1 + pos * z * (c2 + z * c3 / 2)
    elif order == 2:
        return pos * (c2 + z * c3)
    elif order == 3:
        return pos * c3
    return np.zeros_like(x)


def integrate_spline(spline: SplineCoefficients, limits: np.ndarray) -> np.ndarray:
    """Definite integral of the spline between pairs of limits.

    limits: (2, n) array, limits[0] = upper bounds, limits[1] = lower bounds.

    Original: spline/intspline.m
    """
    limits = np.asarray(limits, dtype=float)
    if limits.shape[0] != 2:
        raise ValueError("integrate_spline: limits must have shape (2, n)")

    x = spline.x
    fuzz = np.mean(np.abs(x)) * 1e-12
    n_calls = limits.shape[1]
    area = np.zeros(n_calls)

    for i in range(n_calls):
        upper, lower = limits[0, i], limits[1, i]
        interior = x[(x > lower) & (x < upper)]
        if interior.shape[0] == 0:
            area[i] = (
                evaluate_spline(spline, np.array([upper]), order=-1)[0]
                - evaluate_spline(spline, np.array([lower]), order=-1)[0]
            )
        else:
            pts = np.concatenate([interior - fuzz, [upper]])
            area[i] = (
                np.sum(evaluate_spline(spline, pts, order=-1))
                - evaluate_spline(spline, np.array([lower + fuzz]), order=-1)[0]
            )

    return area


def invert_spline(spline: SplineCoefficients, y: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Invert a (monotonic) spline: find x such that spline(x) = y, via
    Newton's method starting from the bracketing knot.

    Original: spline/invspline.m
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    c = spline
    n = c.x.shape[0]

    decreasing = c.c0[0] > c.c0[1]
    # same 0-indexing adjustment as evaluate_spline -- no "+1"
    if decreasing:
        idx = np.sum(y[None, :] < c.c0[1:, None], axis=0)
    else:
        idx = np.sum(y[None, :] > c.c0[1:, None], axis=0)
    idx = np.clip(idx, 0, n - 1)

    xi = c.x[idx]
    yi = y - c.c0[idx]
    ci = np.column_stack([c.c1[idx], c.c2[idx] / 2, c.c3[idx] / 6])
    cd = ci * np.array([1.0, 2.0, 3.0])

    xt = np.zeros_like(y)
    for _ in range(20):
        pos = (xt > 0).astype(float)
        yt = xt * (ci[:, 0] + pos * xt * (ci[:, 1] + xt * ci[:, 2]))
        dydx = cd[:, 0] + pos * xt * (cd[:, 1] + xt * cd[:, 2])
        xn = xt + (yi - yt) / dydx
        if np.all((np.abs(xn - xt) < tol) & (np.abs(yi - yt) < tol)):
            xt = xn
            break
        xt = xn

    return xi + xt
