"""Reducing a linear system to full row rank by dropping redundant rows.

Not ported from the MATLAB HSF toolbox -- no `.m` file in `hfs-archive`
implements this. Promoted from HSF-Notebooks chapter 11h.

Motivation: `scipy.optimize.minimize(..., method="SLSQP")` builds and
factorizes the Jacobian of the active equality constraints at each step. If
the equality-constraint matrix `A_eq` is rank-deficient (e.g. a duplicated
or linearly-dependent row, easy to end up with when constraints are
assembled programmatically from overlapping groupings), that Jacobian is
singular and SLSQP can silently return the initial guess unchanged rather
than raising -- a hard-to-diagnose failure mode. Pruning `A_eq`/`b_eq` down
to an equivalent full-row-rank system before handing it to the optimizer
avoids this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FullRowRankSystem:
    """An equality system `a @ x == b`, reduced to full row rank."""

    a: np.ndarray
    b: np.ndarray
    kept_rows: np.ndarray


def reduce_to_full_row_rank(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> FullRowRankSystem:
    """Drop rows of the linear equality system `a @ x == b` that don't
    increase the rank of `a`, returning an equivalent full-row-rank system.

    Rows are scanned in order and kept greedily: a row is retained only if
    appending it to the rows already kept increases `matrix_rank`. The
    result `a` (and correspondingly `b`) therefore has `a.shape[0] ==
    np.linalg.matrix_rank(a)`, and the returned system is equivalent to the
    original wherever the original was consistent (dropped rows are exactly
    those linearly dependent on rows already kept).

    Not ported from the MATLAB HSF toolbox -- no `.m` file in `hfs-archive`
    implements this. Promoted from HSF-Notebooks chapter 11h, where it
    guards `scipy.optimize.minimize(..., method="SLSQP")` against a
    rank-deficient equality-constraint Jacobian (which SLSQP otherwise fails
    on silently, returning the initial guess unchanged instead of raising).

    Parameters
    ----------
    a : array_like, shape (m, n)
        Equality-constraint matrix.
    b : array_like, shape (m,)
        Equality-constraint right-hand side.
    tol : float, default 1e-10
        Tolerance passed to `numpy.linalg.matrix_rank` when testing whether
        a candidate row increases rank.

    Returns
    -------
    FullRowRankSystem
        `a`: the retained rows of the constraint matrix, shape (r, n).
        `b`: the correspondingly retained entries of the right-hand side,
        shape (r,).
        `kept_rows`: the 0-indexed row indices (into the original `a`/`b`)
        that were retained, shape (r,).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    keep: list[int] = []
    rows: list[np.ndarray] = []
    for i in range(a.shape[0]):
        cand = np.vstack(rows + [a[i]]) if rows else a[i : i + 1]
        if np.linalg.matrix_rank(cand, tol=tol) > len(rows):
            rows.append(a[i])
            keep.append(i)
    kept_rows = np.array(keep, dtype=int)
    return FullRowRankSystem(a=a[kept_rows], b=b[kept_rows], kept_rows=kept_rows)
