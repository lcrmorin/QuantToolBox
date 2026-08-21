"""Special matrix operations: vec/vech/xpnd, commutation/duplication/elimination.

Ported from QuantToolBox/matrix/{vec,vech,vecr,xpnd,commutation_matrix,
duplication_matrix,elimination_matrix,reshapec,reshaper,diagrv,lowmat,
upmat,design}.m

Key translation notes (apply throughout this module and beyond):

- MATLAB is column-major; NumPy defaults to row-major (C order). The
  original ``vec(X)`` (column-major flatten) is ``X.flatten(order="F")``.
  The original ``vecr(X)`` (row-major flatten, called "vec by rows" in the
  MATLAB code) is simply ``X.flatten(order="C")`` -- no special casing
  needed, it's NumPy's default.
- MATLAB is 1-indexed with inclusive slice ends; all loop bounds below are
  converted to 0-indexed, exclusive-end Python/NumPy equivalents.
- MATLAB "column vectors" (shape (n,1)) are represented here as flat 1-D
  NumPy arrays (shape (n,)), which is the idiomatic NumPy convention.
  Downstream ports should not assume a trailing singleton dimension.
- ``packr`` (drop NaN rows) is not ported as a standalone utility; where it
  was only used to strip NaN placeholders from a padded array (as in
  ``elimination_matrix``), it's inlined via boolean masking.
"""

from __future__ import annotations

import numpy as np


def vec(x: np.ndarray) -> np.ndarray:
    """Column-major ("Fortran order") flatten of a matrix.

    Original: matrix/vec.m
    """
    return np.asarray(x).flatten(order="F")


def vecr(x: np.ndarray) -> np.ndarray:
    """Row-major ("C order") flatten of a matrix.

    Original: matrix/vecr.m -- MATLAB computed this as ``vec(x')``, which
    is algebraically identical to a native row-major flatten of ``x``.
    """
    return np.asarray(x).flatten(order="C")


def vech(a: np.ndarray, method: int | str = 1) -> np.ndarray:
    """Half-vectorization: stack the lower triangle of a square matrix.

    method=1 (default) or 'r': row-wise ordering (matches MATLAB default).
    method=2 or 'c': column-wise ordering (the more common convention
    elsewhere -- e.g. this is what most other vech implementations do).

    Original: matrix/vech.m
    """
    if isinstance(method, str):
        method = 1 if method == "r" else 2

    a = np.asarray(a)
    r, c = a.shape
    if r != c:
        raise ValueError("vech: matrix not square")

    if method == 1:
        at = a.T
        mask = np.triu(np.ones((r, r), dtype=bool))
        return at.flatten(order="F")[mask.flatten(order="F")]
    else:
        return np.concatenate([a[i:r, i] for i in range(r)])


def xpnd(v: np.ndarray, method: int | str = 1) -> np.ndarray:
    """Inverse of vech: expand a half-vectorized vector back to a symmetric matrix.

    method=1 (default) or 'r': row-wise ordering (matches MATLAB default).
    method=2 or 'c': column-wise ordering.

    Original: matrix/xpnd.m
    """
    if isinstance(method, str):
        method = 1 if method == "r" else 2

    v = np.asarray(v).flatten()
    p = v.shape[0]
    n_float = (-1 + np.sqrt(1 + 8 * p)) / 2
    n = round(n_float)
    if abs(n_float - n) > 1e-4:
        raise ValueError("xpnd: the vector does not have the right dimension")

    a = np.zeros((n, n))

    if method == 1:
        for i in range(1, n + 1):
            start = i * (i - 1) // 2
            end = i * (i + 1) // 2
            x = v[start:end]
            a[i - 1, 0:i] = x
            a[0:i, i - 1] = x
    else:
        j = 0
        for i in range(n):
            length = n - i
            x = v[j : j + length]
            a[i:n, i] = x
            a[i, i:n] = x
            j += length

    return a


def commutation_matrix(m: int, n: int) -> np.ndarray:
    """Commutation matrix K_(m,n) such that K @ vec(A) == vec(A.T) for an m x n A.

    Original: matrix/commutation_matrix.m
    """
    p = m * n
    # entry (i, j), 1-indexed: i + j*m  -->  the target column of the 1 in row (i,j)
    idx_matrix = np.arange(1, m + 1)[:, None] + np.arange(0, n * m, m)[None, :]
    v = idx_matrix.flatten(order="C")  # vecr

    k = np.zeros((p, p))
    for i in range(p):
        k[i, v[i] - 1] = 1
    return k


def duplication_matrix(m: int) -> np.ndarray:
    """Duplication matrix D_m such that D @ vech(A) == vec(A) for symmetric A.

    Original: matrix/duplication_matrix.m
    """
    p = m * (m + 1) // 2
    v = np.arange(1, p + 1)
    a = xpnd(v, method=2)
    vv = vec(a)

    d = np.zeros((m * m, p))
    for i in range(m * m):
        d[i, int(vv[i]) - 1] = 1
    return d


def elimination_matrix(m: int) -> np.ndarray:
    """Elimination matrix L_m such that L @ vec(A) == vech(A) for square A.

    Original: matrix/elimination_matrix.m
    """
    p = m * (m + 1) // 2
    row = np.arange(1, m + 1)
    col = np.arange(0, m * m, m)
    v = col[:, None] + row[None, :]

    shift_v = np.full((m, m), np.nan)
    for i in range(m):
        length = m - i
        shift_v[i, 0:length] = v[i, i:m]

    flat = shift_v.flatten(order="C")  # vecr
    flat = flat[~np.isnan(flat)]  # packr

    elim = np.zeros((p, m * m))
    for i in range(p):
        elim[i, int(flat[i]) - 1] = 1
    return elim


def reshapec(v: np.ndarray, n: int, m: int) -> np.ndarray:
    """Reshape (and recycle/truncate as needed) a vector into an n x m matrix,
    column-major.

    Original: matrix/reshapec.m
    """
    v = vec(np.asarray(v))
    r = v.shape[0]
    nm = n * m
    if r > nm:
        v = v[:nm]
    elif r < nm:
        nc = int(np.ceil(nm / r))
        v = np.tile(v, nc)[:nm]
    return v.reshape((n, m), order="F")


def reshaper(v: np.ndarray, n: int, m: int) -> np.ndarray:
    """Reshape (and recycle/truncate as needed) a vector into an n x m matrix,
    row-major.

    Original: matrix/reshaper.m
    """
    v = vecr(np.asarray(v))
    r = v.shape[0]
    nm = n * m
    if r > nm:
        v = v[:nm]
    elif r < nm:
        nc = int(np.ceil(nm / r))
        v = np.tile(v, nc)[:nm]
    return v.reshape((n, m), order="C")


def diagrv(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return a copy of x with its diagonal replaced by v.

    Original: matrix/diagrv.m
    """
    y = np.array(x, copy=True, dtype=float)
    np.fill_diagonal(y, v)
    return y


def lowmat(x: np.ndarray, v: float = 0.0) -> np.ndarray:
    """Return the lower-triangular part of x (including diagonal), with the
    strictly-upper part filled with v.

    Original: matrix/lowmat.m
    """
    x = np.asarray(x)
    r, c = x.shape
    return np.tril(x) + np.triu(np.full((r, c), v), 1)


def upmat(x: np.ndarray, v: float = 0.0) -> np.ndarray:
    """Return the upper-triangular part of x (including diagonal), with the
    strictly-lower part filled with v.

    Original: matrix/upmat.m
    """
    x = np.asarray(x)
    r, c = x.shape
    return np.triu(x) + np.tril(np.full((r, c), v), -1)


def design(v: np.ndarray) -> np.ndarray:
    """Build a 0/1 design (indicator) matrix from a vector of category indices.

    v[i] (1-indexed category, values <= 0 produce an all-zero row) selects
    the column that gets a 1 in row i.

    Original: matrix/design.m
    """
    v = np.asarray(v).flatten()
    n = v.shape[0]
    v_rounded = np.round(v).astype(int)
    m = int(v_rounded.max())
    x = np.zeros((n, m))
    for i in range(n):
        if v_rounded[i] > 0:
            x[i, v_rounded[i] - 1] = 1.0
    return x
