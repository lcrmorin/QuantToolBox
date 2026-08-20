"""Tests for quanttoolbox.linalg.special_matrices.

These check mathematical properties/round-trips (no MATLAB reference
outputs available in this environment) -- if you have access to MATLAB,
consider also asserting exact numeric equality against saved .m outputs
for a few fixed inputs, to catch any subtle indexing divergence.
"""

import numpy as np
import pytest

from quanttoolbox.linalg.special_matrices import (
    commutation_matrix,
    design,
    diagrv,
    duplication_matrix,
    elimination_matrix,
    lowmat,
    reshapec,
    reshaper,
    upmat,
    vec,
    vech,
    vecr,
    xpnd,
)


def test_vec_matches_fortran_flatten(rng):
    x = rng.standard_normal((3, 4))
    assert np.array_equal(vec(x), x.flatten(order="F"))


def test_vecr_matches_c_flatten(rng):
    x = rng.standard_normal((3, 4))
    assert np.array_equal(vecr(x), x.flatten(order="C"))


@pytest.mark.parametrize("method", [1, 2, "r", "c"])
def test_vech_xpnd_roundtrip(rng, method):
    n = 5
    a = rng.standard_normal((n, n))
    a = (a + a.T) / 2  # symmetrize
    v = vech(a, method=method)
    assert v.shape[0] == n * (n + 1) // 2
    a_back = xpnd(v, method=method)
    assert np.allclose(a_back, a)


def test_vech_rejects_non_square():
    with pytest.raises(ValueError):
        vech(np.zeros((3, 4)))


def test_xpnd_rejects_bad_length():
    with pytest.raises(ValueError):
        xpnd(np.zeros(4))  # 4 is not a triangular number


def test_commutation_matrix_property(rng):
    m, n = 3, 4
    a = rng.standard_normal((m, n))
    k = commutation_matrix(m, n)
    assert k.shape == (m * n, m * n)
    assert np.allclose(k @ vec(a), vec(a.T))


def test_commutation_matrix_is_permutation():
    k = commutation_matrix(3, 4)
    # every row/col sums to exactly 1, entries are 0/1
    assert set(np.unique(k)) <= {0.0, 1.0}
    assert np.allclose(k.sum(axis=0), 1.0)
    assert np.allclose(k.sum(axis=1), 1.0)


def test_duplication_matrix_property(rng):
    m = 4
    a = rng.standard_normal((m, m))
    a = (a + a.T) / 2
    d = duplication_matrix(m)
    assert d.shape == (m * m, m * (m + 1) // 2)
    assert np.allclose(d @ vech(a, method=2), vec(a))


def test_elimination_matrix_property(rng):
    m = 4
    a = rng.standard_normal((m, m))
    a = (a + a.T) / 2
    elim = elimination_matrix(m)
    assert elim.shape == (m * (m + 1) // 2, m * m)
    assert np.allclose(elim @ vec(a), vech(a, method=2))


def test_reshapec_exact_fit():
    v = np.arange(6)
    x = reshapec(v, 2, 3)
    assert np.array_equal(x, v.reshape((2, 3), order="F"))


def test_reshapec_truncates():
    v = np.arange(10)
    x = reshapec(v, 2, 3)
    assert x.shape == (2, 3)


def test_reshapec_recycles():
    v = np.arange(3)
    x = reshapec(v, 2, 3)  # needs 6 elements, only 3 given -> tiled
    assert x.shape == (2, 3)


def test_reshaper_matches_transpose_identity():
    v = np.arange(6)
    n, m = 2, 3
    # x = reshape(v,m,n)' in MATLAB == reshape(v,(n,m),order='C') in numpy
    expected = v.reshape((m, n), order="F").T
    assert np.array_equal(reshaper(v, n, m), expected)


def test_diagrv_replaces_diagonal():
    x = np.zeros((3, 3))
    y = diagrv(x, [1.0, 2.0, 3.0])
    assert np.array_equal(np.diag(y), [1.0, 2.0, 3.0])
    # off-diagonal untouched
    assert np.array_equal(y - np.diag(np.diag(y)), np.zeros((3, 3)))


def test_lowmat_upmat_partition_full_matrix():
    x = np.arange(9).reshape(3, 3).astype(float)
    lo = lowmat(x, v=0.0)
    up = upmat(x, v=0.0)
    # diagonal double-counted once each; off-diagonal parts should
    # reconstruct x when combined minus one diagonal copy
    reconstructed = lo + up - np.diag(np.diag(x))
    assert np.allclose(reconstructed, x)


def test_lowmat_fill_value():
    x = np.ones((3, 3))
    lo = lowmat(x, v=-1.0)
    assert lo[0, 1] == -1.0
    assert lo[0, 2] == -1.0
    assert lo[1, 0] == 1.0


def test_design_matrix_basic():
    v = np.array([1, 3, 2, 0, 1])
    x = design(v)
    assert x.shape == (5, 3)
    expected = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )
    assert np.array_equal(x, expected)
