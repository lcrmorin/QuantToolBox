"""Tests for quanttoolbox.linalg.rank_reduction."""

import numpy as np

from quanttoolbox.linalg.rank_reduction import reduce_to_full_row_rank


def test_rank_deficient_system_is_pruned_to_full_row_rank():
    # Row 2 is a linear combination of rows 0 and 1 (row0 + row1), so it's
    # redundant and should be dropped.
    a = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    )
    b = np.array([1.0, 2.0, 3.0])
    assert np.linalg.matrix_rank(a) == 2

    result = reduce_to_full_row_rank(a, b)

    assert result.a.shape[0] == 2
    assert np.linalg.matrix_rank(result.a) == result.a.shape[0]
    assert np.array_equal(result.kept_rows, np.array([0, 1]))
    assert np.allclose(result.a, a[[0, 1]])
    assert np.allclose(result.b, b[[0, 1]])


def test_full_rank_system_passes_through_unchanged():
    a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    b = np.array([1.0, 2.0, 3.0])
    assert np.linalg.matrix_rank(a) == a.shape[0]

    result = reduce_to_full_row_rank(a, b)

    assert result.a.shape == a.shape
    assert np.allclose(result.a, a)
    assert np.allclose(result.b, b)
    assert np.array_equal(result.kept_rows, np.arange(a.shape[0]))


def test_random_rank_deficient_system(rng):
    base = rng.standard_normal((3, 5))
    coeffs = rng.standard_normal(3)
    dependent_row = coeffs @ base
    a = np.vstack([base, dependent_row])
    x_true = rng.standard_normal(5)
    b = a @ x_true

    assert np.linalg.matrix_rank(a) == 3

    result = reduce_to_full_row_rank(a, b)

    assert result.a.shape[0] == 3
    assert np.linalg.matrix_rank(result.a) == 3
    # The reduced system is still solved by the same x (consistent system).
    assert np.isclose(np.linalg.norm(result.a @ x_true - result.b), 0.0, atol=1e-8)
