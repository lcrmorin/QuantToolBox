"""Translated from Examples/matrix/matrix2.m -- checks seven algebraic
identities relating the elimination, duplication, and commutation
matrices, for M = 1..10.

The original compares matrix products against identity/zero matrices with
exact `==`; since these matrices are built from exact 0/1 selections the
products are exact in floating point for the equality- and sum-based
checks, but `inv(D'*D)` (a genuine matrix inversion) is not bit-exact, so
`np.isclose`/`np.allclose` are used throughout instead of `==` to make the
comparison robust the way an equivalent MATLAB run using a tolerance
would be."""

import numpy as np

from quanttoolbox.linalg.special_matrices import (
    commutation_matrix,
    duplication_matrix,
    elimination_matrix,
)

for M in range(1, 11):
    p = M * (M + 1) // 2

    L = elimination_matrix(M)
    D = duplication_matrix(M)
    K = commutation_matrix(M, M)
    K1 = commutation_matrix(M, 1)
    K2 = commutation_matrix(1, M)

    checks = [
        np.allclose(L @ D, np.eye(p)),
        np.allclose(K @ D, D),
        np.array_equal(K1 == K2, K1 == np.eye(M)),
        np.isclose(np.sum(np.diag(K)), M),
        np.isclose(np.sum(np.diag(D.T @ D)), M**2),
        np.allclose(L @ L.T, np.eye(p)),
        np.isclose(np.sum(np.diag(np.linalg.inv(D.T @ D))), M * (M + 3) / 4),
    ]

    verdict = (
        "The propositions are verified" if all(checks) else "The propositions are NOT verified"
    )
    print(f"M={M}: {verdict}")
