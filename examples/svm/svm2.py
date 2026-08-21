"""Translated from Examples/svm/svm2.m -- hard margin SVM classification,
dual formulation (same 15-point dataset as svm1.m)."""

import numpy as np

from quanttoolbox.svm.svm import svm_classification_dual

data = np.array(
    [
        [0.5, 2.5, 1],
        [2.7, 4.2, 1],
        [2.7, 2.0, 1],
        [1.7, 4.2, 1],
        [1.5, 0.7, 1],
        [2.3, 5.3, 1],
        [4.0, 6.9, 1],
        [6.4, 4.5, -1],
        [7.7, 2.2, -1],
        [8.8, 6.0, -1],
        [7.4, 6.5, -1],
        [6.5, 1.7, -1],
        [8.3, 1.3, -1],
        [6.0, 1.3, -1],
        [5.0, 0.5, -1],
    ]
)
y, x = data[:, 2], data[:, 0:2]

r = svm_classification_dual(y, x, c=None)
print("beta0:", round(r.beta0, 8))
print("beta:", np.round(r.beta, 8))
print("margin:", round(r.margin, 8))
print("support vectors (0-indexed):", r.support_vectors)
print("alpha at support vectors:", np.round(r.alpha[r.support_vectors], 8))
