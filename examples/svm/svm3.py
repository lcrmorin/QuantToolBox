"""Translated from Examples/svm/svm3.m -- soft margin SVM classification
(binary hinge loss), primal formulation, at 4 different C values. 17-point
dataset (2 extra, harder-to-separate points added vs. svm1/svm2)."""

import numpy as np

from quanttoolbox.svm.svm import svm_classification_primal

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
        [6.0, 5.0, 1],
        [2.0, 2.0, -1],
    ]
)
y, x = data[:, 2], data[:, 0:2]

for c in [0.01, 0.03, 0.05, 0.30]:
    r = svm_classification_primal(y, x, c=c, loss="hinge")
    print(f"C={c}: beta0={round(r.beta0,6)} beta={np.round(r.beta,6)} margin={round(r.margin,6)}")
