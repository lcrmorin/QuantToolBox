"""Translated from Examples/svm/svm5.m -- hard margin dual on the first
15 points, and soft margin dual (C=0.05) on the full 17-point dataset.
(The original's parameter sweep over C=0..0.4 for plotting purposes is
not re-translated separately -- svm3.py/svm4.py already demonstrate the
same sweep pattern at 4 representative C values.)"""

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
y15, x15 = data[:, 2], data[:, 0:2]
r_hard = svm_classification_dual(y15, x15, c=None)
print("hard margin: beta0=", round(r_hard.beta0, 6), "beta=", np.round(r_hard.beta, 6))

data17 = np.vstack([data, [[6.0, 5.0, 1], [2.0, 2.0, -1]]])
y17, x17 = data17[:, 2], data17[:, 0:2]
r_soft = svm_classification_dual(y17, x17, c=0.05, loss="hinge")
print("soft margin (C=0.05): beta0=", round(r_soft.beta0, 6), "beta=", np.round(r_soft.beta, 6))
