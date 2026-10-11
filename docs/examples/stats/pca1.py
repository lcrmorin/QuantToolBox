"""Explained variance with correlated variables.

Learning goal
-------------
For three standardized variables with correlation 0.8, eigenvalues are 2.6, 0.2, 0.2.
Eigenvector signs and bases within repeated eigenspaces are not unique.

Try it
------
Set the correlation to zero and predict the explained-variance shares.

Source and implementation notes
-------------------------------
Translated from Examples/stats/pca1.m -- PCA on a 3-asset correlation
matrix.
"""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.stats.regression.ols import pca

C = xpnd(np.array([1.00, 0.80, 1.00, 0.80, 0.80, 1.00]), method=1)
result = pca(C)
print("eigenvalues:", np.round(result.eigenvalues, 4))
print("quality (variance share):", np.round(result.quality, 4))
print("loadings:\n", np.round(result.loadings, 4))
