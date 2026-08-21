"""Translated from Examples/stats/elasticnet1.m and elasticnet2.m --
elastic net regression path at two different alpha (L1/L2 mix) values,
on the same 15-observation dataset used in ridge1.m/lasso1.m."""

import numpy as np

from quanttoolbox.stats.regression.lasso import elastic_net_ccd
from quanttoolbox.stats.regression.ols import standardize

data = np.array(
    [
        [3.1, 2.8, 4.3, 0.3, 2.2, 3.5],
        [24.9, 5.9, 3.6, 3.2, 0.7, 6.4],
        [27.3, 6.0, 9.6, 7.6, 9.5, 0.9],
        [25.4, 8.4, 5.4, 1.8, 1.0, 7.1],
        [46.1, 5.2, 7.6, 8.3, 0.6, 4.5],
        [45.7, 6.0, 7.0, 9.6, 0.6, 0.6],
        [47.4, 6.1, 1.0, 8.5, 9.6, 8.6],
        [-1.8, 1.2, 9.6, 2.7, 4.8, 5.8],
        [20.8, 3.2, 5.0, 4.2, 2.7, 3.6],
        [6.8, 0.5, 9.2, 6.9, 9.3, 0.7],
        [12.9, 7.9, 9.1, 1.0, 5.9, 5.4],
        [37.0, 1.8, 1.3, 9.2, 6.1, 8.3],
        [14.7, 7.4, 5.6, 0.9, 5.6, 3.9],
        [-3.2, 2.3, 6.6, 0.0, 3.6, 6.4],
        [44.3, 7.7, 2.2, 6.5, 1.3, 0.7],
    ]
)
y = standardize(data[:, 0])
x = standardize(data[:, 1:6])

for alpha, lam in [(0.50, 1.0), (0.25, 1.0)]:
    beta, path = elastic_net_ccd(y, x, lambda_=lam, alpha=alpha, n_iters=200)
    print(f"alpha={alpha}, lambda={lam}: beta={np.round(beta,4)}")
