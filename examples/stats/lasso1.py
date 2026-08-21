"""Translated from Examples/stats/lasso1.m -- penalized-form lasso path at
5 lambda values (numeric core only; the original's 501-point tau-sweep and
its multi-line coefficient-path plot are dropped -- the same lambda
values and dataset are already used by lasso2.py, which this extends with
R^2/degrees-of-freedom/complexity reporting to match lasso1.m's own
output table)."""

import numpy as np

from quanttoolbox.stats.regression.lasso import lasso_ccd
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

beta_ols = np.linalg.inv(x.T @ x) @ (x.T @ y)
tss = np.mean(y**2)

print("lambda  |beta|         tau=sum|beta| rss     R2      df  complexity")
for lam in [0.0, 0.9, 2.5, 5.5, 7.5]:
    beta, _ = lasso_ccd(y, x, lambda_=lam, n_iters=200)
    u = y - x @ beta
    rss = np.mean(u**2)
    r2 = 1 - rss / tss
    tau = np.sum(np.abs(beta))
    df = int(np.sum(np.abs(beta) >= 1e-6))
    complexity = 1.0 / df if df > 0 else np.inf
    print(
        f"{lam:6.2f}  {np.round(beta, 4)}  tau={tau:.4f}  rss={rss:.4f}  "
        f"R2={r2:.4f}  df={df}  complexity={complexity:.4f}"
    )
