"""OLS, median, LAD, and Huber regression.

Learning goal
-------------
Compare loss functions on the same simulated regression. Read the convergence flag:
close coefficients do not establish successful iterative convergence.

Try it
------
Add one response outlier and compare coefficient changes across estimators.

Source and implementation notes
-------------------------------
Translated from Examples/ects/robust2.m -- OLS vs. median (quantile,
alpha=0.5) regression, LAD regression, and Huber regression, on a
simulated 3-predictor dataset.

The original draws x/u from MATLAB's unseeded `rand`/`randn`; a fixed
seed (`np.random.default_rng(0)`) is substituted here.
"""

import numpy as np

from quanttoolbox.econometrics.estimation import ols_estimation
from quanttoolbox.stats.regression.quantile import quantile_regression
from quanttoolbox.stats.regression.robust import huber_regression, lad_regression

rng = np.random.default_rng(0)
n, k = 100, 3
beta_true = np.arange(1, k + 1, dtype=float)
sigma = 0.20

x = rng.random((n, k))
y = x @ beta_true + sigma * rng.standard_normal(n)

r_ols = ols_estimation(y, x)
u_ols = r_ols.residuals
print("OLS beta:", np.round(r_ols.beta, 4))

alpha = 0.50
beta_med, _, _ = quantile_regression(y, x, alpha)
print("\nMedian regression beta:", np.round(beta_med, 4))

r_lad = lad_regression(y, x)
# LAD's nonsmooth loss can make IRLS stop at its iteration limit.
# The median-regression LP above supplies an objective-value cross-check.
if not r_lad.converged:
    print("LAD IRLS reached its iteration limit; compare absolute loss with the LP fit.")

print("\nLAD regression beta:", np.round(r_lad.beta, 4), "converged:", r_lad.converged)

c = np.quantile(np.abs(u_ols), 0.90)
r_huber = huber_regression(y, x, c=c)
print(
    f"\nHuber regression (c={c:.4f}) beta:",
    np.round(r_huber.beta, 4),
    "converged:",
    r_huber.converged,
)

# Absolute-loss agreement can be useful even when IRLS has not met its stopping rule.
loss_lp = np.abs(y - x @ beta_med).sum()
loss_irls = np.abs(y - x @ r_lad.beta).sum()
print("LAD absolute loss, LP / IRLS:", loss_lp, loss_irls)
