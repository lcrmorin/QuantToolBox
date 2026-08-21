"""Translated from Examples/ects/quantile2.m -- Monte Carlo comparison of
OLS vs. LAD (median quantile regression) under heteroskedastic noise:
mean and standard error of each estimator's slope coefficient across many
simulated datasets (numeric core only; the original's kernel-density plot
of the two estimators' sampling distributions is dropped).

The original draws x/sigma/u from MATLAB's unseeded `rand`/`randn`; a
fixed seed (`np.random.default_rng(0)`) is substituted here."""

import numpy as np

from quanttoolbox.stats.regression.quantile import quantile_regression

rng = np.random.default_rng(0)
n_s = 2000
n_t = 200

b0, b1 = 0.1, 0.2
alpha = 0.50

beta_ols = np.zeros(n_s)
beta_lad = np.zeros(n_s)

for it in range(n_s):
    x = rng.standard_normal(n_t)
    sigma = 0.20 + 0.60 * rng.random(n_t)
    sigma = sigma**2.5
    u = sigma * rng.standard_normal(n_t)
    y = b0 + b1 * x + u

    design = np.column_stack([np.ones(n_t), x])
    beta = np.linalg.inv(design.T @ design) @ (design.T @ y)
    beta_ols[it] = beta[1]

    beta_q, _, _ = quantile_regression(y, design, alpha)
    beta_lad[it] = beta_q[1]

print("parameters      estimate      stderr")
print(f"OLS             {np.mean(beta_ols):.4f}        {np.std(beta_ols, ddof=1):.4f}")
print(f"LAD             {np.mean(beta_lad):.4f}        {np.std(beta_lad, ddof=1):.4f}")
