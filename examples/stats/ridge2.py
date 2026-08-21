"""Translated from Examples/stats/ridge2.m -- tau-targeted ridge regression
(`ridge_tau_targeted`) cross-checked against fixed-lambda ridge (`ridge`),
in both absolute and OLS-relative tau modes, on the same 15-observation
dataset used throughout stats/."""

import numpy as np

from quanttoolbox.stats.regression.ols import standardize
from quanttoolbox.stats.regression.ridge import ridge, ridge_tau_targeted

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
tau_ols = np.sum(beta_ols**2)
print("tau(ols) =", round(tau_ols, 4))

# `ridge()` returns (beta, df, complexity) -- it doesn't report the
# resulting L2 budget, so tau_ridge (the quantity `ridge_tau_targeted`
# was aiming for) is recomputed here as sum(beta**2), to cross-check that
# fixed-lambda ridge at `lambda_out` reproduces the same beta/tau
# `ridge_tau_targeted` found by its grid search.

# Absolute tau targets
tau = np.array([0.7, 0.9, 1.1])
beta_ridge2, lambda_out, df_ridge2, complexity2 = ridge_tau_targeted(y, x, tau)
beta_ridge, df_ridge, complexity = ridge(y, x, lambda_out)
tau_ridge = np.sum(beta_ridge**2, axis=1)

print("\nAbsolute-tau analysis: tau target, tau achieved (fixed-lambda ridge), lambda")
print(np.round(np.column_stack([tau, tau_ridge, lambda_out]), 4))

# Relative tau targets (fraction of tau_ols)
tau = np.array([0.25, 0.50, 0.75, 1.00])
beta_ridge2, lambda_out, df_ridge2, complexity2 = ridge_tau_targeted(y, x, tau, relative=True)
beta_ridge, df_ridge, complexity = ridge(y, x, lambda_out)
tau_ridge_rel = np.sum(beta_ridge**2, axis=1) / tau_ols

print("\nRelative-tau analysis: tau target, tau achieved / tau_ols, lambda")
print(np.round(np.column_stack([tau, tau_ridge_rel, lambda_out]), 4))
print("\nbeta (fixed-lambda ridge) at each relative-tau lambda:")
print(np.round(beta_ridge, 4))
print("\nbeta (tau-targeted ridge):")
print(np.round(beta_ridge2, 4))

# Finer lambda search grid
lambda_search = np.arange(0, 15, 0.001)
beta_ridge2, lambda_out, df_ridge2, complexity2 = ridge_tau_targeted(
    y, x, tau, lambda_search=lambda_search, relative=True
)
beta_ridge, df_ridge, complexity = ridge(y, x, lambda_out)
tau_ridge_rel = np.sum(beta_ridge**2, axis=1) / tau_ols

print("\nRelative-tau analysis, finer lambda grid: tau target, tau achieved / tau_ols, lambda")
print(np.round(np.column_stack([tau, tau_ridge_rel, lambda_out]), 4))
