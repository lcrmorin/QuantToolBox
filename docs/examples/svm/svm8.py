"""Translated from Examples/svm/svm8.m -- OLS, LAD, quantile, and SVM
regression (primal, at two very different C values, LS- and
epsilon-insensitive) compared on a larger (n=1000) simulated dataset.

The original draws x/beta/u from MATLAB's unseeded `rand`/`randn`; a fixed
seed (`np.random.default_rng(0)`) is substituted here for reproducibility
-- this is the fixed-seed substitution the tracker notes svm8.m as still
needing."""

import numpy as np

from quanttoolbox.econometrics.estimation import ols_estimation
from quanttoolbox.stats.regression.quantile import quantile_regression
from quanttoolbox.stats.regression.robust import lad_regression
from quanttoolbox.svm.svm import svm_regression_primal

rng = np.random.default_rng(0)
n, k = 1000, 4
x = rng.random((n, k))
beta_true = 5 * rng.random(k)
u = 0.20 * rng.standard_normal(n)
beta0_true = -3.0
y = beta0_true + x @ beta_true + u

x_design = np.column_stack([np.ones(n), x])

beta_ols = ols_estimation(y, x_design).beta
beta_lad = lad_regression(y, x_design).beta
beta_lad2, _, _ = quantile_regression(y, x_design, tau=0.5)

# SVM regression (primal), C=1
r_ls1 = svm_regression_primal(y, x, c=1)  # LS-SVM
beta_svm_ls = np.concatenate([[r_ls1.beta0], r_ls1.beta])
r_eps1 = svm_regression_primal(y, x, c=1, epsilon=1)  # epsilon-SVM
beta_svm_epsilon = np.concatenate([[r_eps1.beta0], r_eps1.beta])

# SVM regression (primal), C=1000 (effectively unregularized)
r_ls2 = svm_regression_primal(y, x, c=1000)  # LS-SVM
beta_svm_ls2 = np.concatenate([[r_ls2.beta0], r_ls2.beta])
r_eps2 = svm_regression_primal(y, x, c=1000, epsilon=0)  # epsilon-SVM, epsilon=0
beta_svm_epsilon2 = np.concatenate([[r_eps2.beta0], r_eps2.beta])

results = np.column_stack(
    [beta_ols, beta_lad, beta_lad2, beta_svm_ls, beta_svm_epsilon, beta_svm_ls2, beta_svm_epsilon2]
)
print("Comparison of OLS, LAD, quantile(0.5), and SVM estimates")
print(
    "columns: OLS, LAD, Quantile(0.5), SVM-LS(C=1), SVM-eps(C=1,eps=1), SVM-LS(C=1000), SVM-eps(C=1000,eps=0)"
)
print(np.round(results, 3))

print("\nOLS vs. SVM-LS at large C (expect near-identical -- unregularized limit):")
print(np.round(np.column_stack([beta_ols, beta_svm_ls2]), 3))

print(
    "\nQuantile(0.5) vs. SVM-eps at large C, epsilon=0 (expect near-identical -- both approach LAD):"
)
print(np.round(np.column_stack([beta_lad2, beta_svm_epsilon2]), 3))
