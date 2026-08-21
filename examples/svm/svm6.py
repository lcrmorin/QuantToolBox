"""Translated from Examples/svm/svm6.m -- comparison of OLS, LAD, and
SVM-regression (LS-SVM and epsilon-SVM, at two different C values)
estimates on the same 10-obs dataset used in ols1.m/robust1.m."""

import numpy as np

from quanttoolbox.econometrics.estimation import ols_estimation
from quanttoolbox.stats.regression.quantile import quantile_regression
from quanttoolbox.stats.regression.robust import lad_regression
from quanttoolbox.svm.svm import svm_regression_primal

data = np.array(
    [
        [1.5, 1.0, 2.4, 3.6, 0.3],
        [20.4, 1.0, 1.1, 3.8, 5.9],
        [17.1, 1.0, 5.1, 6.3, 6.1],
        [30.9, 1.0, 2.7, 2.4, 9.5],
        [22.2, 1.0, 3.3, 3.0, 7.4],
        [9.1, 1.0, 1.0, 5.4, 4.9],
        [39.2, 1.0, 9.6, 2.8, 8.1],
        [3.1, 1.0, 2.9, 4.4, 1.0],
        [7.2, 1.0, 4.2, 5.6, 1.7],
        [27.6, 1.0, 8.1, 1.7, 5.4],
    ]
)
y = data[:, 0]
x_full = data[:, 1:5]

beta_ols = ols_estimation(y, x_full).beta
beta_lad = lad_regression(y, x_full).beta
beta_lad2, _, _ = quantile_regression(y, x_full, tau=0.5)

x = x_full[:, 1:4]  # drop the intercept and 4th column for SVM (3 predictors)

r_ls1 = svm_regression_primal(y, x, c=1)
r_eps1 = svm_regression_primal(y, x, c=1, epsilon=1)
r_ls2 = svm_regression_primal(y, x, c=100000)
r_eps2 = svm_regression_primal(y, x, c=100000, epsilon=0)

print("OLS:", np.round(beta_ols, 3))
print("LAD:", np.round(beta_lad, 3))
print("Quantile(0.5):", np.round(beta_lad2, 3))
print("SVM LS (C=1):", round(r_ls1.beta0, 3), np.round(r_ls1.beta, 3))
print("SVM eps (C=1, eps=1):", round(r_eps1.beta0, 3), np.round(r_eps1.beta, 3))
print("SVM LS (C=1e5):", round(r_ls2.beta0, 3), np.round(r_ls2.beta, 3))
print("SVM eps (C=1e5, eps=0):", round(r_eps2.beta0, 3), np.round(r_eps2.beta, 3))
