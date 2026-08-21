"""Translated from Examples/ects/ml2.m -- OLS covariance matrix vs.
Gaussian-MLE covariance matrix under all three `ml_estimation` cov
estimators (Hessian, OPG, heteroskedasticity-consistent sandwich), on a
10-observation dataset."""

import numpy as np

from quanttoolbox.econometrics.estimation import ml_estimation, ols_estimation

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
x = data[:, 1:5]

r1 = ols_estimation(y, x)
sigma1 = r1.sigma
theta1 = np.concatenate([r1.beta, [sigma1]])


def logpdf(theta):
    beta = theta[:4]
    sigma2 = theta[4] ** 2
    u = y - x @ beta
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * (u * u) / sigma2


r_hess = ml_estimation(logpdf, theta1, cov="hessian")
r_opg = ml_estimation(logpdf, theta1, cov="opg")
r_hc = ml_estimation(logpdf, theta1, cov="hc")

print("OLS covariance matrix:")
print(np.round(r1.vcv, 5))
print("\nML covariance matrix (Hessian):")
print(np.round(r_hess.vcv, 5))
print("\nML covariance matrix (OPG):")
print(np.round(r_opg.vcv, 5))
print("\nML covariance matrix (heteroskedasticity-consistent):")
print(np.round(r_hc.vcv, 5))
