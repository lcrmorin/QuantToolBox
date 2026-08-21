"""Translated from Examples/ects/ml3.m -- Beta-distribution MLE (LGD --
Loss Given Default -- modeling), comparing the Hessian/OPG/HC covariance
estimators, on 13 observed LGD values."""

import numpy as np
from scipy.stats import beta as beta_dist

from quanttoolbox.econometrics.estimation import ml_estimation

lgd = np.array([0.68, 0.90, 0.22, 0.45, 0.17, 0.25, 0.89, 0.65, 0.75, 0.56, 0.87, 0.92, 0.46])

sv = np.array([1.0, 1.0])


def logpdf(theta):
    theta = np.sqrt(theta**2)  # force positivity, matches the original
    a, b = theta[0], theta[1]
    return np.log(beta_dist.pdf(lgd, a, b))


r_hess = ml_estimation(logpdf, sv, cov="hessian")
r_opg = ml_estimation(logpdf, sv, cov="opg")
r_hc = ml_estimation(logpdf, sv, cov="hc")

print("theta (a, b):", np.round(r_hess.theta, 4))
print("\nML covariance matrix (Hessian):")
print(np.round(r_hess.vcv, 5))
print("\nML covariance matrix (OPG):")
print(np.round(r_opg.vcv, 5))
print("\nML covariance matrix (heteroskedasticity-consistent):")
print(np.round(r_hc.vcv, 5))
