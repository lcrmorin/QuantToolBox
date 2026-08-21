"""Translated from Examples/rpb/test_minvar2.m -- minimum-variance
portfolio with general linear equality/inequality constraints and box
bounds."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.mean_variance import minvar_portfolio
from quanttoolbox.stats.moments import corr_to_cov

sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

a_eq = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
b_eq = np.array([1.0, 0.0])
c_ineq = np.array([[0.0, 0.0, 0.0, -1.0]])
d_ineq = np.array([-0.90])

r = minvar_portfolio(
    cov_matrix, a_eq=a_eq, b_eq=b_eq, c_ineq=c_ineq, d_ineq=d_ineq, lb=-1.50, ub=2.00
)
print("weights:", np.round(r.weights, 3))
print("volatility:", round(r.volatility, 5))
