"""Translated from Examples/rpb/test_lasso5.m -- ridge/lasso/mixed
portfolios with two different ridge/lasso target vectors (equal-weight
and a custom 20/20/30/30 target)."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.optim.quadprog import solve_qp
from quanttoolbox.portfolio.mean_variance import mvo_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)
n = 4
a_eq, b_eq = np.ones((1, n)), np.array([1.0])
gamma_x = 0.5

y1 = np.full(n, 1 / n)
y2 = np.array([0.20, 0.20, 0.30, 0.30])
S_ridge = np.diag(np.diag(cov_matrix))
lambda_lasso = 0.005

x1 = mvo_portfolio(mu, cov_matrix, gamma=gamma_x, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0).weights
x2 = solve_qp(
    cov_matrix, gamma_x * mu, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0, ridge_penalty=(S_ridge, y1)
)
x4 = solve_qp(
    cov_matrix, gamma_x * mu, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0, lasso_penalty=(lambda_lasso, y1)
)
x6 = solve_qp(
    cov_matrix,
    gamma_x * mu,
    a_eq=a_eq,
    b_eq=b_eq,
    lb=0.0,
    ub=1.0,
    ridge_penalty=(S_ridge, y1),
    lasso_penalty=(lambda_lasso, y1),
)
# mixed with DIFFERENT targets for ridge (toward y1) vs lasso (toward y2)
x8 = solve_qp(
    cov_matrix,
    gamma_x * mu,
    a_eq=a_eq,
    b_eq=b_eq,
    lb=0.0,
    ub=1.0,
    ridge_penalty=(S_ridge, y1),
    lasso_penalty=(lambda_lasso, y2),
)

for name, x in [
    ("MVO", x1),
    ("Ridge->y1", x2),
    ("Lasso->y1", x4),
    ("Mixed(ridge->y1,lasso->y1)", x6),
    ("Mixed(ridge->y1,lasso->y2)", x8),
]:
    print(f"{name}: {np.round(x, 4)}")
