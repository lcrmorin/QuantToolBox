"""Translated from Examples/rpb/test_lasso1.m -- Roncalli [2013],
"Introduction to Risk Parity and Budgeting", Example 1 (page 53):
compares an unconstrained gamma-problem MVO portfolio against the same
problem with a budget (sum-to-1) constraint, and ridge/lasso-penalized
variants (toward 0 and toward equal-weight) of the unconstrained
problem.

`compute_mvo_portfolio`/`quadprog_ridge`/`quadprog_lasso` map to
`mvo_portfolio`/`solve_qp(..., ridge_penalty=...)`/`solve_qp(...,
lasso_penalty=...)` as established in test_lasso3.py/test_lasso5.py; the
original's `0, 0` sentinel arguments for "no equality constraint" /
"no bounds" map to simply omitting those keyword arguments."""

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
x0 = np.full(n, 1 / n)

# Case gamma-problem
gamma_x = 0.5

x1 = mvo_portfolio(mu, cov_matrix, gamma=gamma_x).weights
x2 = mvo_portfolio(
    mu, cov_matrix, gamma=gamma_x, a_eq=np.ones((1, n)), b_eq=np.array([1.0])
).weights

lambda_ridge = 0.03
s_ridge = lambda_ridge * np.eye(n)
x3 = solve_qp(cov_matrix, gamma_x * mu, ridge_penalty=(s_ridge, np.zeros(n)))
x4 = solve_qp(cov_matrix, gamma_x * mu, ridge_penalty=(s_ridge, x0))

lambda_lasso = 0.03 / 2
s_lasso = lambda_lasso * np.ones(n)
x5 = solve_qp(cov_matrix, gamma_x * mu, lasso_penalty=(s_lasso, np.zeros(n)))
x6 = solve_qp(cov_matrix, gamma_x * mu, lasso_penalty=(s_lasso, x0))

results = 100 * np.column_stack([x1, x2, x3, x4, x5, x6])
print("            x1      x2      x3      x4      x5      x6")
print(np.round(results, 2))
