"""Translated from Examples/rpb/test_lasso3.m (byte-identical to
test_lasso4.m) -- MVO/ridge/lasso/mixed-penalty portfolios via the
consolidated solve_qp (replacing the original's separate quadprog_ridge/
quadprog_lasso/quadprog_mixed calls -- see optim/quadprog.py docstring)."""

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
gamma_x = 0.5
a_eq, b_eq = np.ones((1, n)), np.array([1.0])

# gamma-problem (plain MVO)
r1 = mvo_portfolio(mu, cov_matrix, gamma=gamma_x, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0)
x1 = r1.weights

# ridge-problem: penalize toward zero, scaled by each asset's own variance
S_ridge = np.diag(np.diag(cov_matrix))
x2 = solve_qp(
    cov_matrix, gamma_x * mu, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0, ridge_penalty=(S_ridge, x0)
)

# lasso-problem: L1 penalty toward equal weight
lambda_lasso = 0.005
x3 = solve_qp(
    cov_matrix, gamma_x * mu, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0, lasso_penalty=(lambda_lasso, x0)
)

# mixed-problem: both ridge and lasso, toward various targets
x4 = solve_qp(
    cov_matrix,
    gamma_x * mu,
    a_eq=a_eq,
    b_eq=b_eq,
    lb=0.0,
    ub=1.0,
    ridge_penalty=(S_ridge, np.zeros(n)),
    lasso_penalty=(lambda_lasso, np.zeros(n)),
)
x5 = solve_qp(
    cov_matrix,
    gamma_x * mu,
    a_eq=a_eq,
    b_eq=b_eq,
    lb=0.0,
    ub=1.0,
    ridge_penalty=(S_ridge, x0),
    lasso_penalty=(lambda_lasso, np.zeros(n)),
)
x6 = solve_qp(
    cov_matrix,
    gamma_x * mu,
    a_eq=a_eq,
    b_eq=b_eq,
    lb=0.0,
    ub=1.0,
    ridge_penalty=(S_ridge, np.zeros(n)),
    lasso_penalty=(lambda_lasso, x0),
)
x7 = solve_qp(
    cov_matrix,
    gamma_x * mu,
    a_eq=a_eq,
    b_eq=b_eq,
    lb=0.0,
    ub=1.0,
    ridge_penalty=(S_ridge, x0),
    lasso_penalty=(lambda_lasso, x0),
)

for name, x in [
    ("MVO", x1),
    ("Ridge", x2),
    ("Lasso", x3),
    ("Mixed(0,0)", x4),
    ("Mixed(EW,0)", x5),
    ("Mixed(0,EW)", x6),
    ("Mixed(EW,EW)", x7),
]:
    print(f"{name}: {np.round(x, 4)}")
