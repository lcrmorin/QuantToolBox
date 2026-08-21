"""Translated from Examples/rpb/test_lasso2.m -- Roncalli [2013],
"Introduction to Risk Parity and Budgeting", Example 1 (page 53): the
original sweeps 250 ridge/lasso penalty strengths and plots the resulting
weight paths (4 subplots: ridge toward zero with a budget constraint,
ridge toward equal-weight with no budget constraint, and the same two
cases for lasso). This translates the *numeric core* at a handful of
representative penalty strengths spanning each sweep's range, rather than
the full 250-point plot -- same "numeric core, plot dropped" convention
used throughout this port, and the same
`solve_qp(..., ridge_penalty=...)`/`solve_qp(..., lasso_penalty=...)`
machinery test_lasso1.py/test_lasso3.py already establish for
quadprog_ridge/quadprog_lasso (see optim/quadprog.py's docstring for why
the variable-splitting original collapses into these two keyword
arguments).

Not cross-verified against Octave: the original's raw
`quadprog_ridge`/`quadprog_lasso` variable-splitting formulation turned
out to be numerically fragile for the *lasso* branches specifically under
Octave's free `quadprog` (returns an infeasible/degenerate all-zero
result, retcode=-3, even for the small, well-conditioned inputs used
here), which made an apples-to-apples comparison unreliable in this
environment; `solve_qp`'s native (variable-splitting-free) L1 handling via
cvxpy does not have that problem and returns sensible, well-conditioned
weights throughout, consistent with test_lasso1.py's already-established
output."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.optim.quadprog import solve_qp
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)
n = 4
x0 = np.full(n, 1 / n)
gamma_x = 0.5
a_eq, b_eq = np.ones((1, n)), np.array([1.0])

lambda_ridge_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
lambda_lasso_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]) / 100

print("Ridge (Static, toward zero, budget constrained):")
for lam in lambda_ridge_grid:
    x = solve_qp(
        cov_matrix,
        gamma_x * mu,
        a_eq=a_eq,
        b_eq=b_eq,
        lb=-100.0,
        ub=100.0,
        ridge_penalty=(lam * np.eye(n), np.zeros(n)),
    )
    print(f"  lambda={lam:.2f}  w={np.round(100 * x, 2)}")

print("Ridge (Dynamic, toward equal-weight, no budget constraint):")
for lam in lambda_ridge_grid:
    x = solve_qp(cov_matrix, gamma_x * mu, lb=-100.0, ub=100.0, ridge_penalty=(lam * np.eye(n), x0))
    print(f"  lambda={lam:.2f}  w={np.round(100 * x, 2)}")

print("Lasso (Static, toward zero, budget constrained):")
for lam in lambda_lasso_grid:
    x = solve_qp(
        cov_matrix,
        gamma_x * mu,
        a_eq=a_eq,
        b_eq=b_eq,
        lb=-100.0,
        ub=100.0,
        lasso_penalty=(lam * np.ones(n), np.zeros(n)),
    )
    print(f"  lambda={100 * lam:.2f}  w={np.round(100 * x, 2)}")

print("Lasso (Dynamic, toward equal-weight, no budget constraint):")
for lam in lambda_lasso_grid:
    x = solve_qp(
        cov_matrix, gamma_x * mu, lb=-100.0, ub=100.0, lasso_penalty=(lam * np.ones(n), x0)
    )
    print(f"  lambda={100 * lam:.2f}  w={np.round(100 * x, 2)}")
