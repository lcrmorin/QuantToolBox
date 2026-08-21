"""Translated from Examples/rpb/test_erc3.m -- Example 17 (page 123),
Roncalli (2013). RB portfolio with unequal target budgets."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.risk_budgeting import risk_contribution, solve_unconstrained
from quanttoolbox.stats.moments import corr_to_cov

sigma = np.array([0.15, 0.20, 0.30, 0.10])
rho = xpnd(np.array([1.00, 0.50, 1.00, 0.00, 0.20, 1.00, -0.10, 0.40, 0.70, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)
x = np.full(4, 0.25)

rc = risk_contribution(x, cov_matrix)
print("equal-weight risk contribution:", rc.risk, np.round(100 * rc.pct_risk_contribution, 2))

b = np.array([0.20, 0.20, 0.30, 0.30])
r = solve_unconstrained(cov_matrix, b=b, method="ccd")
print("RB weights (target budgets 20/20/30/30):", np.round(r.weights, 4))
print("converged:", r.converged, "n_iters:", r.n_iters)
