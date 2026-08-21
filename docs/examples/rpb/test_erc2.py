"""Translated from Examples/rpb/test_erc2.m -- Example 7 (page 80),
Roncalli (2013). Risk contribution decomposition + risk budgeting at
various target budgets."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.risk_budgeting import risk_contribution, solve_unconstrained
from quanttoolbox.stats.moments import corr_to_cov

sigma = np.array([0.30, 0.20, 0.15])
rho = xpnd(np.array([1.00, 0.80, 1.00, 0.50, 0.30, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)
x = np.array([0.50, 0.20, 0.30])

rc = risk_contribution(x, cov_matrix)
print("fixed-x risk contribution:", rc.risk, np.round(100 * rc.pct_risk_contribution, 2))

r_equal = solve_unconstrained(cov_matrix, b=np.full(3, 1 / 3), method="ccd")
print("equal-budget RB weights:", np.round(r_equal.weights, 4))

r_custom = solve_unconstrained(cov_matrix, b=x, method="ccd")
print("target-x-as-budget RB weights:", np.round(r_custom.weights, 4))
