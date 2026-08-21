"""Translated from Examples/rpb/test_box1.m -- ERC and box-constrained
("C-ERC") risk budgeting portfolios at progressively wider bounds around
a starting position."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.risk_budgeting import erc_portfolio, solve_box_constrained
from quanttoolbox.stats.moments import corr_to_cov

x0 = np.array([0.29, 0.25, 0.23, 0.18, 0.05])
sigma = np.array([0.20, 0.20, 0.25, 0.15, 0.25])
rho = xpnd(
    np.array(
        [1.00, 0.40, 1.00, 0.70, 0.75, 1.00, 0.60, 0.55, 0.90, 1.00, 0.70, 0.60, 0.70, 0.65, 1.00]
    ),
    method=1,
)
cov_matrix = corr_to_cov(sigma, rho)

r1 = erc_portfolio(cov_matrix)
print("ERC weights:", np.round(r1.weights, 4))

for delta in [0.02, 0.07, 0.20]:
    x_minus, x_plus = x0 - delta, x0 + delta
    r = solve_box_constrained(cov_matrix, x_minus=x_minus, x_plus=x_plus, x0=x0)
    print(f"box (delta={delta}) weights:", np.round(r.weights, 4), "converged:", r.converged)
