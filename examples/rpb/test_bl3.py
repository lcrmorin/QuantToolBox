"""Translated from Examples/rpb/test_bl3.m -- Roncalli [2013], "Introduction
to Risk Parity and Budgeting", page 24: five Black-Litterman view scenarios
(varying the view itself, its confidence Omega, and the scaling tau), each
turned into a portfolio matched to the *same* target volatility as the
original x0 benchmark -- the sigma-problem branch of
`compute_mvo_portfolio.m`, now covered by `mvo_target_portfolio`. For each
scenario this reports the resulting weights, expected return, volatility
(pinned to sigma0 by construction), active return (alpha) and information
ratio relative to x0.

Note on the displayed IR: the original computes
`IR_x(i) = alpha_x(i)/sigma_x(i)` (active return over the portfolio's *own*
volatility, not tracking error -- both are dimensionless ratios), and then
the whole results matrix is multiplied by 100 for percent-style display
before formatting. Since IR_x is already a ratio, that display convention
ends up printing 100*IR rather than IR itself; this translation keeps that
same convention (labelled `100*IR`) so the printed numbers match the
original table (Roncalli page 26) directly."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.black_litterman import black_litterman_moments, implied_risk_premia
from quanttoolbox.portfolio.mean_variance import mvo_target_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

x0 = np.array([0.40, 0.30, 0.20, 0.10])
sigma0 = np.sqrt(x0 @ cov_matrix @ x0)
r = 0.03
irp = implied_risk_premia(x0, cov_matrix, 0.25)
mu_tilde = r + irp.pi

print(f"x0={np.round(100 * x0, 2)}  mu0={100 * (x0 @ mu_tilde):.2f}  sigma0={100 * sigma0:.2f}")

# (P, Q, Omega, tau) for the 5 view scenarios
p_matrix = np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float)
scenarios = [
    (np.array([0.04, -0.01]), np.diag([0.10**2, 0.05**2]), 1.0),
    (np.array([0.07, -0.01]), np.diag([0.10**2, 0.05**2]), 1.0),
    (np.array([0.04, -0.01]), np.diag([0.20**2, 0.20**2]), 1.0),
    (np.array([0.04, -0.01]), np.diag([0.10**2, 0.05**2]), 0.10),
    (np.array([0.04, -0.01]), np.diag([0.10**2, 0.05**2]), 0.01),
]

for i, (q, omega, tau) in enumerate(scenarios, start=1):
    bl = black_litterman_moments(mu_tilde, tau * cov_matrix, p_matrix, q, omega)
    res = mvo_target_portfolio(bl.mu_bar, cov_matrix, sigma0, problem="sigma", lb=0.0, ub=1.0)[0]
    alpha = (res.weights - x0) @ bl.mu_bar
    te = np.sqrt((res.weights - x0) @ cov_matrix @ (res.weights - x0))
    ir = alpha / res.volatility
    print(
        f"scenario {i}: gamma={res.gamma:.3f}  w={np.round(100 * res.weights, 2)}  "
        f"mu={100 * res.expected_return:.2f}  sigma={100 * res.volatility:.2f}  "
        f"alpha={100 * alpha:.2f}  te={100 * te:.2f}  100*IR={100 * ir:.2f}"
    )
