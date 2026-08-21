"""Translated from Examples/rpb/test_bl4.m -- Roncalli [2013], "Introduction
to Risk Parity and Budgeting", page 24: a single Black-Litterman view
(same P/Q/Omega/tau as test_bl3.py's scenario 1) turned into a *tracking-error*
frontier against the x0 benchmark -- the sigma-problem branch of
`compute_te_portfolio.m`, now covered by `te_target_portfolio`. Six target
tracking-error levels (effectively 0% through 5%) show how active return and
the information ratio (alpha/te, here genuinely dimensionless -- unlike
test_bl3.py's alpha/sigma_x convention) scale with the amount of tracking
error taken."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.black_litterman import black_litterman_moments, implied_risk_premia
from quanttoolbox.portfolio.tracking_error import te_target_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

x0 = np.array([0.40, 0.30, 0.20, 0.10])
r = 0.03
irp = implied_risk_premia(x0, cov_matrix, 0.25)
mu_tilde = r + irp.pi

p_matrix = np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float)
q = np.array([0.04, -0.01])
omega = np.diag([0.10**2, 0.05**2])
bl = black_litterman_moments(mu_tilde, 1.0 * cov_matrix, p_matrix, q, omega)

te_targets = np.array([1e-5, 0.01, 0.02, 0.03, 0.04, 0.05])
results = te_target_portfolio(
    x0, bl.mu_bar, cov_matrix, te_targets, problem="sigma", lb=0.0, ub=1.0
)
for target, res in zip(te_targets, results, strict=False):
    ir = res.active_return / res.tracking_error if res.tracking_error > 0 else float("nan")
    print(
        f"target_te={100 * target:5.2f}  gamma={res.gamma:6.3f}  w={np.round(100 * res.weights, 2)}  "
        f"alpha={100 * res.active_return:6.3f}  te={100 * res.tracking_error:5.2f}  IR={ir:.3f}"
    )
