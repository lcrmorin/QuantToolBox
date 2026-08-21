"""Translated from Examples/rpb/test_bl2.m -- Black-Litterman sensitivity
analysis across 6 scenarios (base case + 5 view/uncertainty/tau variants),
each solved as a fixed-risk-aversion MVO portfolio."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.black_litterman import black_litterman_moments, implied_risk_premia
from quanttoolbox.portfolio.mean_variance import mvo_portfolio
from quanttoolbox.stats.moments import corr_to_cov

sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

x0 = np.array([0.40, 0.30, 0.20, 0.10])
r = 0.03
irp = implied_risk_premia(x0, cov_matrix, sharpe_ratio=0.25)
mu_tilde = r + irp.pi
gamma0 = irp.gamma

scenarios = [
    dict(
        P=np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float),
        Q=np.array([0.04, -0.01]),
        Omega=np.diag([0.10**2, 0.05**2]),
        tau=1,
    ),
    dict(
        P=np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float),
        Q=np.array([0.07, -0.01]),
        Omega=np.diag([0.10**2, 0.05**2]),
        tau=1,
    ),
    dict(
        P=np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float),
        Q=np.array([0.04, -0.01]),
        Omega=np.diag([0.20**2, 0.20**2]),
        tau=1,
    ),
    dict(
        P=np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float),
        Q=np.array([0.04, -0.01]),
        Omega=np.diag([0.10**2, 0.05**2]),
        tau=0.10,
    ),
    dict(
        P=np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float),
        Q=np.array([0.04, -0.01]),
        Omega=np.diag([0.10**2, 0.05**2]),
        tau=0.01,
    ),
]

results = [
    dict(weights=x0, mu=x0 @ mu_tilde, sigma=np.sqrt(x0 @ cov_matrix @ x0), alpha=0.0, te=0.0)
]
for s in scenarios:
    bl = black_litterman_moments(mu_tilde, s["tau"] * cov_matrix, s["P"], s["Q"], s["Omega"])
    mvo = mvo_portfolio(bl.mu_bar, cov_matrix, gamma=gamma0, lb=0.0, ub=1.0)
    alpha = (mvo.weights - x0) @ bl.mu_bar
    te = np.sqrt((mvo.weights - x0) @ cov_matrix @ (mvo.weights - x0))
    results.append(
        dict(weights=mvo.weights, mu=mvo.expected_return, sigma=mvo.volatility, alpha=alpha, te=te)
    )

for i, r_ in enumerate(results):
    print(
        f"scenario {i}: weights={np.round(r_['weights'],4)} mu={round(r_['mu'],5)} te={round(r_['te'],5)}"
    )
