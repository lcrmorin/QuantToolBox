"""Translated from Examples/rpb/test_mvo2.m -- Roncalli [2013], "Introduction
to Risk Parity and Budgeting", Example 1 (pages 7-8): the same 4-asset
mean-variance problem evaluated three ways -- the gamma-problem (pick a
risk-aversion, solve directly), the mu-problem (pick a target expected
return, bisect on gamma to hit it), and the sigma-problem (pick a target
volatility, bisect on gamma to hit it). All three route through
`compute_mvo_portfolio.m`'s three branches; here that's
`mvo_frontier`/`mvo_target_portfolio`.

The original passes `lb=0, ub=0` (MATLAB's "use the default -100/100 wide
bounds" sentinel); passed through explicitly here as `lb=-100, ub=100`."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.mean_variance import mvo_frontier, mvo_target_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

print("1. gamma-problem (page 7)")
gamma_values = np.array([0.00, 0.20, 0.50, 1.00, 2.00, 5.00])
results = mvo_frontier(mu, cov_matrix, gamma_values, lb=-100.0, ub=100.0)
for g, r in zip(gamma_values, results, strict=False):
    print(
        f"  gamma={g:5.2f}  mu={100 * r.expected_return:6.2f}  sigma={100 * r.volatility:6.2f}  "
        f"w={np.round(100 * r.weights, 2)}"
    )

print("\n2. mu-problem (page 8)")
mu_targets = np.array([5.00, 6.00, 7.00, 8.00, 9.00]) / 100
mu_results = mvo_target_portfolio(mu, cov_matrix, mu_targets, problem="mu", lb=-100.0, ub=100.0)
for target, r in zip(mu_targets, mu_results, strict=False):
    print(
        f"  target_mu={100 * target:5.2f}  gamma={r.gamma:6.3f}  mu={100 * r.expected_return:6.2f}  "
        f"sigma={100 * r.volatility:6.2f}  w={np.round(100 * r.weights, 2)}"
    )

print("\n3. sigma-problem (page 8)")
sigma_targets = np.array([15.00, 20.00, 25.00, 30.00, 35.00]) / 100
sigma_results = mvo_target_portfolio(
    mu, cov_matrix, sigma_targets, problem="sigma", lb=-100.0, ub=100.0
)
for target, r in zip(sigma_targets, sigma_results, strict=False):
    print(
        f"  target_sigma={100 * target:5.2f}  gamma={r.gamma:6.3f}  mu={100 * r.expected_return:6.2f}  "
        f"sigma={100 * r.volatility:6.2f}  w={np.round(100 * r.weights, 2)}"
    )
