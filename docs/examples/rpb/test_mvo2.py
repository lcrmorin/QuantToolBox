"""Translated from Examples/rpb/test_mvo2.m -- gamma-problem MVO frontier
(the mu-problem/sigma-problem target-matching sections in the original
use a mode not currently ported -- see mean_variance.py docstring)."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.mean_variance import mvo_frontier
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

gamma_values = np.array([0.00, 0.20, 0.50, 1.00, 2.00, 5.00])
results = mvo_frontier(mu, cov_matrix, gamma_values)
for g, r in zip(gamma_values, results, strict=False):
    print(
        f"gamma={g}: weights={np.round(r.weights,4)} mu={round(r.expected_return,4)} sigma={round(r.volatility,4)}"
    )
