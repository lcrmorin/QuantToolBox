"""Translated from Examples/rpb/test_mvo3.m -- Roncalli [2013], "Introduction
to Risk Parity and Budgeting", Example 1 (page 10): the same sigma-problem
target-matching as test_mvo2.py's section 3, now under three different
weight-bound configurations (unconstrained-ish, long-only, long-only capped
at 40% per asset), showing how the achievable target range narrows as
constraints tighten.

**A genuine bug in the original**, found while cross-checking this example
against Octave: `test_mvo3.m` (unlike `test_mvo2.m` and every other example
in this cluster) never calls `init_global`, which is what sets the global
`BISECTION_Tol` that `bisection.m`'s convergence loop depends on. With
`BISECTION_Tol` undefined, `while max(abs(a-b)) > BISECTION_Tol` compares
against an empty value, which both MATLAB and Octave treat as false --
so the loop runs zero iterations and `bisection` silently returns the raw
bracket midpoint `(0+10)/2 = 5.0` for every target, regardless of what the
target actually is. `compute_mvo_portfolio` then reports this as a
successful solve (retcode=1) even though the resulting portfolio doesn't
come close to the requested volatility target. Running `test_mvo3.m`
standalone in a fresh MATLAB/Octave session reproduces this; running it
right after `test_mvo2.m` in the same session "accidentally" works, because
`clear` (unlike `clear all`) doesn't clear globals, so `test_mvo2.m`'s
earlier `init_global` call leaves `BISECTION_Tol` set. See
`docs/matlab_bugs_found.md` for the full writeup. This translation produces
the *correct* target-matching numbers (as `test_mvo3.m` would if it called
`init_global` like its neighbors do), since `mvo_target_portfolio`'s Python
`bisection` always has a well-defined tolerance."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.mean_variance import mvo_target_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.05, 0.06, 0.08, 0.06])
sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)

sigma_targets = np.array([15.00, 20.00]) / 100

bound_configs = {
    "x1 (lb=-100, ub=100)": (-100.0, 100.0),
    "x2 (lb=0, ub=100, long-only)": (0.0, 100.0),
    "x3 (lb=0, ub=40, long-only, capped)": (0.0, 0.40),
}

for label, (lb, ub) in bound_configs.items():
    print(f"{label}:")
    results = mvo_target_portfolio(mu, cov_matrix, sigma_targets, problem="sigma", lb=lb, ub=ub)
    for target, r in zip(sigma_targets, results, strict=False):
        print(
            f"  target_sigma={100 * target:5.2f}  gamma={r.gamma:6.3f}  mu={100 * r.expected_return:6.2f}  "
            f"sigma={100 * r.volatility:6.2f}  w={np.round(100 * r.weights, 2)}"
        )
