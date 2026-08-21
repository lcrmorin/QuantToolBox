"""Translated from Examples/ects/quantile1.m -- illustrates that
minimizing the pinball (check) loss recovers the same quantile as the
standard sorted-sample quantile estimator, compared against the true
Normal quantile function (numeric core only; the original's comparison
plot is dropped).

The original draws y from MATLAB's unseeded `randn`; a fixed seed
(`np.random.default_rng(0)`) is substituted here. `quantile`/`sort`-based
quantile estimation uses plain `np.quantile` directly (not a ported
quanttoolbox function -- there's nothing toolbox-specific to translate
here, same as other examples that lean on bare NumPy/pandas idioms)."""

import numpy as np
from scipy.optimize import minimize_scalar

from quanttoolbox.stats.distributions import normal_ppf

rng = np.random.default_rng(0)
n_s = 1000
y = rng.standard_normal(n_s)

alpha = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

# Empirical quantile via sorting
q1 = np.quantile(y, alpha)


def pinball_objective(q, data, a):
    u = data - q
    e = u > 0
    return a * np.sum(np.abs(u[e])) + (1 - a) * np.sum(np.abs(u[~e]))


q3 = np.array([minimize_scalar(pinball_objective, args=(y, a)).x for a in alpha])

# True Normal quantiles
q4_true = normal_ppf(alpha)

print(
    "alpha, empirical quantile (sorted), M-estimator quantile (pinball minimization), true Normal quantile:"
)
print(np.round(np.column_stack([alpha, q1, q3, q4_true]), 4))
