"""Translated from Examples/stats/qreg1.m -- linear quantile regression at
9 quantile levels (tau=0.1..0.9) on a simulated 3-predictor dataset.

The original draws x/b/u from MATLAB's unseeded `rand`/`randn`; a fixed
seed (`np.random.default_rng(0)`) is substituted here."""

import numpy as np

from quanttoolbox.stats.regression.quantile import quantile_regression

rng = np.random.default_rng(0)
n, m = 100, 3
tau = np.arange(0.1, 0.91, 0.1)
x = 10 * rng.random((n, m))
b = 10 * rng.standard_normal(m)
sigma = 5.20
u = sigma * rng.standard_normal(n)
y = x @ b + u

beta, u_pos, v_neg = quantile_regression(y, x, tau)
res = y[:, None] - x @ beta

print("tau:", np.round(tau, 2))
print("beta (rows=predictors, cols=tau):")
print(np.round(beta, 4))
print("\nresidual std by tau:", np.round(np.std(res, axis=0), 4))
