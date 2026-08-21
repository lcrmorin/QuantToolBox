"""Translated from Examples/stats/qreg2.m -- local-polynomial kernel mean
and quantile regression (orders 1 and 2) of a nonlinear y=f(x)+noise
relationship, compared against the known population mean/quantile curves
(numeric core only; the original's scatter+curve plot is dropped).

The original explicitly seeds MATLAB's RNG (`rng(123)`); NumPy's generator
is seeded the same way for a comparable (not bit-identical) run."""

import numpy as np

from quanttoolbox.stats.regression.kernel import kernel_mean_regression, kernel_quantile_regression

rng = np.random.default_rng(123)
n = 500
x = rng.random(n)
y = rng.random(n) * (np.cos(2 * np.pi * x - np.pi) + 1)

tau = 0.95
p = 50
z = np.arange(0, 1 + 1e-9, 1 / (p - 1))[:p]

m0 = 0.5 * (np.cos(2 * np.pi * z - np.pi) + 1)
m1 = kernel_mean_regression(y, x, z, order=1)
m2 = kernel_mean_regression(y, x, z, order=2)

q0 = tau * (np.cos(2 * np.pi * z - np.pi) + 1)
q1 = kernel_quantile_regression(y, x, tau, z, order=1)
q2 = kernel_quantile_regression(y, x, tau, z, order=2)

print("z, population mean m0, local-linear m1, local-quadratic m2 (every 10th point):")
print(np.round(np.column_stack([z, m0, m1, m2])[::10], 4))
print("\nz, population q(0.95) q0, local-linear q1, local-quadratic q2 (every 10th point):")
print(np.round(np.column_stack([z, q0, q1, q2])[::10], 4))
