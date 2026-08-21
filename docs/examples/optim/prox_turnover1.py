"""Translated from Examples/optim/prox_turnover1.m -- compares the L1
proximal (soft-thresholding) operator against the L1-ball projection (both
its default sorted-cumsum algorithm and its `proximal_max`-based
alternative) at two different starting points and lambda values.

The original draws `v` from MATLAB's unseeded `rand`; a fixed seed
(`np.random.default_rng(0)`) is substituted here for reproducibility, same
convention used elsewhere in this port (see matrix1.py, building_blocks.md)."""

import numpy as np

from quanttoolbox.optim.projection import projection_l1
from quanttoolbox.optim.proximal import soft_thresholding

rng = np.random.default_rng(0)
v = 5 * (rng.random(10) - 0.5)

lambda_ = 1.20
x0 = np.zeros(10)

x1 = soft_thresholding(v - x0, lambda_) + x0
x2 = projection_l1(v - x0, lambda_) + x0
x3 = projection_l1(v - x0, lambda_, method=0) + x0

x = np.column_stack([v, x1, x2, x3])
print("lambda =", lambda_, " x0 = 0")
print("columns: v, proximal_L1, projection_L1 (method 1), projection_L1 (method 0)")
print(x)
print("column sums of |x - x0|:", np.sum(np.abs(x - x0[:, None]), axis=0))

lambda_ = 2.00
x0 = np.ones(10)
x0 = x0 / np.sum(x0)

x1 = soft_thresholding(v - x0, lambda_) + x0
x2 = projection_l1(v - x0, lambda_) + x0
x3 = projection_l1(v - x0, lambda_, method=0) + x0

x = np.column_stack([v, x1, x2, x3])
print("\nlambda =", lambda_, " x0 = equal-weight (sums to 1)")
print("columns: v, proximal_L1, projection_L1 (method 1), projection_L1 (method 0)")
print(x)
print("column sums of |x - x0|:", np.sum(np.abs(x - x0[:, None]), axis=0))
