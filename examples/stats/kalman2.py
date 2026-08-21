"""Translated from Examples/stats/kalman2.m -- same simulated
time-varying-coefficient regression as kalman1.py, but with the Kalman
filter's initial state a0 fixed at [-20, -20] (deliberately far from the
truth) rather than estimated, to illustrate how quickly the filter
recovers from a poor starting guess. Bounded ML estimation, same
optimizer substitution (`scipy.optimize.minimize` with
`method="L-BFGS-B"`), and the same seeded-RNG substitution as kalman1.py
all apply here too.

Note (preserved faithfully from the original): `sv`/`lb`/`ub` are still
5-dimensional and `theta[0:2]` are still optimized over, exactly as in
kalman1.py -- but since a0 is fixed externally here rather than being
set from `theta[0:2]` inside the likelihood, those two optimizer
dimensions have no effect on the objective. This looks like a leftover
from kalman1.py's script that was never trimmed down; it's kept as-is
rather than "fixed", since the goal is an exact translation."""

import numpy as np
from scipy.optimize import minimize

from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter

rng = np.random.default_rng(0)
nobs = 200

b0 = 10.0 + np.cumsum(1.5 * rng.standard_normal(nobs))
b1 = 4.0 + np.cumsum(0.2 * rng.standard_normal(nobs))

x0 = np.ones(nobs)
x1 = 25.0 * rng.standard_normal(nobs)
u = 2.0 * rng.standard_normal(nobs)
y = x0 * b0 + x1 * b1 + u

z = np.zeros((1, 2, nobs))
z[0, 0, :] = x0
z[0, 1, :] = x1
d = np.zeros((1, nobs))
t_mat = np.repeat(np.eye(2)[:, :, None], nobs, axis=2)
c = np.zeros((2, nobs))
r_mat = np.repeat(np.eye(2)[:, :, None], nobs, axis=2)

a0_fixed = np.array([-20.0, -20.0])


def _neg_sum_logl(theta: np.ndarray) -> float:
    th = theta**2
    p0 = np.diag(th[3:5])
    h = np.full((1, 1, nobs), th[2])
    q = np.repeat(p0[:, :, None], nobs, axis=2)
    ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r_mat, q=q)
    result = kalman_filter(ssm, y[:, None], a0_fixed, p0)
    return -float(np.sum(result.log_l))


sv = np.ones(5)
lb = np.array([-10.0, -10.0, 0.00001, 0.0001, 0.0001])
ub = np.array([10.0, 10.0, 100.0, 100.0, 100.0])

opt_result = minimize(_neg_sum_logl, sv, method="L-BFGS-B", bounds=list(zip(lb, ub, strict=True)))
theta = opt_result.x**2

print("theta (unused a0 dims, sigma_epsilon^2, sigma_b0^2, sigma_b1^2):", np.round(theta, 4))

p0 = np.diag(theta[3:5])
h = np.full((1, 1, nobs), theta[2])
q = np.repeat(p0[:, :, None], nobs, axis=2)
ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r_mat, q=q)
result = kalman_filter(ssm, y[:, None], a0_fixed, p0)
a_cond = result.a_pred

s = np.arange(1, nobs + 1)
print("\ns, true b0, filtered b0, true b1, filtered b1 -- first/last 10:")
print(np.round(np.column_stack([s, b0, a_cond[:, 0], b1, a_cond[:, 1]])[:10], 4))
print(np.round(np.column_stack([s, b0, a_cond[:, 0], b1, a_cond[:, 1]])[-10:], 4))
