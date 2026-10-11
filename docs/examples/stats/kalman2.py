"""Sensitivity to the initial state.

Learning goal
-------------
Repeat the changing-coefficient model with a fixed, poor initial state. Only the three noise scales are optimized; the fixed initial state is not a fitted
parameter. A poor fixed initial state can distort fitted noise scales: in this
run observation noise approaches its lower bound. Convergence does not validate
that initialization assumption.

Try it
------
Change the fixed initial state and compare early predictions with later ones.

Source and implementation notes
-------------------------------
Translated from Examples/stats/kalman2.m -- same simulated
time-varying-coefficient regression as kalman1.py, but with the Kalman
filter's initial state a0 fixed at [-20, -20] (deliberately far from the
truth) rather than estimated, to illustrate how quickly the filter
recovers from a poor starting guess. Bounded ML estimation, same
optimizer substitution (`scipy.optimize.minimize` with
`method="L-BFGS-B"`), and the same seeded-RNG substitution as kalman1.py
all apply here too.

The original optimizes five coordinates although its initial state is fixed,
leaving two coordinates absent from the likelihood. This teaching version
removes those inactive coordinates and fits only the three noise scales.
"""

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
    p0 = np.diag(th[1:3])
    h = np.full((1, 1, nobs), th[0])
    q = np.repeat(p0[:, :, None], nobs, axis=2)
    ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r_mat, q=q)
    result = kalman_filter(ssm, y[:, None], a0_fixed, p0)
    return -float(np.sum(result.log_l))


sv = np.ones(3)
lb = np.array([0.00001, 0.0001, 0.0001])
ub = np.full(3, 100.0)

opt_result = minimize(_neg_sum_logl, sv, method="L-BFGS-B", bounds=list(zip(lb, ub, strict=True)))
print("optimizer converged:", opt_result.success, "message:", opt_result.message)
theta = opt_result.x**2

print("theta (sigma_epsilon^2, sigma_b0^2, sigma_b1^2):", np.round(theta, 4))

p0 = np.diag(theta[1:3])
h = np.full((1, 1, nobs), theta[0])
q = np.repeat(p0[:, :, None], nobs, axis=2)
ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r_mat, q=q)
result = kalman_filter(ssm, y[:, None], a0_fixed, p0)
a_cond = result.a_pred

s = np.arange(1, nobs + 1)
print("\ns, true b0, predicted b0, true b1, predicted b1 -- first/last 10:")
print(np.round(np.column_stack([s, b0, a_cond[:, 0], b1, a_cond[:, 1]])[:10], 4))
print(np.round(np.column_stack([s, b0, a_cond[:, 0], b1, a_cond[:, 1]])[-10:], 4))
