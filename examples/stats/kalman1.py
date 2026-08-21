"""Translated from Examples/stats/kalman1.m -- simulates a 200-
observation time-varying-coefficient regression y_t = b0_t + x1_t*b1_t +
u_t, where b0_t and b1_t are independent random walks starting at 10 and
4 respectively, then recovers (a0, sigma_epsilon, sigma_b0, sigma_b1) by
*bounded* maximum likelihood (the original uses `fmincon` with explicit
lower/upper bounds) and re-runs the time-varying Kalman filter with the
estimated parameters.

`quanttoolbox.econometrics.estimation.ml_estimation` doesn't support box
constraints, so -- to preserve the original's bounded optimization
faithfully rather than silently dropping it -- this calls
`scipy.optimize.minimize(..., method="L-BFGS-B", bounds=...)` directly
on the negative summed Kalman log-likelihood instead. As in the
original, `theta[2:5]` (sigma_epsilon, sigma_b0, sigma_b1) are bounded
to `[1e-5, 100]` (`[1e-4, 100]` for the two sigma_b's) and then *also*
squared inside the likelihood -- `P0`'s diagonal is set from the same
squared theta[3]/theta[4] that also become the (constant) process-noise
covariance Q, i.e. the model assumes the state's initial uncertainty
equals its per-period innovation variance, exactly as the original does.

The original draws from MATLAB's unseeded `randn`; a fixed seed
(`np.random.default_rng(0)`) is substituted here, in the same call order
(b0 innovations, b1 innovations, x1, u) as the original."""

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


def _neg_sum_logl(theta: np.ndarray) -> float:
    th = theta.copy()
    th[2:5] = th[2:5] ** 2
    a0 = th[0:2]
    p0 = np.diag(th[3:5])
    h = np.full((1, 1, nobs), th[2])
    q = np.repeat(p0[:, :, None], nobs, axis=2)
    ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r_mat, q=q)
    result = kalman_filter(ssm, y[:, None], a0, p0)
    return -float(np.sum(result.log_l))


sv = np.ones(5)
lb = np.array([-10.0, -10.0, 0.00001, 0.0001, 0.0001])
ub = np.array([10.0, 10.0, 100.0, 100.0, 100.0])

opt_result = minimize(_neg_sum_logl, sv, method="L-BFGS-B", bounds=list(zip(lb, ub, strict=True)))
theta = opt_result.x.copy()
theta[2:5] = theta[2:5] ** 2

print("theta (a0_1, a0_2, sigma_epsilon^2, sigma_b0^2, sigma_b1^2):", np.round(theta, 4))

a0 = theta[0:2]
p0 = np.diag(theta[3:5])
h = np.full((1, 1, nobs), theta[2])
q = np.repeat(p0[:, :, None], nobs, axis=2)
ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r_mat, q=q)
result = kalman_filter(ssm, y[:, None], a0, p0)
a_cond = result.a_pred

s = np.arange(1, nobs + 1)
print("\ns, true b0, filtered b0, true b1, filtered b1 -- first/last 10:")
print(np.round(np.column_stack([s, b0, a_cond[:, 0], b1, a_cond[:, 1]])[:10], 4))
print(np.round(np.column_stack([s, b0, a_cond[:, 0], b1, a_cond[:, 1]])[-10:], 4))
