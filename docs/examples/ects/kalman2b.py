"""Translated from Examples/ects/kalman2b.m -- Harvey [1990], pages
89-90: frequency-domain (Whittle) maximum-likelihood estimation of the
Local Linear Trend model's (sigma_epsilon, sigma_eta, sigma_zeta) on the
same 61-observation Gnp.asc series as kalman2a.py, then runs the Kalman
filter with the estimated parameters (numeric core only, plot dropped).

The original calls `whittle_local_linear_trend` twice -- once with
`WHITTLE_algorithm = 1` (BFGS) and once with `WHITTLE_algorithm = 3`
("scoring method") -- keeping only the second result. As established in
`whittle1.py`/`whittle.py`'s module docstring, the port only has a single
optimizer path (BFGS), so there is just one call here."""

import numpy as np

from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter
from quanttoolbox.econometrics.whittle import whittle_local_linear_trend

y = np.array(
    [
        116.8,
        120.0,
        123.2,
        130.2,
        131.4,
        125.6,
        124.5,
        134.3,
        135.2,
        151.8,
        146.4,
        139.0,
        127.8,
        147.0,
        165.9,
        165.5,
        179.4,
        190.0,
        189.8,
        190.9,
        203.6,
        183.5,
        169.3,
        144.2,
        141.5,
        154.3,
        169.5,
        193.0,
        203.2,
        192.9,
        209.4,
        227.2,
        263.7,
        297.8,
        337.1,
        361.3,
        355.2,
        312.6,
        309.9,
        323.7,
        324.1,
        255.3,
        383.4,
        395.1,
        412.8,
        406.0,
        438.0,
        446.1,
        452.5,
        447.3,
        475.9,
        487.7,
        497.2,
        529.8,
        551.0,
        581.1,
        617.8,
        658.1,
        675.2,
        706.6,
        724.7,
    ],
    dtype=float,
)
nobs = y.shape[0]

# BFGS method (very sensitive to the starting values -- kept as-is)
sv = 15.0 * np.ones(3)
result_est = whittle_local_linear_trend(y, sv)
theta = result_est.theta

sigma_epsilon, sigma_eta, sigma_zeta = theta
print("theta (sigma_epsilon, sigma_eta, sigma_zeta):", np.round(theta, 4))

ssm = StateSpaceModel(
    z=np.array([[1.0, 0.0]]),
    d=np.array([0.0]),
    h=np.array([[sigma_epsilon**2]]),
    t=np.array([[1.0, 1.0], [0.0, 1.0]]),
    c=np.array([0.0, 0.0]),
    r=np.eye(2),
    q=np.diag([sigma_eta**2, sigma_zeta**2]),
)

a0 = np.array([y[0], 0.0])
p0 = np.zeros((2, 2))
result = kalman_filter(ssm, y[:, None], a0, p0)

t = np.arange(1909, 1909 + nobs)
print("\nt, y, level a(t|t-1), slope a(t|t-1) -- first/last 10 observations:")
print(np.round(np.column_stack([t, y, result.a_pred])[:10], 3))
print(np.round(np.column_stack([t, y, result.a_pred])[-10:], 3))
