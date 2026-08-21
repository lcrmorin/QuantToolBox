"""Translated from Examples/ects/kalman2a.m -- Harvey [1990],
"Forecasting, Structural Time Series and the Kalman Filter", pages
89-90: runs the Kalman filter for a Local Linear Trend (LLT) model
(state = [level, slope]) on 61 annual US GNP observations (1909-1969,
Gnp.asc, embedded directly below), with fixed
(sigma_epsilon, sigma_eta1, sigma_eta2) = (1, 0.5, 0.5) -- numeric core
only, the original's dual-axes plot (level + slope) is dropped.

`results.a_cond` (the filtered-ahead state a(t|t-1), used for the level
sub-plot) maps to `result.a_pred`; `results.a` (the filtered state
a(t|t), computed but only referenced for completeness in the original)
maps to `result.a_filt`."""

import numpy as np

from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter

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

sigma_epsilon = 1.0
sigma_eta1 = 0.5
sigma_eta2 = 0.5

ssm = StateSpaceModel(
    z=np.array([[1.0, 0.0]]),
    d=np.array([0.0]),
    h=np.array([[sigma_epsilon**2]]),
    t=np.array([[1.0, 1.0], [0.0, 1.0]]),
    c=np.array([0.0, 0.0]),
    r=np.eye(2),
    q=np.diag([sigma_eta1**2, sigma_eta2**2]),
)

a0 = np.array([y[0], 0.0])
p0 = np.zeros((2, 2))
result = kalman_filter(ssm, y[:, None], a0, p0)

t = np.arange(1909, 1909 + nobs)
print("t, y, level a(t|t-1), slope a(t|t-1) -- first/last 10 observations:")
print(np.round(np.column_stack([t, y, result.a_pred])[:10], 3))
print(np.round(np.column_stack([t, y, result.a_pred])[-10:], 3))
print("\nsum log-likelihood:", round(float(np.sum(result.log_l)), 4))
