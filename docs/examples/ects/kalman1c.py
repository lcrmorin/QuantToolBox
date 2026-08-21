"""Translated from Examples/ects/kalman1c.m -- Harvey [1990],
"Forecasting, Structural Time Series and the Kalman Filter", pages
89-90: compares time-domain (exact) maximum likelihood against
frequency-domain (Whittle) maximum likelihood estimation of the
local-level model's (sigma_epsilon, sigma_eta), on the same Purse.asc
series as kalman1b.py/panel1.py/whittle1.py.

As in the original, the time-domain likelihood is evaluated with a0 = 0,
P0 = 0 during estimation, but the final filter run used to compare the
two estimates' one-step-ahead predictions uses a0 = y[0], P0 = 0 (the
original's own, slightly inconsistent, choice -- preserved here rather
than "fixed")."""

import numpy as np

from quanttoolbox.econometrics.estimation import ml_estimation
from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter
from quanttoolbox.econometrics.whittle import whittle_local_level

y = np.array(
    [
        10,
        15,
        10,
        10,
        12,
        10,
        7,
        17,
        10,
        14,
        8,
        17,
        14,
        18,
        3,
        9,
        11,
        10,
        6,
        12,
        14,
        10,
        25,
        29,
        33,
        33,
        12,
        19,
        16,
        19,
        19,
        12,
        34,
        15,
        36,
        29,
        26,
        21,
        17,
        19,
        13,
        20,
        24,
        12,
        6,
        14,
        6,
        12,
        9,
        11,
        17,
        12,
        8,
        14,
        14,
        12,
        5,
        8,
        10,
        3,
        16,
        8,
        8,
        7,
        12,
        6,
        10,
        8,
        10,
        5,
        7,
    ],
    dtype=float,
)
nobs = y.shape[0]


def _ssm(sigma_epsilon: float, sigma_eta: float) -> StateSpaceModel:
    return StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[sigma_epsilon**2]]),
        t=np.array([[1.0]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[sigma_eta**2]]),
    )


def ll_ml(theta: np.ndarray) -> np.ndarray:
    sigma_epsilon, sigma_eta = np.abs(theta)
    ssm = _ssm(sigma_epsilon, sigma_eta)
    result = kalman_filter(ssm, y[:, None], np.array([0.0]), np.array([[0.0]]))
    return result.log_l


sv = np.array([3.0, 1.0])

theta1 = ml_estimation(ll_ml, sv).theta
theta2 = whittle_local_level(y, sv).theta

print("theta1 (sigma_epsilon, sigma_eta), time-domain ML:", np.round(theta1, 4))
print("theta2 (sigma_epsilon, sigma_eta), frequency-domain (Whittle) ML:", np.round(theta2, 4))

a0 = np.array([y[0]])
p0 = np.array([[0.0]])

y_cond = np.zeros((nobs, 2))
for i, theta in enumerate((theta1, theta2)):
    sigma_epsilon, sigma_eta = np.abs(theta)
    ssm = _ssm(sigma_epsilon, sigma_eta)
    result = kalman_filter(ssm, y[:, None], a0, p0)
    y_cond[:, i] = result.y_pred[:, 0]

t = np.arange(nobs)
print("\nt, y, y(t|t-1) [time-domain ML], y(t|t-1) [Whittle ML] -- first/last 10 observations:")
print(np.round(np.column_stack([t, y, y_cond])[:10], 3))
print(np.round(np.column_stack([t, y, y_cond])[-10:], 3))
