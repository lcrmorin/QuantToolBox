"""Translated from Examples/ects/kalman4b.m -- same simulated
time-varying-coefficient regression as kalman4a.py, but instead of using
the true (sigma_epsilon, sigma_beta1, sigma_beta2), estimates them by
maximum likelihood (starting from sv = [1, 1, 1]) and re-runs the
time-varying Kalman filter with the estimated variances, comparing the
recovered beta_t path against the simulated true path.

As in kalman1b.py/kalman2c.py/kalman3d.py, `theta = sqrt(theta.^2)` is
translated as `np.abs(theta)`."""

import numpy as np

from quanttoolbox.econometrics.estimation import ml_estimation
from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter

rng = np.random.default_rng(0)
n_t = 200

sigma1 = 0.5
sigma2 = 0.25
sigma = 1.0

beta1 = np.cumsum(sigma1 * rng.standard_normal(n_t))
beta2 = np.cumsum(sigma2 * rng.standard_normal(n_t))
beta = np.column_stack([beta1, beta2])
x = rng.random((n_t, 2))

y = np.sum(x * beta, axis=1) + sigma * rng.standard_normal(n_t)

a0 = np.zeros(2)
p0 = np.zeros((2, 2))

z = x[None, :, :].transpose(0, 2, 1)  # (1, 2, nT)
d = np.zeros((1, n_t))
t_mat = np.repeat(np.eye(2)[:, :, None], n_t, axis=2)
c = np.zeros((2, n_t))
r = np.repeat(np.eye(2)[:, :, None], n_t, axis=2)


def _ssm(sigma_epsilon: float, sigma_beta1: float, sigma_beta2: float) -> StateSpaceModel:
    h = np.full((1, 1, n_t), sigma_epsilon**2)
    q = np.repeat(np.diag([sigma_beta1**2, sigma_beta2**2])[:, :, None], n_t, axis=2)
    return StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r, q=q)


def ssm_logl(theta: np.ndarray) -> np.ndarray:
    sigma_epsilon, sigma_beta1, sigma_beta2 = np.abs(theta)
    ssm = _ssm(sigma_epsilon, sigma_beta1, sigma_beta2)
    result = kalman_filter(ssm, y[:, None], a0, p0)
    return result.log_l


sv = np.ones(3)
ml_result = ml_estimation(ssm_logl, sv)
theta_hat = np.abs(ml_result.theta)

sigma_epsilon_hat, sigma1_hat, sigma2_hat = theta_hat
print(
    "Estimated (sigma_epsilon, sigma_beta1, sigma_beta2):",
    np.round(theta_hat, 4),
    "-- true:",
    [sigma, sigma1, sigma2],
)

ssm_hat = _ssm(sigma_epsilon_hat, sigma1_hat, sigma2_hat)
result = kalman_filter(ssm_hat, y[:, None], a0, p0)
at = result.a_filt

t = np.arange(1, n_t + 1)
print("\nt, true beta1, filtered beta1, true beta2, filtered beta2 -- first/last 10:")
print(np.round(np.column_stack([t, beta[:, 0], at[:, 0], beta[:, 1], at[:, 1]])[:10], 4))
print(np.round(np.column_stack([t, beta[:, 0], at[:, 0], beta[:, 1], at[:, 1]])[-10:], 4))
