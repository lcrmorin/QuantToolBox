"""Translated from Examples/ects/kalman4a.m -- simulates a
200-observation time-varying-coefficient regression y_t = x_t1*beta1_t +
x_t2*beta2_t + eps_t, where beta1_t and beta2_t follow independent
random walks (`recserar(sigma*randn(nT,1), 0, 1.0)` with AR coefficient
1.0 is exactly a cumulative sum starting at 0, translated as
`np.cumsum`), then recovers the beta_t path with a time-varying-Z Kalman
filter (Z_t = x_t, a random-walk state-space model with T=I) and
compares the filtered estimate against the simulated true path.

The original draws from MATLAB's unseeded `randn`/`rand`; a fixed seed
(`np.random.default_rng(0)`) is substituted here."""

import numpy as np

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

z = x[None, :, :].transpose(0, 2, 1)  # (1, 2, nT)
d = np.zeros((1, n_t))
h = np.full((1, 1, n_t), sigma**2)
t_mat = np.repeat(np.eye(2)[:, :, None], n_t, axis=2)
c = np.zeros((2, n_t))
r = np.repeat(np.eye(2)[:, :, None], n_t, axis=2)
q = np.repeat(np.diag([sigma1**2, sigma2**2])[:, :, None], n_t, axis=2)

ssm = StateSpaceModel(z=z, d=d, h=h, t=t_mat, c=c, r=r, q=q)

a0 = np.zeros(2)
p0 = np.zeros((2, 2))
result = kalman_filter(ssm, y[:, None], a0, p0)

at = result.a_filt

t = np.arange(1, n_t + 1)
print("t, true beta1, filtered beta1, true beta2, filtered beta2 -- first/last 10:")
print(np.round(np.column_stack([t, beta[:, 0], at[:, 0], beta[:, 1], at[:, 1]])[:10], 4))
print(np.round(np.column_stack([t, beta[:, 0], at[:, 0], beta[:, 1], at[:, 1]])[-10:], 4))

print("\nRMSE beta1:", round(float(np.sqrt(np.mean((beta[:, 0] - at[:, 0]) ** 2))), 4))
print("RMSE beta2:", round(float(np.sqrt(np.mean((beta[:, 1] - at[:, 1]) ** 2))), 4))
