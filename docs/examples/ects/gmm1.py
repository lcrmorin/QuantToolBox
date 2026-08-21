"""Translated from Examples/ects/gmm1.m -- linear regression estimated
three ways (OLS, Gaussian MLE, GMM with moment conditions matching OLS's
normal equations plus a second-moment condition), each in an unconstrained
and a beta[3]=1 restricted form, on a 10-observation dataset with one
missing y value.

The original's `theta7` (a *second*, independently-coded way of enforcing
the same beta[3]=1 restriction, by hard-wiring it inside the moment
function instead of using the RR/r restriction machinery) is not
re-translated separately -- `theta6` below already demonstrates the
restriction via `gmm_estimation`'s own `restriction=` parameter, the same
mechanism `ols_estimation`/`ml_estimation` use for `theta2`/`theta4`."""

import numpy as np

from quanttoolbox.econometrics.estimation import gmm_estimation, ml_estimation, ols_estimation

data = np.array(
    [
        [np.nan, 1.0, 2.4, 3.6, 0.3],
        [20.4, 1.0, 1.1, 3.8, 5.9],
        [17.1, 1.0, 5.1, 6.3, 6.1],
        [30.9, 1.0, 2.7, 2.4, 9.5],
        [22.2, 1.0, 3.3, 3.0, 7.4],
        [9.1, 1.0, 1.0, 5.4, 4.9],
        [39.2, 1.0, 9.6, 2.8, 8.1],
        [3.1, 1.0, 2.9, 4.4, 1.0],
        [7.2, 1.0, 4.2, 5.6, 1.7],
        [27.6, 1.0, 8.1, 1.7, 5.4],
    ]
)
y = data[:, 0]
x = data[:, 1:5]

# Unconstrained OLS, and OLS with beta[3] restricted to 1
r1 = ols_estimation(y, x)
sigma1 = r1.sigma
rr_ols = np.eye(4, 3)
r_ols = np.array([0.0, 0.0, 0.0, 1.0])
r2 = ols_estimation(y, x, restriction=(rr_ols, r_ols))
sigma2 = r2.sigma

theta1 = np.concatenate([r1.beta, [sigma1]])
theta2 = np.concatenate([r2.beta, [sigma2]])
sv = theta1 + 1.0


def logpdf(theta):
    beta = theta[:4]
    sigma2_ = theta[4] ** 2
    u = y - x @ beta
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2_) - 0.5 * (u * u) / sigma2_


def moments(theta):
    beta = theta[:4]
    sigma2_ = theta[4] ** 2
    u = y - x @ beta
    h = np.zeros((u.shape[0], 5))
    h[:, 0] = u
    h[:, 1] = u * u - sigma2_
    h[:, 2:5] = u[:, None] * x[:, 1:4]
    return h


rr_ml = np.eye(5, 4)
rr_ml[3, 3] = 0
rr_ml[4, 3] = 1
r_ml = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

r3 = ml_estimation(logpdf, sv)
r4 = ml_estimation(logpdf, sv[:4], restriction=(rr_ml, r_ml))
r5 = gmm_estimation(moments, sv)
r6 = gmm_estimation(moments, sv[:4], restriction=(rr_ml, r_ml))

print("Unconstrained: OLS, MLE, GMM")
print(np.round(np.column_stack([theta1, r3.theta, r5.theta]), 5))
print("\nbeta[3]=1 restricted: OLS, MLE, GMM")
print(np.round(np.column_stack([theta2, r4.theta, r6.theta]), 5))
