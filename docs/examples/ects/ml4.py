"""Translated from Examples/ects/ml4.m -- Gaussian-MLE with numerical vs.
analytical Jacobian/Hessian, comparing all three covariance estimators
under both, on a 10-observation dataset. Note theta[4] is parametrized as
sigma^2 directly here (not sigma), matching the original's own warning
comment."""

import numpy as np

from quanttoolbox.econometrics.estimation import ml_estimation

data = np.array(
    [
        [1.1, 1.0, 2.4, 3.6, 0.3],
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


def logpdf(theta):
    beta = theta[:4]
    sigma2 = theta[4]
    u = y - x @ beta
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * (u * u) / sigma2


def jacobian(theta):
    beta = theta[:4]
    sigma2 = theta[4]
    sigma4 = sigma2**2
    u = y - x @ beta
    g_beta = (x * u[:, None]) / sigma2
    g_sigma2 = -1 / (2 * sigma2) + (u * u) / (2 * sigma4)
    return np.column_stack([g_beta, g_sigma2])


def hessian(theta):
    beta = theta[:4]
    sigma2 = theta[4]
    sigma4 = sigma2**2
    sigma6 = sigma2**3
    u = y - x @ beta
    n = y.shape[0]

    h11 = -(x.T @ x) / sigma2
    h12 = -(x.T @ u) / sigma4
    h22 = n / (2 * sigma4) - (u @ u) / sigma6
    h = np.zeros((5, 5))
    h[:4, :4] = h11
    h[:4, 4] = h12
    h[4, :4] = h12
    h[4, 4] = h22
    return h


sv = np.ones(5)

results = {}
for cov in ("hessian", "opg", "hc"):
    results[f"num_{cov}"] = ml_estimation(logpdf, sv, cov=cov)
    results[f"ana_{cov}"] = ml_estimation(
        logpdf, sv, cov=cov, jacobian_fn=jacobian, hessian_fn=hessian
    )

for label, key in [
    ("Numerical Hessian", "num_hessian"),
    ("Numerical OPG", "num_opg"),
    ("Numerical heteroskedasticity-consistent", "num_hc"),
    ("Analytical Hessian", "ana_hessian"),
    ("Analytical OPG", "ana_opg"),
    ("Analytical heteroskedasticity-consistent", "ana_hc"),
]:
    print(f"\nML covariance matrix ({label}):")
    print(np.round(results[key].vcv, 5))
