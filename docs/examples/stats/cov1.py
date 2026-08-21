"""Translated from Examples/stats/cov1.m -- OLS in closed form vs. maximum
likelihood (Gaussian log-likelihood, same model), comparing standard
errors from the Hessian, OPG, and "sandwich" (HC) covariance estimators.

The original's `ml_robust_vcv` helper computing 5 partially-redundant
covariance variants (2 Hessian-based, 1 pure-OPG, 1 sandwich, plus a
duplicate) isn't ported as a standalone function; this package's
`ml_estimation(..., cov=...)` covers the same three conceptual estimators
directly (`cov="hessian"`, `"opg"`, `"hc"`), used here instead.

The original draws x/u from MATLAB's unseeded `rand`/`randn`; a fixed seed
(`np.random.default_rng(0)`) is substituted here. `ml_ols.m` (the
per-observation log-density) is inlined below rather than kept as a
separate file, matching its role as a helper, not a standalone example."""

import numpy as np

from quanttoolbox.econometrics.estimation import ml_estimation

rng = np.random.default_rng(0)
n = 100
beta_true = np.array([1.0, 2.0])
sigma_true = 0.20
x = 10 * (rng.random((n, 2)) - 0.5)
u = sigma_true * rng.standard_normal(n)
y = x @ beta_true + u

beta_hat = np.linalg.inv(x.T @ x) @ (x.T @ y)
u_hat = y - x @ beta_hat
sigma_hat = np.std(u_hat, ddof=1)
cov_beta = (sigma_hat**2) * np.linalg.inv(x.T @ x)
stderr_ols = np.sqrt(np.diag(cov_beta))

print("OLS: beta_true, beta_hat, stderr")
print(np.round(np.column_stack([beta_true, beta_hat, stderr_ols]), 4))


def logpdf(theta, y, x):
    k = x.shape[1]
    beta = theta[:k]
    sigma2 = theta[k] ** 2
    u = y - x @ beta
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * (u**2) / sigma2


sv = np.concatenate([beta_true, [sigma_true]])

r_hess = ml_estimation(lambda th: logpdf(th, y, x), sv, cov="hessian")
r_opg = ml_estimation(lambda th: logpdf(th, y, x), sv, cov="opg")
r_hc = ml_estimation(lambda th: logpdf(th, y, x), sv, cov="hc")

print("\nMLE (Gaussian log-likelihood): sv, theta, stderr(hessian), stderr(opg), stderr(hc)")
print(
    np.round(
        np.column_stack([sv, r_hess.theta, r_hess.stderr, r_opg.stderr, r_hc.stderr]),
        4,
    )
)
print("\nconverged:", r_hess.converged, r_opg.converged, r_hc.converged)
