"""Translated from Examples/ects/robust3.m -- OLS vs. median regression,
LAD regression, and quantile regression at alpha=0.90, both via the IRLS
M-estimator (`quantile_m_regression`) and the exact LP formulation
(`quantile_regression`), on a simulated 3-predictor dataset.

The original draws x/u from MATLAB's unseeded `rand`/`randn`; a fixed
seed (`np.random.default_rng(0)`) is substituted here."""

import numpy as np

from quanttoolbox.econometrics.estimation import ols_estimation
from quanttoolbox.stats.regression.quantile import quantile_regression
from quanttoolbox.stats.regression.robust import lad_regression, quantile_m_regression

rng = np.random.default_rng(0)
n, k = 100, 3
beta_true = np.arange(1, k + 1, dtype=float)
sigma = 0.20

x = rng.random((n, k))
y = x @ beta_true + sigma * rng.standard_normal(n)

r_ols = ols_estimation(y, x)
u_ols = r_ols.residuals
print("OLS beta:", np.round(r_ols.beta, 4))

alpha = 0.50
beta_med, _, _ = quantile_regression(y, x, alpha)
print("\nMedian regression beta:", np.round(beta_med, 4))

r_lad = lad_regression(y, x)
print("\nLAD regression beta:", np.round(r_lad.beta, 4), "converged:", r_lad.converged)

alpha = 0.90
r_qm = quantile_m_regression(y, x, alpha)
print(f"\nQuantile (alpha={alpha}) regression via IRLS M-estimator, beta:", np.round(r_qm.beta, 4))

beta_q, _, _ = quantile_regression(y, x, alpha)
print(f"Quantile (alpha={alpha}) regression via exact LP, beta:        ", np.round(beta_q, 4))
