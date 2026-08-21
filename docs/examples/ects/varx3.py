"""Translated from Examples/ects/varx3.m -- a trivial 5-observation
example comparing plain OLS against `varx_estimate(..., p=0)` on the
same regressor matrix. With no autoregressive lags (p=0), VARX
estimation collapses to OLS, so `theta`/`beta` from the second call
should match `beta` from the first."""

import numpy as np

from quanttoolbox.econometrics.estimation import ols_estimation
from quanttoolbox.econometrics.var import varx_estimate

y = np.array([2, 3, 1, 7, 5], dtype=float)

x = np.array(
    [
        [1, 3, 2],
        [2, 3, 1],
        [7, 1, 7],
        [5, 3, 1],
        [3, 5, 5],
    ],
    dtype=float,
)

ols_result = ols_estimation(y, np.column_stack([np.ones(5), x]))
print("OLS beta:", np.round(ols_result.beta, 4))

varx_result = varx_estimate(y[:, None], np.column_stack([np.ones(5), x]), p=0, method="ls")
print("\nVARX(p=0) theta:", np.round(varx_result.theta, 4))
print("VARX(p=0) beta:", np.round(varx_result.beta, 4))
