"""Translated from Examples/ects/panel1.m -- despite its filename and the
tracker's earlier note ("panel-data example; no ported panel module
yet"), this is actually a local-level (random-walk-plus-noise) Kalman
filter example (Harvey 1990, pp.89-90) applied to a 71-observation
"Purse" series -- it doesn't touch any panel-data functionality. Since
`quanttoolbox.econometrics.kalman` *is* ported, this is translated
properly rather than left as a gap; the tracker entry is corrected
accordingly.

The 71 observations from Purse.asc are embedded directly below, matching
the project's convention of self-contained example scripts (numeric core
only; the original's plot of y_t vs. the one-step-ahead prediction is
dropped)."""

import numpy as np

from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter

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

sigma_epsilon = 5.0
sigma_eta = 2.15

ssm = StateSpaceModel(
    z=np.array([[1.0]]),
    d=np.array([0.0]),
    h=np.array([[sigma_epsilon**2]]),
    t=np.array([[1.0]]),
    c=np.array([0.0]),
    r=np.array([[1.0]]),
    q=np.array([[sigma_eta**2]]),
)

a0 = np.array([y[0]])
p0 = np.array([[0.0]])

result = kalman_filter(ssm, y[:, None], a0, p0)
y_cond = result.y_pred[:, 0]

t = np.arange(nobs)
print("t, y, y(t|t-1) -- first/last 10 observations:")
print(np.round(np.column_stack([t, y, y_cond])[:10], 3))
print(np.round(np.column_stack([t, y, y_cond])[-10:], 3))
print("\nsum log-likelihood:", round(float(np.sum(result.log_l)), 4))
