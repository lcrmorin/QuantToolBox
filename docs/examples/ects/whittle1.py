"""Translated from Examples/ects/whittle1.m -- Harvey [1990],
"Forecasting, Structural Time Series and the Kalman Filter", pages
89-90: estimate a local-level model's variance parameters in the
frequency domain via Whittle maximum likelihood, on the same 71-
observation "Purse" series used in panel1.py.

The original loops `WHITTLE_algorithm` over 1, 2, and 3 to compare three
different optimizer implementations of the same Whittle MLE problem;
`whittle_local_level` (see `whittle.py`'s module docstring) ports only a
single optimizer path (`scipy.optimize.minimize` with BFGS), since all
three MATLAB variants solve the identical optimization problem and
should converge to the same estimate -- so there is only one call here,
not three."""

import numpy as np

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

sv = np.array([3.0, 1.0])

result = whittle_local_level(y, sv)

print("theta (sigma_epsilon, sigma_eta):", np.round(result.theta, 4))
print("stderr:", np.round(result.stderr, 4))
print("sum log-likelihood:", round(result.sum_log_l, 4))
