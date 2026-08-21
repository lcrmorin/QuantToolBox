"""Translated from Examples/ects/varx5a.m -- Judge, Hill, Griffiths,
Lutkepohl & Lee [1988], pages 636-663: a 3-equation simultaneous linear
model estimated four ways -- OLS, GLS-2S, GLS-3S, LIML -- on a
20-observation embedded dataset, recovering the structural Gamma/B
matrices and the implied reduced form PI = -B @ inv(Gamma) at each step.

The original manually iterates feasible GLS across steps 1-3 (each
`varx_cls` call re-estimates Sigma, then feeds it into the next call as
the GLS weight) before switching to `varx_cml` (-> `varx_estimate_cml`,
concentrated/iterated ML) for the LIML step, seeded from step 3's Sigma
via `VARX_Tol = 1e-20`. `varx_estimate_cml` here always starts its own
internal iteration from the identity matrix rather than accepting a
seed Sigma -- since concentrated ML converges to the same fixed point
regardless of starting Sigma (only the number of iterations to get there
differs), this is translated as a plain `varx_estimate_cml(..., tol=
1e-20)` call rather than trying to seed it.

`theta(1:9)` reshaped (row-major, truncating/ignoring the rest) into
Gamma, and `theta(10:24)` reshaped into B, are both translated via the
ported `reshaper` (row-major reshape/recycle) exactly as in the
original."""

import numpy as np

from quanttoolbox.econometrics.var import varx_estimate, varx_estimate_cml
from quanttoolbox.linalg.special_matrices import design, reshaper

data = np.array(
    [
        [1, 3.06, 1.34, 8.48, 28, 359.27, 102.96, 578.49],
        [1, 3.19, 1.44, 9.16, 35, 415.76, 114.38, 650.86],
        [1, 3.3, 1.54, 9.9, 37, 435.11, 118.23, 684.87],
        [1, 3.4, 1.71, 11.02, 36, 440.17, 120.45, 680.47],
        [1, 3.48, 1.89, 11.64, 29, 410.66, 116.25, 642.19],
        [1, 3.6, 1.99, 12.73, 47, 530.33, 140.27, 787.41],
        [1, 3.68, 2.22, 13.88, 50, 557.15, 143.84, 818.06],
        [1, 3.72, 2.43, 14.5, 35, 472.8, 128.2, 712.16],
        [1, 3.92, 2.43, 15.47, 33, 471.76, 126.65, 722.23],
        [1, 4.15, 2.31, 16.61, 40, 538.3, 141.05, 811.44],
        [1, 4.35, 2.39, 17.4, 38, 547.76, 143.71, 816.36],
        [1, 4.37, 2.63, 18.83, 37, 539, 142.37, 807.78],
        [1, 4.59, 2.69, 20.62, 56, 677.6, 173.13, 983.53],
        [1, 5.23, 3.35, 23.76, 88, 943.85, 223.21, 1292.99],
        [1, 6.04, 5.81, 26.52, 62, 893.42, 198.64, 1179.64],
        [1, 6.36, 6.38, 27.45, 51, 871, 191.89, 1134.78],
        [1, 7.04, 6.14, 30.28, 29, 793.93, 181.27, 1053.16],
        [1, 7.81, 6.14, 25.4, 22, 850.36, 180.56, 1085.91],
        [1, 8.09, 6.19, 28.84, 38, 967.42, 208.24, 1246.99],
        [1, 9.24, 6.69, 34.36, 41, 1102.61, 235.43, 1401.94],
    ]
)

x0 = data[:, 0:5]
y = data[:, 5:8]
x = np.column_stack([y, x0])

w = np.array([0, 1, 0, 2, 0, 3, 4, 0, 0, 5, 6, 7, 0, 8, 9, 0, 10, 0, 0, 11, 0, 0, 0, 12])
rr = design(w)
r = np.zeros(24)

# OLS estimate, see JHGLL page 660
sigma = None
labels = ("OLS", "GLS-2S", "GLS-3S", "LIML")

for i in range(4):
    if i == 3:
        result = varx_estimate_cml(y, x, p=0, restriction=(rr, r), tol=1e-20)
    else:
        result = varx_estimate(y, x, p=0, restriction=(rr, r), sigma=sigma, method="ls")
    sigma = result.sigma

    theta = result.theta
    g = reshaper(theta, 3, 3) - np.eye(3)
    b = reshaper(theta[9:24], 5, 3)
    rf = -b @ np.linalg.inv(g)

    print("=" * 60, labels[i])
    print("\nGamma:")
    print(np.round(g, 5))
    print("\nB:")
    print(np.round(b, 5))
    print("\nPI (reduced form):")
    print(np.round(rf, 5))
    print()
