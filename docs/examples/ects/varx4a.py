"""Translated from Examples/ects/varx4a.m -- Judge, Hill, Griffiths,
Lutkepohl & Lee [1988], "Introduction to the Theory and Practice of
Econometrics", pages 444-455: restricted SUR (seemingly-unrelated-
regressions) estimation of a 2-equation system with 10 stacked
coefficients restricted down to 5 free parameters (via `design(w)`), on
a small 20-observation embedded dataset.

Two-step procedure, exactly as the original: first estimate with an
identity residual covariance (`sigma=None`) to obtain a residual
covariance estimate, then re-estimate using that Sigma as the GLS
weighting matrix (`varx_cls` -> `varx_estimate(..., method="ls")` in
both steps; `p=0` means no autoregressive lags, i.e. this is a pure
restricted SUR regression via the VARX machinery, `results.B` here is
just `result.beta` since `p=0` makes `result.phi` empty)."""

import numpy as np

from quanttoolbox.econometrics.var import varx_estimate
from quanttoolbox.linalg.special_matrices import design

data = np.array(
    [
        [40.05292, 1170.6, 97.8, 2.52813, 191.5, 1.8],
        [54.64859, 2015.8, 104.4, 24.91888, 516, 0.8],
        [40.31206, 2803.3, 118, 29.3427, 729, 7.4],
        [84.21099, 2039.7, 156.2, 27.61823, 560.4, 18.1],
        [127.5724, 2256.2, 172.6, 60.35945, 519.9, 23.5],
        [124.8797, 2132.2, 186.6, 50.61588, 628.5, 26.5],
        [96.55514, 1834.1, 220.9, 30.70955, 537.1, 36.2],
        [131.1601, 1588, 287.8, 60.69605, 561.2, 60.8],
        [77.02764, 1749.4, 319.9, 30.00972, 617.2, 84.4],
        [46.96689, 1687.2, 321.3, 42.5075, 626.7, 91.2],
        [100.6597, 2007.7, 319.6, 58.61146, 737.2, 92.4],
        [115.7467, 2208.3, 346, 46.96287, 760.5, 86],
        [114.5826, 1656.7, 456.4, 57.87651, 581.4, 111.1],
        [119.8762, 1604.4, 543.4, 43.22093, 662.3, 130.6],
        [105.5699, 1431.8, 618.3, 22.87143, 583.8, 141.8],
        [148.4266, 1610.5, 647.4, 52.94754, 635.2, 136.7],
        [194.3622, 1819.4, 671.3, 71.2303, 723.8, 129.7],
        [158.2037, 2079.7, 726.1, 61.7255, 864.1, 145.5],
        [163.093, 2371.6, 800.3, 85.13053, 1193.5, 174.8],
        [227.5634, 2759.9, 888.9, 88.27518, 1188.9, 213.5],
    ]
)

y = data[:, [0, 3]]
x = np.column_stack([np.ones(20), data[:, [1, 2, 4, 5]]])

w = np.array([1, 2, 3, 0, 4, 0, 0, 5, 0, 6])
rr = design(w)
r = np.zeros(10)

# First perform a VARX estimation to obtain an estimate of Sigma
result0 = varx_estimate(y, x, p=0, restriction=(rr, r), method="ls")
sigma = result0.sigma

# Then, perform a VARX estimation given the estimated Sigma
result = varx_estimate(y, x, p=0, restriction=(rr, r), sigma=sigma, method="ls")

print("theta:", np.round(result.theta, 5))
print("stderr:", np.round(result.stderr, 5))
print("\nB (= beta, since p=0):")
print(np.round(result.beta, 5))
