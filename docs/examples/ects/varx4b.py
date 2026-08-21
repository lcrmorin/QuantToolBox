"""Translated from Examples/ects/varx4b.m -- Judge, Hill, Griffiths,
Lutkepohl & Lee [1988], pages 460-462: restricted SUR estimation of a
3-equation system (log income on log prices + log quantity, 15 stacked
coefficients restricted to 7/9 free parameters) on a 30-observation
embedded dataset, comparing two different restricted-residual Sigma
estimates used as the GLS weighting matrix for the final restricted SUR
fit.

Three steps, exactly as the original: (1) Sigma1 from a 9-free-parameter
restriction (`w`), (2) Sigma2 from a more tightly restricted 7-free-
parameter version of the same `w` (indices 8 and 12 tied to the same
free parameter as index 4), (3) the final restricted SUR fit run once
with each Sigma."""

import numpy as np

from quanttoolbox.econometrics.var import varx_estimate
from quanttoolbox.linalg.special_matrices import design

data = np.array(
    [
        [10.763, 4.474, 6.629, 487.648, 11.632, 13.194, 45.770],
        [13.033, 10.836, 13.774, 364.877, 12.029, 2.181, 13.393],
        [9.244, 5.856, 4.063, 514.037, 8.196, 5.586, 104.819],
        [4.605, 14.010, 3.868, 760.343, 33.908, 5.231, 137.269],
        [13.045, 11.417, 14.922, 421.746, 4.561, 10.930, 15.914],
        [7.706, 8.755, 14.138, 578.214, 17.594, 11.854, 23.667],
        [7.405, 7.317, 4.794, 561.734, 18.842, 17.045, 62.057],
        [7.519, 6.360, 3.768, 301.470, 11.637, 2.682, 52.262],
        [8.764, 4.188, 8.089, 379.636, 7.645, 13.008, 31.916],
        [13.511, 1.996, 2.708, 478.855, 7.881, 19.623, 123.026],
        [4.943, 7.268, 12.901, 433.741, 9.614, 6.534, 26.255],
        [8.360, 5.839, 11.115, 525.702, 9.067, 9.397, 35.540],
        [5.721, 5.160, 11.220, 513.067, 14.070, 13.188, 32.487],
        [7.225, 9.145, 5.810, 408.666, 15.474, 3.340, 45.838],
        [6.617, 5.034, 5.516, 192.061, 3.041, 4.716, 26.867],
        [14.219, 5.926, 3.707, 462.621, 14.096, 17.141, 43.325],
        [6.769, 8.187, 10.125, 312.659, 4.118, 4.695, 24.330],
        [7.769, 7.193, 2.471, 400.848, 10.489, 7.639, 107.017],
        [9.804, 13.315, 8.976, 392.215, 6.231, 9.089, 23.407],
        [11.063, 6.874, 12.883, 377.724, 6.458, 10.346, 18.254],
        [6.535, 15.533, 4.115, 343.552, 8.736, 3.901, 54.895],
        [11.063, 4.477, 4.962, 301.599, 5.158, 4.350, 45.360],
        [4.016, 9.231, 6.294, 294.112, 16.618, 7.371, 25.318],
        [4.759, 5.907, 8.298, 365.032, 11.342, 6.507, 32.852],
        [5.483, 7.077, 9.638, 256.125, 2.903, 3.770, 22.154],
        [7.890, 9.942, 7.122, 184.798, 3.138, 1.360, 20.575],
        [8.460, 7.043, 4.157, 359.084, 15.315, 6.497, 44.205],
        [6.195, 4.142, 10.040, 629.378, 22.240, 10.963, 44.443],
        [6.743, 3.369, 15.459, 306.527, 10.012, 10.140, 13.251],
        [11.977, 4.806, 6.172, 347.488, 3.982, 8.637, 41.845],
    ]
)
data = np.log(data)

p = data[:, [0, 1, 2]]  # Prices
y = data[:, 3]  # Quantities
q = data[:, [4, 5, 6]]  # Income

big_y = q
big_x = np.column_stack([np.ones(30), p, y])

# First, estimate Sigma from the unrestricted least-squares residuals
w1 = np.array([1, 2, 3, 4, 0, 0, 0, 5, 0, 0, 0, 6, 7, 8, 9])
rr1 = design(w1)
r1 = np.zeros(15)
result1 = varx_estimate(big_y, big_x, p=0, restriction=(rr1, r1), method="ls")
sigma1 = result1.sigma

# Or estimate Sigma from the (more tightly) restricted least-squares
# residuals (indices 8 and 12 tied to free parameter 4)
w2 = np.array([1, 2, 3, 4, 0, 0, 0, 4, 0, 0, 0, 4, 5, 6, 7])
rr2 = design(w2)
r2 = np.zeros(15)
result2 = varx_estimate(big_y, big_x, p=0, restriction=(rr2, r2), method="ls")
sigma2 = result2.sigma

# Then, perform the restricted SUR estimation with each Sigma
sur1 = varx_estimate(big_y, big_x, p=0, restriction=(rr2, r2), sigma=sigma1, method="ls")
print("SUR (Sigma1) theta:", np.round(sur1.theta, 5))

sur2 = varx_estimate(big_y, big_x, p=0, restriction=(rr2, r2), sigma=sigma2, method="ls")
print("\nSUR (Sigma2) theta:", np.round(sur2.theta, 5))
