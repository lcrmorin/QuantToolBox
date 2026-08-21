"""Translated from Examples/maths/pdgm2.m -- same as pdgm1.py, on an
8-observation series."""

import numpy as np

from quanttoolbox.econometrics.whittle import periodogram

y = np.array([0.01, -0.3, 1.4, 1.5, -0.7, 0.3, 0.1, 0.0])

lam, f, i1 = periodogram(y, scaling=False)
_, _, i2 = periodogram(y, scaling=True)

print("FFT coefficients:")
print(f)

print("\nlambda, unscaled periodogram, scaled periodogram:")
print(np.round(np.column_stack([lam, i1, i2]), 10))
