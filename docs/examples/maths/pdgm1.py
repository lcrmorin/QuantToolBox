"""Translated from Examples/maths/pdgm1.m -- raw periodogram of a
4-observation series, unscaled and scaled by 1/(2*pi). Despite the
tracker's earlier note, this is a small numeric example, not a plot;
`pdgm(y)`/`pdgm(y,1)` map directly to
`periodogram(y, scaling=False)`/`periodogram(y, scaling=True)`."""

import numpy as np

from quanttoolbox.econometrics.whittle import periodogram

y = np.array([0.1, 0.3, 0.4, 0.5])

lam, f, i1 = periodogram(y, scaling=False)
_, _, i2 = periodogram(y, scaling=True)

print("FFT coefficients:")
print(f)

print("\nlambda, unscaled periodogram, scaled periodogram:")
print(np.round(np.column_stack([lam, i1, i2]), 10))
