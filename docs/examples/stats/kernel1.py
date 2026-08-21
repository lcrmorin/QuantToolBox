"""Translated from Examples/stats/kernel1.m -- Gaussian-kernel density
estimate of two correlated series (numeric core only; the original's plot
of both density curves is dropped).

The original draws x from MATLAB's unseeded `randn`; a fixed seed
(`np.random.default_rng(0)`) is substituted here."""

import numpy as np

from quanttoolbox.stats.regression.kernel import kernel_density

rng = np.random.default_rng(0)
x1 = rng.standard_normal(100)
x2 = 0.8 * x1 + 1.0
x = np.column_stack([x1, x2])

z = np.linspace(-5, 5, 101)

density, cdf, bandwidths = kernel_density(x, z)

print("Bandwidths (auto-computed, Silverman-type rule):", np.round(bandwidths, 4))
print("\nDensity at z in {-5,-2.5,0,2.5,5} (columns: series 1, series 2):")
idx = [0, 25, 50, 75, 100]
print(np.round(np.column_stack([z[idx], density[idx]]), 4))
