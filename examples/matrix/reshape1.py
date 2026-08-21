"""Translated from Examples/matrix/reshape1.m -- vech/xpnd (both row- and
column-wise orderings) and reshaper/reshapec on a fixed 4x4 matrix.

The original also exercises `shiftr` (a lagged-shift primitive); that
function is not ported (superseded by NumPy roll/slicing idioms -- see
`matrix/shiftr1.m`'s own tracker entry) and is skipped here rather than
reimplemented."""

import numpy as np

from quanttoolbox.linalg.special_matrices import reshapec, reshaper, vech, xpnd

n = 4
col = np.arange(1, n + 1, dtype=float).reshape(-1, 1)  # seqa(1,1,n)
row = (1 + 0.25 * np.arange(n)).reshape(1, -1)  # seqa(1,0.25,n)'
y = col**row
print("y:")
print(y)

v = vech(y)
print("\nvech(y), row-wise (default):", v)
z = xpnd(v)
print("xpnd(v):")
print(z)

v = vech(y, method="c")
print("\nvech(y), column-wise:", v)
z = xpnd(v, method="c")
print("xpnd(v):")
print(z)

print("\nreshaper(y, 3, 2):")
print(reshaper(y, 3, 2))
print("reshaper(y, 5, 1):")
print(reshaper(y, 5, 1))
print("reshaper(y, 1, 5):")
print(reshaper(y, 1, 5))

print("\nreshapec(y, 3, 2):")
print(reshapec(y, 3, 2))
print("reshapec(y, 5, 1):")
print(reshapec(y, 5, 1))
print("reshapec(y, 1, 5):")
print(reshapec(y, 1, 5))
