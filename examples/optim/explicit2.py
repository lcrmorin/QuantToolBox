"""Translated from Examples/optim/explicit2.m -- explicit<->implicit
constraint conversion for a single equality constraint (x[0] = x[1]), plus
a `design` matrix built from a category-index vector."""

import numpy as np

from quanttoolbox.linalg.special_matrices import design
from quanttoolbox.optim.bisection import explicit_to_implicit, implicit_to_explicit

CC = np.array([[1.0, -1.0, 0, 0, 0, 0, 0, 0]])  # constraint: x[0] - x[1] = 0
c = np.array([0.0])

RR, r = explicit_to_implicit(CC, c)
CC2, c2 = implicit_to_explicit(RR, r)

print("R:")
print(RR)
print("\nr:", r)
print("\nC:")
print(CC2)
print("\nc:", c2)

w = np.concatenate([[1.0, 1.0], np.arange(2, 8, dtype=float)])  # seqa(2,1,6)
RR2 = design(w)
r2 = np.zeros(8)

print("\nR (design(w)):")
print(RR2)
print("\nr:", r2)
