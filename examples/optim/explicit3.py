"""Translated from Examples/optim/explicit3.m -- explicit-to-implicit
conversion for three simultaneous zero-restrictions on an 8-parameter
vector (as would arise e.g. from AR1_12=0, MA1_11=0, MA1_21=0 in a VAR
specification)."""

import numpy as np

from quanttoolbox.optim.bisection import explicit_to_implicit

CC = np.zeros((3, 8))
CC[0, 2] = 1  # beta[2] = 0  (AR1_12)
CC[1, 4] = 1  # beta[4] = 0  (MA1_11)
CC[2, 5] = 1  # beta[5] = 0  (MA1_21)
c = np.zeros(3)

RR, r = explicit_to_implicit(CC, c)

print("R:")
print(RR)
print("\nr:", r)
