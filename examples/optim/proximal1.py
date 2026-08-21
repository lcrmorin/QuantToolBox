"""Translated from Examples/optim/proximal1.m -- proximal projections onto
boxes, single/multiple inequality constraints, single/multiple equality
constraints, and combined linear-constraint sets, on a fixed random
starting point.

The original compares `Proximal_Algorithm = 1` (closed-form, where one
exists) against `Proximal_Algorithm = 2` (an alternate/QP-based
computation of the *same* projection) side by side. During porting, the
redundant `Proximal_Algorithm == 2` branches for `proximal_bounds` and
`proximal_equality` were dropped (see `proximal.py`'s module docstring --
they solve the exact same problem the closed form already solves exactly,
so keeping both added no information), and `proximal_inequality`/
`proximal_linear_constraints` only ever had one (Dykstra) implementation.
So there is no second variant left to compare here -- each constraint
type is projected onto once below, rather than twice.

The original draws x from MATLAB's unseeded `rand`; a fixed seed
(`np.random.default_rng(0)`) is substituted here."""

import numpy as np

from quanttoolbox.optim.proximal import (
    proximal_bounds,
    proximal_equality,
    proximal_inequality,
    proximal_linear_constraints,
)

rng = np.random.default_rng(0)
n = 10
x = rng.random(n)

print("x:", np.round(x, 4))

# Lower & upper bounds
lb, ub = np.zeros(n), 0.5 * np.ones(n)
x1 = proximal_bounds(x, lb, ub)
print("\nBounds [0, 0.5]:", np.round(x1, 4))

# One inequality constraint
c_ineq = np.zeros((1, n))
c_ineq[0, :4] = 0.25
d_ineq = np.array([0.5])
x1, rc = proximal_inequality(x, c_ineq, d_ineq)
print("\nInequality (1 constraint):", np.round(x1, 4), "retcode:", rc)

# Two inequality constraints
c_ineq = np.zeros((2, n))
d_ineq = np.zeros(2)
c_ineq[0, :4] = 0.25
c_ineq[1, 3:6] = -np.array([1, 2, 2])
d_ineq[0] = 0.5
d_ineq[1] = -2.0
x1, rc = proximal_inequality(x, c_ineq, d_ineq)
print("\nInequality (2 constraints):", np.round(x1, 4), "retcode:", rc)

# Four inequality constraints
c_ineq = np.zeros((4, n))
d_ineq = np.zeros(4)
c_ineq[0, :4] = 0.25
c_ineq[1, 3:6] = -np.array([1, 2, 2])
c_ineq[2, 9] = 1
c_ineq[3, 9] = -1
d_ineq[0] = 0.5
d_ineq[1] = -2.0
d_ineq[2] = 0.1
d_ineq[3] = -0.1
x1, rc = proximal_inequality(x, c_ineq, d_ineq)
print("\nInequality (4 constraints):", np.round(x1, 4), "retcode:", rc)

# One equality constraint
a_eq = np.ones((1, n))
b_eq = np.array([4.0])
x1, rc = proximal_equality(x, a_eq, b_eq)
print("\nEquality (1 constraint):", np.round(x1, 4), "retcode:", rc)

# Two equality constraints
a_eq = np.zeros((2, n))
b_eq = np.zeros(2)
a_eq[0, :] = 1.0
a_eq[1, [0, 1]] = [1, -1]
b_eq[0] = 4.0
x1, rc = proximal_equality(x, a_eq, b_eq)
print("\nEquality (2 constraints):", np.round(x1, 4), "retcode:", rc)

# Linear constraints: 1 equality + 1 inequality + bounds
a_eq = np.ones((1, n))
b_eq = np.array([4.0])
c_ineq = np.zeros((1, n))
d_ineq = np.array([0.5])
c_ineq[0, :4] = 0.25
lb, ub = np.zeros(n), 0.5 * np.ones(n)
x1, rc = proximal_linear_constraints(
    x, a_eq=a_eq, b_eq=b_eq, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb, ub=ub
)
print("\nLinear constraints (1 eq, 1 ineq, bounds):", np.round(x1, 4), "retcode:", rc)

# Linear constraints: 2 equality + 2 inequality + bounds
a_eq = np.zeros((2, n))
b_eq = np.zeros(2)
a_eq[0, :] = 1.0
a_eq[1, [0, 1]] = [1, -1]
b_eq[0] = 4.0
c_ineq = np.zeros((2, n))
d_ineq = np.zeros(2)
c_ineq[0, :4] = 0.25
c_ineq[1, 3:6] = -np.array([1, 2, 2])
d_ineq[0] = 0.5
d_ineq[1] = -2.0
x1, rc = proximal_linear_constraints(
    x, a_eq=a_eq, b_eq=b_eq, c_ineq=c_ineq, d_ineq=d_ineq, lb=lb, ub=ub
)
print("\nLinear constraints (2 eq, 2 ineq, bounds):", np.round(x1, 4), "retcode:", rc)
