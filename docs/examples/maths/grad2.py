"""Translated from Examples/maths/grad2.m -- numerical vs. analytical
gradient of a scalar function of two variables that mixes polynomial,
log, and exponential terms."""

import numpy as np

from quanttoolbox.maths.numerical_diff import numerical_gradient


def fun(x):
    x1, x2 = x[0], x[1]
    return 3 * x1**2 + 6 * x1 + 7 + np.log(x1) + x1 * x2 + x2**2 + np.exp(x2)


def grad_analytical(x):
    x1, x2 = x[0], x[1]
    return np.array([6 * x1 + 6 + 1.0 / x1 + x2, x1 + 2 * x2 + np.exp(x2)])


x0 = np.array([0.5, 0.001])
g = grad_analytical(x0)

g1 = numerical_gradient(fun, x0)
print("Forward difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))

g1 = numerical_gradient(fun, x0, method="central")
print("Central difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))
