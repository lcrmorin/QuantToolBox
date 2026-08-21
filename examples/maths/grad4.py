"""Translated from Examples/maths/grad4.m -- numerical vs. analytical
gradient of the separable, elementwise function f(x) = 2*x * exp(x^2/3).

Same elementwise-via-sum approach as grad3.py's translation (see that
file's docstring for why)."""

import numpy as np

from quanttoolbox.maths.numerical_diff import numerical_gradient


def fun_elementwise(x):
    return 2 * x * np.exp(x**2 / 3)


def fun_sum(x):
    return np.sum(fun_elementwise(x))


def grad_analytical(x):
    return 2 * np.exp(x**2 / 3) + 2 * x * (2 * x / 3) * np.exp(x**2 / 3)


x0 = np.array([2.5, 3.0, 3.5])
g = grad_analytical(x0)

g1 = numerical_gradient(fun_sum, x0)
print("Forward difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))

g1 = numerical_gradient(fun_sum, x0, method="central")
print("Central difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))
