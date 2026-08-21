"""Translated from Examples/maths/grad3.m -- numerical vs. analytical
gradient of the separable, elementwise function f(x) = x^2 * exp(x^2/3).

The original MATLAB `numerical_gradient` accepts an elementwise-vectorized
function and returns an elementwise gradient. This package's
`numerical_gradient` is scalar-valued only (see `numerical_jacobian` for
the general vector case), so the elementwise function is summed first --
since each term depends on a single x_i, the gradient of the sum w.r.t.
x_i equals the elementwise derivative at x_i (see also grad1.m's
translation in building_blocks.md, which uses the same trick)."""

import numpy as np

from quanttoolbox.maths.numerical_diff import numerical_gradient


def fun_elementwise(x):
    return x**2 * np.exp(x**2 / 3)


def fun_sum(x):
    return np.sum(fun_elementwise(x))


def grad_analytical(x):
    return 2 * x * np.exp(x**2 / 3) + x**2 * (2 * x / 3) * np.exp(x**2 / 3)


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
