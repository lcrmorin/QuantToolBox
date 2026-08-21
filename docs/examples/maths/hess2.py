"""Translated from Examples/maths/hess2.m -- numerical gradient and Hessian
of the same function as grad2.m, at a point with a very small second
coordinate (x2=1e-5), compared against their known analytical forms."""

import numpy as np

from quanttoolbox.maths.numerical_diff import numerical_gradient, numerical_hessian


def fun(x):
    x1, x2 = x[0], x[1]
    return 3 * x1**2 + 6 * x1 + 7 + np.log(x1) + x1 * x2 + x2**2 + np.exp(x2)


def grad_analytical(x):
    x1, x2 = x[0], x[1]
    return np.array([6 * x1 + 6 + 1.0 / x1 + x2, x1 + 2 * x2 + np.exp(x2)])


def hess_analytical(x):
    x1, x2 = x[0], x[1]
    h = np.zeros((2, 2))
    h[0, 0] = 6 - 1.0 / (x1**2)
    h[1, 0] = 1.0
    h[0, 1] = h[1, 0]
    h[1, 1] = 2 + np.exp(x2)
    return h


x0 = np.array([0.5, 0.00001])
g = grad_analytical(x0)
h = hess_analytical(x0)

g1 = numerical_gradient(fun, x0)
print("Gradient, forward difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))

h1 = numerical_hessian(fun, x0, dh=6e-5)
print("\nNumerical Hessian:")
print(h1)
print("\nAnalytical Hessian:")
print(h)
