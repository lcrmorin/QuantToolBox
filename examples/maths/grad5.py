"""Translated from Examples/maths/grad5.m -- two equivalent formulations of
the same gradient: fun(x) = 0.5*x'x + exp(x)'*(1/x) (already scalar-valued,
so it needs no summing trick), and fun2(x) = sum(0.5*x_i^2 + exp(x_i)/x_i)
(explicitly written as a sum of separable terms). Both have gradient
g(x) = x + exp(x) .* (1/x - 1/x^2), confirming the two formulations agree
numerically."""

import numpy as np

from quanttoolbox.maths.numerical_diff import numerical_gradient


def fun(x):
    return 0.5 * x @ x + np.exp(x) @ (1.0 / x)


def fun2(x):
    return np.sum(0.5 * x * x + np.exp(x) / x)


def grad_analytical(x):
    return x + np.exp(x) * (1.0 / x - 1.0 / (x * x))


x0 = np.array([1.0, 2.0, 3.0])
g = grad_analytical(x0)

print("fun(x) = 0.5*x'x + exp(x)'*(1/x)")
g1 = numerical_gradient(fun, x0)
print("Forward difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))

g1 = numerical_gradient(fun, x0, method="central")
print("Central difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))

print("\nfun2(x) = sum(0.5*x^2 + exp(x)/x)  (same gradient, written as a sum)")
g1 = numerical_gradient(fun2, x0)
print("Forward difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))

g1 = numerical_gradient(fun2, x0, method="central")
print("Central difference (numerical, analytical, |diff|):")
print(np.column_stack([g1, g, np.abs(g - g1)]))
print("d =", np.max(np.abs(g - g1)))
