"""Tests for quanttoolbox.maths.numerical_diff."""

import numpy as np

from quanttoolbox.maths.numerical_diff import (
    numerical_gradient,
    numerical_hessian,
    numerical_jacobian,
    sign_operator,
)


def _f_scalar(x):
    # f(x) = x1^2 + 2*x2^2 + x1*x2
    # grad = [2x1+x2, 4x2+x1], hess = [[2,1],[1,4]]
    return x[0] ** 2 + 2 * x[1] ** 2 + x[0] * x[1]


def _f_vector(x):
    return np.array([x[0] ** 2, x[0] * x[1], x[1] ** 2])


def test_numerical_gradient_forward_matches_analytical():
    x0 = np.array([1.0, 2.0])
    g = numerical_gradient(_f_scalar, x0, method="forward")
    assert np.allclose(g, [4.0, 9.0], atol=1e-3)


def test_numerical_gradient_central_more_accurate_than_forward():
    # use a non-quadratic function -- for a purely quadratic function the
    # forward-difference leading error term can vanish exactly, which
    # isn't representative of the general case.
    def f(x):
        return np.exp(x[0]) + np.sin(x[1])

    x0 = np.array([0.5, 1.0])
    true_grad = np.array([np.exp(0.5), np.cos(1.0)])
    g_central = numerical_gradient(f, x0, method="central")
    g_forward = numerical_gradient(f, x0, method="forward")
    err_central = np.max(np.abs(g_central - true_grad))
    err_forward = np.max(np.abs(g_forward - true_grad))
    assert err_central <= err_forward


def test_numerical_jacobian_matches_analytical():
    x0 = np.array([1.0, 2.0])
    j = numerical_jacobian(_f_vector, x0, method="central")
    expected = np.array([[2.0, 0.0], [2.0, 1.0], [0.0, 4.0]])
    assert np.allclose(j, expected, atol=1e-3)


def test_numerical_hessian_matches_analytical():
    x0 = np.array([1.0, 2.0])
    h = numerical_hessian(_f_scalar, x0)
    expected = np.array([[2.0, 1.0], [1.0, 4.0]])
    assert np.allclose(h, expected, atol=1e-2)


def test_numerical_hessian_symmetric():
    x0 = np.array([0.5, -0.3, 1.2])

    def f(x):
        return np.sum(x**3) + x[0] * x[1] * x[2]

    h = numerical_hessian(f, x0)
    assert np.allclose(h, h.T, atol=1e-6)


def test_sign_operator_basic():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(sign_operator(x), [-1.0, 0.0, 1.0])


def test_sign_operator_matches_numpy_sign():
    x = np.array([-5.0, -0.001, 0.0, 0.001, 5.0])
    assert np.array_equal(sign_operator(x), np.sign(x))
