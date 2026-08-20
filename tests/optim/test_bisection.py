"""Tests for quanttoolbox.optim.bisection."""

import numpy as np

from quanttoolbox.optim.bisection import (
    bisection,
    bisection2,
    explicit_to_implicit,
    implicit_to_explicit,
)


def test_bisection_finds_root_scalar():
    def f(x):
        return x**2 - 4  # root at 2 (in bracket [0,5])

    root = bisection(f, 0.0, 5.0)
    assert np.isclose(root, 2.0, atol=1e-6)


def test_bisection_finds_negative_root():
    def f(x):
        return x**2 - 4  # root at -2 (in bracket [-5,0])

    root = bisection(f, -5.0, 0.0)
    assert np.isclose(root, -2.0, atol=1e-6)


def test_bisection_returns_nan_if_no_sign_change():
    def f(x):
        return x**2 + 1  # always positive, no root

    root = bisection(f, 0.0, 5.0)
    assert np.isnan(root)


def test_bisection_vectorized():
    def f(x):
        return x**2 - np.array([4.0, 9.0])

    roots = bisection(f, np.array([0.0, 0.0]), np.array([5.0, 5.0]))
    assert np.allclose(roots, [2.0, 3.0], atol=1e-6)


def test_bisection_exact_endpoint():
    def f(x):
        return x - 2.0

    root = bisection(f, 2.0, 5.0)  # ya == 0 exactly
    assert np.isclose(root, 2.0)


def test_bisection2_finds_root_with_state():
    # state z just tracks call count here, doesn't affect the root
    def f(x, z):
        return x**2 - 4, z + 1

    root, z_final = bisection2(f, 0.0, 5.0, z0=0.0)
    assert np.isclose(root, 2.0, atol=1e-6)
    assert z_final > 0


def test_explicit_implicit_roundtrip():
    # CC @ x = c defines a plane; convert to null-space form and back
    cc = np.array([[1.0, 1.0, 1.0]])  # x1+x2+x3 = 1
    c = np.array([1.0])

    rr, r0 = explicit_to_implicit(cc, c)
    # any x = r0 + RR @ t should satisfy CC @ x = c for any t
    t = np.array([0.5, -0.3])
    x = r0 + rr @ t
    assert np.isclose(cc @ x, c, atol=1e-6)


def test_implicit_to_explicit_recovers_constraint():
    # start from a null-space basis for the plane x1+x2+x3=const
    cc_orig = np.array([[1.0, 1.0, 1.0]])
    c_orig = np.array([1.0])
    rr, r0 = explicit_to_implicit(cc_orig, c_orig)

    cc_back, c_back = implicit_to_explicit(rr, r0)
    # cc_back should be parallel to cc_orig (same null space complement)
    ratio = cc_back[0] / cc_orig[0]
    assert np.allclose(ratio, ratio[0], atol=1e-6)
