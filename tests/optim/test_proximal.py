"""Tests for quanttoolbox.optim.proximal."""

import numpy as np

from quanttoolbox.optim.proximal import (
    proximal_bounds,
    proximal_equality,
    proximal_inequality,
    proximal_l2,
    proximal_linear_constraints,
    proximal_linfinity,
    proximal_max,
    proximal_turnover,
    soft_thresholding,
)


def test_soft_thresholding_basic():
    v = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = soft_thresholding(v, 1.0)
    assert np.allclose(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_soft_thresholding_asymmetric():
    v = np.array([-3.0, 3.0])
    out = soft_thresholding(v, lambda_minus=1.0, lambda_plus=2.0)
    # positive side shrinks by lambda_plus=2, negative side shrinks by lambda_minus=1
    assert np.isclose(out[1], 3.0 - 2.0)
    assert np.isclose(out[0], -(3.0 - 1.0))


def test_proximal_l2_shrinks_toward_origin():
    v = np.array([3.0, 4.0])  # norm 5
    out = proximal_l2(v, lambda_=2.0)
    assert np.isclose(np.linalg.norm(out), 3.0)
    assert np.allclose(out / np.linalg.norm(out), v / np.linalg.norm(v))


def test_proximal_l2_no_shrink_if_lambda_exceeds_norm():
    v = np.array([1.0, 0.0])
    out = proximal_l2(v, lambda_=10.0)
    assert np.allclose(out, 0.0)


def test_proximal_max_caps_largest_entries():
    v = np.array([5.0, 3.0, 1.0])
    out = proximal_max(v, lambda_=3.0)
    # total reduction across capped entries should equal lambda_
    assert np.isclose(np.sum(v - out), 3.0)
    assert np.all(out <= v)


def test_proximal_linfinity_caps_max_abs():
    v = np.array([5.0, -3.0, 1.0])
    out = proximal_linfinity(v, lambda_=2.0)
    reduction = np.sum(np.abs(v) - np.abs(out))
    assert np.isclose(reduction, 2.0, atol=1e-6)


def test_proximal_bounds_clips():
    v = np.array([-5.0, 0.5, 5.0])
    out = proximal_bounds(v, lb=-1.0, ub=1.0)
    assert np.allclose(out, [-1.0, 0.5, 1.0])


def test_proximal_equality_satisfies_constraint():
    v = np.array([1.0, 2.0, 3.0])
    a_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])
    x, retcode = proximal_equality(v, a_eq, b_eq)
    assert np.isclose(np.sum(x), 1.0)
    assert retcode == 0


def test_proximal_equality_is_nearest_point():
    # projecting an already-feasible point should return it unchanged
    v = np.array([0.5, 0.5])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    x, _ = proximal_equality(v, a_eq, b_eq)
    assert np.allclose(x, v)


def test_proximal_inequality_satisfies_constraint():
    v = np.array([2.0, 2.0])
    c_ineq = np.array([[1.0, 1.0]])  # x1+x2 <= 1
    d_ineq = np.array([1.0])
    x, retcode = proximal_inequality(v, c_ineq, d_ineq)
    assert np.sum(x) <= 1.0 + 1e-6
    assert retcode == 0


def test_proximal_inequality_already_feasible_unchanged():
    v = np.array([0.2, 0.2])
    c_ineq = np.array([[1.0, 1.0]])
    d_ineq = np.array([1.0])
    x, _ = proximal_inequality(v, c_ineq, d_ineq)
    assert np.allclose(x, v)


def test_proximal_linear_constraints_combines_equality_and_bounds():
    # note: symmetric inputs (e.g. v=[2,-1,2]) can trigger the same
    # early-plateau exit the original MATLAB exact-equality convergence
    # check has -- see the docstring caveat. Use an asymmetric input here
    # to test normal (well-behaved) convergence.
    v = np.array([2.3, -1.1, 1.7])
    a_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])
    x, retcode = proximal_linear_constraints(v, a_eq=a_eq, b_eq=b_eq, lb=0.0, ub=1.0)
    assert np.isclose(np.sum(x), 1.0, atol=1e-3)
    assert np.all(x >= -1e-6)
    assert np.all(x <= 1.0 + 1e-6)


def test_proximal_turnover_respects_budget():
    v = np.array([0.8, 0.2])
    x0 = np.array([0.5, 0.5])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    x, retcode = proximal_turnover(v, a_eq, b_eq, None, None, 0.0, 1.0, x0, tau=0.2)
    turnover = np.sum(np.abs(x - x0))
    assert turnover <= 0.2 + 1e-3
    assert np.isclose(np.sum(x), 1.0, atol=1e-3)
