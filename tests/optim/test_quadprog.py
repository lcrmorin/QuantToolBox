"""Tests for quanttoolbox.optim.quadprog."""

import numpy as np

from quanttoolbox.optim.quadprog import qp_bc_ccd, qp_hyperplane, solve_qp


def test_solve_qp_unconstrained_matches_closed_form():
    q = np.array([[2.0, 0.0], [0.0, 2.0]])
    r = np.array([1.0, 2.0])
    x = solve_qp(q, r)
    # unconstrained min of 0.5 x'Qx - R'x -> x = Q^-1 R
    expected = np.linalg.solve(q, r)
    assert np.allclose(x, expected, atol=1e-5)


def test_solve_qp_equality_constraint():
    q = np.eye(2)
    r = np.array([0.0, 0.0])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    x = solve_qp(q, r, a_eq=a_eq, b_eq=b_eq)
    assert np.isclose(np.sum(x), 1.0, atol=1e-5)
    # nearest point on the line to origin, weighted equally -> [0.5, 0.5]
    assert np.allclose(x, [0.5, 0.5], atol=1e-4)


def test_solve_qp_default_budget_constraint():
    q = np.eye(3)
    r = np.zeros(3)
    x = solve_qp(q, r, default_budget_constraint=True)
    assert np.isclose(np.sum(x), 1.0, atol=1e-5)


def test_solve_qp_box_bounds():
    q = np.eye(2)
    r = np.array([10.0, 10.0])  # wants to go large positive
    x = solve_qp(q, r, lb=0.0, ub=1.0)
    assert np.allclose(x, [1.0, 1.0], atol=1e-4)


def test_solve_qp_inequality_constraint():
    q = np.eye(2)
    r = np.array([5.0, 5.0])
    c_ineq = np.array([[1.0, 1.0]])
    d_ineq = np.array([1.0])
    x = solve_qp(q, r, c_ineq=c_ineq, d_ineq=d_ineq)
    assert np.sum(x) <= 1.0 + 1e-4


def test_solve_qp_ridge_penalty_shrinks_toward_target(rng):
    n = 3
    q = np.eye(n)
    r = np.array([5.0, 5.0, 5.0])
    x_target = np.zeros(n)

    x_no_ridge = solve_qp(q, r)
    x_ridge = solve_qp(q, r, ridge_penalty=(10.0, x_target))
    assert np.linalg.norm(x_ridge - x_target) < np.linalg.norm(x_no_ridge - x_target)


def test_solve_qp_lasso_penalty_induces_sparsity():
    n = 5
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, n))
    q = a.T @ a + 0.1 * np.eye(n)
    r = a.T @ (a @ np.array([1.0, -1.0, 0.5, 0.0, 0.0]))
    x_target = np.zeros(n)

    x_no_lasso = solve_qp(q, r)
    x_lasso = solve_qp(q, r, lasso_penalty=(2.0, x_target))
    n_nonzero_before = np.sum(np.abs(x_no_lasso) > 1e-4)
    n_nonzero_after = np.sum(np.abs(x_lasso) > 1e-4)
    assert n_nonzero_after <= n_nonzero_before


def test_solve_qp_turnover_constraint_respected():
    q = np.eye(3)
    r = np.array([5.0, -5.0, 0.0])
    x0 = np.array([0.2, 0.3, 0.5])
    a_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])
    x = solve_qp(q, r, a_eq=a_eq, b_eq=b_eq, turnover=(x0, 0.3))
    turnover = np.sum(np.abs(x - x0))
    assert turnover <= 0.3 + 1e-4


def test_solve_qp_mixed_ridge_and_lasso():
    n = 3
    q = np.eye(n)
    r = np.array([3.0, 3.0, 3.0])
    x = solve_qp(q, r, ridge_penalty=(1.0, np.zeros(n)), lasso_penalty=(1.0, np.zeros(n)))
    assert x is not None
    assert x.shape == (n,)


def test_qp_bc_ccd_matches_solve_qp_on_box():
    rng = np.random.default_rng(1)
    n = 4
    a = rng.standard_normal((10, n))
    q = a.T @ a + 0.5 * np.eye(n)
    r = rng.standard_normal(n)

    x_cvxpy = solve_qp(q, r, lb=-1.0, ub=1.0)
    x_ccd, path = qp_bc_ccd(q, r, x_minus=-1.0, x_plus=1.0, n_iters=200)
    assert np.allclose(x_cvxpy, x_ccd, atol=1e-2)


def test_qp_bc_ccd_unconstrained_matches_closed_form():
    q = np.array([[2.0, 0.5], [0.5, 2.0]])
    r = np.array([1.0, 1.0])
    x_ccd, path = qp_bc_ccd(q, r, n_iters=200)
    expected = np.linalg.solve(q, r)
    assert np.allclose(x_ccd, expected, atol=1e-4)


def test_qp_hyperplane_satisfies_constraint():
    q = np.array([[2.0, 0.0], [0.0, 2.0]])
    r = np.array([1.0, 1.0])
    a = np.array([1.0, 1.0])
    b = 1.0
    x, lam = qp_hyperplane(q, r, a, b)
    assert np.isclose(a @ x, b, atol=1e-8)


def test_qp_hyperplane_matches_solve_qp():
    q = np.array([[3.0, 1.0], [1.0, 2.0]])
    r = np.array([2.0, -1.0])
    a = np.array([1.0, 2.0])
    b = 0.5
    x_closed, _ = qp_hyperplane(q, r, a, b)
    x_cvxpy = solve_qp(q, r, a_eq=a[None, :], b_eq=np.array([b]))
    assert np.allclose(x_closed, x_cvxpy, atol=1e-4)
