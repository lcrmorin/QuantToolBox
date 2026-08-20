"""Tests for quanttoolbox.optim.projection."""

import numpy as np

from quanttoolbox.optim.projection import (
    projection_box_l2,
    projection_l1,
    projection_l2,
    projection_linfinity,
)


def test_projection_l1_within_ball_unchanged():
    v = np.array([0.1, 0.2, 0.1])  # L1 norm 0.4
    out = projection_l1(v, radius=1.0)
    assert np.allclose(out, v)


def test_projection_l1_outside_ball_projects_to_boundary():
    v = np.array([2.0, 2.0])
    out = projection_l1(v, radius=1.0)
    assert np.isclose(np.sum(np.abs(out)), 1.0, atol=1e-6)


def test_projection_l1_methods_agree():
    v = np.array([3.0, -1.0, 2.0, 0.5])
    out1 = projection_l1(v, radius=2.0, method=1)
    out2 = projection_l1(v, radius=2.0, method=2)
    assert np.allclose(out1, out2, atol=1e-8)


def test_projection_l2_shrinks_appropriately():
    v = np.array([3.0, 4.0])  # norm 5
    out = projection_l2(v, lambda_=2.0)
    # projection_l2 = v - proximal_l2(v, lambda) -- removes the shrunk part
    assert np.isclose(np.linalg.norm(out), 2.0)


def test_projection_linfinity_clips():
    v = np.array([5.0, -5.0, 0.5])
    out = projection_linfinity(v, radius=2.0)
    assert np.allclose(out, [2.0, -2.0, 0.5])


def test_projection_box_l2_satisfies_box():
    v = np.array([3.0, -3.0])
    c = np.array([0.0, 0.0])
    x, retcode = projection_box_l2(v, x_minus=-1.0, x_plus=1.0, c=c, lambda_=5.0)
    assert np.all(x >= -1.0 - 1e-6)
    assert np.all(x <= 1.0 + 1e-6)


def test_projection_box_l2_within_constraints_stable():
    v = np.array([0.1, -0.1])
    c = np.array([0.0, 0.0])
    x, retcode = projection_box_l2(v, x_minus=-1.0, x_plus=1.0, c=c, lambda_=5.0)
    # already well within both box and ball -- should stay close to v
    assert np.allclose(x, v, atol=1e-3)
