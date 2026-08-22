"""Tests for quanttoolbox.credit.markov_chain."""

import numpy as np

from quanttoolbox.credit.markov_chain import expected_hitting_time


def test_expected_hitting_time_two_state_matches_hand_derived_formula():
    # Two-state chain 0<->1, absorbing target {1}: h(0) = 1/p, where p is
    # the probability of moving 0 -> 1 in one step (standard geometric-
    # waiting-time result).
    p01 = 0.2
    p = np.array([[1 - p01, p01], [0.0, 1.0]])
    h = expected_hitting_time(p, target_states=[1])
    assert np.isclose(h[0], 1.0 / p01)
    assert h[1] == 0.0


def test_expected_hitting_time_zero_at_target_states():
    rng = np.random.default_rng(1)
    n = 5
    p = rng.random((n, n))
    p = p / p.sum(axis=1, keepdims=True)
    h = expected_hitting_time(p, target_states=[2, 4])
    assert h[2] == 0.0
    assert h[4] == 0.0


def test_expected_hitting_time_satisfies_first_step_equation():
    rng = np.random.default_rng(2)
    n = 6
    p = rng.random((n, n))
    p = p / p.sum(axis=1, keepdims=True)
    target = [0]
    h = expected_hitting_time(p, target_states=target)

    other = np.setdiff1d(np.arange(n), target)
    # h[i] == 1 + sum_j p[i,j] * h[j] for every non-target state i.
    lhs = h[other]
    rhs = 1.0 + p[np.ix_(other, np.arange(n))] @ h
    assert np.allclose(lhs, rhs)


def test_expected_hitting_time_all_states_are_targets_returns_zero():
    n = 3
    p = np.eye(n)
    h = expected_hitting_time(p, target_states=np.arange(n))
    assert np.allclose(h, 0.0)
