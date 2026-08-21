"""Tests for quanttoolbox.sustainable_finance.entropy."""

import numpy as np
from scipy.linalg import expm

from quanttoolbox.sustainable_finance.entropy import (
    estimate_markov_generator,
    shannon_entropy,
    shannon_entropy_markov_chain,
)


def test_shannon_entropy_single_variable_matches_hand_computation():
    p_x = np.array([0.5, 0.25, 0.25])
    expected = -np.sum(p_x * np.log(p_x))

    i_x, i_y, i_xy, i_x_y = shannon_entropy(p_x)

    assert np.isclose(i_x, expected)
    assert i_y == 0.0
    assert i_xy == 0.0
    assert np.isclose(i_x_y, i_x)


def test_shannon_entropy_uniform_distribution_matches_log_n():
    n = 4
    p_x = np.full(n, 1.0 / n)
    i_x, _, _, _ = shannon_entropy(p_x)
    assert np.isclose(i_x, np.log(n))


def test_shannon_entropy_independent_joint_gives_zero_mutual_information():
    p_x = np.array([0.5, 0.5])
    p_y = np.array([0.3, 0.7])
    p_xy = np.outer(p_x, p_y)

    i_x, i_y, i_xy, i_x_y = shannon_entropy(p_xy)

    assert np.isclose(i_x, -np.sum(p_x * np.log(p_x)))
    assert np.isclose(i_y, -np.sum(p_y * np.log(p_y)))
    assert np.isclose(i_xy, 0.0, atol=1e-10)
    assert np.isclose(i_x_y, i_x + i_y)


def test_shannon_entropy_perfectly_dependent_joint_gives_mutual_information_equal_to_marginal():
    # p_xy diagonal: X determines Y exactly, so I(X;Y) = I(X) = I(Y) = I(X,Y).
    p_xy = np.diag([0.5, 0.3, 0.2])
    i_x, i_y, i_xy, i_x_y = shannon_entropy(p_xy)

    assert np.isclose(i_x, i_y)
    assert np.isclose(i_xy, i_x)
    assert np.isclose(i_x_y, i_x)


def test_shannon_entropy_zero_probability_entries_do_not_produce_nan():
    p_x = np.array([1.0, 0.0, 0.0])
    i_x, _, _, _ = shannon_entropy(p_x)
    assert np.isclose(i_x, 0.0)
    assert not np.isnan(i_x)


def test_shannon_entropy_markov_chain_at_t_zero_gives_perfectly_dependent_joint():
    lambda_ = np.array([[-0.5, 0.5], [0.3, -0.3]])
    i_x, i_y, i_xy, i_x_y = shannon_entropy_markov_chain(lambda_, t=0.0)

    assert np.isclose(i_x[0], i_y[0])
    assert np.isclose(i_xy[0], i_x[0])


def test_shannon_entropy_markov_chain_matches_direct_expm_computation():
    lambda_ = np.array([[-0.5, 0.5], [0.3, -0.3]])
    t = np.array([1.0, 2.0])

    p_inf = expm(lambda_ * 1000.0)
    pi = p_inf[0, :]

    expected_i_x = []
    expected_i_xy_joint = []
    for tt in t:
        p_t = expm(lambda_ * tt)
        p_xy = pi[:, None] * p_t
        p_x = np.sum(p_xy, axis=1)
        expected_i_x.append(-np.sum(np.where(p_x > 0, p_x * np.log(p_x), 0.0)))
        expected_i_xy_joint.append(-np.sum(np.where(p_xy > 0, p_xy * np.log(p_xy), 0.0)))

    i_x, i_y, i_xy, i_x_y = shannon_entropy_markov_chain(lambda_, t)
    assert np.allclose(i_x, expected_i_x)
    assert np.allclose(i_x_y, expected_i_xy_joint)


def test_estimate_markov_generator_lambda1_preserves_row_sums():
    # Israel-Rosenthal-Wei Lambda1: min(x,0) + max(x,0) = x identically, so
    # Lambda1's row sums exactly equal Lambda's row sums.
    lam = np.array(
        [
            [-0.6, 0.5, 0.2],
            [0.3, -0.5, 0.1],
            [-0.1, 0.4, -0.2],
        ]
    )
    lambda1, _ = estimate_markov_generator(lam)
    assert np.allclose(np.sum(lambda1, axis=1), np.sum(lam, axis=1))


def test_estimate_markov_generator_lambda1_has_no_negative_off_diagonal_entries():
    lam = np.array(
        [
            [-0.6, 0.5, 0.2],
            [0.3, -0.5, 0.1],
            [-0.1, 0.4, -0.2],
        ]
    )
    lambda1, lambda2 = estimate_markov_generator(lam)

    off_diag_mask = ~np.eye(3, dtype=bool)
    assert np.all(lambda1[off_diag_mask] >= 0.0)
    assert np.all(lambda2[off_diag_mask] >= -1e-12)


def test_estimate_markov_generator_valid_generator_is_left_unchanged_by_lambda1():
    # a generator that already has non-negative off-diagonal entries and
    # zero row sums should pass through Lambda1 unchanged.
    lam = np.array(
        [
            [-0.8, 0.5, 0.3],
            [0.2, -0.6, 0.4],
            [0.1, 0.3, -0.4],
        ]
    )
    assert np.allclose(np.sum(lam, axis=1), 0.0)

    lambda1, _ = estimate_markov_generator(lam)
    assert np.allclose(lambda1, lam)


def test_estimate_markov_generator_lambda2_also_preserves_row_sums():
    # Lambda2's redistribution is also row-sum-preserving in general (not
    # just when the input already happens to sum to zero) -- verified
    # algebraically: for rows with positive off-diagonal mass (g[i] > 0),
    # sum_j Lambda2(i,j) = lam[i,i] + row_pos_sum[i] - b[i], which is
    # exactly sum_j Lambda(i,j).
    lam = np.array(
        [
            [-0.6, 0.5, 0.2],
            [0.3, -0.5, 0.1],
            [-0.1, 0.4, -0.2],
        ]
    )
    _, lambda2 = estimate_markov_generator(lam)
    assert np.allclose(np.sum(lambda2, axis=1), np.sum(lam, axis=1))


def test_estimate_markov_generator_zero_sum_input_stays_zero_sum():
    # When the input already has zero row sums (the typical case for a
    # generator estimated from a valid transition matrix that merely has
    # sign violations), both repair methods preserve that.
    lam = np.array(
        [
            [-0.9, 0.5, 0.4],
            [0.3, -0.6, 0.3],
            [-0.1, 0.3, -0.2],
        ]
    )
    assert np.allclose(np.sum(lam, axis=1), 0.0)

    lambda1, lambda2 = estimate_markov_generator(lam)
    assert np.allclose(np.sum(lambda1, axis=1), 0.0, atol=1e-10)
    assert np.allclose(np.sum(lambda2, axis=1), 0.0, atol=1e-10)
