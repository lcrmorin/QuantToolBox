"""Tests for quanttoolbox.stats.dose_response."""

import numpy as np
from scipy.stats import norm

from quanttoolbox.stats.dose_response import (
    drc_hormetic1,
    drc_hormetic2,
    drc_log_logistic,
    drc_log_normal,
    drc_weibull1,
    drc_weibull2,
)


def test_drc_log_logistic_matches_hand_computation():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max = 1.0, 2.0, 0.0, 100.0

    expected = y_min + (y_max - y_min) / (1.0 + np.exp(-beta * (np.log(x) - np.log(alpha))))
    assert np.allclose(drc_log_logistic(x, alpha, beta, y_min, y_max), expected)


def test_drc_log_logistic_at_alpha_is_the_midpoint():
    # At x = alpha (the inflection dose), the curve sits exactly halfway
    # between y_min and y_max regardless of beta.
    alpha, y_min, y_max = 2.0, 0.0, 100.0
    for beta in [0.5, 1.0, 3.0]:
        y = drc_log_logistic(np.array([alpha]), alpha, beta, y_min, y_max)[0]
        assert np.isclose(y, (y_min + y_max) / 2)


def test_drc_log_logistic_monotone_increasing_for_positive_beta():
    # y = y_min + (y_max - y_min) / (1 + exp(-beta * (log(x) - log(alpha)))):
    # as x grows past alpha, the exponential term shrinks toward 0, so the
    # curve rises from y_min toward y_max (for beta > 0).
    x = np.linspace(0.1, 10, 50)
    y = drc_log_logistic(x, alpha=1.0, beta=2.0, y_min=0.0, y_max=100.0)
    assert np.all(np.diff(y) >= 0)


def test_drc_log_normal_matches_hand_computation():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max = 1.5, 1.2, 10.0, 90.0

    expected = y_min + (y_max - y_min) * norm.cdf(beta * (np.log(x) - np.log(alpha)))
    assert np.allclose(drc_log_normal(x, alpha, beta, y_min, y_max), expected)


def test_drc_log_normal_at_alpha_is_the_midpoint():
    alpha, y_min, y_max = 3.0, 0.0, 100.0
    y = drc_log_normal(np.array([alpha]), alpha, 1.5, y_min, y_max)[0]
    assert np.isclose(y, (y_min + y_max) / 2)


def test_drc_weibull1_matches_hand_computation():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max = 1.0, 1.5, 5.0, 95.0

    expected = y_min + (y_max - y_min) * np.exp(-np.exp(beta * (np.log(x) - np.log(alpha))))
    assert np.allclose(drc_weibull1(x, alpha, beta, y_min, y_max), expected)


def test_drc_weibull2_matches_hand_computation():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max = 1.0, 1.5, 5.0, 95.0

    expected = y_min + (y_max - y_min) * (1.0 - np.exp(-np.exp(beta * (np.log(x) - np.log(alpha)))))
    assert np.allclose(drc_weibull2(x, alpha, beta, y_min, y_max), expected)


def test_drc_weibull1_and_weibull2_are_complementary():
    # weibull1(x) + weibull2(x) == y_min + y_max, by construction (one uses
    # exp(-u), the other 1 - exp(-u), for the same u).
    x = np.array([0.3, 1.0, 3.0])
    alpha, beta, y_min, y_max = 1.0, 2.0, 0.0, 100.0
    y1 = drc_weibull1(x, alpha, beta, y_min, y_max)
    y2 = drc_weibull2(x, alpha, beta, y_min, y_max)
    assert np.allclose(y1 + y2, y_min + y_max)


def test_drc_hormetic1_reduces_to_log_logistic_at_gamma_zero():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max = 1.0, 2.0, 0.0, 100.0

    hormetic = drc_hormetic1(x, alpha, beta, y_min, y_max, gamma_=0.0)
    plain = drc_log_logistic(x, alpha, beta, y_min, y_max)
    assert np.allclose(hormetic, plain)


def test_drc_hormetic1_matches_hand_computation():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max, gamma_ = 1.0, 2.0, 0.0, 100.0, 5.0

    y = 1.0 + np.exp(-beta * (np.log(x) - np.log(alpha)))
    expected = y_min + (y_max - y_min + gamma_ * x) / y
    assert np.allclose(drc_hormetic1(x, alpha, beta, y_min, y_max, gamma_), expected)


def test_drc_hormetic2_reduces_to_log_logistic_at_gamma_zero():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max = 1.0, 2.0, 0.0, 100.0

    hormetic = drc_hormetic2(x, alpha, beta, y_min, y_max, gamma_=0.0, delta=1.0)
    plain = drc_log_logistic(x, alpha, beta, y_min, y_max)
    assert np.allclose(hormetic, plain)


def test_drc_hormetic2_matches_hand_computation():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    alpha, beta, y_min, y_max, gamma_, delta = 1.0, 2.0, 0.0, 100.0, 5.0, 1.5

    y = 1.0 + np.exp(-beta * (np.log(x) - np.log(alpha)))
    expected = y_min + (y_max - y_min + gamma_ * np.exp(-1.0 / x**delta)) / y
    assert np.allclose(drc_hormetic2(x, alpha, beta, y_min, y_max, gamma_, delta), expected)
