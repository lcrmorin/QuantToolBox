"""Tests for quanttoolbox.credit.vasicek."""

import numpy as np
from scipy.stats import norm

from quanttoolbox.credit.vasicek import (
    invcdf_default_rate,
    thresholds_from_matrix,
    vasicek_density,
)


def test_thresholds_from_matrix_round_trips_transition_probabilities():
    rng = np.random.default_rng(0)
    p = rng.random((5, 5)) + np.eye(5) * 3  # diagonal-heavy, like a real rating matrix

    result = thresholds_from_matrix(p, inf_limit=10.0)

    p_ij = p / p.sum(axis=1, keepdims=True)
    assert np.allclose(result.p_ij, p_ij)

    p_from_thresholds = norm.cdf(result.z2) - norm.cdf(result.z1)
    assert np.allclose(p_from_thresholds, p_ij, atol=1e-8)


def test_thresholds_from_matrix_caps_outer_buckets_at_inf_limit():
    # A matrix with an exact 0% and an exact 100% transition cell.
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = thresholds_from_matrix(p, inf_limit=10.0)
    assert np.all(np.isfinite(result.z1))
    assert np.all(np.isfinite(result.z2))
    assert np.max(np.abs(result.z1)) <= 10.0
    assert np.max(np.abs(result.z2)) <= 10.0


def test_invcdf_default_rate_matches_hand_derived_closed_form():
    alpha, default_prob, rho = 0.999, 0.02, 0.15
    expected = norm.cdf(
        (norm.ppf(default_prob) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho)
    )
    assert np.isclose(invcdf_default_rate(alpha, default_prob, rho), expected)


def test_invcdf_default_rate_recovers_default_prob_as_rho_vanishes():
    # As rho -> 0, the systematic factor stops mattering and the quantile
    # collapses to the unconditional default_prob at every confidence level.
    default_prob = 0.03
    for alpha in (0.5, 0.9, 0.999):
        assert np.isclose(invcdf_default_rate(alpha, default_prob, 1e-9), default_prob, atol=1e-4)


def test_invcdf_default_rate_increasing_in_alpha():
    default_prob, rho = 0.02, 0.2
    alphas = np.array([0.5, 0.9, 0.99, 0.999])
    values = invcdf_default_rate(alphas, default_prob, rho)
    assert np.all(np.diff(values) > 0)


def test_vasicek_density_matches_hand_derived_closed_form():
    d, default_prob, rho = 0.03, 0.02, 0.15
    x = norm.ppf(d)
    expected = np.sqrt((1 - rho) / rho) * np.exp(
        0.5 * x**2 - (np.sqrt(1 - rho) * x - norm.ppf(default_prob)) ** 2 / (2 * rho)
    )
    assert np.isclose(vasicek_density(d, default_prob, rho), expected)


def test_vasicek_density_integrates_to_one():
    default_prob, rho = 0.05, 0.2
    d_grid = np.linspace(1e-5, 1 - 1e-5, 20_000)
    pdf = vasicek_density(d_grid, default_prob, rho)
    area = np.trapezoid(pdf, d_grid)
    assert np.isclose(area, 1.0, atol=1e-2)
