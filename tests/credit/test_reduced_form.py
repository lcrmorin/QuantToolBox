"""Tests for quanttoolbox.credit.reduced_form."""

import numpy as np
from scipy.linalg import expm, logm

from quanttoolbox.credit.reduced_form import (
    build_climate_stressed_generator,
    cdf_exponential,
    density_markov_generator,
    hazard_markov_generator,
    inv_exponential,
    pdf_exponential,
    rnd_exponential,
    survival_exponential,
    survival_markov_generator,
)
from quanttoolbox.sustainable_finance.entropy import estimate_markov_generator


def test_survival_exponential_homogeneous_matches_closed_form():
    lam = np.array([0.05, 0.1])
    t = np.array([0.0, 1.0, 2.0, 5.0])

    result = survival_exponential(t, lam)
    expected = np.exp(-np.outer(t, lam))
    assert np.allclose(result, expected)


def test_cdf_and_survival_exponential_sum_to_one():
    lam = np.array([0.05, 0.1])
    t = np.array([0.5, 1.0, 3.0])

    s = survival_exponential(t, lam)
    f = cdf_exponential(t, lam)
    assert np.allclose(s + f, 1.0)


def test_pdf_exponential_homogeneous_matches_hazard_times_survival():
    lam = np.array([0.05, 0.1])
    t = np.array([0.5, 1.0, 3.0])

    s = survival_exponential(t, lam)
    f = pdf_exponential(t, lam)
    assert np.allclose(f, lam[None, :] * s)


def test_inv_exponential_homogeneous_round_trips_cdf():
    lam = np.array([0.05, 0.1])
    t = np.array([1.0, 2.0, 5.0])

    p = cdf_exponential(t, lam)
    t_back = inv_exponential(p, lam)
    assert np.allclose(t_back, t[:, None], atol=1e-6)


def test_survival_exponential_piecewise_matches_hand_computed_hazard_integral():
    # knots at 1, 2, 3 (last extended to +inf); hazard 0.05/0.10/0.20
    lambda_pw = np.array([[1.0, 0.05], [2.0, 0.10], [3.0, 0.20]])
    t = np.array([0.5, 1.0, 1.5, 2.5, 10.0])

    def hand_survival(tt):
        if tt <= 1.0:
            h = 0.05 * tt
        elif tt <= 2.0:
            h = 0.05 * 1.0 + 0.10 * (tt - 1.0)
        else:
            h = 0.05 * 1.0 + 0.10 * 1.0 + 0.20 * (tt - 2.0)
        return np.exp(-h)

    result = survival_exponential(t, lambda_pw)
    expected = np.array([hand_survival(tt) for tt in t])
    assert np.allclose(result[:, 0], expected)


def test_pdf_exponential_piecewise_matches_hazard_times_survival():
    lambda_pw = np.array([[1.0, 0.05], [2.0, 0.10], [3.0, 0.20]])
    t = np.array([0.5, 1.0, 1.5, 2.5, 10.0])

    def hazard(tt):
        if tt < 1.0:
            return 0.05
        if tt < 2.0:
            return 0.10
        return 0.20

    s = survival_exponential(t, lambda_pw)
    f = pdf_exponential(t, lambda_pw)
    expected = np.array([hazard(tt) * ss for tt, ss in zip(t, s[:, 0], strict=True)])
    assert np.allclose(f[:, 0], expected)


def test_inv_exponential_piecewise_round_trips_cdf():
    lambda_pw = np.array([[1.0, 0.05], [2.0, 0.10], [3.0, 0.20]])
    t = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 10.0])

    p = cdf_exponential(t, lambda_pw)
    t_back = inv_exponential(p, lambda_pw)
    assert np.allclose(t_back[:, 0], t, atol=1e-6)


def test_inv_exponential_broadcasts_single_probability_column_across_scenarios():
    lambda_pw = np.array([[1.0, 0.05, 0.5], [2.0, 0.10, 0.3], [3.0, 0.20, 0.4]])
    t = np.array([0.5, 1.0, 1.5, 2.5, 3.5])

    s = survival_exponential(t, lambda_pw)
    p_col0 = 1.0 - s[:, [0]]

    t_back = inv_exponential(p_col0, lambda_pw)
    assert t_back.shape == (5, 2)
    assert np.allclose(t_back[:, 0], t, atol=1e-6)


def test_inv_exponential_extreme_probabilities_return_nan():
    lam = np.array([0.05])
    p = np.array([0.0, 1.0, 0.5])

    result = inv_exponential(p, lam)
    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 0])
    assert not np.isnan(result[2, 0])


def test_rnd_exponential_generated_default_times_are_nonnegative_and_reproducible():
    lam = np.array([0.1])
    t1 = rnd_exponential(1000, 1, lam, random_state=42)
    t2 = rnd_exponential(1000, 1, lam, random_state=42)

    assert np.all(t1 >= 0.0)
    assert np.allclose(t1, t2)


def test_rnd_exponential_pregenerated_uniforms_matches_direct_inversion():
    lam = np.array([0.1])
    u = np.array([[0.2], [0.5], [0.8]])

    via_rnd = rnd_exponential(u, 0, lam)
    via_inv = inv_exponential(u, lam)
    assert np.allclose(via_rnd, via_inv, equal_nan=True)


def test_rnd_exponential_sample_mean_matches_theoretical_mean():
    lam = np.array([0.2])
    samples = rnd_exponential(20000, 1, lam, random_state=0)
    # mean of an Exp(lambda) distribution is 1/lambda
    assert np.isclose(np.nanmean(samples), 1.0 / lam[0], rtol=0.05)


def test_survival_markov_generator_matches_direct_expm_computation():
    lambda_matrix = np.array(
        [
            [-0.3, 0.25, 0.05],
            [0.1, -0.35, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )
    t = np.array([0.0, 0.5, 1.0, 3.0])

    result = survival_markov_generator(t, lambda_matrix)
    for i, tt in enumerate(t):
        if tt == 0.0:
            expected = np.ones(3)
        else:
            m = expm(tt * lambda_matrix)
            expected = 1.0 - m[:, 2]
        assert np.allclose(result[i], expected)


def test_density_markov_generator_at_t_zero_matches_continuous_limit():
    # f(0) has a special-cased branch (returns Lambda[:, -1] directly,
    # skipping the matrix exponential); it should agree with the general
    # formula's limit as t -> 0 (expm(0) = I), and with evaluating the
    # general formula directly at t = 0.
    lambda_matrix = np.array(
        [
            [-0.3, 0.25, 0.05],
            [0.1, -0.35, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )
    f_at_zero = density_markov_generator(0.0, lambda_matrix)
    assert np.allclose(f_at_zero, lambda_matrix[:, -1][None, :])

    general_formula_at_zero = lambda_matrix @ expm(0.0 * lambda_matrix)
    assert np.allclose(f_at_zero[0], general_formula_at_zero[:, -1])

    # Mixed array of t values including an exact zero exercises both
    # branches of the per-element loop in a single call.
    t = np.array([0.0, 1.0])
    f_mixed = density_markov_generator(t, lambda_matrix)
    assert np.allclose(f_mixed[0], lambda_matrix[:, -1])
    assert np.allclose(f_mixed[1], (lambda_matrix @ expm(1.0 * lambda_matrix))[:, -1])


def test_hazard_markov_generator_matches_density_over_survival():
    lambda_matrix = np.array(
        [
            [-0.3, 0.25, 0.05],
            [0.1, -0.35, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )
    t = np.array([0.5, 1.0, 3.0])

    s = survival_markov_generator(t, lambda_matrix)
    f = density_markov_generator(t, lambda_matrix)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = hazard_markov_generator(t, lambda_matrix)
        expected = f / s
    # column 2 (the absorbing default state itself) is 0/0 = nan: an
    # already-defaulted obligor has no well-defined default hazard.
    assert np.allclose(h, expected, equal_nan=True)
    assert np.all(np.isnan(h[:, 2]))
    assert np.all(np.isfinite(h[:, :2]))


def test_build_climate_stressed_generator_preserves_row_sums_zero():
    lambda_matrix = np.array(
        [
            [-0.3, 0.25, 0.05],
            [0.1, -0.35, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )
    for beta in [0.0, 0.5, 1.0, 2.5, 5.0]:
        lambda_climate = build_climate_stressed_generator(lambda_matrix, beta)
        assert np.allclose(lambda_climate.sum(axis=1), 0.0, atol=1e-12)


def test_build_climate_stressed_generator_scales_default_column_and_rebalances_diagonal():
    lambda_matrix = np.array(
        [
            [-0.3, 0.25, 0.05],
            [0.1, -0.35, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )
    beta = 2.5
    lambda_climate = build_climate_stressed_generator(lambda_matrix, beta)

    stress = beta * lambda_matrix[:, -1]
    expected = lambda_matrix.copy()
    expected[:, -1] = expected[:, -1] + stress
    expected = expected - np.diag(stress)

    assert np.allclose(lambda_climate, expected)
    # off-diagonal entries in non-default columns are untouched by the
    # stress scaling itself
    assert np.isclose(lambda_climate[0, 1], lambda_matrix[0, 1])
    assert np.isclose(lambda_climate[2, 0], lambda_matrix[2, 0])
    # diagonal is shifted by exactly -stress
    assert np.allclose(np.diag(lambda_climate) - np.diag(lambda_matrix), -stress)


def test_build_climate_stressed_generator_matches_notebook_call_site():
    # Reproduces the exact P -> Lambda derivation and beta_25 = 2.5 call
    # site from HSF-Notebooks chapter 13f (cell defining beta_25).
    p = np.array(
        [
            [92.82, 6.50, 0.56, 0.06, 0.06, 0.00, 0.00, 0.00],
            [0.63, 91.87, 6.64, 0.65, 0.06, 0.11, 0.04, 0.00],
            [0.08, 2.26, 91.66, 5.11, 0.61, 0.23, 0.01, 0.04],
            [0.05, 0.27, 5.84, 87.74, 4.74, 0.98, 0.16, 0.22],
            [0.04, 0.11, 0.64, 7.85, 81.14, 8.27, 0.89, 1.06],
            [0.00, 0.11, 0.30, 0.42, 6.75, 83.07, 3.86, 5.49],
            [0.19, 0.00, 0.38, 0.75, 2.44, 12.03, 60.71, 23.50],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00],
        ]
    )
    p = p / p.sum(axis=1, keepdims=True)

    lambda_raw = logm(p)
    lambda_matrix, _ = estimate_markov_generator(lambda_raw)

    beta_25 = 2.5
    lambda_climate = build_climate_stressed_generator(lambda_matrix, beta_25)

    # generator-matrix validity: every row still sums to zero
    assert np.allclose(lambda_climate.sum(axis=1), 0.0, atol=1e-10)

    # cross-check against the notebook's own formula, evaluated directly
    k = lambda_matrix.shape[1]
    stress = beta_25 * lambda_matrix[:, k - 1]
    expected = lambda_matrix.copy()
    expected[:, k - 1] = expected[:, k - 1] + stress
    expected = expected - np.diag(stress)
    assert np.allclose(lambda_climate, expected)

    # spot-check against a value independently computed from the notebook's
    # own P -> Lambda -> Lambda_climate pipeline (CCC row, CCC column)
    assert np.isclose(lambda_climate[6, 6], -1.2398547456, atol=1e-6)
