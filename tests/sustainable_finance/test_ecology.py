"""Tests for quanttoolbox.sustainable_finance.ecology."""

import numpy as np
import pytest
from scipy.optimize import brentq

from quanttoolbox.sustainable_finance.ecology import (
    endemics_area_relationship,
    fn_delta_t,
    hurlbert,
    species_abundance_distribution,
    species_area_relationship,
)


def _notebook_species_abundance_distribution(n_i, brk=None):
    # A verbatim reimplementation of HSF-Notebooks chapter 05c's own
    # (independent, not-imported-from-quanttoolbox) reimplementation of
    # `species_abundance_distribution` (notebook 05c, cell 1), kept
    # separate from quanttoolbox's implementation so it serves as a
    # genuine cross-check rather than testing the code against itself.
    n_i = np.asarray(n_i, dtype=float)
    if brk is None:
        j, s = np.unique(n_i, return_counts=True)
        return j, s
    if isinstance(brk, str) and brk.lower() == "octave":
        k = np.floor(np.log2(n_i)).astype(int)
        k_max = k.max()
        octaves = np.arange(0, k_max + 1)
        s = np.array([(k == kk).sum() for kk in octaves])
        edges = 2.0 ** (octaves + 1)
        return octaves, s, edges
    brk = np.asarray(brk, dtype=float)
    edges = np.concatenate([[0.0], brk])
    s, _ = np.histogram(n_i, bins=edges)
    return brk, s


# Real per-species tree-abundance counts from Whittaker's Siskiyou
# Mountains vegetation survey (chap5_biodiversity_whittaker_siskiyou.xlsx,
# sheet "4_Tree_Data", grouped by species and summed), as read and grouped
# in HSF-Notebooks chapter 05c, cell 10. Several counts are non-integer
# (e.g. 894.001, 8.001) -- real field-survey data, not a synthetic example
# -- which is exactly what exposed the old toolbox binning convention's
# disagreement with the notebook's numpy-native one (see
# species_abundance_distribution's docstring).
_N_I_WHITTAKER = np.array(
    [
        4172.0, 1720.0, 2.0, 894.001, 244.0, 140.0, 32.0, 10.0,
        63.001999999999995, 56.0, 77.001, 3.0, 1636.0, 1816.0, 61.0, 12.0,
        966.0, 818.0, 27.0, 4.0, 1886.0, 409.002, 8.001, 102.0,
        925.001, 1599.0, 39.0, 364.0, 8439.0, 4.001, 11.001999999999999, 10.0,
        1.0, 262.0, 24.0, 934.0, 452.0, 1155.0, 123.0, 1.0,
        3725.0, 4207.0, 72.0, 38.0, 44.0, 13.0, 277.003, 685.0,
        685.002, 63.0, 5.0, 350.0, 53.0, 1025.001, 439.001, 1.0,
        535.0, 1.0, 1.0,
    ]
)  # fmt: skip

# Real per-species tree-abundance counts from Condit's Barro Colorado
# Island 50-hectare forest census (chap5_biodiversity_condit_bci.xlsx,
# sheet "full matrix", BCI-prefixed columns summed per species, zero
# counts dropped), as read in HSF-Notebooks chapter 05c, cell 22.
_N_I_BCI = np.array(
    [
        3, 2, 1, 92, 23, 156, 1, 1, 27, 983, 3, 22, 28, 27, 236, 21, 52, 201, 39, 1,
        294, 188, 1, 55, 23, 100, 3, 2, 54, 87, 19, 264, 25, 2, 39, 38, 2, 3, 1, 2,
        85, 25, 22, 13, 1, 63, 325, 364, 55, 98, 1, 12, 4, 47, 88, 13, 16, 33, 285, 21,
        2, 26, 18, 81, 12, 55, 177, 1717, 1, 7, 3, 4, 7, 3, 23, 5, 6, 92, 12, 23,
        99, 10, 376, 68, 244, 38, 85, 644, 13, 229, 7, 288, 5, 681, 101, 41, 52, 10, 49, 98,
        2, 20, 10, 67, 5, 26, 76, 14, 14, 236, 33, 51, 12, 27, 5, 14, 10, 64, 121, 93,
        1, 25, 15, 167, 10, 2, 21, 8, 70, 1, 7, 5, 7, 33, 10, 4, 5, 29, 36, 22,
        184, 788, 33, 1, 5, 3, 21, 16, 45, 9, 1, 61, 43, 1, 9, 15, 755, 13, 2, 203,
        31, 345, 111, 50, 381, 8, 4, 2, 80, 724, 4, 248, 17, 3, 33, 2, 1, 289, 13, 16,
        78, 346, 12, 28, 8, 29, 63, 26, 218, 118, 26, 30, 68, 322, 98, 1, 3, 28, 43, 379,
        7, 12, 2, 5, 40, 15, 2, 82, 1681, 1, 147, 32, 58, 163, 617, 25, 164, 1, 12, 143,
        149, 67, 45, 1, 10,
    ],
    dtype=float,
)  # fmt: skip


def test_species_area_relationship_full_area_gives_total_species():
    n_i = np.array([3.0, 5.0, 1.0, 10.0])
    total_species = n_i.shape[0]

    result = species_area_relationship(n_i, None, area_total=100.0, area=100.0)
    assert np.isclose(result, total_species)


def test_species_area_relationship_zero_area_gives_zero_species():
    n_i = np.array([3.0, 5.0, 1.0, 10.0])

    result = species_area_relationship(n_i, None, area_total=100.0, area=0.0)
    assert np.isclose(result, 0.0)


def test_species_area_relationship_array_of_areas_is_monotone_increasing():
    n_i = np.array([3.0, 5.0, 1.0, 10.0])
    areas = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

    result = species_area_relationship(n_i, None, area_total=100.0, area=areas)
    assert np.all(np.diff(result) >= 0)


def test_species_area_relationship_histogram_mode_matches_n_i_mode():
    n_i = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    j, s, _ = species_abundance_distribution(n_i)

    area_total, area = 100.0, 40.0
    direct = species_area_relationship(n_i, None, area_total, area)
    via_hist = species_area_relationship(j, s, area_total, area)
    assert np.isclose(direct, via_hist)


def test_endemics_area_relationship_full_area_gives_total_species():
    n_i = np.array([3.0, 5.0, 1.0, 10.0])
    total_species = n_i.shape[0]

    result = endemics_area_relationship(n_i, None, area_total=100.0, area=100.0)
    assert np.isclose(result, total_species)


def test_endemics_area_relationship_zero_area_gives_zero_endemics():
    n_i = np.array([3.0, 5.0, 1.0, 10.0])

    result = endemics_area_relationship(n_i, None, area_total=100.0, area=0.0)
    assert np.isclose(result, 0.0)


def test_endemics_area_relationship_histogram_mode_matches_n_i_mode():
    n_i = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    j, s, _ = species_abundance_distribution(n_i)

    area_total, area = 100.0, 40.0
    direct = endemics_area_relationship(n_i, None, area_total, area)
    via_hist = endemics_area_relationship(j, s, area_total, area)
    assert np.isclose(direct, via_hist)


def test_species_abundance_distribution_default_mode_matches_hand_count():
    n_i = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    j, s, breaks = species_abundance_distribution(n_i)

    assert breaks is None
    assert np.allclose(j, [1.0, 2.0, 3.0])
    assert np.allclose(s, [2.0, 1.0, 3.0])
    assert np.sum(s) == n_i.shape[0]


def test_species_abundance_distribution_octave_mode_covers_all_individuals():
    n_i = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 20.0])
    j, s, breaks = species_abundance_distribution(n_i, breaks="octave")

    assert breaks is not None
    assert np.sum(s) == n_i.shape[0]
    assert j.shape == s.shape


def test_species_abundance_distribution_octave_mode_matches_floor_log2_convention():
    # Locks in the corrected numpy-native octave convention: class k
    # (k=0,1,2,...) is n in [2**k, 2**(k+1)), i.e. k = floor(log2(n)) --
    # n=1 -> k=0; n=2,3 -> k=1; n=4..7 -> k=2; n=8..15 -> k=3; n=20 -> k=4.
    n_i = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 20.0])
    j, s, breaks = species_abundance_distribution(n_i, breaks="octave")

    assert np.array_equal(j, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(s, [1.0, 2.0, 1.0, 2.0, 1.0])
    assert np.array_equal(breaks, [2.0, 4.0, 8.0, 16.0, 32.0])


def test_species_abundance_distribution_octave_mode_rejects_other_strings():
    with pytest.raises(ValueError, match="breaks must be"):
        species_abundance_distribution(np.array([1.0, 2.0]), breaks="linear")


def test_species_abundance_distribution_custom_breaks_covers_all_individuals():
    n_i = np.array([1.0, 4.0, 6.0, 9.0, 12.0])
    breaks = np.array([2.0, 5.0, 10.0, 15.0])

    j, s, out_breaks = species_abundance_distribution(n_i, breaks=breaks)

    assert np.array_equal(out_breaks, breaks)
    assert np.sum(s) == n_i.shape[0]
    assert j.shape == (4,)


def test_species_abundance_distribution_custom_breaks_matches_np_histogram_convention():
    # Locks in the corrected numpy-native convention: bins over
    # [0, *breaks], half-open [lower, upper) except the last bin, which is
    # closed at both ends -- exactly `np.histogram(n_i, bins=[0,*breaks])`.
    # n=1 -> [0,2); n=4 -> [2,5); n=6,9 -> [5,10); n=12 -> [10,15] (closed).
    n_i = np.array([1.0, 4.0, 6.0, 9.0, 12.0])
    breaks = np.array([2.0, 5.0, 10.0, 15.0])

    j, s, out_breaks = species_abundance_distribution(n_i, breaks=breaks)

    assert np.array_equal(s, [1.0, 1.0, 2.0, 1.0])
    assert np.allclose(j, [1.0, 3.5, 7.5, 12.5])
    expected_s, _ = np.histogram(n_i, bins=np.concatenate([[0.0], breaks]))
    assert np.array_equal(s, expected_s)


def test_species_abundance_distribution_custom_breaks_boundary_value_goes_to_upper_bin():
    # The old (now-corrected) toolbox convention was inclusive at each
    # bin's *top* (n_i<=1 landed in the first bucket); numpy's convention
    # is inclusive at the *bottom* instead (bins are half-open [lo,hi)
    # except the last), so a count exactly on an interior breakpoint now
    # goes to the bin ABOVE it, not the one below.
    n_i = np.array([1.0])
    breaks = np.array([1.0, 5.0])

    j, s, out_breaks = species_abundance_distribution(n_i, breaks=breaks)

    assert np.array_equal(s, [0.0, 1.0])


def test_hurlbert_methods_1_and_2_agree():
    n_i = np.array([50.0, 30.0, 20.0, 10.0, 5.0])
    m = 40

    result1 = hurlbert(n_i, m, method=1)
    result2 = hurlbert(n_i, m, method=2)
    assert np.isclose(result1, result2, rtol=1e-8)


def test_hurlbert_full_sample_gives_total_species_count():
    n_i = np.array([50.0, 30.0, 20.0, 10.0, 5.0])
    n = int(np.sum(n_i))

    result = hurlbert(n_i, m=n)
    assert np.isclose(result, n_i.shape[0])


def test_hurlbert_sample_of_one_always_gives_exactly_one_species():
    # Drawing a single individual always yields exactly one species, so
    # E[S(1)] = 1 regardless of the abundance distribution -- by hand:
    # sac = sum_s (1 - (n - n_s)/n) = sum_s (n_s / n) = n / n = 1.
    n_i = np.array([50.0, 30.0, 20.0])
    result = hurlbert(n_i, m=1)
    assert np.isclose(result, 1.0)


# --- species_abundance_distribution: real-data cross-check against HSF-Notebooks
# chapter 05c's own (independent) implementation, per Task 3. -----------------


def test_species_abundance_distribution_matches_notebook_05c_on_whittaker_octave_mode():
    # HSF-Notebooks chapter 05c, cell 18 -- octave distribution of
    # Whittaker's Siskiyou Mountains tree survey. This is exactly the
    # dataset (non-integer counts included) that exposed the old toolbox
    # convention's disagreement with the notebook's before this fix.
    j, s, breaks = species_abundance_distribution(_N_I_WHITTAKER, breaks="octave")
    j_nb, s_nb, breaks_nb = _notebook_species_abundance_distribution(_N_I_WHITTAKER, "octave")

    assert np.array_equal(j, j_nb.astype(float))
    assert np.array_equal(s, s_nb.astype(float))
    assert np.array_equal(breaks, breaks_nb)
    assert np.sum(s) == _N_I_WHITTAKER.shape[0]


def test_species_abundance_distribution_matches_notebook_05c_on_whittaker_custom_breaks():
    # HSF-Notebooks chapter 05c, cell 10 -- custom-breakpoint distribution
    # of the same Whittaker data, with the notebook's own breakpoints
    # (brk = 50, 100, ..., 1000).
    brk = np.arange(50, 1050, 50, dtype=float)

    j, s, out_breaks = species_abundance_distribution(_N_I_WHITTAKER, breaks=brk)
    brk_nb, s_nb = _notebook_species_abundance_distribution(_N_I_WHITTAKER, brk)

    assert np.array_equal(s, s_nb.astype(float))
    assert np.array_equal(out_breaks, brk_nb)


def test_species_abundance_distribution_matches_notebook_05c_on_bci_octave_mode():
    # HSF-Notebooks chapter 05c, cell 22 -- octave distribution of
    # Condit's Barro Colorado Island 50-hectare forest census (BCI
    # columns).
    j, s, breaks = species_abundance_distribution(_N_I_BCI, breaks="octave")
    j_nb, s_nb, breaks_nb = _notebook_species_abundance_distribution(_N_I_BCI, "octave")

    assert np.array_equal(j, j_nb.astype(float))
    assert np.array_equal(s, s_nb.astype(float))
    assert np.array_equal(breaks, breaks_nb)
    assert np.sum(s) == _N_I_BCI.shape[0]


# --- fn_delta_t -----------------------------------------------------------


def test_fn_delta_t_lambda_and_mu_long_boundary_values():
    # HSF-Notebooks chapter 16c, cell 29's own boundary sanity checks:
    # lambda(0)=lambda_0, lambda(S_s)=0, mu_long(0)=0, mu_long(S_s)=mu_s.
    S_s = 200.0
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 0.70, 0.01

    delta_0, lambda_0_val, mu_long_0, mu_short_0, mu_0 = fn_delta_t(
        0.0, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s
    )
    delta_s, lambda_s_val, mu_long_s, mu_short_s, mu_s_val = fn_delta_t(
        S_s, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s
    )

    assert np.isclose(lambda_0_val, lambda_0)
    assert np.isclose(lambda_s_val, 0.0, atol=1e-12)
    assert np.isclose(mu_long_0, 0.0, atol=1e-12)
    assert np.isclose(mu_long_s, mu_s)


def test_fn_delta_t_lambda_non_increasing_mu_long_non_decreasing():
    S_s = 200.0
    S_t = np.arange(0, S_s + 1)
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 0.70, 0.01

    _, lambda_t, mu_long_t, _, _ = fn_delta_t(S_t, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)

    assert np.all(np.diff(lambda_t) <= 1e-9)
    assert np.all(np.diff(mu_long_t) >= -1e-9)


def test_fn_delta_t_single_equilibrium_matches_notebook_16c_q1f_case():
    # HSF-Notebooks chapter 16c Q1.f's own single-equilibrium parameters.
    # The root S* found independently here (via scipy's brentq, rather
    # than reusing quanttoolbox's own bisection routine) must be a
    # genuine zero of delta(S), with delta changing sign around it.
    S_s = 200.0
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 0.70, 0.01

    def delta_scalar(x):
        return fn_delta_t(x, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)[0]

    s_star = brentq(delta_scalar, 0.0, S_s)
    delta_star = delta_scalar(s_star)

    assert np.isclose(delta_star, 0.0, atol=1e-8)
    assert delta_scalar(s_star - 1.0) > 0.0 > delta_scalar(s_star + 1.0)


def test_fn_delta_t_bistable_case_matches_notebook_16c_q1g_case():
    # HSF-Notebooks chapter 16c Q1.g's own bistable parameters: two
    # interior equilibria, S1* < S2*.
    S_s = 200.0
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 1.70, 0.02

    def delta_scalar(x):
        return fn_delta_t(x, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)[0]

    s1_star = brentq(delta_scalar, 0.0, 50.0)
    s2_star = brentq(delta_scalar, 50.0, 150.0)

    assert np.isclose(delta_scalar(s1_star), 0.0, atol=1e-8)
    assert np.isclose(delta_scalar(s2_star), 0.0, atol=1e-8)
    assert s1_star < s2_star


def test_fn_delta_t_negative_s_t_is_clipped_to_zero():
    S_s = 200.0
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 0.70, 0.01

    at_negative = fn_delta_t(-5.0, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)
    at_zero = fn_delta_t(0.0, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)

    for neg_val, zero_val in zip(at_negative, at_zero, strict=True):
        assert np.isclose(neg_val, zero_val)


def test_fn_delta_t_scalar_and_array_calls_agree():
    S_s = 200.0
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 0.70, 0.01

    scalar_result = fn_delta_t(50.0, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)
    array_result = fn_delta_t(np.array([50.0]), lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s)

    for scalar_val, array_val in zip(scalar_result, array_result, strict=True):
        assert np.isclose(scalar_val, array_val[0])


def test_fn_delta_t_mu_t_equals_sum_of_long_and_short_components():
    S_s = 200.0
    S_t = np.arange(0, S_s + 1, 10)
    lambda_0, beta1 = 1.0, 0.5
    mu_s, beta2 = 1.0, 2.0
    beta3, beta4 = 0.70, 0.01

    delta_t, lambda_t, mu_long_t, mu_short_t, mu_t = fn_delta_t(
        S_t, lambda_0, beta1, mu_s, beta2, beta3, beta4, S_s
    )

    assert np.allclose(mu_t, mu_long_t + mu_short_t)
    assert np.allclose(delta_t, lambda_t - mu_t)
