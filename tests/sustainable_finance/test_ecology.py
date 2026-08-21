"""Tests for quanttoolbox.sustainable_finance.ecology."""

import numpy as np

from quanttoolbox.sustainable_finance.ecology import (
    endemics_area_relationship,
    hurlbert,
    species_abundance_distribution,
    species_area_relationship,
)


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


def test_species_abundance_distribution_octave_mode_rejects_other_strings():
    import pytest

    with pytest.raises(ValueError, match="breaks must be"):
        species_abundance_distribution(np.array([1.0, 2.0]), breaks="linear")


def test_species_abundance_distribution_custom_breaks_covers_all_individuals():
    n_i = np.array([1.0, 4.0, 6.0, 9.0, 12.0])
    breaks = np.array([2.0, 5.0, 10.0, 15.0])

    j, s, out_breaks = species_abundance_distribution(n_i, breaks=breaks)

    assert np.array_equal(out_breaks, breaks)
    assert np.sum(s) == n_i.shape[0]
    assert j.shape == (4,)


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
