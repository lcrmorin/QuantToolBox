"""Tests for quanttoolbox.stats.scoring.

Cross-checked both against hand-rolled formulas and against the real data
used in HSF-Notebooks chapter 02b's own worked example (cell 19 of
02b_scoring_methodology.ipynb), which is the same x1/x2 pair as the
matching MATLAB example in hfs-archive's chap2_scoring9.m.
"""

import numpy as np
from scipy import stats as scipy_stats

from quanttoolbox.stats.scoring import (
    aggregate,
    bz_score,
    qscore,
    qz_score,
    zq_score,
    zscore,
)

# Same x1/x2 arrays as HSF-Notebooks chapter 02b cell 19 and MATLAB
# chap2_scoring9.m's worked example.
X1 = np.array([94.0, 38.6, 30.6, 74.4, 97.1, 57.1, 132.4, 92.5, 64.9])
X2 = np.array([-3.0, -5.5, 5.6, -1.3, -16.8, -3.5, 8.5, -9.1, -4.6])


def test_qscore_matches_hand_rank_formula():
    scaling = 100.0
    expected = scipy_stats.rankdata(X1) / (len(X1) + 1) * scaling
    assert np.allclose(qscore(X1, scaling), expected)


def test_qscore_default_scaling_is_one():
    assert np.allclose(qscore(X1), scipy_stats.rankdata(X1) / (len(X1) + 1))


def test_qscore_mean_is_one_half_regardless_of_data():
    # Ranks are a permutation of 1..n, so mean(rank)/(n+1) == 0.5 exactly,
    # for any input (a structural invariant of the rank formula).
    for x in (X1, X2, np.array([5.0, 1.0, 1.0, 3.0])):
        assert np.isclose(qscore(x).mean(), 0.5)


def test_zscore_matches_hand_standardization():
    expected = (X1 - X1.mean()) / X1.std()
    assert np.allclose(zscore(X1), expected)


def test_zscore_has_zero_mean_and_unit_std():
    z = zscore(X2)
    assert np.isclose(z.mean(), 0.0, atol=1e-12)
    assert np.isclose(z.std(), 1.0)


def test_zscore_matches_chapter15b_standardize_formula():
    # Chapter 15b's `standardize(a) = (a - a.mean()) / a.std(ddof=0)` for a
    # 1-D array is the exact same formula as zscore -- numpy's `.std()`
    # already defaults to ddof=0, confirming the two chapters share one
    # z-score formula under different names.
    def standardize_ch15b(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return (a - a.mean()) / a.std(ddof=0)

    for x in (X1, X2):
        assert np.allclose(zscore(x), standardize_ch15b(x))


def test_aggregate_is_weighted_sum():
    s1 = qscore(X1, 100.0)
    s2 = qscore(X2, 100.0)
    w1, w2 = 0.7, 0.3
    assert np.allclose(aggregate(s1, s2, w1, w2), w1 * s1 + w2 * s2)


def test_qz_score_matches_hand_formula():
    scaling = 100.0
    z = (X1 - X1.mean()) / X1.std()
    expected = scaling * scipy_stats.norm.cdf(z)
    assert np.allclose(qz_score(X1, scaling), expected)


def test_qz_score_bounded_by_scaling():
    scaling = 100.0
    s = qz_score(X2, scaling)
    assert np.all(s >= 0.0)
    assert np.all(s <= scaling)


def test_zq_score_matches_hand_formula():
    q = scipy_stats.rankdata(X1) / (len(X1) + 1)
    expected = scipy_stats.norm.ppf(q)
    assert np.allclose(zq_score(X1), expected)


def test_bz_score_default_matches_hand_formula_beta_2_2():
    z = (X1 - X1.mean()) / X1.std()
    u = scipy_stats.norm.cdf(z)
    expected = scipy_stats.beta.ppf(u, 2.0, 2.0)
    assert np.allclose(bz_score(X1), expected)
    assert np.allclose(bz_score(X1, 2.0, 2.0), expected)


def test_bz_score_skew_variant_matches_hand_formula_beta_2p5_1p5():
    # This is the MATLAB `bz_skew_function` case (a=2.5, b=1.5), which
    # `bz_score`'s generic a/b subsumes.
    z = (X2 - X2.mean()) / X2.std()
    u = scipy_stats.norm.cdf(z)
    expected = scipy_stats.beta.ppf(u, 2.5, 1.5)
    assert np.allclose(bz_score(X2, 2.5, 1.5), expected)


def test_bz_score_bounded_in_unit_interval():
    for x, a, b in [(X1, 2.0, 2.0), (X2, 2.5, 1.5)]:
        s = bz_score(x, a, b)
        assert np.all(s >= 0.0)
        assert np.all(s <= 1.0)


def test_aggregated_and_rescored_matches_notebook_cell19_workflow():
    # Reproduces the shape of HSF-Notebooks chapter 02b cell 19 / MATLAB
    # chap2_scoring9.m's worked example: score x1 and x2 several ways,
    # aggregate 0.7/0.3, then re-score the aggregate in its own space.
    scaling = 100.0
    w1, w2 = 0.7, 0.3

    q1, q2 = qscore(X1, scaling), qscore(X2, scaling)
    q12 = aggregate(q1, q2, w1, w2)
    q_final = qscore(q12, scaling)
    assert np.allclose(q_final, scipy_stats.rankdata(q12) / (len(q12) + 1) * scaling)

    z1, z2 = zscore(X1), zscore(X2)
    z12 = aggregate(z1, z2, w1, w2)
    z_final = zscore(z12)
    assert np.isclose(z_final.mean(), 0.0, atol=1e-12)
    assert np.isclose(z_final.std(), 1.0)

    bz1, bz2 = bz_score(X1), bz_score(X2)
    bz12 = aggregate(bz1, bz2, w1, w2)
    bz_final = bz_score(bz12)
    assert np.all((bz_final >= 0.0) & (bz_final <= 1.0))
