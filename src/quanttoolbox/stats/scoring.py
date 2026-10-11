"""Rank, z-score, and Beta scoring transforms with weighted aggregation.

Adapted from the local scoring functions in
``HSF/2. ESG Scoring/chap2_scoring9.m`` and ``chap2_scoring10.m``, via
HSF-Notebooks chapter 02b. The transforms retain the notebook formulas
and share a consistent interface. ``bz_score`` generalizes the original
Beta(2, 2) and Beta(2.5, 1.5) transforms to caller-supplied shape parameters.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from quanttoolbox.stats.distributions import normal_cdf, normal_ppf


def qscore(x: np.ndarray, scaling: float = 1.0) -> np.ndarray:
    """Rescaled-rank score: rank(x) / (n + 1), times `scaling`.

    Original: `qscore_function` in chap2_scoring9.m / chap2_scoring10.m;
    HSF-Notebooks chapter 02b `qscore`.
    """
    x = np.asarray(x, dtype=float)
    q = scipy_stats.rankdata(x) / (len(x) + 1)
    return scaling * q


def zscore(x: np.ndarray) -> np.ndarray:
    """Standard z-score: (x - mean(x)) / std(x).

    Original: `zscore_function` in chap2_scoring9.m / chap2_scoring10.m;
    HSF-Notebooks chapter 02b `zscore` (the same formula is also reused,
    under the name `standardize`, in chapter 15b).
    """
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def aggregate(s1: np.ndarray, s2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """Weighted aggregation of two already-computed scores: w1*s1 + w2*s2.

    Original: `aggregate_function` in chap2_scoring9.m; HSF-Notebooks
    chapter 02b `aggregate`.
    """
    return w1 * np.asarray(s1) + w2 * np.asarray(s2)


def qz_score(x: np.ndarray, scaling: float = 1.0) -> np.ndarray | float:
    """z-score mapped through the normal CDF, times `scaling`.

    Original: `qz_function` in chap2_scoring9.m; HSF-Notebooks chapter 02b
    `qz_score`.
    """
    return scaling * normal_cdf(zscore(x))


def zq_score(x: np.ndarray) -> np.ndarray | float:
    """q-score (scaling=1) mapped through the normal quantile function.

    Original: `zq_function` in chap2_scoring9.m; HSF-Notebooks chapter 02b
    `zq_score`.
    """
    return normal_ppf(qscore(x, 1.0))


def bz_score(x: np.ndarray, a: float = 2.0, b: float = 2.0) -> np.ndarray:
    """z-score -> normal CDF -> Beta(a, b) quantile function.

    Original: `bz_function` (a=2, b=2) and `bz_skew_function` (a=2.5,
    b=1.5) in chap2_scoring9.m, generalized here into a single function
    taking the Beta shape parameters directly; HSF-Notebooks chapter 02b
    `bz_score`.
    """
    u = normal_cdf(zscore(x))
    return scipy_stats.beta.ppf(u, a, b)
