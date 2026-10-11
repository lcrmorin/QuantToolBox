"""Single-factor Gaussian credit thresholds and Vasicek default-rate measures.

Adapted from HSF-Notebooks chapters 13g, 13i, and 13j using the Vasicek
(2002) large-homogeneous-portfolio formulas, rather than a MATLAB port.
``invcdf_default_rate`` and ``vasicek_density`` provide the quantile and
density of the limiting portfolio default-rate distribution.

``thresholds_from_matrix`` uses finite outer thresholds of +/-10 by
default. Smaller ``inf_limit`` values truncate more probability mass, so
threshold-implied probabilities can differ from the normalized input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class SingleFactorThresholdsResult:
    """Gaussian thresholds for each origin/destination rating pair.

    ``z1`` and ``z2`` bound each destination bucket; ``p_ij`` contains the
    row-normalized transition probabilities. ``Phi(z2) - Phi(z1)`` recovers
    those probabilities up to finite-tail truncation and rounding error.
    """

    z1: np.ndarray
    z2: np.ndarray
    p_ij: np.ndarray


def thresholds_from_matrix(p: np.ndarray, inf_limit: float = 10.0) -> SingleFactorThresholdsResult:
    """Convert a row-stochastic rating-transition matrix `p` into
    single-factor Gaussian-copula conditioning thresholds bracketing each
    destination rating.

    Cumulative-probability thresholds ``z_ij = Phi^-1(cumsum(p_ij))``,
    with the 0%/100% (and any floating-point-overshoot NaN) cells capped
    at +/- `inf_limit` rather than +/- infinity.
    """
    p = np.asarray(p, dtype=float)
    k = p.shape[0]
    p_ij = p / p.sum(axis=1, keepdims=True)
    p_cum = np.cumsum(p_ij, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_ij = norm.ppf(p_cum)
    z_ij = np.where(np.isnan(z_ij), inf_limit, z_ij)
    z_ij = np.where(z_ij == -np.inf, -inf_limit, z_ij)
    z_ij = np.where(z_ij == np.inf, inf_limit, z_ij)

    z1 = np.hstack([-inf_limit * np.ones((k, 1)), z_ij[:, :-1]])
    z2 = np.hstack([z_ij[:, :-1], inf_limit * np.ones((k, 1))])
    return SingleFactorThresholdsResult(z1=z1, z2=z2, p_ij=p_ij)


def invcdf_default_rate(
    alpha: np.ndarray | float, default_prob: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray | float:
    """Vasicek single-factor / large-homogeneous-portfolio default-rate
    quantile at confidence level `alpha` (the Basel IRB "worst-case
    default rate"), given an obligor-level unconditional default
    probability `default_prob` and asset correlation `rho`.
    """
    return norm.cdf((norm.ppf(default_prob) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))


def vasicek_density(
    d: np.ndarray | float, default_prob: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray | float:
    """Closed-form limiting density of the single-factor Vasicek
    portfolio default rate `d`, given obligor-level unconditional default
    probability `default_prob` and asset correlation `rho`.
    """
    x = norm.ppf(d)
    return np.sqrt((1 - rho) / rho) * np.exp(
        0.5 * x**2 - (np.sqrt(1 - rho) * x - norm.ppf(default_prob)) ** 2 / (2 * rho)
    )
