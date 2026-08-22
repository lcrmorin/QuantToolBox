"""Vasicek (2002) single-factor Gaussian-copula credit-portfolio model:
discretizing a rating-transition matrix into single-factor conditioning
thresholds, and the resulting large-homogeneous-portfolio default-rate
quantile and density.

Not ported from the MATLAB HSF toolbox -- no `.m` file in `hfs-archive`
implements this family of functions. These are the standard closed-form
Vasicek (2002) results (`invcdf_default_rate`, `vasicek_density`) and a
direct single-factor-copula discretization of a transition matrix
(`thresholds_from_matrix`), matching the Python formulas already used in
this book's own notebooks (chapters 13g/13i/13j of the HSF-Notebooks
series) rather than a translation of an existing source file.

Translation notes:

- `thresholds_from_matrix` follows the calling notebooks' own +/-10
  sentinel convention for the open ends of the outermost rating buckets
  (rather than +/- infinity): `Phi(10)`/`Phi(-10)` are already within
  ~1e-23 of 1/0, so this is a deliberate finite-sentinel choice (matching
  the notebooks' own round-trip check `Phi(z2) - Phi(z1) == p_ij` to
  float precision), not an approximation any caller would notice at a
  smaller `inf_limit`.
- `invcdf_default_rate` is the closed-form single-factor /
  large-homogeneous-portfolio (asymptotic) default-rate quantile at
  confidence `alpha` -- the "worst-case default rate" (WCDR) formula
  behind the Basel II/III IRB capital requirement, and the quantile
  function of the distribution `vasicek_density` is the density of.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class SingleFactorThresholdsResult:
    """Single-factor Gaussian-copula conditioning thresholds for each
    destination rating, discretized from a row-stochastic transition
    matrix: `z1`/`z2` bracket the systematic-factor range landing in each
    destination rating, so that `Phi(z2) - Phi(z1) == p_ij` (the
    row-normalized transition probability actually used) up to float
    precision."""

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
