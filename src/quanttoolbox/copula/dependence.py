"""Kendall's tau and Spearman's rho for the copula families in
`families.py`, the two special functions (Debye, dilogarithm) their
closed forms need, and the empirical dependogram (pseudo-observations
via marginal ranks).

Ported from HSF toolbox `copula/{KendallCopula*,SpearmanCopula*,
DebyeFunction,diLogFunction,dependogram}.m`.

Architecture -- avoiding the original's per-family boilerplate:

- `SpearmanCopula.m` is already, in the original, a *generic* estimator:
  ``rho = 12 * integral2(C, 0, 1, 0, 1) - 3`` for *any* copula CDF `C`.
  `SpearmanCopulaClayton.m`/`SpearmanCopulaGumbel.m` are just this generic
  function called with `cdfCopulaClayton`/`cdfCopulaGumbel` plugged in
  (looping over a `theta` array) -- because Clayton and Gumbel have no
  closed-form Spearman's rho. That generic estimator is `spearman_rho_
  numeric` here (`scipy.integrate.dblquad` in place of `integral2`);
  `clayton_rho`/`gumbel_rho` are one-line calls to it, and it works
  unchanged for *any* other family in `families.py` that lacks a closed
  form (AMH, Plackett, ... -- pass e.g. ``lambda u1, u2: families.
  plackett_cdf(u1, u2, theta)``), not just the two the original wired it
  up for.
- Frank and Gaussian *do* have closed forms (`SpearmanCopulaFrank.m`,
  `SpearmanCopulaNormal.m`), so `frank_rho`/`gaussian_rho` use those
  directly instead of the (much slower) numeric integral.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import dblquad, quad
from scipy.stats import rankdata


def debye_function(x: np.ndarray | float, k: int) -> np.ndarray:
    """The Debye function ``D_k(x) = (k / x^k) * integral_0^x t^k / (e^t
    - 1) dt``, extended to `x < 0` via the reflection identity ``D_k(-x)
    = D_k(x) + k*x / (k+1)``. Used by `frank_tau`/`frank_rho`.

    Original: copula/DebyeFunction.m
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x_arr)
    for i, xx in enumerate(x_arr.ravel()):
        is_nonneg = xx >= 0.0
        xa = abs(xx)
        if xa == 0.0:
            out.ravel()[i] = 1.0 / (k + 1.0)
            continue
        integral, _ = quad(lambda t: t**k / np.expm1(t), 0.0, xa)
        value = (k / xa**k) * integral
        if not is_nonneg:
            value = value + k * xa / (1.0 + k)
        out.ravel()[i] = value
    return out if np.asarray(x).ndim > 0 else out[0]


def dilog_function(x: np.ndarray | float) -> np.ndarray:
    """``integral_x^1 log(t) / (1 - t) dt`` (related to the dilogarithm
    ``Li2``). Not used by any closed form in this module (the original
    likewise never calls it from `KendallCopula*.m`/`SpearmanCopula*.m`)
    -- ported as a standalone special function for completeness.

    Original: copula/diLogFunction.m
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x_arr)
    for i, xx in enumerate(x_arr.ravel()):
        integral, _ = quad(lambda t: np.log(t) / (1.0 - t), xx, 1.0)
        out.ravel()[i] = integral
    return out if np.asarray(x).ndim > 0 else out[0]


def clayton_tau(theta: np.ndarray | float) -> np.ndarray:
    """Kendall's tau of the Clayton copula: ``theta / (theta + 2)``
    (verified to match `statsmodels`).

    Original: copula/KendallCopulaClayton.m
    """
    theta = np.asarray(theta, dtype=float)
    return theta / (theta + 2.0)


def frank_tau(theta: np.ndarray | float) -> np.ndarray:
    """Kendall's tau of the Frank copula: ``1 - 4*(1 - D_1(theta)) /
    theta`` (verified to match `statsmodels`).

    Original: copula/KendallCopulaFrank.m
    """
    theta = np.asarray(theta, dtype=float)
    d1 = debye_function(theta, 1)
    return 1.0 - 4.0 * (1.0 - d1) / theta


def gumbel_tau(theta: np.ndarray | float) -> np.ndarray:
    """Kendall's tau of the Gumbel copula: ``1 - 1/theta`` (verified to
    match `statsmodels`).

    Original: copula/KendallCopulaGumbel.m
    """
    theta = np.asarray(theta, dtype=float)
    return 1.0 - 1.0 / theta


def gaussian_tau(rho: np.ndarray | float) -> np.ndarray:
    """Kendall's tau of the Gaussian copula: ``2 * asin(rho) / pi``
    (verified to match `statsmodels`).

    Original: copula/KendallCopulaNormal.m
    """
    rho = np.asarray(rho, dtype=float)
    return 2.0 * np.arcsin(rho) / np.pi


def spearman_rho_numeric(cdf_fn: Callable[[float, float], float]) -> float:
    """Spearman's rho of an arbitrary bivariate copula `cdf_fn(u1, u2)`,
    via the generic identity ``rho = 12 * integral_0^1 integral_0^1
    C(u1, u2) du1 du2 - 3``. Works for any copula CDF, not just the ones
    with a Python wrapper below -- see module docstring.

    Original: copula/SpearmanCopula.m
    """
    integral, _ = dblquad(lambda u2, u1: cdf_fn(u1, u2), 0.0, 1.0, 0.0, 1.0)
    return 12.0 * integral - 3.0


def clayton_rho(theta: float) -> float:
    """Spearman's rho of the Clayton copula (no closed form -- numeric
    integration via `spearman_rho_numeric`).

    Original: copula/SpearmanCopulaClayton.m
    """
    from quanttoolbox.copula.families import clayton_cdf

    return spearman_rho_numeric(lambda u1, u2: float(clayton_cdf(u1, u2, theta)))


def gumbel_rho(theta: float) -> float:
    """Spearman's rho of the Gumbel copula (no closed form -- numeric
    integration via `spearman_rho_numeric`).

    Original: copula/SpearmanCopulaGumbel.m
    """
    from quanttoolbox.copula.families import gumbel_cdf

    return spearman_rho_numeric(lambda u1, u2: float(gumbel_cdf(u1, u2, theta)))


def frank_rho(theta: np.ndarray | float) -> np.ndarray:
    """Spearman's rho of the Frank copula: ``1 - 12*(D_1(theta) -
    D_2(theta)) / theta``.

    Original: copula/SpearmanCopulaFrank.m
    """
    theta = np.asarray(theta, dtype=float)
    d1 = debye_function(theta, 1)
    d2 = debye_function(theta, 2)
    return 1.0 - 12.0 * (d1 - d2) / theta


def gaussian_rho(rho: np.ndarray | float) -> np.ndarray:
    """Spearman's rho of the Gaussian copula: ``6 * asin(rho/2) / pi``.

    Original: copula/SpearmanCopulaNormal.m
    """
    rho = np.asarray(rho, dtype=float)
    return 6.0 * np.arcsin(rho / 2.0) / np.pi


def dependogram(data: np.ndarray) -> np.ndarray:
    """Pseudo-observations of the empirical copula: each column of
    `data` (shape ``(n_obs, n_dim)``) replaced by its marginal ranks
    (ties averaged), scaled to ``(0, 1)`` by ``rank / (n_obs + 1)``.

    Original: copula/dependogram.m
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    ranks = np.apply_along_axis(rankdata, 0, data)
    return ranks / (n + 1.0)
