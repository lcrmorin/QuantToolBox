"""Simulation for the copula families in `families.py`: closed-form
quantile inversion where the original derived one (AMH, Frank), a
generic root-finding conditional-inversion engine otherwise (Gumbel, and
any other bivariate family), direct Cholesky-based simulation for the
Gaussian/Student copulas, and an empirical-quantile transform for turning
uniform copula draws into draws matching an observed marginal sample
(NORTA-style simulation).

Ported from HSF toolbox `copula/{rndCopula2,rndCopulaAMH,rndCopulaFrank,
rndCopulaGumbel,rndCopulaNormal,rndCopulaNormal2,rndCopulaStudent,
rndCopulaStudent2,rndnCopula,rndCopulaEmpiricalQuantile}.m`.

Architecture:

- `rndCopula2.m` is already, in the original, a *generic* bivariate
  copula simulator: given any conditional-CDF function handle
  ``cndCopula2(u1, u2)`` (``= dC(u1,u2)/du1``), it draws ``u1, v2 ~
  U(0,1)`` and solves ``cndCopula2(u1, u2) = v2`` for `u2` by bisection.
  `rndCopulaGumbel.m` is exactly this generic engine, called with
  Gumbel's own conditional CDF plugged in, because Gumbel has no
  closed-form quantile inversion. That generic engine is
  `simulate_from_conditional_cdf` here, built on this package's own
  `quanttoolbox.optim.bisection.bisection` (already vectorized) rather
  than a second hand-rolled bisection loop -- and, like the original, it
  works for *any* bivariate family with a conditional-CDF function, not
  just Gumbel: pass e.g. ``families.amh_conditional_cdf`` to simulate
  from AMH this way instead of `simulate_amh`'s closed-form inversion.
- AMH and Frank *do* have closed-form conditional-CDF inversions
  (`rndCopulaAMH.m`'s quadratic-formula root, `rndCopulaFrank.m`'s direct
  log-formula), so `simulate_amh`/`simulate_frank` use those directly --
  faster and exact, no root-finding needed.
- `rndCopulaNormal.m` (n-dim) and `rndCopulaNormal2.m` (bivariate, a
  ``rho in (-1, 1)`` special case of the same Cholesky construction) are
  merged into one `simulate_gaussian_copula`; same for
  `rndCopulaStudent.m`/`rndCopulaStudent2.m` -> `simulate_student_copula`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm as normdist
from scipy.stats import t as tdist

from quanttoolbox.config import BisectionConfig
from quanttoolbox.optim.bisection import bisection


def simulate_from_conditional_cdf(
    conditional_cdf: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_samples: int,
    random_state: object = None,
    config: BisectionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `n_samples` draws `(u1, u2)` from a bivariate copula given
    only its conditional CDF ``conditional_cdf(u1, u2) = dC(u1, u2) /
    du1``: draws `u1, v2 ~ U(0,1)` and solves ``conditional_cdf(u1, u2) =
    v2`` for `u2` by bisection. Works for any family exposing a
    conditional CDF (`families.amh_conditional_cdf`,
    `families.gaussian_copula_conditional_cdf`, a hand-written one for a
    family not in `families.py` at all, ...), not just Gumbel -- see
    module docstring.

    Original: copula/rndCopula2.m (generic engine) and rndCopulaGumbel.m
    (Gumbel plugged into it)
    """
    if config is None:
        config = BisectionConfig()
    rng = np.random.default_rng(random_state)
    u1 = rng.random(n_samples)
    v2 = rng.random(n_samples)

    eps = np.finfo(float).eps
    a = np.full(n_samples, eps)
    b = np.full(n_samples, 1.0 - eps)
    u2 = bisection(lambda u: conditional_cdf(u1, u) - v2, a, b, config)
    return u1, np.asarray(u2)


def simulate_gumbel(
    theta: float, n_samples: int, random_state: object = None, config: BisectionConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from the Gumbel copula via `simulate_from_conditional_cdf`
    (no closed-form quantile inversion exists for Gumbel).

    Original: copula/rndCopulaGumbel.m
    """

    def conditional_cdf(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        u1t = -np.log(u1)
        u2t = -np.log(u2)
        w = u1t**theta + u2t**theta
        beta = 1.0 / theta
        return np.exp(-(w**beta)) * (1.0 + (u2t / u1t) ** theta) ** (beta - 1.0) / u1

    return simulate_from_conditional_cdf(conditional_cdf, n_samples, random_state, config)


def simulate_amh(
    theta: float, n_samples: int, random_state: object = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from the AMH copula via its closed-form conditional-CDF
    inversion (the smaller root of a quadratic in `u2`).

    Original: copula/rndCopulaAMH.m
    """
    rng = np.random.default_rng(random_state)
    u1 = rng.random(n_samples)
    v = rng.random(n_samples)

    a = v * theta**2 * (1.0 - u1) ** 2 - theta
    b = 2.0 * theta * v * (1.0 - theta + theta * u1) * (1.0 - u1) - (1.0 - theta)
    c = v * (1.0 - theta + theta * u1) ** 2
    delta = b**2 - 4.0 * a * c

    u2 = (-b - np.sqrt(delta)) / (2.0 * a)
    return u1, u2


def simulate_frank(
    theta: float, n_samples: int, random_state: object = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from the Frank copula via its closed-form conditional-CDF
    inversion.

    Original: copula/rndCopulaFrank.m
    """
    rng = np.random.default_rng(random_state)
    u1 = rng.random(n_samples)
    v2 = rng.random(n_samples)

    u2 = (
        -np.log(1.0 + v2 * (np.exp(-theta) - 1.0) / (v2 + (1.0 - v2) * np.exp(-theta * u1))) / theta
    )
    return u1, u2


def simulate_gaussian_copula(
    corr: np.ndarray, n_samples: int, random_state: object = None
) -> np.ndarray:
    """Simulate from the Gaussian copula with correlation matrix `corr`
    (``k_dim x k_dim``, ``k_dim >= 2``): Cholesky-correlate standard
    normal draws, then map through the standard normal CDF.

    Original: copula/rndCopulaNormal.m (n-dim) and rndCopulaNormal2.m
    (bivariate) -- merged, see module docstring.
    """
    corr = np.asarray(corr, dtype=float)
    rng = np.random.default_rng(random_state)
    z = rng.standard_normal((n_samples, corr.shape[0]))
    correlated = z @ np.linalg.cholesky(corr).T
    return normdist.cdf(correlated)


def simulate_student_copula(
    corr: np.ndarray, nu: float, n_samples: int, random_state: object = None
) -> np.ndarray:
    """Simulate from the Student-t copula with correlation matrix `corr`
    and `nu` degrees of freedom: Cholesky-correlate standard normal
    draws, scale by an independent chi-squared draw, then map through
    the Student-t CDF.

    Original: copula/rndCopulaStudent.m (n-dim) and rndCopulaStudent2.m
    (bivariate) -- merged, see module docstring.
    """
    corr = np.asarray(corr, dtype=float)
    rng = np.random.default_rng(random_state)
    n = rng.standard_normal((n_samples, corr.shape[0]))
    chi2 = rng.chisquare(nu, size=n_samples)

    correlated = n @ np.linalg.cholesky(corr).T
    scaled = correlated / np.sqrt(chi2 / nu)[:, None]
    return tdist.cdf(scaled, nu)


def empirical_quantile_transform(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Transform uniform copula draws `u` (shape ``(n_samples, n_dim)``)
    into draws from the empirical (linearly-interpolated) marginal
    distribution of an observed sample `x` -- either shape ``(n_ref,)``
    (the same reference sample reused for every column of `u`) or shape
    ``(n_ref, n_dim)`` (one reference sample per column). NORTA-style
    simulation: pairs a copula's dependence structure with an arbitrary
    observed marginal instead of a named distribution family.

    Original: copula/rndCopulaEmpiricalQuantile.m
    """
    u = np.asarray(u, dtype=float)
    n_dim = u.shape[1]
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = np.tile(x[:, None], (1, n_dim))

    n_ref = x.shape[0]
    y = np.zeros_like(u)
    for i in range(n_dim):
        z = np.sort(x[:, i])
        z_padded = np.concatenate(([z[0]], z, [z[-1]]))
        w = n_ref * u[:, i]
        wt = np.floor(w).astype(int)
        frac = w - wt
        wt = wt + 1  # 1-based index into z_padded, matching the original
        y[:, i] = z_padded[wt] + frac * (z_padded[wt + 1] - z_padded[wt])
    return y
