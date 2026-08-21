"""Dose-response curve models (toxicology/ecotoxicology): log-logistic,
log-normal, and Weibull sigmoidal curves, plus two "hormetic" variants that
add a low-dose stimulatory term to the log-logistic curve.

Ported from HSF toolbox `stats/{drcHormetic1,drcHormetic2,drcLogLogistic,
drcLogNormal,drcWeibull1,drcWeibull2}.m`.

Translation notes:

- All six curves are small, closed-form functions with no `scipy`
  equivalent (`scipy` has no dose-response-curve module) -- ported
  algorithm-for-algorithm. `drc_log_normal` reuses this package's own
  `quanttoolbox.stats.distributions.normal_cdf` in place of the original's
  `cdfn`.
- `alpha` is the curve's inflection-point dose (ED50-style location
  parameter), `beta` its slope, `y_min`/`y_max` the lower/upper response
  asymptotes; `gamma_`/`delta` (hormetic variants only) control the extra
  low-dose stimulatory term. Parameter names and order match the
  originals exactly.
"""

from __future__ import annotations

import numpy as np

from quanttoolbox.stats.distributions import normal_cdf


def drc_log_logistic(
    x: np.ndarray, alpha: float, beta: float, y_min: float, y_max: float
) -> np.ndarray:
    """Log-logistic dose-response curve.

    Original: stats/drcLogLogistic.m
    """
    x = np.asarray(x, dtype=float)
    y = 1.0 + np.exp(-beta * (np.log(x) - np.log(alpha)))
    return y_min + (y_max - y_min) / y


def drc_log_normal(
    x: np.ndarray, alpha: float, beta: float, y_min: float, y_max: float
) -> np.ndarray:
    """Log-normal dose-response curve.

    Original: stats/drcLogNormal.m
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(normal_cdf(beta * (np.log(x) - np.log(alpha))))
    return y_min + (y_max - y_min) * y


def drc_weibull1(
    x: np.ndarray, alpha: float, beta: float, y_min: float, y_max: float
) -> np.ndarray:
    """Weibull (type I) dose-response curve.

    Original: stats/drcWeibull1.m
    """
    x = np.asarray(x, dtype=float)
    y = np.exp(-np.exp(beta * (np.log(x) - np.log(alpha))))
    return y_min + (y_max - y_min) * y


def drc_weibull2(
    x: np.ndarray, alpha: float, beta: float, y_min: float, y_max: float
) -> np.ndarray:
    """Weibull (type II) dose-response curve.

    Original: stats/drcWeibull2.m
    """
    x = np.asarray(x, dtype=float)
    y = 1.0 - np.exp(-np.exp(beta * (np.log(x) - np.log(alpha))))
    return y_min + (y_max - y_min) * y


def drc_hormetic1(
    x: np.ndarray, alpha: float, beta: float, y_min: float, y_max: float, gamma_: float
) -> np.ndarray:
    """Hormetic (type I) dose-response curve: adds a linear-in-dose
    stimulatory term ``gamma_ * x`` to the log-logistic curve's numerator,
    producing a low-dose "hump" before the usual sigmoidal decline.

    Original: stats/drcHormetic1.m
    """
    x = np.asarray(x, dtype=float)
    y = 1.0 + np.exp(-beta * (np.log(x) - np.log(alpha)))
    return y_min + (y_max - y_min + gamma_ * x) / y


def drc_hormetic2(
    x: np.ndarray,
    alpha: float,
    beta: float,
    y_min: float,
    y_max: float,
    gamma_: float,
    delta: float,
) -> np.ndarray:
    """Hormetic (type II) dose-response curve: like `drc_hormetic1`, but
    the stimulatory term decays as ``exp(-1 / x**delta)`` instead of
    growing linearly in `x`.

    Original: stats/drcHormetic2.m
    """
    x = np.asarray(x, dtype=float)
    y = 1.0 + np.exp(-beta * (np.log(x) - np.log(alpha)))
    return y_min + (y_max - y_min + gamma_ * np.exp(-1.0 / x**delta)) / y
