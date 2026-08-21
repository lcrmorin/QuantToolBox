"""Unit-root testing (Augmented Dickey-Fuller).

Ported from QuantToolBox/ects/adf_test.m

Translation notes:

- The original hand-rolls the ADF regression and hardcodes ~15x50 tables
  of MacKinnon critical values (interpolated for sample size and
  significance level) across three model specifications (no constant, a
  constant, and a constant + trend) -- several hundred numeric literals.
  ``statsmodels.tsa.stattools.adfuller`` implements the same test using
  the same underlying MacKinnon (1994/2010) critical-value approximation,
  is the standard, actively-maintained Python implementation, and is
  already a project dependency -- so this wraps it directly instead of
  transcribing the original's tables.
- The original scans lag orders 0..pLags and all three specifications
  ("n"/"c"/"ct") in one call, returning a (3, pLags+1) result grid;
  ``adf_test`` here reproduces that scan structure, calling
  ``adfuller(..., autolag=None, maxlag=p)`` for each (specification, lag)
  combination to match the original's fixed-lag (not automatically
  selected) behavior.
- ``wald_test`` (originally also in ects/) lives in
  ``econometrics.estimation`` instead, alongside the estimators it's most
  often used with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller

_SPECIFICATIONS = ("n", "c", "ct")  # no constant, constant, constant+trend


@dataclass
class ADFTestResult:
    specification: np.ndarray  # ("n","c","ct") labels, length 3
    lags: np.ndarray  # lag values tested, length n_lags
    tau: np.ndarray  # (3, n_lags) ADF test statistics
    p_value: np.ndarray  # (3, n_lags) MacKinnon p-values
    critical_values: np.ndarray  # (3, n_lags, 3) critical values at 1%/5%/10%


def adf_test(y: np.ndarray, max_lags: int) -> ADFTestResult:
    """Augmented Dickey-Fuller unit-root test, scanned over lag orders
    0..max_lags and all three specifications (no constant, constant,
    constant + trend).

    Original: ects/adf_test.m

    Uses statsmodels.tsa.stattools.adfuller -- see module docstring.
    """
    y = np.asarray(y, dtype=float).flatten()
    y = y[~np.isnan(y)]

    lags = np.arange(0, max_lags + 1)
    n_specs = len(_SPECIFICATIONS)
    n_lags = lags.shape[0]

    tau = np.full((n_specs, n_lags), np.nan)
    p_value = np.full((n_specs, n_lags), np.nan)
    critical_values = np.full((n_specs, n_lags, 3), np.nan)

    for i, spec in enumerate(_SPECIFICATIONS):
        for j, lag in enumerate(lags):
            try:
                stat, pval, _, _, crit = adfuller(
                    y, maxlag=int(lag), regression=spec, autolag=None, store=False, regresults=False
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            tau[i, j] = stat
            p_value[i, j] = pval
            critical_values[i, j] = [crit["1%"], crit["5%"], crit["10%"]]

    return ADFTestResult(
        specification=np.array(_SPECIFICATIONS),
        lags=lags,
        tau=tau,
        p_value=p_value,
        critical_values=critical_values,
    )
