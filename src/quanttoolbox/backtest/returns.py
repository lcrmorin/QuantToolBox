"""Price/return series conversion: simple returns, cumulative price indices,
funded/unfunded conversions, and capitalized LIBOR indices.

Ported from QuantToolBox/backtest/{price2return,return2price,
price2unfunded,unfunded2price,capitalized_libor,capitalized_libor_plus}.m

Translation notes:

- ``price2return2.m`` in the original is dead/incomplete code (it
  references undefined variables ``Dates``/``day_of_week`` and shares its
  function name with ``price2return.m``, so it could never have been
  called under normal MATLAB name resolution) -- it is NOT ported here.
- ``lagn``/``lag1`` (n-period lag) are replaced by array slicing
  (``arr[:-n]``) or ``pandas.Series.shift(n)`` where a NaN-padded result is
  wanted.
- ``findnomiss``/``fillmiss(..., 4)`` (MATLAB: find first/last non-missing
  row per column, forward/back-fill gaps) map to pandas' ``.ffill()`` /
  ``.bfill()`` plus ``first_valid_index()`` / ``last_valid_index()``.
- All functions accept and return 2-D arrays (rows = dates, columns =
  assets) to match the original's column-oriented convention, even for a
  single series -- pass a (n, 1) array or reshape as needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_to_return(x: np.ndarray, n_lags: int = 1) -> np.ndarray:
    """Simple (arithmetic) returns over n_lags periods: x[t]/x[t-n] - 1.

    Original: backtest/price2return.m
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.full_like(x, np.nan)
    y[n_lags:] = x[n_lags:] / x[:-n_lags] - 1.0
    return y


def return_to_price(x: np.ndarray, keep_interior_nan: bool = False) -> np.ndarray:
    """Convert a return series back into a cumulative price index starting
    at 100, handling leading/trailing NaN gaps per column.

    Original: backtest/return2price.m

    Parameters
    ----------
    keep_interior_nan : if True, interior single-period gaps in the input
        (immediately after the first valid observation) are preserved as
        NaN in the output rather than treated as a zero return.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    r, c = x.shape

    x_filled = np.where(np.isnan(x), 0.0, x)
    y = 100 * np.cumprod(1 + x_filled, axis=0)

    for j in range(c):
        col = x[:, j]
        valid = np.where(~np.isnan(col))[0]
        if valid.size == 0:
            y[:, j] = np.nan
            continue
        fnm, lnm = valid[0], valid[-1]
        if fnm >= 2:
            y[: fnm - 1, j] = np.nan
        if lnm < r - 1:
            y[lnm + 1 :, j] = np.nan

    if keep_interior_nan:
        for j in range(c):
            col = x[:, j]
            valid = np.where(~np.isnan(col))[0]
            if valid.size == 0:
                continue
            fnm = valid[0]
            missing = np.isnan(col)
            if fnm >= 1:
                missing[fnm - 1] = False
            y[missing, j] = np.nan

    # renormalize each column to start at 100 from its first valid value
    for j in range(c):
        valid = np.where(~np.isnan(y[:, j]))[0]
        if valid.size > 0:
            y[:, j] = 100 * y[:, j] / y[valid[0], j]

    return y


def _fill_missing_ffill_bfill(x: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaN gaps (MATLAB fillmiss(x, 4))."""
    df = pd.DataFrame(x)
    return df.ffill().bfill().to_numpy()


def _first_last_valid(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column index of first/last non-NaN row (MATLAB findnomiss)."""
    r, c = x.shape
    fnm = np.zeros(c, dtype=int)
    lnm = np.zeros(c, dtype=int)
    for j in range(c):
        valid = np.where(~np.isnan(x[:, j]))[0]
        fnm[j] = valid[0] if valid.size else -1
        lnm[j] = valid[-1] if valid.size else -1
    return fnm, lnm


def price_to_unfunded(
    funded_prices: np.ndarray, libor_index: np.ndarray, method: int = 1
) -> np.ndarray:
    """Convert funded (total-return) prices to unfunded (excess-return)
    prices by subtracting the LIBOR return each period.

    method=1 (default): mask output using the funded series' own
    first/last valid range. method=2: mask output using the input's exact
    NaN pattern instead.

    Original: backtest/price2unfunded.m
    """
    funded_prices = np.asarray(funded_prices, dtype=float)
    if funded_prices.ndim == 1:
        funded_prices = funded_prices[:, None]
    libor_index = np.asarray(libor_index, dtype=float)
    if libor_index.ndim == 1:
        libor_index = libor_index[:, None]

    r, c = funded_prices.shape
    if method == 2:
        cnd = np.isnan(funded_prices)
    else:
        fnm, lnm = _first_last_valid(funded_prices)

    funded_filled = _fill_missing_ffill_bfill(funded_prices)
    libor_filled = _fill_missing_ffill_bfill(libor_index)

    r_funded = price_to_return(funded_filled, 1)
    r_libor = price_to_return(libor_filled, 1)
    r_unfunded = np.where(np.isnan(r_funded - r_libor), 0.0, r_funded - r_libor)
    unfunded_prices = 100 * np.cumprod(1 + r_unfunded, axis=0)

    if method == 2:
        unfunded_prices[cnd] = np.nan
    else:
        for j in range(c):
            if fnm[j] > 0:
                unfunded_prices[: fnm[j], j] = np.nan
            if lnm[j] < r - 1:
                unfunded_prices[lnm[j] + 1 :, j] = np.nan

    return unfunded_prices


def unfunded_to_price(
    unfunded_prices: np.ndarray, libor_index: np.ndarray, method: int = 1
) -> np.ndarray:
    """Convert unfunded (excess-return) prices to funded (total-return)
    prices by adding back the LIBOR return each period.

    Original: backtest/unfunded2price.m
    """
    unfunded_prices = np.asarray(unfunded_prices, dtype=float)
    if unfunded_prices.ndim == 1:
        unfunded_prices = unfunded_prices[:, None]
    libor_index = np.asarray(libor_index, dtype=float)
    if libor_index.ndim == 1:
        libor_index = libor_index[:, None]

    r, c = unfunded_prices.shape
    if method == 2:
        cnd = np.isnan(unfunded_prices)
    else:
        fnm, lnm = _first_last_valid(unfunded_prices)

    unfunded_filled = _fill_missing_ffill_bfill(unfunded_prices)
    libor_filled = _fill_missing_ffill_bfill(libor_index)

    r_unfunded = price_to_return(unfunded_filled, 1)
    r_libor = price_to_return(libor_filled, 1)
    r_funded = np.where(np.isnan(r_unfunded + r_libor), 0.0, r_unfunded + r_libor)
    funded_prices = 100 * np.cumprod(1 + r_funded, axis=0)

    if method == 2:
        funded_prices[cnd] = np.nan
    else:
        for j in range(c):
            if fnm[j] > 0:
                funded_prices[: fnm[j], j] = np.nan
            if lnm[j] < r - 1:
                funded_prices[lnm[j] + 1 :, j] = np.nan

    return funded_prices


def capitalized_libor(
    dates: pd.DatetimeIndex, libor_rates: np.ndarray, method: int = 1
) -> np.ndarray:
    """Build a capitalized (compounding) index from a LIBOR-style annualized
    rate series, using actual/365.25 day-count compounding.

    Original: backtest/capitalized_libor.m
    """
    dates = pd.DatetimeIndex(dates)
    libor_rates = np.asarray(libor_rates, dtype=float)
    if libor_rates.ndim == 1:
        libor_rates = libor_rates[:, None]

    n_dates, n_cols = libor_rates.shape
    day_nums = (dates - dates[0]).days.to_numpy()

    if method == 2:
        indx_missing = np.isnan(libor_rates)
    else:
        fnm, lnm = _first_last_valid(libor_rates)

    rates_filled = _fill_missing_ffill_bfill(libor_rates)
    rates_filled = np.where(np.isnan(rates_filled), 0.0, rates_filled)

    libor_index = np.zeros((n_dates, n_cols))
    libor_index[0, :] = 100.0
    for t in range(1, n_dates):
        dt = (day_nums[t] - day_nums[t - 1]) / 365.25
        libor_index[t, :] = libor_index[t - 1, :] * (1 + rates_filled[t - 1, :] * dt)

    if method == 2:
        libor_index[indx_missing] = np.nan
    else:
        for j in range(n_cols):
            if fnm[j] > 0:
                libor_index[: fnm[j], j] = np.nan
            if lnm[j] < n_dates - 1:
                libor_index[lnm[j] + 1 :, j] = np.nan

    return libor_index


def capitalized_libor_plus(
    dates: pd.DatetimeIndex, libor_index: np.ndarray, plus: np.ndarray
) -> np.ndarray:
    """Build a capitalized "LIBOR + spread" index from an existing LIBOR
    index and a per-series spread (in annualized terms).

    Original: backtest/capitalized_libor_plus.m
    """
    dates = pd.DatetimeIndex(dates)
    day_nums = (dates - dates[0]).days.to_numpy()
    n_dates = dates.shape[0]

    libor_index = np.asarray(libor_index, dtype=float).flatten()
    plus = np.atleast_1d(np.asarray(plus, dtype=float))
    n_cols = plus.shape[0]

    libor_filled = _fill_missing_ffill_bfill(libor_index[:, None]).flatten()
    valid = np.where(~np.isnan(libor_index))[0]
    fnm = valid[0] if valid.size else 0
    lnm = valid[-1] if valid.size else n_dates - 1

    libor_rates = price_to_return(libor_filled[:, None], 1).flatten()
    libor_rates[fnm] = 0.0
    libor_rates = np.where(np.isnan(libor_rates), 0.0, libor_rates)

    libor_index_plus = np.zeros((n_dates, n_cols))
    libor_index_plus[0, :] = 100.0
    for t in range(1, n_dates):
        dt = (day_nums[t] - day_nums[t - 1]) / 365.25
        libor_index_plus[t, :] = libor_index_plus[t - 1, :] * (1 + libor_rates[t] + plus * dt)

    if fnm > 0:
        libor_index_plus[:fnm, :] = np.nan
    if lnm < n_dates - 1:
        libor_index_plus[lnm + 1 :, :] = np.nan

    libor_index_plus = 100.0 * libor_index_plus / libor_index_plus[fnm, :]
    return libor_index_plus
