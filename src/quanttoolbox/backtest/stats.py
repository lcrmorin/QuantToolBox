"""Drawdown, turnover, average-return, and monthly/yearly performance statistics.

Ported from QuantToolBox/backtest/{maximum_drawdown,static_turnover,
annualized_turnover,average_return,index_repeated_data,monthly_statistics,
yearly_statistics}.m

Translation notes:

- ``maximum_drawdown`` returns 0-indexed start/end row positions (not
  MATLAB's 1-indexed), so downstream code should adjust index arithmetic
  accordingly (e.g. when mapping back to a Dates array).
- ``monthly_statistics``/``yearly_statistics`` reproduce the originals'
  "reindex onto a full trading calendar, forward-fill gaps" pattern using
  pandas (``reindex`` + ``ffill``) instead of MATLAB's manual
  ``indnv``/``fillmiss`` combination.
- Both statistics functions accept a ``begin_date``/``end_date`` as either
  a ``pandas.Timestamp`` or an integer YYYYMMDD (0 meaning "use the full
  available range"), matching the original's dual calling convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quanttoolbox.dates.rebalancing import generate_trading_dates


def maximum_drawdown(
    x: np.ndarray, relative: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maximum drawdown of each column of a price/index series.

    Original: backtest/maximum_drawdown.m

    Returns
    -------
    max_dd : maximum drawdown per column (negative or zero).
    start_dd : 0-indexed row where the drawdown period started (peak).
    end_dd : 0-indexed row where the maximum drawdown was reached (trough).
    tau_dd : drawdown duration in rows (end_dd - start_dd + 1).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, c = x.shape

    running_max = np.full((n, c), np.nan)
    for i in range(1, n):
        running_max[i, :] = np.nanmax(x[: i + 1, :], axis=0)

    dd = running_max - x
    dd = np.where(np.isnan(dd), 0.0, dd)
    if relative:
        dd = dd / running_max
        dd = np.where(np.isnan(dd), 0.0, dd)

    max_dd = np.max(dd, axis=0)
    end_dd = np.argmax(dd, axis=0)

    start_dd = np.zeros(c, dtype=int)
    for j in range(c):
        i = end_dd[j]
        while i > 0 and dd[i, j] != 0:
            i -= 1
        start_dd[j] = i

    tau_dd = end_dd - start_dd + 1
    return -max_dd, start_dd, end_dd, tau_dd


def static_turnover(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Turnover between consecutive rows of x (or between x and y if given).

    Original: backtest/static_turnover.m
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        if x.ndim == 1:
            x = x[:, None]
        diff = np.abs(x[1:] - x[:-1])
        return np.sum(diff, axis=1)
    y = np.asarray(y, dtype=float)
    return np.sum(np.abs(x - y), axis=0)


def annualized_turnover(
    dates: pd.DatetimeIndex, turnover: np.ndarray, by_year: bool = False
) -> np.ndarray:
    """Annualize a turnover series, either as a single average-per-year
    figure over the whole sample, or broken out year by year.

    Original: backtest/annualized_turnover.m
    """
    dates = pd.DatetimeIndex(dates)
    turnover = np.asarray(turnover, dtype=float)
    if turnover.ndim == 1:
        turnover = turnover[:, None]

    if not by_year:
        valid = ~np.isnan(turnover).any(axis=1)
        to_valid = turnover[valid]
        dt_years = (dates[valid][-1] - dates[valid][0]).days / 365.25
        return np.sum(to_valid, axis=0) / dt_years

    years = dates.year.to_numpy()
    unique_years = np.unique(years)
    n_years = unique_years.shape[0]
    n_series = turnover.shape[1]

    tau = np.zeros((n_years, n_series))
    for i, yr in enumerate(unique_years):
        mask = years == yr
        for j in range(n_series):
            col = turnover[mask, j]
            if np.all(np.isnan(col)):
                tau[i, j] = np.nan
            else:
                tau[i, j] = np.nansum(col)
    return tau


def average_return(r: np.ndarray, n_lags: int) -> np.ndarray:
    """Trailing n_lags-period moving average of a return series (NaN-filled
    as zero before averaging).

    Original: backtest/average_return.m
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    r_filled = np.where(np.isnan(r), 0.0, r)
    n_dates, n_cols = r.shape

    x = np.full((n_dates, n_cols), np.nan)
    for i in range(n_lags - 1, n_dates):
        x[i, :] = np.mean(r_filled[i - n_lags + 1 : i + 1, :], axis=0)
    return x


def index_repeated_data(x: np.ndarray, precision: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify rows that repeat the previous row's value (after rounding to
    `precision` decimal places) -- useful for spotting stale/non-trading
    price data.

    Original: backtest/index_repeated_data.m

    Returns
    -------
    idx_repeated : 0-indexed positions where the value repeats the prior row.
    idx_changed : 0-indexed positions where the value differs from the prior row.
    is_repeated : boolean mask, same length as x.
    """
    x = np.asarray(x, dtype=float).flatten()
    scaled = np.round(x * 10**precision)

    y = np.concatenate([[np.nan], scaled[:-1]])
    cnd = scaled == y
    both_nan = np.isnan(scaled) & np.isnan(y)
    cnd = cnd | both_nan
    cnd[0] = False  # no prior row to compare against

    idx = np.arange(x.shape[0])
    return idx[cnd], idx[~cnd], cnd


@dataclass
class MonthlyStats:
    years: np.ndarray
    months: np.ndarray
    mu: np.ndarray


@dataclass
class YearlyStats:
    years: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    max_dd: np.ndarray = field(default_factory=lambda: np.array([]))
    mu_te: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_te: np.ndarray = field(default_factory=lambda: np.array([]))
    beta: np.ndarray = field(default_factory=lambda: np.array([]))
    rho: np.ndarray = field(default_factory=lambda: np.array([]))


def _reindex_to_full_calendar(
    dates: pd.DatetimeIndex, values: np.ndarray, begin: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Reindex values onto a full daily calendar between begin/end, then
    forward/back-fill gaps -- mirrors the original's generate_trading_dates
    + indnv + fillmiss(...,4) pattern."""
    _, full_dates = generate_trading_dates(begin, end)
    series = pd.DataFrame(values, index=dates).reindex(full_dates)
    series = series.ffill().bfill()
    return full_dates, series.to_numpy()


def monthly_statistics(
    dates: pd.DatetimeIndex,
    backtests: np.ndarray,
    begin_date: pd.Timestamp | int = 0,
    end_date: pd.Timestamp | int = 0,
) -> tuple[MonthlyStats, YearlyStats]:
    """Monthly and yearly returns/volatility/drawdown statistics for one or
    more backtest series.

    Original: backtest/monthly_statistics.m
    """
    dates = pd.DatetimeIndex(dates)
    backtests = np.asarray(backtests, dtype=float)
    if backtests.ndim == 1:
        backtests = backtests[:, None]

    begin = dates[0] if begin_date in (0, None) else pd.Timestamp(begin_date)
    end = dates[-1] if end_date in (0, None) else pd.Timestamp(end_date)

    mask = (dates >= begin) & (dates <= end)
    dates, backtests = dates[mask], backtests[mask]

    full_dates, backtests = _reindex_to_full_calendar(dates, backtests, begin, end)
    n_cols = backtests.shape[1]

    monthly_period = full_dates.year * 100 + full_dates.month
    unique_months = np.unique(monthly_period)
    n_months = unique_months.shape[0]
    month_to_row = {m: i for i, m in enumerate(unique_months)}

    mu = np.full((n_months, n_cols), np.nan)
    for j in range(n_cols):
        col = backtests[:, j]
        valid = np.where(~np.isnan(col))[0]
        if valid.size == 0:
            continue
        fnm, lnm = valid[0], valid[-1]
        x0 = col[fnm]
        for t in range(fnm + 1, lnm + 1):
            if monthly_period[t] != monthly_period[t - 1]:
                x1 = col[t - 1]
                mu[month_to_row[monthly_period[t - 1]], j] = x1 / x0 - 1.0
                x0 = x1
        if lnm > fnm and monthly_period[lnm] == monthly_period[lnm - 1]:
            mu[month_to_row[monthly_period[lnm]], j] = col[lnm] / x0 - 1.0

    years_arr = unique_months // 100
    months_arr = unique_months % 100
    monthly_stats = MonthlyStats(years=years_arr, months=months_arr, mu=mu)

    unique_years = np.unique(years_arr)
    n_years = unique_years.shape[0]
    mu_y = np.full((n_years, n_cols), np.nan)
    sigma_y = np.full((n_years, n_cols), np.nan)
    max_dd_y = np.full((n_years, n_cols), np.nan)

    num_dates_full = full_dates.year * 10000 + full_dates.month * 100 + full_dates.day
    for j in range(n_cols):
        for t, yr in enumerate(unique_years):
            begin_year = (yr - 1) * 10000 + 1231
            end_year = begin_year + 10000
            mask_y = (num_dates_full >= begin_year) & (num_dates_full <= end_year)
            data = backtests[mask_y, j]
            data = data[~np.isnan(data)]
            if data.size == 0:
                continue
            mu_y[t, j] = data[-1] / data[0] - 1.0
            r = price_to_return_1d(data)
            sigma_y[t, j] = np.sqrt(260) * np.nanstd(r[1:], ddof=1)
            prices = return_to_price_1d(r)
            dd_result = maximum_drawdown(prices, relative=True)
            max_dd_y[t, j] = dd_result[0][0]

    yearly_stats = YearlyStats(years=unique_years, mu=mu_y, sigma=sigma_y, max_dd=max_dd_y)
    return monthly_stats, yearly_stats


def yearly_statistics(
    dates: pd.DatetimeIndex,
    backtests: np.ndarray,
    benchmark: np.ndarray,
    begin_date: pd.Timestamp | int = 0,
    end_date: pd.Timestamp | int = 0,
) -> YearlyStats:
    """Yearly return/volatility/tracking-error/beta/correlation statistics
    for one or more backtest series against a benchmark.

    Original: backtest/yearly_statistics.m
    """
    dates = pd.DatetimeIndex(dates)
    backtests = np.asarray(backtests, dtype=float)
    if backtests.ndim == 1:
        backtests = backtests[:, None]
    benchmark = np.asarray(benchmark, dtype=float).flatten()

    begin = dates[0] if begin_date in (0, None) else pd.Timestamp(begin_date)
    end = dates[-1] if end_date in (0, None) else pd.Timestamp(end_date)

    mask = (dates >= begin) & (dates <= end)
    dates, backtests, benchmark = dates[mask], backtests[mask], benchmark[mask]

    full_dates, backtests = _reindex_to_full_calendar(dates, backtests, begin, end)
    _, benchmark_2d = _reindex_to_full_calendar(dates, benchmark[:, None], begin, end)
    benchmark = benchmark_2d.flatten()
    n_cols = backtests.shape[1]

    years_full = full_dates.year.to_numpy()
    unique_years = np.unique(years_full)
    n_years = unique_years.shape[0]

    mu = np.full((n_years, n_cols), np.nan)
    sigma = np.full((n_years, n_cols), np.nan)
    mu_te = np.full((n_years, n_cols), np.nan)
    sigma_te = np.full((n_years, n_cols), np.nan)
    max_dd = np.full((n_years, n_cols), np.nan)
    beta = np.full((n_years, n_cols), np.nan)
    rho = np.full((n_years, n_cols), np.nan)

    num_dates_full = full_dates.year * 10000 + full_dates.month * 100 + full_dates.day
    for j in range(n_cols):
        for t, yr in enumerate(unique_years):
            begin_year = (yr - 1) * 10000 + 1231
            end_year = begin_year + 10000
            mask_y = (num_dates_full >= begin_year) & (num_dates_full <= end_year)
            b = backtests[mask_y, j]
            bench = benchmark[mask_y]
            valid = ~np.isnan(b) & ~np.isnan(bench)
            b, bench = b[valid], bench[valid]
            if b.size <= 1:
                continue

            mu[t, j] = b[-1] / b[0] - 1.0
            mu_te[t, j] = (b[-1] / b[0]) / (bench[-1] / bench[0]) - 1.0

            r_b = price_to_return_1d(b)
            r_bench = price_to_return_1d(bench)
            sigma[t, j] = np.sqrt(260) * np.nanstd(r_b[1:], ddof=1)
            prices = return_to_price_1d(r_b)
            dd_result = maximum_drawdown(prices, relative=True)
            max_dd[t, j] = dd_result[0][0]

            e = r_b - r_bench
            sigma_te[t, j] = np.sqrt(260) * np.nanstd(e[1:], ddof=1)

            joint = np.column_stack([r_bench, r_b])
            joint = joint[~np.isnan(joint).any(axis=1)]
            if joint.shape[0] > 1:
                cov = np.cov(joint, rowvar=False)
                with np.errstate(invalid="ignore", divide="ignore"):
                    beta[t, j] = cov[1, 0] / cov[1, 1]
                    rho[t, j] = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])

    return YearlyStats(
        years=unique_years,
        mu=mu,
        sigma=sigma,
        mu_te=mu_te,
        sigma_te=sigma_te,
        max_dd=max_dd,
        beta=beta,
        rho=rho,
    )


def price_to_return_1d(x: np.ndarray) -> np.ndarray:
    """1-D convenience wrapper around price_to_return for internal use here."""
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)
    y[1:] = x[1:] / x[:-1] - 1.0
    return y


def return_to_price_1d(r: np.ndarray) -> np.ndarray:
    """1-D convenience wrapper around return_to_price for internal use here."""
    r_filled = np.where(np.isnan(r), 0.0, r)
    return 100 * np.cumprod(1 + r_filled)
