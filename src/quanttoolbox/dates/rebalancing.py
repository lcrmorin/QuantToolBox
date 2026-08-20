"""Rebalancing-date calendar generation.

Ported from QuantToolbox/dates/{generic_rebalancing,annual_rebalancing,
monthly_rebalancing,quarterly_rebalancing,semi_annual_rebalancing,
weekly_rebalancing,generate_trading_dates}.m

Translation notes:

- All rebalancing functions take a ``pandas.DatetimeIndex`` of available
  trading dates and return (mask, rebalancing_dates) instead of MATLAB's
  (RB, RB_Dates, RB_Days) triple -- the weekday *name* (RB_Days) is easily
  recovered from ``rebalancing_dates.day_name()`` and isn't returned as a
  separate value here.
- "Business day" here means Monday-Friday only, matching MATLAB's default
  ``isbusday``/``busdate`` behavior (no holiday calendar). If you need a
  real trading calendar (exchange holidays), pass a custom
  ``pandas.tseries.offsets.CustomBusinessDay`` calendar into
  ``pandas.bdate_range`` upstream and feed those dates in as ``dates``.
- The original's month-end target is the *previous business day* if the
  calendar end-of-month falls on a weekend; here that becomes a simple
  weekday rollback.
- The final rebalancing date snapped from the input ``dates`` is the
  closest available date on-or-before each computed target (an "asof"
  match), mirroring the original's ``indnv(..., 2)`` nearest-lower lookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prev_business_day(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Roll Saturday/Sunday dates back to the preceding Friday."""
    weekday = dates.weekday.to_numpy()
    shift_days = np.where(weekday == 5, 1, np.where(weekday == 6, 2, 0))
    return dates - pd.to_timedelta(shift_days, unit="D")


def _snap_to_available(target_dates: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each target date, find the closest available date on or before it."""
    dates = dates.sort_values()
    pos = dates.searchsorted(target_dates, side="right") - 1
    valid = pos >= 0
    return pd.DatetimeIndex(sorted(set(dates[pos[valid]])))


def generic_rebalancing(
    dates: pd.DatetimeIndex, frequency: int
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate rebalancing dates at the given month-frequency (1=monthly,
    3=quarterly, 6=semi-annual, 12=annual), snapped to available dates.

    Original: dates/generic_rebalancing.m

    Returns
    -------
    mask : bool ndarray, same length/order as ``dates``, True on rebalancing dates.
    rebalancing_dates : DatetimeIndex of the selected rebalancing dates.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()

    full_range = pd.date_range(dates[0], dates[-1], freq="D")
    year = full_range.year.to_numpy()
    month = full_range.month.to_numpy()
    if frequency != 1:
        month = (np.ceil(month / frequency) * frequency).astype(int)

    period_key = year * 100 + month
    eom = pd.to_datetime({"year": year, "month": month, "day": 1}) + pd.offsets.MonthEnd(0)
    eom_adj = _prev_business_day(pd.DatetimeIndex(eom))

    # one target date per distinct (year, grouped-month) period
    period_targets = pd.Series(eom_adj).groupby(period_key).first()
    target_dates = pd.DatetimeIndex(period_targets.to_numpy())

    rb_dates = _snap_to_available(target_dates, dates)
    mask = dates.isin(rb_dates)
    return mask, rb_dates


def monthly_rebalancing(dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Original: dates/monthly_rebalancing.m"""
    return generic_rebalancing(dates, 1)


def quarterly_rebalancing(dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Original: dates/quarterly_rebalancing.m"""
    return generic_rebalancing(dates, 3)


def semi_annual_rebalancing(dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Original: dates/semi_annual_rebalancing.m"""
    return generic_rebalancing(dates, 6)


def annual_rebalancing(dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Original: dates/annual_rebalancing.m"""
    return generic_rebalancing(dates, 12)


def weekly_rebalancing(
    dates: pd.DatetimeIndex, day_of_week: int
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate weekly rebalancing dates on a given weekday, snapped to
    available dates.

    ``day_of_week`` uses MATLAB's ``weekday`` convention: 1=Sunday,
    2=Monday, ..., 7=Saturday (to preserve call-site compatibility with the
    original). Internally converted to pandas' Monday=0 convention.

    Original: dates/weekly_rebalancing.m
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    full_range = pd.date_range(dates[0], dates[-1], freq="D")

    # MATLAB weekday: 1=Sun..7=Sat  ->  pandas weekday: 0=Mon..6=Sun
    pandas_weekday = (day_of_week - 2) % 7

    target_dates = full_range[full_range.weekday == pandas_weekday]
    rb_dates = _snap_to_available(target_dates, dates)
    mask = dates.isin(rb_dates)
    return mask, rb_dates


def generate_trading_dates(
    begin_date: pd.Timestamp | int, end_date: pd.Timestamp | int, business_days_only: bool = False
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate a calendar of dates between begin_date and end_date,
    optionally restricted to business days (Mon-Fri, no holiday calendar).

    Original: dates/generate_trading_dates.m

    Returns
    -------
    yyyymmdd : int ndarray of dates in YYYYMMDD format.
    dates : DatetimeIndex of the same dates.
    """
    date1 = pd.Timestamp(begin_date) if not isinstance(begin_date, pd.Timestamp) else begin_date
    date2 = pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date

    dates = pd.date_range(date1, date2, freq="D")
    if business_days_only:
        dates = dates[dates.weekday < 5]

    yyyymmdd = (dates.year * 10000 + dates.month * 100 + dates.day).to_numpy()
    return yyyymmdd, dates
