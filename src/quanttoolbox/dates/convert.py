"""Excel <-> Python date conversion and date-format helpers.

Ported from QuantToolbox/dates/{Excel2Matlab_Dates,Matlab2Excel_Dates,
is_yyyymmdd,numdate,datenum2,excel_column}.m

Translation notes:

- MATLAB's ``datenum`` epoch (day 1 = 1-Jan-0000, proleptic) is NOT used
  here. Excel's own serial-date epoch (day 1 = 1-Jan-1900, with the famous
  1900-leap-year bug baked in) is handled directly against
  ``pandas.Timestamp`` via a fixed offset, which is simpler and avoids
  MATLAB's separate datenum epoch entirely.
- Per the original warning: Excel's serial dates are only valid for dates
  after 1900-02-29 (which doesn't actually exist -- Excel incorrectly
  treats 1900 as a leap year). Do not use these helpers for dates before
  1900-03-01.
- Where MATLAB used ``datetime`` arrays, this module uses
  ``pandas.Timestamp`` / ``pandas.DatetimeIndex`` throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")  # so that serial 1 == 1900-01-01,
# and the fictitious 1900-02-29 is naturally absorbed by the +2 offset used
# below, matching Excel's (buggy) day count.


def excel_to_datetime(
    excel_dates: int | float | np.ndarray | list, date_format: str = "%d/%m/%Y"
) -> pd.DatetimeIndex | pd.Timestamp:
    """Convert Excel serial date numbers (or formatted date strings) to
    pandas Timestamps.

    Original: dates/Excel2Matlab_Dates.m
    """
    if isinstance(excel_dates, str) or (
        isinstance(excel_dates, (list | np.ndarray))
        and len(excel_dates) > 0
        and isinstance(np.asarray(excel_dates).flat[0], str)
    ):
        return pd.to_datetime(excel_dates, format=date_format)

    excel_dates = np.asarray(excel_dates)
    result = _EXCEL_EPOCH + pd.to_timedelta(excel_dates, unit="D")
    if result.ndim == 0 or (hasattr(excel_dates, "shape") and excel_dates.shape == ()):
        return pd.Timestamp(result)
    return pd.DatetimeIndex(result)


def datetime_to_excel(dates: pd.Timestamp | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    """Convert pandas Timestamps (or a YYYYMMDD-encoded numeric array) to
    Excel serial date numbers. Always returns a float array.

    Original: dates/Matlab2Excel_Dates.m
    """
    if isinstance(dates, (pd.Timestamp | pd.DatetimeIndex)):
        idx = pd.DatetimeIndex([dates]) if isinstance(dates, pd.Timestamp) else dates
        delta = idx - _EXCEL_EPOCH
        return delta.days.to_numpy().astype(float)

    arr = np.asarray(dates)
    test, yyyy, mm, dd = is_yyyymmdd(arr)
    if test:
        idx = pd.DatetimeIndex(pd.to_datetime({"year": yyyy, "month": mm, "day": dd}))
        delta = idx - _EXCEL_EPOCH
        return delta.days.to_numpy().astype(float)

    # already serial-like numeric input; nothing to convert
    return arr.astype(float)


def is_yyyymmdd(x: np.ndarray | int | float) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Test whether a numeric array is encoded as YYYYMMDD integers, and if
    so, decompose it into (year, month, day) components.

    Original: dates/is_yyyymmdd.m
    """
    x = np.asarray(x, dtype=float)

    if np.any(np.fix(x) != x):
        return False, np.array([]), np.array([]), np.array([])

    dd = x - 100 * np.floor(x / 100)
    x2 = np.floor((x - dd) / 100)
    mm = x2 - 100 * np.floor(x2 / 100)
    yyyy = np.floor((x2 - mm) / 100)

    test = bool(
        dd.min() >= 1
        and dd.max() <= 31
        and mm.min() >= 1
        and mm.max() <= 12
        and yyyy.min() >= 1900
        and yyyy.max() <= 2100
    )
    return test, yyyy.astype(int), mm.astype(int), dd.astype(int)


def to_yyyymmdd(dates: pd.DatetimeIndex | pd.Timestamp) -> np.ndarray:
    """Convert pandas dates to an integer YYYYMMDD array.

    Original: dates/datenum2.m
    """
    idx = pd.DatetimeIndex([dates]) if isinstance(dates, pd.Timestamp) else pd.DatetimeIndex(dates)
    return (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy()


def parse_date_serial(x: np.ndarray | int | float) -> pd.DatetimeIndex:
    """Parse a numeric date-like array that may be either YYYYMMDD-encoded
    or a plain serial day count, returning pandas Timestamps.

    Original: dates/numdate.m
    """
    x = np.asarray(x, dtype=float)
    test, yyyy, mm, dd = is_yyyymmdd(x)
    if test:
        return pd.DatetimeIndex(pd.to_datetime({"year": yyyy, "month": mm, "day": dd}))
    # fall back: treat as Excel-style serial day count
    return pd.DatetimeIndex(excel_to_datetime(x))


def excel_column(x: int) -> str:
    """Convert a 1-indexed column number to its Excel column letter (1='A',
    27='AA', ...).

    Original: dates/excel_column.m
    """
    if x < 1:
        raise ValueError("excel_column: x must be >= 1")
    letters = ""
    n = x
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters
