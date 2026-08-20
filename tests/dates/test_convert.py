"""Tests for quanttoolbox.dates.convert."""

import numpy as np
import pandas as pd
import pytest

from quanttoolbox.dates.convert import (
    datetime_to_excel,
    excel_column,
    excel_to_datetime,
    is_yyyymmdd,
    parse_date_serial,
    to_yyyymmdd,
)


def test_excel_to_datetime_known_value():
    # Excel serial 44927 == 2023-01-01 (verified against Excel/openpyxl)
    result = excel_to_datetime(np.array([44927]))
    assert result[0] == pd.Timestamp("2023-01-01")


def test_excel_datetime_roundtrip():
    original = pd.DatetimeIndex(["2020-01-01", "2021-06-15", "2023-12-31"])
    excel = datetime_to_excel(original)
    back = excel_to_datetime(excel)
    assert list(back) == list(original)


def test_is_yyyymmdd_true_case():
    test, yyyy, mm, dd = is_yyyymmdd(np.array([20230115, 20200229]))
    assert test
    assert list(yyyy) == [2023, 2020]
    assert list(mm) == [1, 2]
    assert list(dd) == [15, 29]


def test_is_yyyymmdd_false_case_out_of_range_month():
    test, *_ = is_yyyymmdd(np.array([20231315]))  # month 13 invalid
    assert not test


def test_is_yyyymmdd_false_case_non_integer():
    test, *_ = is_yyyymmdd(np.array([20230115.5]))
    assert not test


def test_to_yyyymmdd():
    dates = pd.DatetimeIndex(["2023-01-15", "2020-02-29"])
    result = to_yyyymmdd(dates)
    assert list(result) == [20230115, 20200229]


def test_parse_date_serial_yyyymmdd_input():
    result = parse_date_serial(np.array([20230115]))
    assert result[0] == pd.Timestamp("2023-01-15")


def test_parse_date_serial_excel_serial_input():
    # 44927 is not a plausible yyyymmdd (way below 1900xxxx), falls back to excel serial
    result = parse_date_serial(np.array([44927]))
    assert result[0] == pd.Timestamp("2023-01-01")


@pytest.mark.parametrize(
    "x,expected",
    [(1, "A"), (26, "Z"), (27, "AA"), (52, "AZ"), (53, "BA"), (702, "ZZ"), (703, "AAA")],
)
def test_excel_column(x, expected):
    assert excel_column(x) == expected


def test_excel_column_rejects_zero():
    with pytest.raises(ValueError):
        excel_column(0)
