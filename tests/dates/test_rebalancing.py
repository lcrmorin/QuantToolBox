"""Tests for quanttoolbox.dates.rebalancing."""

import numpy as np
import pandas as pd
import pytest

from quanttoolbox.dates.rebalancing import (
    annual_rebalancing,
    generate_trading_dates,
    generic_rebalancing,
    monthly_rebalancing,
    quarterly_rebalancing,
    semi_annual_rebalancing,
    weekly_rebalancing,
)


@pytest.fixture
def two_year_bdates():
    return pd.bdate_range("2022-01-01", "2023-12-31")


def test_monthly_rebalancing_count(two_year_bdates):
    mask, rb_dates = monthly_rebalancing(two_year_bdates)
    # 24 months in a 2-year span -> 24 rebalancing dates
    assert len(rb_dates) == 24
    assert mask.sum() == 24
    assert mask.shape[0] == len(two_year_bdates)


def test_monthly_rebalancing_dates_are_near_month_end(two_year_bdates):
    _, rb_dates = monthly_rebalancing(two_year_bdates)
    for d in rb_dates:
        month_end = d + pd.offsets.MonthEnd(0)
        assert (month_end - d).days <= 3


def test_quarterly_rebalancing_count(two_year_bdates):
    _, rb_dates = quarterly_rebalancing(two_year_bdates)
    assert len(rb_dates) == 8  # 8 quarters in 2 years


def test_semi_annual_rebalancing_count(two_year_bdates):
    _, rb_dates = semi_annual_rebalancing(two_year_bdates)
    assert len(rb_dates) == 4


def test_annual_rebalancing_count(two_year_bdates):
    _, rb_dates = annual_rebalancing(two_year_bdates)
    assert len(rb_dates) == 2


def test_rebalancing_dates_are_subset_of_input(two_year_bdates):
    _, rb_dates = quarterly_rebalancing(two_year_bdates)
    assert set(rb_dates).issubset(set(two_year_bdates))


def test_weekly_rebalancing_matches_weekday(two_year_bdates):
    # MATLAB weekday 2 == Monday
    mask, rb_dates = weekly_rebalancing(two_year_bdates, day_of_week=2)
    assert all(d.weekday() == 0 for d in rb_dates)  # pandas Monday == 0
    assert mask.sum() == len(rb_dates)


def test_weekly_rebalancing_friday():
    # MATLAB weekday 6 == Friday
    dates = pd.bdate_range("2023-01-01", "2023-01-31")
    _, rb_dates = weekly_rebalancing(dates, day_of_week=6)
    assert all(d.weekday() == 4 for d in rb_dates)  # pandas Friday == 4


def test_generic_rebalancing_matches_convenience_wrappers(two_year_bdates):
    mask_a, dates_a = generic_rebalancing(two_year_bdates, 3)
    mask_b, dates_b = quarterly_rebalancing(two_year_bdates)
    assert np.array_equal(mask_a, mask_b)
    assert list(dates_a) == list(dates_b)


def test_generate_trading_dates_all_days():
    yyyymmdd, dates = generate_trading_dates("2023-01-01", "2023-01-10")
    assert len(dates) == 10
    assert yyyymmdd[0] == 20230101
    assert yyyymmdd[-1] == 20230110


def test_generate_trading_dates_business_days_only():
    # 2023-01-01 is a Sunday, 2023-01-07 is a Saturday
    yyyymmdd, dates = generate_trading_dates("2023-01-01", "2023-01-07", business_days_only=True)
    assert all(d.weekday() < 5 for d in dates)
    assert 20230101 not in yyyymmdd  # Sunday excluded
    assert 20230107 not in yyyymmdd  # Saturday excluded
