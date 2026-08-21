"""Translated from Examples/backtest/backtest2.m -- generate_backtest at
three rebalancing schedules (single rebalance at t0, four fixed
positional rebalances, and a schedule mixing an actual date subset with
two out-of-range placeholder dates) on a simulated 3-asset equal-weight
portfolio, with one price set to NaN in the third test.

The original draws R from MATLAB's unseeded `randn`; a fixed seed
(`np.random.default_rng(0)`) is substituted here for reproducibility,
same convention used elsewhere in this port. Plotting is dropped (see
`docs/migration_map.md`'s example translation tracker Notes column
convention)."""

import numpy as np

from quanttoolbox.backtest.reporting import generate_backtest
from quanttoolbox.backtest.returns import return_to_price
from quanttoolbox.dates.convert import parse_date_serial
from quanttoolbox.dates.rebalancing import generate_trading_dates

# generate_trading_dates takes pd.Timestamp (or a bare int, which it feeds
# straight to pd.Timestamp(int) -- interpreted as a nanosecond epoch, NOT
# a YYYYMMDD date). parse_date_serial(...) is used here to get an actual
# 2016-01-01/2016-12-31 pair instead.
d1, d2 = parse_date_serial([20160101, 20161231])
_, dates = generate_trading_dates(d1, d2, business_days_only=True)
n_dates = len(dates)

n_assets = 3
sigma = 0.20 / np.sqrt(260)

rng = np.random.default_rng(0)
r = sigma * rng.standard_normal((n_dates, n_assets))
indices = return_to_price(r)

weights = np.full((n_dates, n_assets), 1.0 / n_assets)

# Test 1: single rebalance at t0
rb_dates1 = dates[:1]
result1 = generate_backtest(dates, weights, indices, rb_dates1)
y1 = np.sum(indices * weights, axis=1)
y1 = 100 * y1 / y1[0]
print("Test 1 (single rebalance): backtest vs. buy&hold, first/last 3 rows")
print(np.column_stack([result1.backtest, y1])[:3])
print(np.column_stack([result1.backtest, y1])[-3:])

# Test 2: four fixed positional rebalances (MATLAB 1-indexed [1,50,150,200])
rb_dates2 = np.array([1, 50, 150, 200]) - 1
result2 = generate_backtest(dates, weights, indices, rb_dates2)
y2 = np.sum(indices * weights, axis=1)
y2 = 100 * y2 / y2[0]
print("\nTest 2 (4 fixed rebalances): backtest vs. buy&hold, first/last 3 rows")
print(np.column_stack([result2.backtest, y2])[:3])
print(np.column_stack([result2.backtest, y2])[-3:])

# Test 3: an actual date subset plus two out-of-range placeholder dates
# (20150101/20170101, both before/after the calendar -- they simply won't
# match anything in `dates`), and a NaN price injected on day 3.
rb_idx = np.array([10, 50, 150, 200]) - 1
placeholder_before, placeholder_after = parse_date_serial([20150101, 20170101])
rb_dates3 = dates[rb_idx].insert(0, placeholder_before)
rb_dates3 = rb_dates3.insert(len(rb_dates3), placeholder_after)
indices3 = indices.copy()
indices3[2, 0] = np.nan
result3 = generate_backtest(dates, weights, indices3, rb_dates3)
y3 = np.sum(indices3 * weights, axis=1)
y3 = 100 * y3 / y3[0]
print("\nTest 3 (date subset + NaN price): backtest vs. buy&hold, first/last 3 rows")
print(np.column_stack([result3.backtest, y3])[:3])
print(np.column_stack([result3.backtest, y3])[-3:])
