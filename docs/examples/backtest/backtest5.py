"""Translated from Examples/backtest/backtest5.m -- generate_backtest with
a flat transaction cost (0.01) under monthly rebalancing (the original
builds weekly and daily rebalancing schedules too, but overwrites both
before use, so only the monthly one -- `RB_Dates = ones(nDates,1)`, i.e.
every date is a rebalance -- is actually exercised; reproduced as-is)."""

import numpy as np

from quanttoolbox.backtest.reporting import generate_backtest
from quanttoolbox.backtest.returns import return_to_price
from quanttoolbox.dates.convert import parse_date_serial
from quanttoolbox.dates.rebalancing import generate_trading_dates

# See backtest2.py's translation note: generate_trading_dates needs an
# actual pd.Timestamp, not a bare YYYYMMDD int, hence parse_date_serial.
d1, d2 = parse_date_serial([20160101, 20161231])
_, dates = generate_trading_dates(d1, d2, business_days_only=True)
n_dates = len(dates)

n_assets = 3
sigma = 0.20 / np.sqrt(260)

rng = np.random.default_rng(123456789)
r = sigma * rng.standard_normal((n_dates, n_assets))
indices = return_to_price(r)

weights = np.ones((n_dates, n_assets))
weights = weights / np.sum(weights, axis=1, keepdims=True)

rb_dates = np.ones(
    n_dates
)  # "monthly rebalancing" per the original's final overwrite -> every date

result1 = generate_backtest(dates, weights, indices, rb_dates)
tc = 0.01
result2 = generate_backtest(dates, weights, indices, rb_dates, tc_bid_ask=tc)

print("backtest without / with transaction costs, turnover, TC (first 5 rows):")
print(
    np.round(
        np.column_stack(
            [result1.backtest, result2.backtest, result2.turnover, result2.transaction_costs]
        )[:5],
        4,
    )
)

to_total = np.nansum(result2.turnover)
cost1 = 1 - result2.backtest[-1] / result1.backtest[-1]
cost2 = to_total * tc
print("\nTotal turnover:", round(to_total, 4))
print("Cost1 (1 - final wealth ratio):", round(cost1, 5))
print("Cost2 (total turnover * tc):", round(cost2, 5))
