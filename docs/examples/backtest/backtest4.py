"""Translated from Examples/backtest/backtest4.m -- generate_backtest with
per-asset bid/ask transaction costs, comparing turnover computed inside
the backtest against `static_turnover` computed independently from the
rebalance-date weights.

The original explicitly seeds MATLAB's RNG (`rng(123456789)`); NumPy's
generator is seeded the same way for a comparable (not bit-identical --
different RNG algorithms) run."""

import numpy as np

from quanttoolbox.backtest.reporting import generate_backtest
from quanttoolbox.backtest.returns import return_to_price
from quanttoolbox.backtest.stats import static_turnover
from quanttoolbox.dates.convert import parse_date_serial
from quanttoolbox.dates.rebalancing import generate_trading_dates

# See backtest2.py's translation note: generate_trading_dates needs an
# actual pd.Timestamp, not a bare YYYYMMDD int, hence parse_date_serial.
d1, d2 = parse_date_serial([20160101, 20160304])
_, dates = generate_trading_dates(d1, d2, business_days_only=True)
n_dates = len(dates)

n_assets = 3
sigma = 0.20 / np.sqrt(260)

rng = np.random.default_rng(123456789)
r = sigma * rng.standard_normal((n_dates, n_assets))
indices = return_to_price(r)
indices = np.ones_like(indices)  # matches the original: prices reset to a flat 1.0

weights = rng.random((n_dates, n_assets))
weights = weights / np.sum(weights, axis=1, keepdims=True)

rb_dates = np.arange(5, 30, 5) - 1  # seqa(5,5,5) -> positions 5,10,15,20,25 (1-indexed)

result1 = generate_backtest(dates, weights, indices, rb_dates)
result2 = generate_backtest(
    dates, weights, indices, rb_dates, tc_bid_ask=np.array([0.01, 0.01, 0.05])
)

print("backtest without / with transaction costs, and turnover/TC (first 8 rows):")
print(
    np.round(
        np.column_stack(
            [result1.backtest, result2.backtest, result2.turnover, result2.transaction_costs]
        )[:8],
        4,
    )
)

rb_mask = result2.rebalancing[:, 0] == 1
w_rb = weights[rb_mask]
to_from_backtest = result2.turnover[rb_mask]
# static_turnover gives one turnover per consecutive pair of rebalance
# weights (n_rb - 1 values), vs. the backtest's own per-rebalance turnover
# series (n_rb values, first entry 0 at the initial rebalance) -- printed
# separately since they don't align row-for-row.
to_static = static_turnover(w_rb)
print("\n100*weights at rebalance dates, turnover (from backtest):")
print(np.round(np.column_stack([100 * w_rb, to_from_backtest]), 4))
print("\nturnover between consecutive rebalances (static_turnover, independent cross-check):")
print(np.round(to_static, 4))
