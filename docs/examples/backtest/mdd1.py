"""Translated from Examples/backtest/mdd1.m -- maximum_drawdown on 3
simulated price indices (numeric core only; the original's plot marking
the drawdown peak/trough on each series is dropped).

The original explicitly seeds MATLAB's RNG (`rng(1234567)`); NumPy's
generator is seeded the same way for a comparable (not bit-identical)
run."""

import numpy as np

from quanttoolbox.backtest.returns import return_to_price
from quanttoolbox.backtest.stats import maximum_drawdown
from quanttoolbox.dates.convert import parse_date_serial
from quanttoolbox.dates.rebalancing import generate_trading_dates

# See backtest2.py's translation note: generate_trading_dates needs an
# actual pd.Timestamp, not a bare YYYYMMDD int, hence parse_date_serial.
d1, d2 = parse_date_serial([20160101, 20161231])
_, dates = generate_trading_dates(d1, d2, business_days_only=True)
n_dates = len(dates)

n_assets = 3
sigma = 0.20 / np.sqrt(260)

rng = np.random.default_rng(1234567)
r = sigma * rng.standard_normal((n_dates, n_assets))
indices = return_to_price(r)

max_dd, start_dd, end_dd, tau_dd = maximum_drawdown(indices, relative=True)

print("Maximum drawdown per asset:", np.round(max_dd, 4))
print("Start date (peak):", [str(dates[i].date()) for i in start_dd])
print("End date (trough):", [str(dates[i].date()) for i in end_dd])
print("Duration (trading days):", tau_dd)
