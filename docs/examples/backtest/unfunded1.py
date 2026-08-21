"""Translated from Examples/backtest/unfunded1.m -- compares a funded
(price-return) backtest against economically equivalent unfunded
(excess-return) formulations, three different ways:
Backtest1 -- pure funded weights on funded prices.
Backtest2 -- funded weights plus an explicit cash/LIBOR leg, funded prices.
Backtest3 -- generate_backtest_funded_unfunded with the LIBOR leg funded
and the risky assets unfunded (price_to_unfunded'd), which should match
Backtest1/Backtest2 economically since unfunding + re-funding is neutral.

The original draws R from MATLAB's unseeded `randn`; a fixed seed
(`np.random.default_rng(0)`) is substituted here. Plotting is dropped."""

import numpy as np

from quanttoolbox.backtest.reporting import generate_backtest, generate_backtest_funded_unfunded
from quanttoolbox.backtest.returns import price_to_unfunded

n_dates = 3000
dates = np.arange(1, n_dates + 1)
rb = np.zeros(n_dates)
rb[np.arange(1, n_dates, 5) - 1] = 1.0

sigma = 0.20 * np.sqrt(1 / 260)
rng = np.random.default_rng(0)
r = sigma * rng.standard_normal((n_dates, 2))
prices_funded = 100 * np.cumprod(1 + r, axis=0)

rate = 0.03
r_libor = rate * np.sqrt(1 / 260) * np.ones((n_dates, 1))
libor_index = 100 * np.cumprod(1 + r_libor, axis=0)
prices_unfunded = price_to_unfunded(prices_funded, libor_index, method=1)

weights = np.tile([0.5, 0.5], (n_dates, 1))

w1 = weights
backtest1 = generate_backtest(dates, w1, prices_funded, rb).backtest

w2 = np.column_stack([weights, 1 - np.sum(weights, axis=1)])
backtest2 = generate_backtest(dates, w2, np.column_stack([prices_funded, libor_index]), rb).backtest

weights_libor = np.ones((n_dates, 1))
w3 = weights
backtest3 = generate_backtest_funded_unfunded(
    dates, weights_libor, libor_index, w3, prices_unfunded, rb
).backtest

print(
    "Backtest1 (funded), Backtest2 (funded + cash leg), Backtest3 (unfunded formulation) -- first/last 5 rows:"
)
print(np.round(np.column_stack([backtest1, backtest2, backtest3])[:5], 3))
print(np.round(np.column_stack([backtest1, backtest2, backtest3])[-5:], 3))
