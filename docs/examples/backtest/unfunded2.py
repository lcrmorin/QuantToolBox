"""Translated from Examples/backtest/unfunded2.m -- a tiny (n=8),
hand-traceable version of unfunded1.py's funded/unfunded round-trip:
checks that price_to_unfunded -> unfunded_to_price recovers the original
funded prices, and that the three backtest formulations agree, on a
deterministic constant-return series (no randomness involved, so no
fixed-seed substitution is needed here)."""

import numpy as np

from quanttoolbox.backtest.reporting import generate_backtest, generate_backtest_funded_unfunded
from quanttoolbox.backtest.returns import price_to_return, price_to_unfunded, unfunded_to_price

n_dates = 8
dates = np.arange(1, n_dates + 1)
rb = np.ones(n_dates)  # every date is a rebalance

r = 0.03 * np.ones((n_dates, 1))
prices_funded = 100 * np.cumprod(1 + r, axis=0)
prices_funded = 100 * prices_funded / prices_funded[0]
r1 = price_to_return(prices_funded, 1)

r_libor = 0.01 * np.ones((n_dates, 1))
libor_index = 100 * np.cumprod(1 + r_libor, axis=0)
prices_unfunded = price_to_unfunded(prices_funded, libor_index, method=1)
r2 = price_to_return(prices_unfunded, 1)
r3 = price_to_return(libor_index, 1)
p3 = unfunded_to_price(prices_unfunded, libor_index)

print("returns: r (input), r1 (from prices_funded), r2 (unfunded), r3 (libor):")
print(np.round(np.column_stack([r, r1, r2, r3]), 5))
print("\nprices: funded, round-tripped-back-to-funded (p3), libor:")
print(np.round(np.column_stack([prices_funded, p3, libor_index]), 4))

weights = np.ones((n_dates, 1))
w1 = weights
backtest1 = generate_backtest(dates, w1, prices_funded, rb).backtest

w2 = np.column_stack([weights, 1 - np.sum(weights, axis=1)])
backtest2 = generate_backtest(dates, w2, np.column_stack([prices_funded, libor_index]), rb).backtest

weights_libor = np.ones((n_dates, 1))
w3 = weights
backtest3 = generate_backtest_funded_unfunded(
    dates, weights_libor, libor_index, w3, prices_unfunded, rb
).backtest

print("\nBacktest1 (funded), Backtest2 (funded + cash leg), Backtest3 (unfunded formulation):")
print(np.round(np.column_stack([backtest1, backtest2, backtest3]), 4))
