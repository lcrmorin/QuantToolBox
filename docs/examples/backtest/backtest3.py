"""Translated from Examples/backtest/backtest3.m -- generate_backtest at
four different rebalancing frequencies (buy-and-hold, every period, every
2 periods, every 3 periods) on a fixed 7-date, 2-asset price series."""

import numpy as np
import pandas as pd

from quanttoolbox.backtest.reporting import generate_backtest

indices = np.array(
    [
        [100, 100],
        [150, 50],
        [250, 100],
        [100, 150],
        [110, 125],
        [160, 140],
        [150, 150],
    ],
    dtype=float,
)
weights = np.full((7, 2), 0.50)
dates = pd.bdate_range("2023-01-02", periods=7)

# buy & hold (single rebalance at t=0)
rb_mask = np.zeros(7, dtype=bool)
rb_mask[0] = True
r1 = generate_backtest(dates, weights, indices, dates[rb_mask])
print("buy & hold:", np.round(r1.backtest, 3))

# rebalance every period
r2 = generate_backtest(dates, weights, indices, dates)
print("every period:", np.round(r2.backtest, 3))

# rebalance every 2 periods (indices 0,2,4,6 -> 1st,3rd,5th,7th)
rb_mask3 = np.zeros(7, dtype=bool)
rb_mask3[[0, 2, 4, 6]] = True
r3 = generate_backtest(dates, weights, indices, dates[rb_mask3])
print("every 2 periods:", np.round(r3.backtest, 3))

# rebalance every 3 periods (indices 0,3,6 -> 1st,4th,7th)
rb_mask4 = np.zeros(7, dtype=bool)
rb_mask4[[0, 3, 6]] = True
r4 = generate_backtest(dates, weights, indices, dates[rb_mask4])
print("every 3 periods:", np.round(r4.backtest, 3))
