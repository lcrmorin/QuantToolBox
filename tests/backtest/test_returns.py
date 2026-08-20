"""Tests for quanttoolbox.backtest.returns."""

import numpy as np
import pandas as pd

from quanttoolbox.backtest.returns import (
    capitalized_libor,
    capitalized_libor_plus,
    price_to_return,
    price_to_unfunded,
    return_to_price,
    unfunded_to_price,
)


def test_price_to_return_basic():
    x = np.array([[100.0], [110.0], [99.0]])
    r = price_to_return(x, n_lags=1)
    assert np.isnan(r[0, 0])
    assert np.isclose(r[1, 0], 0.10)
    assert np.isclose(r[2, 0], 99 / 110 - 1)


def test_price_to_return_multi_lag():
    x = np.array([[100.0], [110.0], [121.0]])
    r = price_to_return(x, n_lags=2)
    assert np.isnan(r[0, 0])
    assert np.isnan(r[1, 0])
    assert np.isclose(r[2, 0], 121 / 100 - 1)


def test_return_to_price_roundtrip():
    prices = np.array([[100.0], [105.0], [110.25], [99.0]])
    r = price_to_return(prices, 1)
    back = return_to_price(r)
    # since prices[0] == 100 already, the round trip (which renormalizes
    # to start at 100) should reproduce the original series exactly
    assert np.allclose(back, prices, atol=1e-6)


def test_return_to_price_starts_at_100():
    r = np.array([[np.nan], [0.05], [0.02], [-0.01]])
    y = return_to_price(r)
    valid = ~np.isnan(y[:, 0])
    assert np.isclose(y[valid, 0][0], 100.0)


def test_price_to_unfunded_zero_libor_matches_funded_returns():
    n = 50
    funded = 100 * np.cumprod(1 + np.full((n, 1), 0.001), axis=0)
    libor = np.full((n, 1), 100.0)  # flat LIBOR index -> zero libor return
    unfunded = price_to_unfunded(funded, libor)
    r_funded = price_to_return(funded, 1)
    r_unfunded = price_to_return(unfunded, 1)
    assert np.allclose(r_funded[1:], r_unfunded[1:], atol=1e-8)


def test_unfunded_to_price_is_inverse_of_price_to_unfunded():
    rng = np.random.default_rng(0)
    n = 100
    funded_r = rng.normal(0.0005, 0.01, (n, 1))
    funded_r[0] = np.nan
    funded = return_to_price(funded_r)
    libor_r = rng.normal(0.0001, 0.0005, (n, 1))
    libor_r[0] = np.nan
    libor = return_to_price(libor_r)

    unfunded = price_to_unfunded(funded, libor)
    funded_back = unfunded_to_price(unfunded, libor)

    valid = ~np.isnan(funded[:, 0]) & ~np.isnan(funded_back[:, 0])
    # both series start at 100 and should track closely (round-trip up to
    # renormalization at the first valid point)
    ratio = funded[valid, 0] / funded_back[valid, 0]
    assert np.allclose(ratio, ratio[0], atol=1e-6)


def test_capitalized_libor_compounds_correctly():
    dates = pd.bdate_range("2023-01-01", periods=5)
    rates = np.full((5, 1), 0.05)  # flat 5% annualized rate
    index = capitalized_libor(dates, rates)
    assert index[0, 0] == 100.0
    assert index[-1, 0] > 100.0
    # roughly compounding at 5%/yr over the period
    dt_total = (dates[-1] - dates[0]).days / 365.25
    expected = 100 * (1.05) ** dt_total  # rough check, not exact due to daily compounding
    assert abs(index[-1, 0] - expected) / expected < 0.01


def test_capitalized_libor_plus_adds_spread():
    dates = pd.bdate_range("2023-01-01", periods=10)
    base_rates = np.full((10, 1), 0.02)
    base_index = capitalized_libor(dates, base_rates)

    plus = np.array([0.0, 0.01])  # 0bps and 100bps spread
    plus_index = capitalized_libor_plus(dates, base_index[:, 0], plus)

    assert plus_index.shape == (10, 2)
    # the series with a positive spread should outperform the zero-spread one
    assert plus_index[-1, 1] > plus_index[-1, 0]
