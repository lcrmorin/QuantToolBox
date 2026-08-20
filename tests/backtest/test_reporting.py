"""Tests for quanttoolbox.backtest.reporting."""

import numpy as np
import pandas as pd

from quanttoolbox.backtest.reporting import (
    backtest_reporting,
    generate_backtest,
    generate_backtest_funded_unfunded,
)


def test_generate_backtest_single_asset_buy_and_hold():
    dates = pd.bdate_range("2023-01-01", periods=20)
    n = len(dates)
    prices = np.linspace(100, 120, n)[:, None]  # rising price
    weights = np.ones((n, 1))  # always 100% invested
    rb_dates = dates[[0]]  # rebalance only at the start

    result = generate_backtest(dates, weights, prices, rb_dates)
    # buy-and-hold in a single rising asset -> backtest should track the
    # price's relative performance exactly (base 100)
    expected = 100 * prices[:, 0] / prices[0, 0]
    assert np.allclose(result.backtest, expected, atol=1e-6)


def test_generate_backtest_two_asset_equal_weight_no_rebalance_drift():
    dates = pd.bdate_range("2023-01-01", periods=10)
    n = len(dates)
    # asset 1 flat, asset 2 doubles
    prices = np.column_stack([np.full(n, 100.0), np.linspace(100, 200, n)])
    weights = np.full((n, 2), 0.5)
    rb_dates = dates[[0]]

    result = generate_backtest(dates, weights, prices, rb_dates)
    # at t=0: 50 units in each leg (50 from asset1 @100 = 0.5 units, 50 from asset2 @100=0.5 units)
    expected_end = 50 * (prices[-1, 0] / prices[0, 0]) + 50 * (prices[-1, 1] / prices[0, 1])
    assert np.isclose(result.backtest[-1], expected_end, atol=1e-6)


def test_generate_backtest_rebalancing_resets_weights():
    dates = pd.bdate_range("2023-01-01", periods=10)
    n = len(dates)
    # asset1 doubles then flat, asset2 flat then doubles
    prices = np.ones((n, 2)) * 100.0
    prices[5:, 0] = 200.0  # asset1 jumps at t=5
    prices[:5, 1] = 100.0
    prices[5:, 1] = 100.0
    weights = np.full((n, 2), 0.5)
    rb_dates = dates[[0, 5]]  # rebalance at start and at t=5 (back to 50/50)

    result = generate_backtest(dates, weights, prices, rb_dates)
    assert not np.any(np.isnan(result.backtest))
    assert result.rebalancing[0, 0] == 1.0  # requested RB flag set at t=0
    assert result.rebalancing[5, 0] == 1.0  # and at t=5


def test_generate_backtest_with_transaction_costs_reduces_wealth():
    dates = pd.bdate_range("2023-01-01", periods=15)
    n = len(dates)
    prices = np.column_stack([np.full(n, 100.0), np.full(n, 100.0)])
    # alternate weights each rebalance to force turnover
    weights = np.zeros((n, 2))
    weights[:5, :] = [1.0, 0.0]
    weights[5:10, :] = [0.0, 1.0]
    weights[10:, :] = [1.0, 0.0]
    rb_dates = dates[[0, 5, 10]]

    result_no_tc = generate_backtest(dates, weights, prices, rb_dates)
    result_tc = generate_backtest(
        dates, weights, prices, rb_dates, tc_bid_ask=np.array([0.01, 0.01])
    )

    # with flat prices, no-TC backtest should stay at 100 throughout;
    # with TC, wealth should be strictly lower after paying transaction costs
    assert np.isclose(result_no_tc.backtest[-1], 100.0, atol=1e-6)
    assert result_tc.backtest[-1] < 100.0
    assert result_tc.turnover is not None
    assert result_tc.transaction_costs is not None


def test_generate_backtest_missing_price_excludes_asset():
    dates = pd.bdate_range("2023-01-01", periods=10)
    n = len(dates)
    prices = np.column_stack([np.full(n, 100.0), np.full(n, 100.0)])
    prices[0, 1] = np.nan  # asset2 has no starting price
    weights = np.full((n, 2), 0.5)
    rb_dates = dates[[0]]

    result = generate_backtest(dates, weights, prices, rb_dates)
    # asset2 excluded -> should behave like 100% in asset1 (flat -> stays 100)
    assert np.allclose(result.backtest, 100.0, atol=1e-6)


def test_generate_backtest_funded_unfunded_funded_only():
    dates = pd.bdate_range("2023-01-01", periods=10)
    n = len(dates)
    prices_funded = np.linspace(100, 110, n)[:, None]
    weights_funded = np.ones((n, 1))
    rb_dates = dates[[0]]

    result = generate_backtest_funded_unfunded(dates, weights_funded, prices_funded, 0, 0, rb_dates)
    expected = 100 * prices_funded[:, 0] / prices_funded[0, 0]
    assert np.allclose(result.backtest, expected, atol=1e-6)


def test_generate_backtest_funded_unfunded_unfunded_only():
    dates = pd.bdate_range("2023-01-01", periods=10)
    n = len(dates)
    # unfunded leg: excess return series
    prices_unfunded = np.linspace(100, 105, n)[:, None]
    weights_unfunded = np.ones((n, 1))
    rb_dates = dates[[0]]

    result = generate_backtest_funded_unfunded(
        dates, 0, 0, weights_unfunded, prices_unfunded, rb_dates
    )
    r_unfunded = prices_unfunded[:, 0] / prices_unfunded[0, 0] - 1.0
    expected = 100 * (1 + r_unfunded)
    assert np.allclose(result.backtest, expected, atol=1e-6)


def test_backtest_reporting_basic_daily():
    dates = pd.bdate_range("2020-01-01", "2023-12-31")
    n = len(dates)
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.01, n)
    returns[0] = 0.0
    backtest = 100 * np.cumprod(1 + returns)

    report = backtest_reporting(dates, backtest)
    assert report is not None
    assert report.frequency_label == "daily"
    assert report.frequency == 260
    assert report.mu.shape[0] == 1
    assert not np.isnan(report.sharpe_ratio[0])


def test_backtest_reporting_with_benchmark_beta_near_one():
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    n = len(dates)
    rng = np.random.default_rng(1)
    market_returns = rng.normal(0.0003, 0.01, n)
    market_returns[0] = 0.0
    benchmark = 100 * np.cumprod(1 + market_returns)
    backtest = benchmark.copy()  # identical -> beta should be 1

    report = backtest_reporting(dates, backtest[:, None], b_index=benchmark)
    assert np.isclose(report.beta[0], 1.0, atol=1e-6)
    assert np.isclose(report.rho[0], 1.0, atol=1e-6)
    assert np.isclose(report.mu_tracking_error[0], 0.0, atol=1e-6)


def test_backtest_reporting_irregular_frequency_returns_none():
    # random irregular dates -> average gap won't classify cleanly
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2020-01-01")
    offsets = np.sort(rng.integers(0, 1000, 50))
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=int(o)) for o in offsets])
    backtest = np.linspace(100, 110, 50)
    report = backtest_reporting(dates, backtest)
    assert report is None
