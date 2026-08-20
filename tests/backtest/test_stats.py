"""Tests for quanttoolbox.backtest.stats."""

import numpy as np
import pandas as pd

from quanttoolbox.backtest.stats import (
    annualized_turnover,
    average_return,
    index_repeated_data,
    maximum_drawdown,
    monthly_statistics,
    static_turnover,
    yearly_statistics,
)


def test_maximum_drawdown_known_series():
    # peak at 110 (row1), trough at 90 (row3): drawdown = -20 absolute, or -18.18% relative
    x = np.array([[100.0], [110.0], [95.0], [90.0], [105.0]])
    max_dd, start_dd, end_dd, tau_dd = maximum_drawdown(x, relative=False)
    assert np.isclose(max_dd[0], -20.0)
    assert end_dd[0] == 3
    assert start_dd[0] == 1


def test_maximum_drawdown_relative():
    x = np.array([[100.0], [110.0], [95.0], [90.0], [105.0]])
    max_dd, start_dd, end_dd, tau_dd = maximum_drawdown(x, relative=True)
    assert np.isclose(max_dd[0], -20.0 / 110.0, atol=1e-6)


def test_maximum_drawdown_no_drawdown():
    x = np.array([[100.0], [101.0], [102.0], [103.0]])
    max_dd, *_ = maximum_drawdown(x)
    assert np.isclose(max_dd[0], 0.0)


def test_static_turnover_single_series():
    x = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
    tau = static_turnover(x)
    assert tau.shape[0] == 2  # n-1 transitions
    assert np.isclose(tau[0], 0.1 + 0.1)
    assert np.isclose(tau[1], 0.2 + 0.2)


def test_static_turnover_two_series():
    x = np.array([0.3, 0.7])
    y = np.array([0.5, 0.5])
    tau = static_turnover(x, y)
    assert np.isclose(tau, 0.2 + 0.2)


def test_annualized_turnover_whole_sample():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    turnover = np.full((len(dates), 1), 0.01)  # constant daily turnover
    tau = annualized_turnover(dates, turnover, by_year=False)
    dt_years = (dates[-1] - dates[0]).days / 365.25
    expected = (0.01 * len(dates)) / dt_years
    assert np.isclose(tau[0], expected, rtol=1e-6)


def test_annualized_turnover_by_year():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    turnover = np.full((len(dates), 1), 0.01)
    tau = annualized_turnover(dates, turnover, by_year=True)
    assert tau.shape[0] == 2  # 2020, 2021


def test_average_return_basic():
    r = np.array([[0.01], [0.02], [0.03], [0.04]])
    avg = average_return(r, n_lags=2)
    assert np.isnan(avg[0, 0])
    assert np.isclose(avg[1, 0], (0.01 + 0.02) / 2)
    assert np.isclose(avg[2, 0], (0.02 + 0.03) / 2)
    assert np.isclose(avg[3, 0], (0.03 + 0.04) / 2)


def test_index_repeated_data():
    x = np.array([1.001, 1.001, 1.002, 1.002, 1.002, 1.005])
    idx_yes, idx_no, cnd = index_repeated_data(x, precision=2)
    # rounded to 2 decimals: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] -- all identical after rounding
    assert not cnd[0]  # first row never counted as repeated
    assert np.all(cnd[1:])  # all subsequent rows repeat at this precision


def test_index_repeated_data_high_precision_no_repeats():
    x = np.array([1.001, 1.002, 1.003, 1.004])
    idx_yes, idx_no, cnd = index_repeated_data(x, precision=3)
    assert idx_yes.shape[0] == 0  # no repeats at full precision


def test_monthly_statistics_shapes():
    dates = pd.bdate_range("2022-01-01", "2023-12-31")
    n = len(dates)
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0003, 0.01, n)
    returns[0] = 0.0
    backtest = 100 * np.cumprod(1 + returns)

    monthly_stats, yearly_stats = monthly_statistics(dates, backtest[:, None])
    assert monthly_stats.mu.shape[1] == 1
    assert yearly_stats.mu.shape[0] == 2  # 2022, 2023
    # most months should have a valid (non-nan) return
    assert np.sum(~np.isnan(monthly_stats.mu)) > 20


def test_yearly_statistics_beta_one_when_identical_to_benchmark():
    dates = pd.bdate_range("2022-01-01", "2022-12-31")
    n = len(dates)
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.01, n)
    returns[0] = 0.0
    backtest = 100 * np.cumprod(1 + returns)
    benchmark = backtest.copy()  # identical to the backtest itself

    result = yearly_statistics(dates, backtest[:, None], benchmark)
    assert np.allclose(result.beta[~np.isnan(result.beta)], 1.0, atol=1e-6)
    assert np.allclose(result.rho[~np.isnan(result.rho)], 1.0, atol=1e-6)
    assert np.allclose(result.mu_te[~np.isnan(result.mu_te)], 0.0, atol=1e-6)
