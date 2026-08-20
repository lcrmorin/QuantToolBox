"""Backtest simulation engines and comprehensive performance reporting.

Ported from QuantToolbox/backtest/{generate_backtest,generate_backtest2,
backtest_reporting}.m

Translation notes:

- ``generate_backtest`` (buy-and-hold-between-rebalances simulator, with
  optional per-asset bid/ask transaction costs) is ported preserving its
  loop-over-rebalancing-periods structure, since each period's asset
  universe (nonzero weights, valid prices) can differ and doesn't
  vectorize cleanly across periods.
- ``backtest_reporting`` is entirely numeric in the original (it builds up
  a results struct); the MATLAB `disp`/printing that would normally
  accompany a report script isn't part of this function, so nothing was
  dropped here -- the returned ``BacktestReport`` dataclass holds every
  field the original struct held.
- Frequency auto-detection (daily/weekly/monthly, from the average gap
  between dates) is preserved exactly, including the original's
  hard failure (empty result) for irregular calendars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quanttoolbox.backtest.stats import (
    MonthlyStats,
    YearlyStats,
    maximum_drawdown,
    monthly_statistics,
    price_to_return_1d,
    yearly_statistics,
)


@dataclass
class BacktestResult:
    backtest: np.ndarray  # (n_dates,) wealth index, base 100
    rebalancing: np.ndarray  # (n_dates, 2): [requested RB flag, effective RB flag]
    turnover: np.ndarray | None  # (n_dates,), only if transaction costs used
    transaction_costs: np.ndarray | None  # (n_dates,), only if transaction costs used


def generate_backtest(
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
    prices: np.ndarray,
    rb_dates: pd.DatetimeIndex | np.ndarray,
    tc_bid_ask: np.ndarray | None = None,
) -> BacktestResult:
    """Simulate a buy-and-hold-between-rebalances portfolio backtest.

    On each rebalancing date, the portfolio is reset to the target
    `weights` for that date (using prices at that date), then held
    (drifting with asset prices) until the next rebalancing date.
    Assets with NaN price or NaN/zero weight at a rebalance are excluded
    for that period. If `tc_bid_ask` is given, per-unit bid/ask
    transaction costs are charged on turnover at each rebalance, and the
    simulation tracks actual unit holdings rather than pure weights.

    Original: backtest/generate_backtest.m

    Parameters
    ----------
    dates : full daily calendar of the backtest.
    weights : (n_dates, n_assets) target weights (only rows at rebalancing
        dates are used).
    prices : (n_dates, n_assets) asset price levels.
    rb_dates : rebalancing dates, either as a DatetimeIndex (subset of
        `dates`) or a boolean/0-1 mask array aligned with `dates`.
    tc_bid_ask : optional (n_assets,) or (n_assets, 2) [bid, ask] per-unit
        transaction cost rates.
    """
    dates = pd.DatetimeIndex(dates)
    weights = np.asarray(weights, dtype=float)
    prices = np.asarray(prices, dtype=float)
    n_dates = dates.shape[0]
    n_assets = weights.shape[1]

    rb = np.zeros(n_dates, dtype=bool)
    if isinstance(rb_dates, pd.DatetimeIndex | pd.Series) or (
        isinstance(rb_dates, np.ndarray) and np.issubdtype(rb_dates.dtype, np.datetime64)
    ):
        rb[dates.isin(pd.DatetimeIndex(rb_dates))] = True
    else:
        rb_arr = np.asarray(rb_dates)
        if rb_arr.shape[0] == n_dates:
            rb = rb_arr.astype(bool)
        else:
            rb[rb_arr.astype(int)] = True
    rb[-1] = True

    rb_idx = np.where(rb)[0]
    n_rb = rb_idx.shape[0]
    rb_effective = np.zeros(n_dates, dtype=bool)

    use_tc = tc_bid_ask is not None
    if use_tc:
        tc_bid_ask = np.asarray(tc_bid_ask, dtype=float)
        if tc_bid_ask.ndim == 2 and tc_bid_ask.shape[1] == 2:
            if tc_bid_ask.shape[0] == 1:
                tc_bid = np.full(n_assets, tc_bid_ask[0, 0])
                tc_ask = np.full(n_assets, tc_bid_ask[0, 1])
            else:
                tc_bid = tc_bid_ask[:, 0]
                tc_ask = tc_bid_ask[:, 1]
        else:
            tc_flat = tc_bid_ask.flatten()
            tc_bid = np.full(n_assets, tc_flat[0]) if tc_flat.shape[0] == 1 else tc_flat
            tc_ask = tc_bid.copy()

    backtest = np.full(n_dates, np.nan)
    turnover = np.zeros(n_dates) if use_tc else np.full(n_dates, np.nan)
    tc_series = np.zeros(n_dates) if use_tc else np.full(n_dates, np.nan)
    wealth = 100.0

    indx_t_begin = rb_idx[0]
    if indx_t_begin > 0:
        rb[:indx_t_begin] = False
        if use_tc:
            turnover[:indx_t_begin] = np.nan
            tc_series[:indx_t_begin] = np.nan

    n_previous = None
    rebalancing_begin = True

    for i in range(n_rb - 1):
        rb_effective[indx_t_begin] = True
        indx_t_end = rb_idx[i + 1]
        indx_t = np.arange(indx_t_begin, indx_t_end + 1)

        w_t = weights[indx_t_begin, :].copy()
        nonzero = np.where(w_t != 0)[0]
        if nonzero.size == 0:
            indx_t_begin = indx_t_end
            continue

        if use_tc:
            active = np.arange(n_assets)
        else:
            active = nonzero
            w_t = w_t[active]

        isnan_w = np.isnan(w_t)
        p_begin = prices[indx_t_begin, active]
        p_window = prices[np.ix_(indx_t, active)] / p_begin[None, :]
        p_end = prices[indx_t_end, active]
        isnan_begin = np.isnan(p_begin)
        isnan_end = np.isnan(p_end)

        zero_mask = isnan_begin | isnan_end | isnan_w
        if np.any(zero_mask):
            w_t = np.where(zero_mask, 0.0, w_t)
            p_window = p_window.copy()
            p_window[:, zero_mask] = 1.0

        w_sum = np.sum(w_t)
        if w_sum != 0:
            w_t = w_t / w_sum

        if use_tc:
            n_t_units = (wealth * w_t) / p_window[0, :]
            if i == 0 or rebalancing_begin:
                tc_t, to_t = 0.0, 0.0
                rebalancing_begin = False
            else:
                tc_bid_t = np.maximum(n_t_units - n_previous, 0.0) * p_window[0, :] * tc_bid
                tc_ask_t = np.maximum(n_previous - n_t_units, 0.0) * p_window[0, :] * tc_ask
                tc_t = np.sum(tc_bid_t + tc_ask_t)
                to_t = np.sum(np.abs(n_t_units - n_previous) * p_window[0, :]) / wealth
            backtest_t = p_window @ n_t_units - tc_t
            backtest[indx_t] = backtest_t
            n_previous = n_t_units
            turnover[indx_t[0]] = to_t
            tc_series[indx_t[0]] = tc_t
        else:
            backtest_t = p_window @ w_t
            backtest[indx_t] = wealth * backtest_t

        wealth = backtest[indx_t[-1]]
        indx_t_begin = indx_t_end

    rebalancing = np.column_stack([rb, rb_effective]).astype(float)
    return BacktestResult(
        backtest=backtest,
        rebalancing=rebalancing,
        turnover=turnover if use_tc else None,
        transaction_costs=tc_series if use_tc else None,
    )


def generate_backtest_funded_unfunded(
    dates: pd.DatetimeIndex,
    weights_funded: np.ndarray | float,
    prices_funded: np.ndarray | float,
    weights_unfunded: np.ndarray | float,
    prices_unfunded: np.ndarray | float,
    rb_dates: pd.DatetimeIndex | np.ndarray,
) -> BacktestResult:
    """Simulate a backtest combining a "funded" (price-return) leg and an
    "unfunded" (excess-return, i.e. financed) leg, rebalanced together.

    Pass ``weights_funded=0`` (or ``weights_unfunded=0``) to disable that
    leg entirely, matching the original's scalar-zero sentinel convention.

    Original: backtest/generate_backtest2.m
    """
    dates = pd.DatetimeIndex(dates)
    n_dates = dates.shape[0]

    rb = np.zeros(n_dates, dtype=bool)
    if isinstance(rb_dates, pd.DatetimeIndex | pd.Series) or (
        isinstance(rb_dates, np.ndarray) and np.issubdtype(rb_dates.dtype, np.datetime64)
    ):
        rb[dates.isin(pd.DatetimeIndex(rb_dates))] = True
    else:
        rb_arr = np.asarray(rb_dates)
        if rb_arr.shape[0] == n_dates:
            rb = rb_arr.astype(bool)
        else:
            rb[rb_arr.astype(int)] = True
    rb[-1] = True

    rb_idx = np.where(rb)[0]
    n_rb = rb_idx.shape[0]
    rb_effective = np.zeros(n_dates, dtype=bool)

    def _prep(
        weights: np.ndarray | float, prices: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if np.isscalar(weights) and weights == 0:
            return np.ones((n_dates, 1)), np.ones((n_dates, 1)), 1
        w = np.asarray(weights, dtype=float)
        p = np.asarray(prices, dtype=float)
        return w, p, w.shape[1]

    weights_funded, prices_funded, n_funded = _prep(weights_funded, prices_funded)
    weights_unfunded, prices_unfunded, n_unfunded = _prep(weights_unfunded, prices_unfunded)

    backtest = np.full(n_dates, np.nan)
    wealth = 100.0
    indx_t_begin = rb_idx[0]
    if indx_t_begin > 0:
        rb[:indx_t_begin] = False

    for i in range(n_rb - 1):
        rb_effective[indx_t_begin] = True
        indx_t_end = rb_idx[i + 1]
        indx_t = np.arange(indx_t_begin, indx_t_end + 1)

        w_funded = weights_funded[indx_t_begin, :]
        p_begin_funded = prices_funded[indx_t_begin, :]
        p_funded = prices_funded[indx_t, :] / p_begin_funded[None, :]
        n_units_funded = (wealth * w_funded) / p_funded[0, :]
        backtest_t = p_funded @ n_units_funded

        w_unfunded = weights_unfunded[indx_t_begin, :]
        wealth_unfunded_t = wealth * w_unfunded
        p_begin_unfunded = prices_unfunded[indx_t_begin, :]
        r_unfunded = prices_unfunded[indx_t, :] / p_begin_unfunded[None, :] - 1.0
        backtest_t = backtest_t + r_unfunded @ wealth_unfunded_t

        backtest[indx_t] = backtest_t
        wealth = backtest[indx_t[-1]]
        indx_t_begin = indx_t_end

    rebalancing = np.column_stack([rb, rb_effective]).astype(float)
    return BacktestResult(
        backtest=backtest, rebalancing=rebalancing, turnover=None, transaction_costs=None
    )


@dataclass
class BacktestReport:
    frequency_label: str
    frequency: int
    begin_date: int
    end_date: int
    time_period_years: float
    mu: np.ndarray
    mu_benchmark: float
    mu_risk_free: float
    mu_tracking_error: np.ndarray
    sigma: np.ndarray
    sigma_benchmark: float
    sigma_risk_free: float
    sigma_tracking_error: np.ndarray
    sharpe_ratio: np.ndarray
    sharpe_ratio_benchmark: float
    information_ratio: np.ndarray
    beta: np.ndarray
    rho: np.ndarray
    max_dd: np.ndarray
    max_dd_benchmark: float
    monthly_stats: MonthlyStats
    yearly_stats: YearlyStats
    yearly_stats_te: YearlyStats


def _detect_frequency(dates: pd.DatetimeIndex) -> tuple[int, str] | None:
    gaps = np.diff(dates.to_numpy().astype("datetime64[D]").astype(float))
    mean_gap = np.mean(gaps)
    if 0.9 < mean_gap < 1.7:
        return 260, "daily"
    if 5 < mean_gap < 9:
        return 52, "weekly"
    if 27 < mean_gap < 33:
        return 12, "monthly"
    return None


def backtest_reporting(
    dates: pd.DatetimeIndex,
    backtest: np.ndarray,
    b_index: np.ndarray | None = None,
    r_index: np.ndarray | None = None,
    begin_date: pd.Timestamp | int = 0,
    end_date: pd.Timestamp | int = 0,
) -> BacktestReport | None:
    """Comprehensive performance report: annualized return/vol, Sharpe and
    information ratios, beta/correlation to a benchmark, maximum drawdown,
    and monthly/yearly breakdowns.

    Original: backtest/backtest_reporting.m

    Returns None if the input calendar's frequency can't be classified as
    daily/weekly/monthly (matching the original's early-return behavior).
    """
    dates = pd.DatetimeIndex(dates)
    backtest = np.asarray(backtest, dtype=float)
    if backtest.ndim == 1:
        backtest = backtest[:, None]
    n_dates = dates.shape[0]

    freq_result = _detect_frequency(dates)
    if freq_result is None:
        return None
    frequency, freq_label = freq_result

    begin = dates[0] if begin_date in (0, None) else pd.Timestamp(begin_date)
    end = dates[-1] if end_date in (0, None) else pd.Timestamp(end_date)

    use_benchmark = b_index is not None and not np.isscalar(b_index)
    if b_index is None or np.isscalar(b_index):
        b_index = np.full(n_dates, 100.0)
    else:
        b_index = np.asarray(b_index, dtype=float).flatten()
    if r_index is None or np.isscalar(r_index):
        r_index = np.full(n_dates, 100.0)
    else:
        r_index = np.asarray(r_index, dtype=float).flatten()

    monthly_stats, yearly_stats = monthly_statistics(dates, backtest, begin, end)
    yearly_stats_te = yearly_statistics(dates, backtest, b_index, begin, end)

    valid_bt = ~np.isnan(backtest).all(axis=1)
    first_valid = np.argmax(valid_bt)
    last_valid = n_dates - 1 - np.argmax(valid_bt[::-1])
    begin = max(begin, dates[first_valid])
    end = min(end, dates[last_valid])

    if use_benchmark:
        valid_b = ~np.isnan(b_index)
        first_valid_b = np.argmax(valid_b)
        last_valid_b = n_dates - 1 - np.argmax(valid_b[::-1])
        begin = max(begin, dates[first_valid_b])
        end = min(end, dates[last_valid_b])

    mask = (dates >= begin) & (dates <= end)
    dates_m = dates[mask]
    backtest_m = backtest[mask]
    b_index_m = b_index[mask]
    r_index_m = r_index[mask]

    dt_years = (dates_m[-1] - dates_m[0]).days / 365.25

    mu = (backtest_m[-1, :] / backtest_m[0, :]) ** (1 / dt_years) - 1.0
    mu_r = (r_index_m[-1] / r_index_m[0]) ** (1 / dt_years) - 1.0
    if use_benchmark:
        mu_b = (b_index_m[-1] / b_index_m[0]) ** (1 / dt_years) - 1.0
        mu_te = ((backtest_m[-1, :] / backtest_m[0, :]) / (b_index_m[-1] / b_index_m[0])) ** (
            1 / dt_years
        ) - 1.0
    else:
        mu_b, mu_te = np.nan, np.full(backtest_m.shape[1], np.nan)

    r_bt = price_to_return_1d_columns(backtest_m)
    r_r = price_to_return_1d(r_index_m)
    r_b = price_to_return_1d(b_index_m)
    e = r_bt - r_b[:, None]

    sigma = np.sqrt(frequency) * np.nanstd(r_bt, axis=0, ddof=1)
    sigma_r = np.sqrt(frequency) * np.nanstd(r_r, ddof=1)
    if use_benchmark:
        sigma_b = np.sqrt(frequency) * np.nanstd(r_b, ddof=1)
        sigma_te = np.sqrt(frequency) * np.nanstd(e, axis=0, ddof=1)
    else:
        sigma_b, sigma_te = np.nan, np.full(backtest_m.shape[1], np.nan)

    sharpe_ratio = (mu - mu_r) / sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        if use_benchmark:
            sharpe_ratio_b = (mu_b - mu_r) / sigma_b
            information_ratio = mu_te / sigma_te
        else:
            sharpe_ratio_b, information_ratio = np.nan, np.full(backtest_m.shape[1], np.nan)

    joint_r = np.column_stack([r_b, r_bt])
    valid_r = ~np.isnan(joint_r).any(axis=1)
    joint_r = joint_r[valid_r]
    cov = np.cov(joint_r, rowvar=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cov[1:, 0] / cov[0, 0]
        rho = cov[1:, 0] / (np.sqrt(cov[0, 0]) * np.sqrt(np.diag(cov)[1:]))

    max_dd, *_ = maximum_drawdown(backtest_m, relative=True)
    if use_benchmark:
        max_dd_b_arr, *_ = maximum_drawdown(b_index_m[:, None], relative=True)
        max_dd_b: float = float(max_dd_b_arr[0])
    else:
        max_dd_b = float("nan")

    return BacktestReport(
        frequency_label=freq_label,
        frequency=frequency,
        begin_date=int(dates_m[0].year * 10000 + dates_m[0].month * 100 + dates_m[0].day),
        end_date=int(dates_m[-1].year * 10000 + dates_m[-1].month * 100 + dates_m[-1].day),
        time_period_years=dt_years,
        mu=mu,
        mu_benchmark=mu_b,
        mu_risk_free=mu_r,
        mu_tracking_error=mu_te,
        sigma=sigma,
        sigma_benchmark=sigma_b,
        sigma_risk_free=sigma_r,
        sigma_tracking_error=sigma_te,
        sharpe_ratio=sharpe_ratio,
        sharpe_ratio_benchmark=sharpe_ratio_b,
        information_ratio=information_ratio,
        beta=beta,
        rho=rho,
        max_dd=max_dd,
        max_dd_benchmark=max_dd_b,
        monthly_stats=monthly_stats,
        yearly_stats=yearly_stats,
        yearly_stats_te=yearly_stats_te,
    )


def price_to_return_1d_columns(x: np.ndarray) -> np.ndarray:
    """Column-wise version of price_to_return_1d for a 2-D (n_dates, n_cols) array."""
    y = np.full_like(x, np.nan)
    y[1:, :] = x[1:, :] / x[:-1, :] - 1.0
    return y
