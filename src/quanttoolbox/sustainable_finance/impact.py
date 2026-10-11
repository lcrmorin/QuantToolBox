"""Allocate blended-finance losses and coupons to junior and senior tranches.

Original: ``HSF/5. Impact Investing/blended_finance_simulation.m``,
adapted via HSF-Notebooks chapter 05a.

While cumulative defaults satisfy ``nt_default <= D_junior / Ni``, the
senior tranche receives its minimum coupon and the junior tranche receives
the residual. Once the junior loss buffer is exhausted, all coupon income
goes to the senior tranche.

Only the primary cash-flow series are returned. The original's redundant
indicator-based check series are covered by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlendedFinanceResult:
    """Per-period cash-flow waterfall series for a two-tranche
    blended-finance portfolio: cumulative losses (`lt`), cumulative
    coupon income (`gt`), cumulative return (`rt`), and annualized return
    (`rt_bar`), each for the portfolio as a whole and for the junior and
    senior tranches individually. `t` holds the period index (1-based),
    matching the length of `nt_default`.
    """

    t: np.ndarray
    lt: np.ndarray
    lt_junior: np.ndarray
    lt_senior: np.ndarray
    gt: np.ndarray
    gt_junior: np.ndarray
    gt_senior: np.ndarray
    rt: np.ndarray
    rt_junior: np.ndarray
    rt_senior: np.ndarray
    rt_bar: np.ndarray
    rt_bar_junior: np.ndarray
    rt_bar_senior: np.ndarray


def blended_finance_simulation(
    n0: float,
    d_junior: float,
    ni: float,
    ri: float,
    ci: float,
    coupon_rate_senior: float,
    nt_default: np.ndarray,
) -> BlendedFinanceResult:
    """Two-tranche (junior/senior) cash-flow waterfall for a blended-finance
    portfolio, given a cumulative-default path `nt_default` (one entry per
    period).

    `n0` is the total portfolio notional, `d_junior` the junior
    (first-loss) tranche size, `ni` the notional per underlying loan/bond,
    `ri` the recovery rate on default, `ci` the coupon rate paid by the
    underlying loans/bonds, and `coupon_rate_senior` the minimum coupon
    rate promised to the senior tranche. Losses are allocated first to the
    junior tranche up to its size, with any excess passed to the senior
    tranche; coupon income is split between the two tranches as described
    in the module docstring.

    Original: HSF/5. Impact Investing/blended_finance_simulation.m
    """
    nt_default = np.asarray(nt_default, dtype=float)
    t = np.arange(1, nt_default.shape[0] + 1)

    n_loans = n0 / ni
    a_junior = 0.0
    a_senior = d_junior
    d_senior = n0
    nt_default_junior = d_junior / ni

    lt = ni * (1.0 - ri) * nt_default
    lt_junior = np.minimum(lt, d_junior)
    lt_senior = np.maximum(lt - a_senior, 0.0)

    delta_gt = ci * ni * (n_loans - nt_default)
    minimum_income_senior = coupon_rate_senior * (d_senior - a_senior)
    delta_gt_senior = np.where(nt_default <= nt_default_junior, minimum_income_senior, delta_gt)
    delta_gt_junior = delta_gt - delta_gt_senior

    gt = np.cumsum(delta_gt)
    gt_junior = np.cumsum(delta_gt_junior)
    gt_senior = np.cumsum(delta_gt_senior)

    rt = (gt - lt) / n0
    rt_junior = (gt_junior - lt_junior) / (d_junior - a_junior)
    rt_senior = (gt_senior - lt_senior) / (d_senior - a_senior)

    rt_bar = rt / t
    rt_bar_junior = rt_junior / t
    rt_bar_senior = rt_senior / t

    return BlendedFinanceResult(
        t=t,
        lt=lt,
        lt_junior=lt_junior,
        lt_senior=lt_senior,
        gt=gt,
        gt_junior=gt_junior,
        gt_senior=gt_senior,
        rt=rt,
        rt_junior=rt_junior,
        rt_senior=rt_senior,
        rt_bar=rt_bar,
        rt_bar_junior=rt_bar_junior,
        rt_bar_senior=rt_bar_senior,
    )
