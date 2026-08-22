"""Tests for quanttoolbox.sustainable_finance.impact."""

import numpy as np

from quanttoolbox.sustainable_finance.impact import blended_finance_simulation


def _reference_check_series(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default):
    """Re-derive the primary series via the original MATLAB function's
    indicator-based "_check" formulas (cnd1/cnd2), as an independent
    cross-check of the min/max/where-based implementation under test."""
    nt_default = np.asarray(nt_default, dtype=float)
    a_junior = 0.0
    a_senior = d_junior
    d_senior = n0
    nt_default_junior = d_junior / ni

    lt = ni * (1.0 - ri) * nt_default

    cnd1 = (a_junior <= lt) & (lt <= d_junior)
    cnd2 = lt > d_junior
    lt_junior_check = (lt - a_junior) * cnd1 + (d_junior - a_junior) * cnd2

    cnd1 = (a_senior <= lt) & (lt <= d_senior)
    cnd2 = lt > d_senior
    lt_senior_check = (lt - a_senior) * cnd1 + (d_senior - a_senior) * cnd2

    n_loans = n0 / ni
    delta_gt = ci * ni * (n_loans - nt_default)
    minimum_income_senior = coupon_rate_senior * (d_senior - a_senior)
    delta_gt_junior_check = (delta_gt - minimum_income_senior) * (nt_default <= nt_default_junior)
    delta_gt_senior_check = delta_gt - delta_gt_junior_check

    return lt_junior_check, lt_senior_check, delta_gt_junior_check, delta_gt_senior_check


def test_matches_matlab_indicator_based_check_formulas():
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [0, 1, 2, 2, 4, 5, 5, 6, 7, 8]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)
    lt_junior_check, lt_senior_check, delta_gt_junior_check, delta_gt_senior_check = (
        _reference_check_series(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)
    )

    assert np.allclose(result.lt_junior, lt_junior_check)
    assert np.allclose(result.lt_senior, lt_senior_check)
    assert np.allclose(result.gt_junior, np.cumsum(delta_gt_junior_check))
    assert np.allclose(result.gt_senior, np.cumsum(delta_gt_senior_check))


def test_losses_and_income_reconcile_to_the_portfolio_totals():
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [4, 7, 10, 20, 24, 27, 29, 30, 33, 34]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)

    assert np.allclose(result.lt_junior + result.lt_senior, result.lt)
    assert np.allclose(result.gt_junior + result.gt_senior, result.gt)
    assert np.allclose(result.t, np.arange(1, 11))


def test_mild_scenario_matches_notebook_output():
    # Real regression data from HSF-Notebooks 05a_impact_market_instruments.ipynb.
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [0, 1, 2, 2, 4, 5, 5, 6, 7, 8]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)

    assert np.isclose(100 * result.rt_bar[-1], 8.80, atol=5e-3)
    assert np.isclose(100 * result.rt_bar_junior[-1], 28.00, atol=5e-3)
    assert np.isclose(100 * result.rt_bar_senior[-1], 4.00, atol=5e-3)


def test_severe_scenario_matches_notebook_output():
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [6, 10, 16, 20, 25, 30, 32, 36, 40, 41]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)

    assert np.isclose(100 * result.rt_bar[-1], 3.34, atol=5e-3)
    assert np.isclose(100 * result.rt_bar_junior[-1], 1.00, atol=5e-3)
    assert np.isclose(100 * result.rt_bar_senior[-1], 3.92, atol=5e-3)


def test_moderate_scenario_matches_notebook_output():
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [4, 7, 10, 20, 24, 27, 29, 30, 33, 34]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)

    assert np.isclose(100 * result.rt_bar[-1], 4.42, atol=5e-3)
    assert np.isclose(100 * result.rt_bar_junior[-1], 1.55, atol=5e-3)
    assert np.isclose(100 * result.rt_bar_senior[-1], 5.14, atol=5e-3)


def test_senior_gets_only_minimum_income_while_junior_buffer_intact():
    # nt_default_junior = D_junior / Ni = 100 / 5 = 20. In the mild scenario
    # cumulative defaults never exceed 8 < 20, so the senior tranche should
    # never receive more than its guaranteed minimum coupon per period.
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [0, 1, 2, 2, 4, 5, 5, 6, 7, 8]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)

    minimum_income_senior = coupon_rate_senior * (n0 - d_junior)
    expected_gt_senior = np.cumsum(np.full(len(nt_default), minimum_income_senior))
    assert np.allclose(result.gt_senior, expected_gt_senior)


def test_no_defaults_gives_zero_loss_and_full_coupon_return():
    n0, d_junior, ni, ri, ci, coupon_rate_senior = 500.0, 100.0, 5.0, 0.0, 0.10, 0.04
    nt_default = [0, 0, 0]

    result = blended_finance_simulation(n0, d_junior, ni, ri, ci, coupon_rate_senior, nt_default)

    assert np.allclose(result.lt, 0.0)
    n_loans = n0 / ni
    expected_gt = np.cumsum(np.full(3, ci * ni * n_loans))
    assert np.allclose(result.gt, expected_gt)
    assert np.allclose(result.rt, expected_gt / n0)
