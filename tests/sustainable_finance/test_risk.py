"""Tests for quanttoolbox.sustainable_finance.risk."""

import numpy as np

from quanttoolbox.sustainable_finance.risk import (
    bond_portfolio_metrics,
    quadratic_form,
    quadratic_form_risk,
)


def test_quadratic_form_matches_hand_computation():
    x = np.array([1.0, 2.0])
    q = np.array([[2.0, 0.0], [0.0, 3.0]])
    r = np.array([0.5, 1.0])
    c = 4.0

    expected = 0.5 * x @ q @ x - x @ r + c
    assert np.isclose(quadratic_form(x, q, r, c), expected)


def test_quadratic_form_risk_single_sector_single_asset():
    sector = np.array([1])
    risk = np.array([2.0])
    risk_star = np.array([3.0])

    result = quadratic_form_risk(sector, risk, risk_star)

    assert result.n == 1
    assert result.n_sector == 1
    assert np.isclose(result.q[0, 0], risk[0] ** 2)
    assert np.isclose(result.r[0], risk[0] * risk_star[0])
    assert np.isclose(result.c, 0.5 * risk_star[0] ** 2)


def test_quadratic_form_risk_two_sectors_sum_to_full_q_r_c():
    sector = np.array([1, 1, 2])
    risk = np.array([1.0, 2.0, 3.0])
    risk_star = np.array([0.5, 1.5])  # one target per sector

    result = quadratic_form_risk(sector, risk, risk_star)

    # sector contributions should sum to the totals
    assert np.allclose(result.q_j.sum(axis=2), result.q)
    assert np.allclose(result.r_j.sum(axis=1), result.r)
    assert np.isclose(result.c_j.sum(), result.c)

    # sector 1 (indices 0, 1): s_j_risk = [1, 2, 0]
    expected_q_sector1 = np.outer([1.0, 2.0, 0.0], [1.0, 2.0, 0.0])
    assert np.allclose(result.q_j[:, :, 0], expected_q_sector1)


def test_quadratic_form_risk_qf_evaluates_at_w_when_given():
    sector = np.array([1, 2])
    risk = np.array([2.0, 3.0])
    risk_star = np.array([1.0, 1.0])
    w = np.array([0.5, 0.5])

    result = quadratic_form_risk(sector, risk, risk_star, w)
    expected = quadratic_form(w, result.q, result.r, result.c)
    assert np.isclose(result.qf, expected)


def test_quadratic_form_risk_qf_is_nan_without_w():
    result = quadratic_form_risk(np.array([1]), np.array([1.0]), np.array([1.0]))
    assert np.isnan(result.qf)


def test_bond_portfolio_metrics_matches_hand_computation():
    sector = np.array([1, 1, 2])
    md = np.array([2.0, 4.0, 6.0])
    dts = np.array([1.0, 1.0, 2.0])
    w = np.array([0.5, 0.25, 0.25])

    metrics = bond_portfolio_metrics(sector, md, dts, w)

    assert np.isclose(metrics.md_portfolio, np.sum(w * md))
    assert np.isclose(metrics.dts_portfolio, np.sum(w * dts))
    assert np.allclose(metrics.unique_sector, [1, 2])
    # sector 1 = indices 0, 1
    assert np.isclose(metrics.md_by_sector[0], w[0] * md[0] + w[1] * md[1])
    assert np.isclose(metrics.dts_by_sector[0], w[0] * dts[0] + w[1] * dts[1])
    # sector 2 = index 2
    assert np.isclose(metrics.md_by_sector[1], w[2] * md[2])
    # sector sums should match the portfolio total
    assert np.isclose(metrics.md_by_sector.sum(), metrics.md_portfolio)
    assert np.isclose(metrics.dts_by_sector.sum(), metrics.dts_portfolio)
