"""Tests for quanttoolbox.bond.pricing."""

import numpy as np
import pytest

from quanttoolbox.bond.pricing import (
    bond_convexity,
    bond_duration,
    bond_portfolio_quadratic_form,
    bond_portfolio_quadratic_form_vs_benchmark,
    bond_price,
    bond_ytm,
    coupon_yield,
)
from quanttoolbox.config import BisectionConfig


def test_bond_price_zero_coupon_continuous_matches_closed_form():
    price = bond_price(5.0, [100.0], rate=0.05, method=1)
    assert np.isclose(price, 100.0 * np.exp(-0.05 * 5.0))


def test_bond_price_zero_coupon_discrete_matches_closed_form():
    price = bond_price(5.0, [100.0], rate=0.05, method=2)
    assert np.isclose(price, 100.0 / (1.05**5))


def test_bond_price_sums_multiple_cash_flows():
    t = np.array([1.0, 2.0, 3.0])
    ct = np.array([5.0, 5.0, 105.0])
    rate = 0.04
    price = bond_price(t, ct, rate, method=1)
    expected = np.sum(ct * np.exp(-t * rate))
    assert np.isclose(price, expected)


def test_bond_price_higher_rate_gives_lower_price():
    t = np.array([1.0, 2.0, 3.0])
    ct = np.array([5.0, 5.0, 105.0])
    assert bond_price(t, ct, 0.08, method=1) < bond_price(t, ct, 0.02, method=1)


def test_bond_ytm_round_trips_a_known_rate():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ct = np.array([5.0, 5.0, 5.0, 5.0, 105.0])
    true_rate = 0.037
    price = bond_price(t, ct, true_rate, method=1)

    ytm = bond_ytm(t, ct, price)
    assert np.isclose(ytm, true_rate, atol=1e-4)


def test_bond_ytm_round_trips_a_known_rate_discrete_compounding():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ct = np.array([5.0, 5.0, 5.0, 5.0, 105.0])
    true_rate = 0.037
    price = bond_price(t, ct, true_rate, method=2)

    ytm = bond_ytm(t, ct, price, method=2)
    assert np.isclose(ytm, true_rate, atol=1e-4)


def test_bond_ytm_zero_coupon():
    t = 10.0
    ct = [100.0]
    true_rate = 0.06
    price = bond_price(t, ct, true_rate, method=1)

    ytm = bond_ytm(t, ct, price)
    assert np.isclose(ytm, true_rate, atol=1e-4)


def test_coupon_yield_matches_direct_formula():
    t = np.array([1.0, 2.0, 3.0])
    ct = np.array([5.0, 5.0, 105.0])
    rate = 0.03

    yield_ = coupon_yield(t, ct, rate)
    price = bond_price(t, ct, rate, method=1)
    assert np.isclose(yield_, ct[0] / price)


def test_bond_portfolio_quadratic_form_isolates_md_when_dts_weight_is_zero():
    sector = np.array([1, 1, 2])
    md = np.array([2.0, 4.0, 6.0])
    md_star = np.array([3.0, 6.0])
    dts = np.array([1.0, 1.0, 2.0])
    dts_star = np.array([1.0, 2.0])
    carry = np.array([0.01, 0.02, 0.03])
    w = np.array([0.5, 0.25, 0.25])

    result = bond_portfolio_quadratic_form(
        sector,
        varphi_md=1.0,
        md=md,
        md_star=md_star,
        varphi_dts=0.0,
        dts=dts,
        dts_star=dts_star,
        gamma_carry=0.0,
        carry=carry,
        w=w,
    )

    assert np.allclose(result.q, result.md.q)
    assert np.allclose(result.r, result.md.r)
    assert np.isclose(result.c, result.md.c)
    assert np.isclose(result.qf, result.md.qf)


def test_bond_portfolio_quadratic_form_matches_manual_combination():
    sector = np.array([1, 2])
    md = np.array([2.0, 3.0])
    md_star = np.array([1.0, 1.0])
    dts = np.array([0.5, 0.5])
    dts_star = np.array([0.2, 0.2])
    gamma_carry = 2.0
    carry = np.array([0.01, 0.02])
    w = np.array([0.6, 0.4])
    varphi_md, varphi_dts = 0.7, 0.3

    result = bond_portfolio_quadratic_form(
        sector, varphi_md, md, md_star, varphi_dts, dts, dts_star, gamma_carry, carry, w
    )

    expected_q = varphi_md * result.md.q + varphi_dts * result.dts.q
    expected_r = gamma_carry * carry + varphi_md * result.md.r + varphi_dts * result.dts.r
    expected_c = varphi_md * result.md.c + varphi_dts * result.dts.c
    expected_qf = 0.5 * w @ expected_q @ w - w @ expected_r + expected_c

    assert np.allclose(result.q, expected_q)
    assert np.allclose(result.r, expected_r)
    assert np.isclose(result.c, expected_c)
    assert np.isclose(result.qf, expected_qf)


def test_bond_portfolio_quadratic_form_vs_benchmark_reduces_to_plain_form_at_zero_benchmark():
    # With b=0 and varphi_as=0, the "vs benchmark" form should reduce
    # exactly to bond_portfolio_quadratic_form's Q/R/c, since Q_MD_b/R_MD_b/
    # c_MD_b (etc.) collapse to Q_MD/R_MD/c_MD when b=0.
    sector = np.array([1, 1, 2])
    md = np.array([2.0, 4.0, 6.0])
    md_star = np.array([3.0, 6.0])
    dts = np.array([1.0, 1.0, 2.0])
    dts_star = np.array([1.0, 2.0])
    gamma_carry = 1.5
    carry = np.array([0.01, 0.02, 0.03])
    w = np.array([0.5, 0.25, 0.25])
    b = np.zeros(3)
    varphi_md, varphi_dts = 0.6, 0.4

    plain = bond_portfolio_quadratic_form(
        sector, varphi_md, md, md_star, varphi_dts, dts, dts_star, gamma_carry, carry, w
    )
    vs_bench = bond_portfolio_quadratic_form_vs_benchmark(
        sector,
        varphi_as=0.0,
        varphi_md=varphi_md,
        md=md,
        md_star=md_star,
        varphi_dts=varphi_dts,
        dts=dts,
        dts_star=dts_star,
        gamma_carry=gamma_carry,
        carry=carry,
        w=w,
        b=b,
    )

    assert np.allclose(plain.q, vs_bench.q)
    assert np.allclose(plain.r, vs_bench.r)
    assert np.isclose(plain.c, vs_bench.c)
    assert np.isclose(plain.qf, vs_bench.qf)


def test_bond_portfolio_quadratic_form_vs_benchmark_defaults_star_to_zero():
    sector = np.array([1, 2])
    md = np.array([2.0, 3.0])
    dts = np.array([0.5, 0.5])
    carry = np.array([0.0, 0.0])
    w = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])

    result = bond_portfolio_quadratic_form_vs_benchmark(
        sector,
        varphi_as=0.1,
        varphi_md=0.5,
        md=md,
        md_star=None,
        varphi_dts=0.4,
        dts=dts,
        dts_star=None,
        gamma_carry=0.0,
        carry=carry,
        w=w,
        b=b,
    )
    assert np.isfinite(result.qf)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("rate", [-0.8, -0.02, 0.0, 0.04, 2.5])
def test_yield_recovers_closed_form_zero_coupon(method, rate):
    price = 100 * (np.exp(-3 * rate) if method == 1 else (1 + rate) ** -3)
    actual = bond_ytm(3, [100], price, BisectionConfig(tol=1e-12), method)
    assert actual == pytest.approx(rate, abs=1e-11)


@pytest.mark.parametrize("method", [1, 2])
def test_yield_handles_immediate_payment_and_fractional_times(method):
    times = np.array([0, 0.25, 1.5, 7.75])
    flows = np.array([7, 0, 3, 103])
    price = bond_price(times, flows, -0.015, method)
    actual = bond_ytm(times, flows, price, BisectionConfig(tol=1e-12), method)
    assert actual == pytest.approx(-0.015, abs=1e-11)


def test_yield_search_does_not_overflow_for_long_maturity():
    # The initial negative-rate bracket would overflow direct price evaluation.
    price = 100 * np.exp(10)
    assert bond_ytm(1000, [100], price) == pytest.approx(-0.01, abs=1e-5)


@pytest.mark.parametrize(
    "times,flows,rate,method,match",
    [
        ([], [], 0.03, 1, "nonempty"),
        ([1, 2], [100], 0.03, 1, "matching"),
        ([[1], [2]], [5, 105], 0.03, 1, "1-D"),
        ([-1], [100], 0.03, 1, "nonnegative"),
        ([np.nan], [100], 0.03, 1, "finite"),
        ([1], [np.inf], 0.03, 1, "finite"),
        ([1], [100], np.nan, 1, "finite scalar"),
        ([1], [100], [0.03], 1, "finite scalar"),
        ([1], [100], -1, 2, "greater than -1"),
        ([1], [100], 0.03, 3, "method"),
    ],
)
def test_price_rejects_invalid_inputs(times, flows, rate, method, match):
    with pytest.raises(ValueError, match=match):
        bond_price(times, flows, rate, method)


@pytest.mark.parametrize(
    "times,flows,price,match",
    [
        ([1, 2], [-5, 105], 100, "nonnegative"),
        ([0], [100], 100, "future"),
        ([1], [0], 100, "future"),
        ([0, 1], [10, 100], 10, "price"),
        ([1], [100], 0, "price"),
        ([1], [100], np.inf, "price"),
    ],
)
def test_yield_rejects_nonunique_or_impossible_inputs(times, flows, price, match):
    with pytest.raises(ValueError, match=match):
        bond_ytm(times, flows, price)


@pytest.mark.parametrize(
    "config",
    [
        BisectionConfig(tol=0),
        BisectionConfig(tol=np.nan),
        BisectionConfig(max_iters=0),
        BisectionConfig(max_iters=1.5),
    ],
)
def test_yield_rejects_invalid_solver_configuration(config):
    with pytest.raises(ValueError, match="config"):
        bond_ytm(5, [100], 90, config)


def test_yield_reports_nonconvergence():
    with pytest.raises(RuntimeError, match="converge"):
        bond_ytm([1, 5], [5, 105], 90, BisectionConfig(tol=1e-14, max_iters=1), 2)


def test_annual_yield_reports_unrepresentable_boundary():
    with pytest.raises(RuntimeError, match="too close to -1"):
        bond_ytm(0.001, [1], 100, method=2)


def test_price_accepts_signed_flows_and_scalar_maturity():
    assert bond_price(0, [-10, 30], 0.02) == 20
    assert bond_price(1, 100, 0.0) == 100


def test_price_reports_overflow_but_ignores_zero_payments():
    with pytest.raises(OverflowError, match="present value"):
        bond_price(1000, [100], -1)
    assert bond_price([0, 1000], [100, 0], -1) == 100


@pytest.mark.parametrize("method", [1, 2])
def test_zero_coupon_duration_and_convexity(method):
    rate, maturity = 0.04, 5
    divisor = 1 if method == 1 else 1 + rate
    assert bond_duration(maturity, [100], rate, method) == pytest.approx(maturity / divisor)
    assert bond_duration(maturity, [100], rate, method, modified=False) == pytest.approx(maturity)
    expected = maturity**2 if method == 1 else maturity * (maturity + 1) / divisor**2
    assert bond_convexity(maturity, [100], rate, method) == pytest.approx(expected)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("rate", [-0.02, 0.05])
def test_sensitivities_match_numerical_price_derivatives(method, rate):
    times, flows = [0, 0.5, 1.5, 5], [2, 3, 3, 103]
    h = 1e-4
    price = bond_price(times, flows, rate, method)
    up = bond_price(times, flows, rate + h, method)
    down = bond_price(times, flows, rate - h, method)
    assert bond_duration(times, flows, rate, method) == pytest.approx(
        -(up - down) / (2 * h * price), rel=1e-6
    )
    assert bond_convexity(times, flows, rate, method) == pytest.approx(
        (up - 2 * price + down) / (h**2 * price), rel=1e-6
    )


@pytest.mark.parametrize("function", [bond_duration, bond_convexity])
@pytest.mark.parametrize("flows", [[-5, 105], [0, 0]])
def test_sensitivities_reject_invalid_weights(function, flows):
    with pytest.raises(ValueError, match="nonnegative"):
        function([1, 2], flows, 0.03)


def test_coupon_yield_rejects_zero_price():
    with pytest.raises(ValueError, match="positive present value"):
        coupon_yield(1, [0], 0.03)
