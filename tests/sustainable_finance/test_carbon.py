"""Tests for quanttoolbox.sustainable_finance.carbon."""

import numpy as np
from scipy.integrate import quad

from quanttoolbox.sustainable_finance.carbon import (
    carbon_budget,
    carbon_budget_compound_reduction,
    carbon_budget_linear,
    carbon_budget_piecewise,
)


def test_carbon_budget_linear_matches_numerical_integral():
    t0, t, beta0, beta1 = 2020.0, 2030.0, 5.0, 0.2

    expected, _ = quad(lambda s: beta0 + beta1 * s, t0, t)
    assert np.isclose(carbon_budget_linear(t0, t, beta0, beta1), expected)


def test_carbon_budget_linear_zero_slope_is_rectangle():
    assert np.isclose(carbon_budget_linear(0.0, 10.0, 5.0, 0.0), 50.0)


def test_carbon_budget_method1_linear_rate_matches_numerical_integral():
    t0, t, ce_t0, r = 2020.0, 2030.0, 40.0, 0.5

    def ce(s):
        return ce_t0 - r * (s - t0)

    expected, _ = quad(ce, t0, t)
    assert np.isclose(carbon_budget(t0, t, ce_t0, r, method=1), expected)


def test_carbon_budget_method2_compound_rate_matches_numerical_integral():
    t0, t, ce_t0, r = 2020.0, 2030.0, 40.0, 0.05

    def ce(s):
        return ce_t0 * (1.0 - r) ** (s - t0)

    expected, _ = quad(ce, t0, t)
    assert np.isclose(carbon_budget(t0, t, ce_t0, r, method=2), expected)


def test_carbon_budget_method3_growth_rate_matches_numerical_integral():
    t0, t, ce_t0, r = 2020.0, 2030.0, 40.0, 0.03

    def ce(s):
        return ce_t0 * np.exp(-r * (s - t0))

    expected, _ = quad(ce, t0, t)
    assert np.isclose(carbon_budget(t0, t, ce_t0, r, method=3), expected)


def test_carbon_budget_linear_reduction_is_special_case_of_carbon_budget():
    # carbon_budget_linear_reduction.m (not ported separately) is the
    # method=1 branch of carbon_budget_Reduction.m under r = reduction * ce_t0.
    t0, t, ce_t0, reduction = 2020.0, 2025.0, 100.0, 0.02
    r = reduction * ce_t0
    dt = t - t0

    direct = dt * ce_t0 - 0.5 * dt**2 * reduction * ce_t0
    via_carbon_budget = carbon_budget(t0, t, ce_t0, r=r, method=1)
    assert np.isclose(direct, via_carbon_budget)


def test_carbon_budget_piecewise_matches_numerical_integral_of_interpolant():
    t_k = np.array([2020.0, 2025.0, 2030.0, 2040.0])
    ce_k = np.array([40.0, 30.0, 20.0, 5.0])
    t0, t = 2022.0, 2035.0

    def ce(s):
        return float(np.interp(s, t_k, ce_k))

    expected, _ = quad(ce, t0, t)
    result = carbon_budget_piecewise(t0, t, t_k, ce_k)
    assert np.isclose(result, expected, rtol=1e-6)


def test_carbon_budget_piecewise_full_range_matches_trapezoid_rule():
    t_k = np.array([0.0, 1.0, 2.0, 3.0])
    ce_k = np.array([10.0, 8.0, 6.0, 6.0])

    result = carbon_budget_piecewise(0.0, 3.0, t_k, ce_k)
    expected = np.trapezoid(ce_k, t_k)
    assert np.isclose(result, expected)


def test_carbon_budget_compound_reduction_without_gdp_matches_numerical_integral():
    t0, t, delta_r, r_minus, ce_t0 = 2020.0, 2030.0, 0.04, 0.1, 50.0

    def ce(s):
        return (1.0 - r_minus) * ce_t0 * (1.0 - delta_r) ** (s - t0)

    expected, _ = quad(ce, t0, t)
    result = carbon_budget_compound_reduction(t0, t, delta_r, r_minus, ce_t0)
    assert np.isclose(result, expected)


def test_carbon_budget_compound_reduction_with_gdp_matches_numerical_integral():
    t0, t, delta_r, r_minus, ce_t0, g_y = 2020.0, 2030.0, 0.04, 0.1, 50.0, 0.02

    def ce(s):
        return (1.0 - r_minus) * ce_t0 * (1.0 + g_y) ** (s - t0) * (1.0 - delta_r) ** (s - t0)

    expected, _ = quad(ce, t0, t)
    result = carbon_budget_compound_reduction(t0, t, delta_r, r_minus, ce_t0, g_y=g_y)
    assert np.isclose(result, expected)
