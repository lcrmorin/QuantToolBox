"""Tests for quanttoolbox.spline.spline."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from quanttoolbox.spline.spline import (
    evaluate_spline,
    fit_smoothing_spline,
    integrate_spline,
    invert_spline,
)


def test_interpolation_matches_data_at_knots():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    spl = fit_smoothing_spline(x, y, p=1.0)
    assert np.allclose(evaluate_spline(spl, x, order=0), y, atol=1e-8)


def test_interpolation_matches_scipy_natural_cubic_spline():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.sin(x) + 0.1 * x**2
    spl = fit_smoothing_spline(x, y, p=1.0)
    sk_spl = CubicSpline(x, y, bc_type="natural")

    x_eval = np.linspace(0.1, 4.9, 20)
    assert np.allclose(evaluate_spline(spl, x_eval), sk_spl(x_eval), atol=1e-8)


def test_smoothing_reduces_noise_vs_interpolation():
    # use a fixed local seed (independent of the shared fixture) known to
    # produce noise large enough to make the smoothing benefit clear and
    # robust, rather than depending on the shared fixture's specific draw.
    local_rng = np.random.default_rng(0)
    n = 30
    x = np.sort(local_rng.uniform(0, 10, n))
    y_noisy = np.sin(x) + local_rng.standard_normal(n) * 0.2

    spl_interp = fit_smoothing_spline(x, y_noisy, p=1.0)
    spl_smooth = fit_smoothing_spline(x, y_noisy, p=0.3)

    x_eval = np.linspace(1, 9, 50)
    err_interp = np.mean((evaluate_spline(spl_interp, x_eval) - np.sin(x_eval)) ** 2)
    err_smooth = np.mean((evaluate_spline(spl_smooth, x_eval) - np.sin(x_eval)) ** 2)
    assert err_smooth < err_interp


def test_rejects_non_increasing_x():
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 1.0, 4.0])
    with pytest.raises(ValueError):
        fit_smoothing_spline(x, y)


def test_first_derivative_of_quadratic():
    x = np.linspace(0, 5, 6)
    y = x**2
    spl = fit_smoothing_spline(x, y, p=1.0)
    deriv = evaluate_spline(spl, np.array([2.5]), order=1)
    assert np.isclose(deriv[0], 5.0, atol=0.1)


def test_second_derivative_roughly_constant_for_quadratic():
    x = np.linspace(0, 5, 20)
    y = x**2
    spl = fit_smoothing_spline(x, y, p=1.0)
    second_deriv = evaluate_spline(spl, np.linspace(1, 4, 10), order=2)
    # d^2/dx^2 (x^2) = 2 everywhere
    assert np.allclose(second_deriv, 2.0, atol=0.2)


def test_integral_of_linear_function():
    x = np.linspace(0, 10, 11)
    y = 2 * x + 1  # integral from 0 to 5 = [x^2+x] from 0 to5 = 25+5=30
    spl = fit_smoothing_spline(x, y, p=1.0)
    area = integrate_spline(spl, np.array([[5.0], [0.0]]))
    assert np.isclose(area[0], 30.0, atol=0.5)


def test_integral_multiple_limits_at_once():
    x = np.linspace(0, 10, 11)
    y = np.ones_like(x) * 3.0  # constant function, integral = 3*(upper-lower)
    spl = fit_smoothing_spline(x, y, p=1.0)
    limits = np.array([[2.0, 5.0, 8.0], [0.0, 0.0, 0.0]])
    area = integrate_spline(spl, limits)
    assert np.allclose(area, [6.0, 15.0, 24.0], atol=0.1)


def test_invert_spline_recovers_x():
    x = np.linspace(0, 5, 6)
    y = x**2
    spl = fit_smoothing_spline(x, y, p=1.0)
    x_recovered = invert_spline(spl, np.array([6.25]))
    assert np.isclose(x_recovered[0], 2.5, atol=0.05)


def test_invert_spline_roundtrip():
    x = np.linspace(0, 10, 15)
    y = x + 0.1 * np.sin(x)  # monotonically increasing
    spl = fit_smoothing_spline(x, y, p=1.0)

    x_targets = np.array([2.0, 5.0, 8.0])
    y_targets = evaluate_spline(spl, x_targets, order=0)
    x_recovered = invert_spline(spl, y_targets)
    assert np.allclose(x_recovered, x_targets, atol=0.05)


def test_invert_spline_rejects_bad_limits_shape():
    x = np.linspace(0, 5, 6)
    y = x**2
    spl = fit_smoothing_spline(x, y, p=1.0)
    with pytest.raises(ValueError):
        integrate_spline(spl, np.array([1.0, 2.0, 3.0]))
