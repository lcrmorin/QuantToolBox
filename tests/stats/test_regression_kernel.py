"""Tests for quanttoolbox.stats.regression.kernel."""

import numpy as np
from scipy.stats import norm

from quanttoolbox.stats.regression.kernel import (
    kernel_density,
    kernel_mean_regression,
    kernel_payoff_regression,
    kernel_quantile_regression,
)


def test_kernel_density_integrates_to_one(rng):
    data = rng.standard_normal(2000)
    x = np.linspace(-4, 4, 400)
    density, cdf, bw = kernel_density(data, x)
    integral = np.trapezoid(density[:, 0], x)
    assert abs(integral - 1.0) < 0.05


def test_kernel_density_approximates_normal_pdf(rng):
    data = rng.standard_normal(5000)
    x = np.array([-1.0, 0.0, 1.0])
    density, cdf, bw = kernel_density(data, x)
    ref = norm.pdf(x)
    assert np.allclose(density[:, 0], ref, atol=0.1)


def test_kernel_mean_regression_recovers_linear_trend(rng):
    n = 2000
    x = rng.uniform(-3, 3, n)
    y = 2.0 * x + rng.standard_normal(n) * 0.1
    z = np.array([-1.0, 0.0, 1.0])
    m = kernel_mean_regression(y, x, z, order=1)
    assert np.allclose(m, 2.0 * z, atol=0.3)


def test_kernel_payoff_regression_default_grid(rng):
    n = 500
    x = rng.uniform(-2, 2, n)
    y = x**2 + rng.standard_normal(n) * 0.05
    z, q = kernel_payoff_regression(x, y, order=2, n_points=50)
    assert z.shape[0] == 50
    assert q.shape[0] == 50
    # payoff should be roughly convex (U-shaped) like x^2
    assert q[0] > q[25] or q[-1] > q[25]


def test_kernel_quantile_regression_median_close_to_mean(rng):
    n = 3000
    x = rng.uniform(-2, 2, n)
    y = 1.5 * x + rng.standard_normal(n) * 0.2  # symmetric noise
    z = np.array([0.0])
    q_median = kernel_quantile_regression(y, x, tau=0.5, z=z, order=1)
    m = kernel_mean_regression(y, x, z, order=1)
    assert np.allclose(q_median, m, atol=0.3)
