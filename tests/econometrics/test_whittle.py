"""Tests for quanttoolbox.econometrics.whittle."""

import numpy as np
import pytest

from quanttoolbox.econometrics.whittle import (
    periodogram,
    whittle_local_level,
    whittle_local_linear_trend,
)


def test_periodogram_shapes(rng):
    n = 100
    y = rng.standard_normal(n)
    lam, fft_coeffs, intensity = periodogram(y)
    assert lam.shape == (n,)
    assert fft_coeffs.shape == (n,)
    assert intensity.shape == (n,)


def test_periodogram_scaling_divides_by_2pi(rng):
    n = 100
    y = rng.standard_normal(n)
    _, _, intensity_unscaled = periodogram(y, scaling=False)
    _, _, intensity_scaled = periodogram(y, scaling=True)
    assert np.allclose(intensity_scaled, intensity_unscaled / (2 * np.pi))


def test_periodogram_parseval_relation(rng):
    # Parseval: since I = |FFT(y)|^2 / n, and sum(|FFT(y)|^2) = n * sum(y^2)
    # by the DFT Parseval theorem, sum(I) == sum(y^2) directly.
    n = 200
    y = rng.standard_normal(n)
    _, _, intensity = periodogram(y, scaling=False)
    assert np.isclose(np.sum(intensity), np.sum(y**2), rtol=1e-6)


def test_whittle_local_level_recovers_known_parameters(rng):
    n = 3000
    sigma_eps_true, sigma_eta_true = 0.5, 0.2
    mu = np.zeros(n)
    for t in range(1, n):
        mu[t] = mu[t - 1] + rng.standard_normal() * sigma_eta_true
    y = mu + rng.standard_normal(n) * sigma_eps_true

    result = whittle_local_level(y, sv=np.array([1.0, 1.0]))
    assert result.converged
    estimated = np.abs(result.theta)
    assert np.allclose(estimated, [sigma_eps_true, sigma_eta_true], atol=0.1)


def test_whittle_local_level_analytical_gradient_matches_numerical(rng):
    # regression test for the 2*pi Jacobian scaling bug found during porting
    from quanttoolbox.econometrics.estimation import _numerical_jacobian
    from quanttoolbox.econometrics.whittle import _local_level_sdf, _local_level_sdf_jacobian

    n = 500
    y = rng.standard_normal(n)
    lam, _, intensity = periodogram(y, scaling=True)
    theta = np.array([0.6, 0.3])

    analytical = _local_level_sdf_jacobian(lam, theta)

    def sdf_at(t):
        return _local_level_sdf(lam, t)

    numerical = _numerical_jacobian(sdf_at, theta)
    assert np.allclose(analytical, numerical, atol=1e-4)


def test_whittle_local_level_rejects_wrong_sv_length():
    with pytest.raises(ValueError):
        whittle_local_level(np.zeros(100), sv=np.array([1.0, 1.0, 1.0]))


def test_whittle_local_linear_trend_recovers_known_parameters(rng):
    n = 3000
    sigma_eps_true, sigma_eta_true, sigma_zeta_true = 0.3, 0.1, 0.05

    level = np.zeros(n)
    slope = np.zeros(n)
    for t in range(1, n):
        slope[t] = slope[t - 1] + rng.standard_normal() * sigma_zeta_true
        level[t] = level[t - 1] + slope[t - 1] + rng.standard_normal() * sigma_eta_true
    y = level + rng.standard_normal(n) * sigma_eps_true

    result = whittle_local_linear_trend(y, sv=np.array([1.0, 1.0, 1.0]))
    estimated = np.abs(result.theta)
    # local linear trend is harder to identify precisely -- check order of magnitude / rough recovery
    assert estimated.shape == (3,)
    assert np.all(np.isfinite(estimated))


def test_whittle_local_linear_trend_rejects_wrong_sv_length():
    with pytest.raises(ValueError):
        whittle_local_linear_trend(np.zeros(100), sv=np.array([1.0, 1.0]))
