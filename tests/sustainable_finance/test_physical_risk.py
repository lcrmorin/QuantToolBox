"""Tests for quanttoolbox.sustainable_finance.physical_risk."""

import numpy as np

from quanttoolbox.config import BisectionConfig
from quanttoolbox.optim.bisection import bisection
from quanttoolbox.sustainable_finance.physical_risk import (
    AngularResolution,
    HorizontalResolutionResult,
    IceAlbedoRelaxationResult,
    WiechelProjection,
    compute_horizontal_resolution,
    ice_albedo,
    ice_albedo_tau,
    resolution_to_angular,
    wiechel_forward,
)

# ---------------------------------------------------------------------------
# ice_albedo / ice_albedo_tau
# ---------------------------------------------------------------------------

# Concrete parameter set used throughout HSF-Notebooks chapter 08c's own
# worked example.
_C = 4e8 / (24 * 3600 * 365.25)
_S0, _ETA = 1368.0, 3.0
_ALPHA_COLD, _T_COLD = 0.7, 260.0
_ALPHA_WARM, _T_WARM = 0.30, 295.0
_EPSILON, _SIGMA = 0.78, 5.67e-8


def _notebook_ice_albedo_formula(
    t_s, alpha_cold, t_cold, alpha_warm, t_warm, eta, s0, epsilon, sigma, c, d_e=0.0
):
    """Independent re-derivation of the notebook's own formula (not just a
    call to the module under test), used as a cross-check."""
    t_s = np.asarray(t_s, dtype=float)
    alpha = np.where(
        t_s < t_cold,
        alpha_cold,
        np.where(
            t_s > t_warm,
            alpha_warm,
            alpha_warm
            + (alpha_cold - alpha_warm)
            * (np.clip(t_warm - t_s, 0, None) ** eta)
            / ((t_warm - t_cold) ** eta),
        ),
    )
    f_solar = 0.25 * s0 * (1 - alpha)
    f_lw = (2 - epsilon) / 2 * sigma * t_s**4
    return (f_solar - f_lw + d_e) / c


def test_ice_albedo_matches_independent_formula_random_inputs():
    rng = np.random.default_rng(0)
    t_s = rng.uniform(200, 350, size=25)

    got = ice_albedo(
        t_s, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, _ETA, _S0, _EPSILON, _SIGMA, _C
    )
    expected = _notebook_ice_albedo_formula(
        t_s, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, _ETA, _S0, _EPSILON, _SIGMA, _C
    )
    assert np.allclose(got, expected)


def test_ice_albedo_cold_and_warm_branches_use_pure_alpha_cold_and_alpha_warm():
    # Deep in the cold branch, albedo == alpha_cold; deep in the warm
    # branch, albedo == alpha_warm -- check this indirectly through the
    # closed-form energy balance (no albedo blending term).
    t_cold_deep, t_warm_deep = 100.0, 400.0

    got_cold = ice_albedo(
        t_cold_deep, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, _ETA, _S0, _EPSILON, _SIGMA, _C
    )
    expected_cold = (
        0.25 * _S0 * (1 - _ALPHA_COLD) - (2 - _EPSILON) / 2 * _SIGMA * t_cold_deep**4
    ) / _C
    assert np.isclose(got_cold, expected_cold)

    got_warm = ice_albedo(
        t_warm_deep, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, _ETA, _S0, _EPSILON, _SIGMA, _C
    )
    expected_warm = (
        0.25 * _S0 * (1 - _ALPHA_WARM) - (2 - _EPSILON) / 2 * _SIGMA * t_warm_deep**4
    ) / _C
    assert np.isclose(got_warm, expected_warm)


def test_ice_albedo_equilibria_match_notebook_bisection_data():
    # HSF-Notebooks chapter 08c cell output, eta=3 (the chapter's default):
    # "Equilibria (K): [233.38 268.43 288.13]" -- reproduced here to higher
    # precision via the same bisection routine the notebook uses.
    def fun1(t_s):
        return ice_albedo(
            t_s, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, _ETA, _S0, _EPSILON, _SIGMA, _C, 0.0
        )

    cfg = BisectionConfig(tol=1e-10, max_iters=200)
    t1 = bisection(fun1, 200, 250, cfg)
    t2 = bisection(fun1, 250, 280, cfg)
    t3 = bisection(fun1, 280, 350, cfg)

    assert np.isclose(t1, 233.3773, atol=1e-3)
    assert np.isclose(t2, 268.4350, atol=1e-3)
    assert np.isclose(t3, 288.1255, atol=1e-3)
    # And each root is genuinely a root.
    assert np.isclose(fun1(t1), 0.0, atol=1e-6)
    assert np.isclose(fun1(t2), 0.0, atol=1e-6)
    assert np.isclose(fun1(t3), 0.0, atol=1e-6)


def _notebook_ice_albedo_tau_formula(
    t_s, alpha_cold, t_cold, alpha_warm, t_warm, eta, s0, epsilon, sigma, c
):
    """Independent re-derivation (scalar/1-D case) of tau/lambda, used as a
    cross-check against the module's (possibly column-broadcasting)
    implementation."""
    t_s = np.asarray(t_s, dtype=float)
    in_range = (t_s >= t_cold) & (t_s <= t_warm)
    d_alpha = np.where(
        in_range,
        -eta
        * (alpha_cold - alpha_warm)
        * (np.clip(t_warm - t_s, 0, None) ** (eta - 1))
        / ((t_warm - t_cold) ** eta),
        0.0,
    )
    lambda_ = -0.25 * s0 * d_alpha - 4 * ((2 - epsilon) / 2) * sigma * t_s**3
    tau = np.abs(c / lambda_)
    return tau, lambda_


def test_ice_albedo_tau_matches_independent_formula_scalar_inputs():
    for t_s in (233.3773, 268.4350, 288.1255):
        result = ice_albedo_tau(
            np.array([t_s]),
            _ALPHA_COLD,
            _T_COLD,
            _ALPHA_WARM,
            _T_WARM,
            _ETA,
            _S0,
            _EPSILON,
            _SIGMA,
            _C,
        )
        assert isinstance(result, IceAlbedoRelaxationResult)
        expected_tau, expected_lambda = _notebook_ice_albedo_tau_formula(
            np.array([t_s]),
            _ALPHA_COLD,
            _T_COLD,
            _ALPHA_WARM,
            _T_WARM,
            _ETA,
            _S0,
            _EPSILON,
            _SIGMA,
            _C,
        )
        assert np.allclose(result.tau, expected_tau)
        assert np.allclose(result.lambda_, expected_lambda)


def test_ice_albedo_tau_at_equilibria_matches_notebook_lambda_values():
    # Notebook cell output: "lambda_total: [-1.76  4.08 -2.86]" at the
    # three eta=3 equilibria.
    t_eq = np.array([233.3773, 268.4350, 288.1255])
    result = ice_albedo_tau(
        t_eq, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, _ETA, _S0, _EPSILON, _SIGMA, _C
    )

    assert np.allclose(result.lambda_, [-1.76, 4.08, -2.86], atol=0.01)
    # Stable equilibria (lambda_ < 0) are T1 and T3; the unstable one is T2.
    assert result.lambda_[0] < 0
    assert result.lambda_[1] > 0
    assert result.lambda_[2] < 0
    assert np.allclose(result.tau, np.abs(_C / result.lambda_))


def test_ice_albedo_tau_broadcasts_1d_params_as_columns_against_2d_t_s():
    # Mirrors the notebook's own sweep usage: T_s is (n, 3) (one row per
    # swept-parameter value, one column per equilibrium branch), eta is
    # (n,) and must broadcast as a column.
    n = 5
    eta_vals = np.linspace(1.0, 4.0, n)
    t_s = np.tile(np.array([233.0, 268.0, 288.0]), (n, 1))

    result = ice_albedo_tau(
        t_s, _ALPHA_COLD, _T_COLD, _ALPHA_WARM, _T_WARM, eta_vals, _S0, _EPSILON, _SIGMA, _C
    )

    assert result.tau.shape == (n, 3)
    assert result.lambda_.shape == (n, 3)
    # Cross-check row-by-row against the independent scalar formula.
    for i in range(n):
        expected_tau, expected_lambda = _notebook_ice_albedo_tau_formula(
            t_s[i],
            _ALPHA_COLD,
            _T_COLD,
            _ALPHA_WARM,
            _T_WARM,
            eta_vals[i],
            _S0,
            _EPSILON,
            _SIGMA,
            _C,
        )
        assert np.allclose(result.tau[i], expected_tau)
        assert np.allclose(result.lambda_[i], expected_lambda)


# ---------------------------------------------------------------------------
# compute_horizontal_resolution
# ---------------------------------------------------------------------------


def test_compute_horizontal_resolution_matches_notebook_table():
    # HSF-Notebooks chapter 12c's own table (delta_phi == delta_lambda, km,
    # rounded to the nearest integer).
    delta = np.array([10, 5.6, 4.0, 2.8, 2.5, 1.9, 1.4, 1.1, 0.75, 0.375])
    expected_r0 = np.array([1112.0, 623.0, 445.0, 311.0, 278.0, 211.0, 156.0, 122.0, 83.0, 42.0])
    expected_r45 = np.array([935.0, 524.0, 374.0, 262.0, 234.0, 178.0, 131.0, 103.0, 70.0, 35.0])
    expected_ravg = np.array([848.0, 475.0, 339.0, 238.0, 212.0, 161.0, 119.0, 93.0, 64.0, 32.0])

    result = compute_horizontal_resolution(delta, delta)

    assert isinstance(result, HorizontalResolutionResult)
    assert np.allclose(np.round(result.r_0), expected_r0)
    assert np.allclose(np.round(result.r_45), expected_r45)
    assert np.allclose(np.round(result.r_avg), expected_ravg)


def test_compute_horizontal_resolution_matches_matlab_source_formula():
    # Independent re-derivation of HSF/12. Physical Risk/
    # compute_horizontal_resolution.m directly (length_phi=111.2, phi=0
    # and phi=pi/4, R_avg = 0.7628 * R_0), not just a call to the module.
    delta_phi, delta_lambda = 2.5, 3.75
    length_phi = 111.2
    expected_r0 = length_phi * np.sqrt(delta_phi * delta_lambda * np.cos(0.0))
    expected_r45 = length_phi * np.sqrt(delta_phi * delta_lambda * np.cos(np.pi / 4))
    expected_ravg = 0.7628 * expected_r0

    result = compute_horizontal_resolution(delta_phi, delta_lambda)

    assert np.isclose(result.r_0, expected_r0)
    assert np.isclose(result.r_45, expected_r45)
    assert np.isclose(result.r_avg, expected_ravg)


def test_compute_horizontal_resolution_r45_smaller_than_r0():
    # Resolution shrinks toward the poles as meridians converge (cos(phi) -> 0).
    result = compute_horizontal_resolution(4.0, 5.0)
    assert result.r_45 < result.r_0


# ---------------------------------------------------------------------------
# resolution_to_angular
# ---------------------------------------------------------------------------


def test_resolution_to_angular_rhomboidal_notations():
    assert resolution_to_angular("R15") == AngularResolution(delta_phi=4.5, delta_lambda=7.5)
    assert resolution_to_angular("R21") == AngularResolution(delta_phi=3.2, delta_lambda=5.6)
    assert resolution_to_angular("R30") == AngularResolution(delta_phi=2.25, delta_lambda=3.75)


def test_resolution_to_angular_triangular_notations_have_equal_phi_lambda():
    for label, expected_delta in [("T21", 5.6), ("T85", 1.4), ("T126", 0.9375), ("T319", 0.375)]:
        result = resolution_to_angular(label)
        assert result.delta_phi == expected_delta
        assert result.delta_lambda == expected_delta


def test_resolution_to_angular_plain_degree_string():
    result = resolution_to_angular("4°x5°")
    assert result.delta_phi == 4.0
    assert result.delta_lambda == 5.0

    result2 = resolution_to_angular("2.5°x3.75°")
    assert result2.delta_phi == 2.5
    assert result2.delta_lambda == 3.75


def test_resolution_to_angular_unrecognized_label_returns_nan_pair():
    result = resolution_to_angular("not-a-resolution")
    assert np.isnan(result.delta_phi)
    assert np.isnan(result.delta_lambda)


def test_resolution_to_angular_feeds_compute_horizontal_resolution():
    # End-to-end check matching HSF-Notebooks chapter 12d's own pipeline:
    # resolution label -> angular pair -> horizontal resolution (km).
    angular = resolution_to_angular("T63")
    result = compute_horizontal_resolution(angular.delta_phi, angular.delta_lambda)
    assert result.r_0 > result.r_45 > 0


# ---------------------------------------------------------------------------
# wiechel_forward
# ---------------------------------------------------------------------------


def _notebook_wiechel_formula(lon_deg, lat_deg, R=1.0):
    lam = np.radians(lon_deg)
    phi = np.radians(lat_deg)
    x = R * (np.sin(lam) * np.cos(phi) - (1 - np.sin(phi)) * np.cos(lam))
    y = -R * (np.cos(lam) * np.cos(phi) + (1 - np.sin(phi)) * np.sin(lam))
    return x, y


def test_wiechel_forward_matches_independent_formula_random_inputs():
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 180, size=30)
    lat = rng.uniform(0, 90, size=30)

    result = wiechel_forward(lon, lat, R=2.0)
    expected_x, expected_y = _notebook_wiechel_formula(lon, lat, R=2.0)

    assert isinstance(result, WiechelProjection)
    assert np.allclose(result.x, expected_x)
    assert np.allclose(result.y, expected_y)


def test_wiechel_forward_north_pole_maps_to_origin():
    result = wiechel_forward(lon_deg=np.array([0.0, 90.0, 180.0, -45.0]), lat_deg=90.0)
    assert np.allclose(result.x, 0.0, atol=1e-12)
    assert np.allclose(result.y, 0.0, atol=1e-12)


def test_wiechel_forward_default_radius_is_one():
    result_default = wiechel_forward(30.0, 20.0)
    result_explicit = wiechel_forward(30.0, 20.0, R=1.0)
    assert np.isclose(result_default.x, result_explicit.x)
    assert np.isclose(result_default.y, result_explicit.y)


def test_wiechel_forward_satisfies_documented_equal_area_identity():
    # x**2 + y**2 == 2 * R**2 * (1 - sin(phi)) -- the same radial equal-area
    # function as the polar Lambert azimuthal projection, per HSF-Notebooks
    # chapter 12b's own derivation.
    rng = np.random.default_rng(2)
    lon = rng.uniform(-180, 180, size=20)
    lat = rng.uniform(0, 90, size=20)
    R = 3.0

    result = wiechel_forward(lon, lat, R=R)
    phi = np.radians(lat)
    lhs = result.x**2 + result.y**2
    rhs = 2 * R**2 * (1 - np.sin(phi))
    assert np.allclose(lhs, rhs)
