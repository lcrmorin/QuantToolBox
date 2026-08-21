"""Tests for quanttoolbox.sustainable_finance.climate."""

import numpy as np

from quanttoolbox.sustainable_finance.climate import (
    dice_temperature_matrix,
    dice_temperature_simulation,
)


def test_dice_temperature_matrix_shapes_and_known_constants():
    result = dice_temperature_matrix(1.0)

    assert result.phi_cc.shape == (3, 3)
    assert result.b_cc.shape == (3,)
    assert result.xi_t.shape == (2, 2)
    assert result.b_t.shape == (2,)
    assert np.isclose(result.xi1, 0.098)
    assert np.isclose(result.xi2, 3.8 / 2.9)
    assert np.isclose(result.lambda_, result.xi2)
    assert np.isclose(result.beta, result.xi3)


def test_dice_temperature_matrix_method2_fifth_power_round_trips_xi_t_5():
    # Original hsf/dice_temperature_matrix.m always computes
    # mpower(Xi_T_5, 1/5) for mtd==2 regardless of Delta_t (verified against
    # the .m source) -- so (xi_t)^5 should reconstruct xi_t_5.
    result = dice_temperature_matrix(1.0, method=2)

    xi_t_5th_power = np.linalg.matrix_power(result.xi_t, 5)
    assert np.allclose(xi_t_5th_power, result.xi_t_5, atol=1e-6)


def test_dice_temperature_matrix_default_scale_matches_explicit_seconds_per_year():
    default = dice_temperature_matrix(1.0)
    explicit = dice_temperature_matrix(1.0, scale=365.25 * 24 * 3600)
    assert np.allclose(default.xi_t, explicit.xi_t)


def test_dice_temperature_simulation_shape_and_deterministic_time_column():
    t0, t_end, delta_t = 2015.0, 2025.0, 1.0

    def y_fn(t):
        return 100.0 * (1.02 ** (t - t0))

    def mu_fn(t):
        return 0.03 * (t - t0)

    results = dice_temperature_simulation(t0, t_end, delta_t, y_fn, mu_fn)

    n_iters = int(round((t_end - t0) / delta_t))
    assert results.shape == (n_iters + 1, 10)
    assert np.allclose(results[:, 0], np.arange(t0, t_end + delta_t / 2, delta_t))
    assert results[0, 0] == t0


def test_dice_temperature_simulation_zero_mitigation_gives_positive_emissions():
    t0, t_end, delta_t = 2015.0, 2020.0, 1.0

    def y_fn(t):
        return 100.0

    def mu_fn(t):
        return 0.0

    results = dice_temperature_simulation(t0, t_end, delta_t, y_fn, mu_fn)
    ce_column = results[:, 1]
    assert np.all(ce_column > 0)


def test_dice_temperature_simulation_full_mitigation_only_land_emissions_decay():
    t0, t_end, delta_t = 2015.0, 2020.0, 1.0

    def y_fn(t):
        return 100.0

    def mu_fn(t):
        return 1.0

    results = dice_temperature_simulation(t0, t_end, delta_t, y_fn, mu_fn)
    ce_land_only = results[:, 1]
    # with full mitigation, industrial emissions are zero and land emissions
    # decay geometrically -- so total emissions should be strictly decreasing
    assert np.all(np.diff(ce_land_only) < 0)
