"""Tests for quanttoolbox.econometrics.tests."""

import numpy as np

from quanttoolbox.econometrics.tests import adf_test


def test_adf_fails_to_reject_unit_root_for_random_walk(rng):
    n = 500
    rw = np.cumsum(rng.standard_normal(n))
    result = adf_test(rw, max_lags=2)
    # constant-only spec (index 1): should not reject at the 5% level
    assert np.all(result.p_value[1] > 0.05)


def test_adf_rejects_unit_root_for_stationary_ar1(rng):
    n = 500
    phi = 0.5
    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = phi * ar1[t - 1] + rng.standard_normal()
    result = adf_test(ar1, max_lags=2)
    assert np.all(result.p_value[1] < 0.01)


def test_adf_result_shapes(rng):
    n = 300
    y = rng.standard_normal(n)
    result = adf_test(y, max_lags=3)
    assert result.tau.shape == (3, 4)  # 3 specs, lags 0..3
    assert result.p_value.shape == (3, 4)
    assert result.critical_values.shape == (3, 4, 3)
    assert list(result.specification) == ["n", "c", "ct"]
    assert list(result.lags) == [0, 1, 2, 3]


def test_adf_critical_values_ordered(rng):
    n = 300
    y = rng.standard_normal(n)
    result = adf_test(y, max_lags=1)
    # 1% critical value should be more negative than 5%, which is more negative than 10%
    cv = result.critical_values[1, 0]  # constant spec, lag 0
    assert cv[0] < cv[1] < cv[2]
