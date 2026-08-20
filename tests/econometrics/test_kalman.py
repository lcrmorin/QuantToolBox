"""Tests for quanttoolbox.econometrics.kalman."""

import numpy as np
import pytest

from quanttoolbox.econometrics.kalman import StateSpaceModel, kalman_filter, steady_state


def test_state_space_model_valid_time_invariant():
    ssm = StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[0.5]]),
        t=np.array([[0.9]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[1.0]]),
    )
    assert not ssm.time_varying
    assert ssm.n == 1 and ssm.m == 1 and ssm.g == 1


def test_state_space_model_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        StateSpaceModel(
            z=np.array([[1.0, 0.0]]),
            d=np.array([0.0]),
            h=np.array([[0.5]]),
            t=np.array([[0.9]]),
            c=np.array([0.0]),
            r=np.array([[1.0]]),
            q=np.array([[1.0]]),
        )


def test_steady_state_matches_analytical_ar1():
    phi = 0.5
    q_var = 1.0
    ssm = StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[0.1]]),
        t=np.array([[phi]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[q_var]]),
    )
    a_bar, p_bar = steady_state(ssm)
    assert np.isclose(a_bar[0], 0.0)
    assert np.isclose(p_bar[0, 0], q_var / (1 - phi**2), atol=1e-8)


def test_steady_state_rejects_unstable_model():
    ssm = StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[0.1]]),
        t=np.array([[1.0]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[1.0]]),
    )
    with pytest.raises(ValueError):
        steady_state(ssm)


def test_kalman_filter_tracks_local_level_state(rng):
    n = 500
    sigma_eps, sigma_eta = 0.5, 0.1
    true_a = np.zeros(n)
    for t in range(1, n):
        true_a[t] = true_a[t - 1] + rng.standard_normal() * sigma_eta
    y = true_a + rng.standard_normal(n) * sigma_eps

    ssm = StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[sigma_eps**2]]),
        t=np.array([[1.0]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[sigma_eta**2]]),
    )
    result = kalman_filter(ssm, y[:, None], a0=np.array([0.0]), p0=np.array([[1e6]]))
    correlation = np.corrcoef(result.a_filt[:, 0], true_a)[0, 1]
    assert correlation > 0.85


def test_kalman_filter_shapes(rng):
    n = 100
    ssm = StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[0.5]]),
        t=np.array([[0.9]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[1.0]]),
    )
    y = rng.standard_normal((n, 1))
    result = kalman_filter(ssm, y, a0=np.array([0.0]), p0=np.array([[1.0]]))
    assert result.a_filt.shape == (n, 1)
    assert result.p_filt.shape == (1, 1, n)
    assert result.log_l.shape == (n,)


def test_kalman_filter_time_varying_matches_time_invariant(rng):
    n = 50
    sigma_eps, sigma_eta = 0.5, 0.1
    y = rng.standard_normal(n)

    ssm_const = StateSpaceModel(
        z=np.array([[1.0]]),
        d=np.array([0.0]),
        h=np.array([[sigma_eps**2]]),
        t=np.array([[1.0]]),
        c=np.array([0.0]),
        r=np.array([[1.0]]),
        q=np.array([[sigma_eta**2]]),
    )
    ssm_tv = StateSpaceModel(
        z=np.tile(np.array([[1.0]])[:, :, None], (1, 1, n)),
        d=np.tile(np.array([0.0])[:, None], (1, n)),
        h=np.tile(np.array([[sigma_eps**2]])[:, :, None], (1, 1, n)),
        t=np.tile(np.array([[1.0]])[:, :, None], (1, 1, n)),
        c=np.tile(np.array([0.0])[:, None], (1, n)),
        r=np.tile(np.array([[1.0]])[:, :, None], (1, 1, n)),
        q=np.tile(np.array([[sigma_eta**2]])[:, :, None], (1, 1, n)),
    )
    r_const = kalman_filter(ssm_const, y[:, None], a0=np.array([0.0]), p0=np.array([[1.0]]))
    r_tv = kalman_filter(ssm_tv, y[:, None], a0=np.array([0.0]), p0=np.array([[1.0]]))
    assert np.allclose(r_const.a_filt, r_tv.a_filt)
    assert np.allclose(r_const.log_l, r_tv.log_l)
