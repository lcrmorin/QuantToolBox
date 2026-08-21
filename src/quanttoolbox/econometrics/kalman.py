"""Linear-Gaussian state-space models and the Kalman filter.

Ported from QuantToolBox/ects/{state_space_model,ssm_set,ssm_steady_state,
Kalman_filtering}.m

Model convention (matching the original):

    measurement:  y_t = Z_t @ a_t + d_t + eps_t,   eps_t ~ N(0, H_t)
    transition:   a_t = T_t @ a_{t-1} + c_t + R_t @ eta_t,   eta_t ~ N(0, Q_t)

Translation notes:

- ``state_space_model``/``ssm_set`` (construct and validate a state-space
  model, optionally time-varying via 3-D arrays) are consolidated into
  the ``StateSpaceModel`` dataclass, whose ``__post_init__`` performs the
  same dimension-consistency checks the original returned as a
  ``retcode`` flag.
- ``Kalman_filtering.m``'s main loop is ported directly; MATLAB's
  ``try/catch`` around the innovation-covariance inverse (falling back to
  Moore-Penrose pseudo-inverse if singular) is replicated exactly with
  ``numpy.linalg.inv``/``numpy.linalg.pinv``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quanttoolbox.linalg.special_matrices import reshapec, vec


@dataclass
class StateSpaceModel:
    """Linear-Gaussian state-space model. Pass time-invariant matrices (2-D
    arrays) for a constant model, or time-varying matrices (3-D arrays,
    last axis = time) for a time-varying one.

    Original: ects/state_space_model.m (+ ects/ssm_set.m for allocating
    empty time-varying matrix blocks -- just use ``np.zeros((n, m, nobs))``
    etc. directly in Python instead of a dedicated allocator function)
    """

    z: np.ndarray  # (n, m) or (n, m, nobs) -- measurement loading
    d: np.ndarray  # (n,) or (n, nobs) -- measurement intercept
    h: np.ndarray  # (n, n) or (n, n, nobs) -- measurement noise covariance
    t: np.ndarray  # (m, m) or (m, m, nobs) -- transition matrix
    c: np.ndarray  # (m,) or (m, nobs) -- transition intercept
    r: np.ndarray  # (m, g) or (m, g, nobs) -- state noise loading
    q: np.ndarray  # (g, g) or (g, g, nobs) -- state noise covariance

    time_varying: bool = field(init=False)
    n: int = field(init=False)
    m: int = field(init=False)
    g: int = field(init=False)
    nobs: int | None = field(init=False)

    def __post_init__(self) -> None:
        self.time_varying = self.z.ndim == 3
        if not self.time_varying:
            self.n, self.m = self.z.shape
            self.g = self.r.shape[1]
            self.nobs = None
            ok = (
                self.d.shape == (self.n,)
                and self.h.shape == (self.n, self.n)
                and self.t.shape == (self.m, self.m)
                and self.c.shape == (self.m,)
                and self.r.shape == (self.m, self.g)
                and self.q.shape == (self.g, self.g)
            )
        else:
            self.n, self.m, self.nobs = self.z.shape
            self.g = self.r.shape[1]
            ok = (
                self.d.shape == (self.n, self.nobs)
                and self.h.shape == (self.n, self.n, self.nobs)
                and self.t.shape == (self.m, self.m, self.nobs)
                and self.c.shape == (self.m, self.nobs)
                and self.r.shape == (self.m, self.g, self.nobs)
                and self.q.shape == (self.g, self.g, self.nobs)
            )
        if not ok:
            raise ValueError("StateSpaceModel: inconsistent matrix dimensions")


def steady_state(ssm: StateSpaceModel) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state (unconditional) mean and covariance of the state, for a
    time-invariant, stable state-space model: a_bar = (I-T)^-1 c,
    vec(P_bar) = (I - T ⊗ T)^-1 vec(R Q R').

    Original: ects/ssm_steady_state.m
    """
    if ssm.time_varying:
        raise ValueError("steady_state: model is time-varying")

    m = ssm.m
    w0 = np.eye(m) - ssm.t
    if np.allclose(w0, 0):
        raise ValueError("steady_state: the model is not stable (T == I)")
    try:
        w0_inv = np.linalg.inv(w0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("steady_state: the model is not stable (I - T singular)") from exc

    a_bar = w0_inv @ ssm.c

    if np.allclose(ssm.q, 0):
        return a_bar, np.zeros((m, m))

    w1 = np.kron(ssm.t, ssm.t)
    w2 = np.eye(m * m) - w1
    try:
        w2_inv = np.linalg.inv(w2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("steady_state: the model is not stable (I - T⊗T singular)") from exc

    w3 = w2_inv @ vec(ssm.r @ ssm.q @ ssm.r.T)
    p_bar = reshapec(w3, m, m)
    return a_bar, p_bar


@dataclass
class KalmanFilterResult:
    y_pred: np.ndarray  # (nobs, n) predicted observations y(t|t-1)
    v: np.ndarray  # (nobs, n) innovations
    f: np.ndarray  # (n, n, nobs) innovation covariance
    a_pred: np.ndarray  # (nobs, m) predicted state a(t|t-1)
    p_pred: np.ndarray  # (m, m, nobs) predicted state covariance
    a_filt: np.ndarray  # (nobs, m) filtered state a(t|t)
    p_filt: np.ndarray  # (m, m, nobs) filtered state covariance
    log_l: np.ndarray  # (nobs,) per-observation log-likelihood contribution


def kalman_filter(
    ssm: StateSpaceModel, y: np.ndarray, a0: np.ndarray, p0: np.ndarray
) -> KalmanFilterResult:
    """Run the Kalman filter for the given state-space model, observations
    y, and initial state mean/covariance (a0, P0).

    Original: ects/Kalman_filtering.m
    """
    y = np.asarray(y, dtype=float)
    n, m = ssm.n, ssm.m
    nobs = ssm.nobs if ssm.nobs is not None else y.shape[0]

    if y.shape != (nobs, n):
        raise ValueError("kalman_filter: y has the wrong dimensions")

    if not ssm.time_varying:
        zt, dt, ht, tt, ct, rt, qt = ssm.z, ssm.d, ssm.h, ssm.t, ssm.c, ssm.r, ssm.q
        rqr = rt @ qt @ rt.T

    at, pt = np.asarray(a0, dtype=float).copy(), np.asarray(p0, dtype=float).copy()

    y_pred = np.zeros((nobs, n))
    v = np.zeros((nobs, n))
    f = np.zeros((n, n, nobs))
    a_pred = np.zeros((nobs, m))
    p_pred = np.zeros((m, m, nobs))
    a_filt = np.zeros((nobs, m))
    p_filt = np.zeros((m, m, nobs))
    log_l = np.zeros(nobs)

    for i in range(nobs):
        yt = y[i]

        if ssm.time_varying:
            zt, dt, ht = ssm.z[:, :, i], ssm.d[:, i], ssm.h[:, :, i]
            tt, ct, rt, qt = ssm.t[:, :, i], ssm.c[:, i], ssm.r[:, :, i], ssm.q[:, :, i]
            rqr = rt @ qt @ rt.T

        # prediction
        at1 = tt @ at + ct
        pt1 = tt @ pt @ tt.T + rqr

        # innovation
        yt1 = zt @ at1 + dt
        vt = yt - yt1

        # updating
        ft = zt @ pt1 @ zt.T + ht
        try:
            inv_ft = np.linalg.inv(ft)
        except np.linalg.LinAlgError:
            inv_ft = np.zeros((n, n)) if np.allclose(ft, 0) else np.linalg.pinv(ft)

        a_mat = pt1 @ zt.T
        b_mat = a_mat @ inv_ft

        at = at1 + b_mat @ vt
        pt = pt1 - b_mat @ a_mat.T

        det_ft = np.linalg.det(ft)
        if det_ft <= 0:
            det_ft = 1e-10

        y_pred[i] = yt1
        v[i] = vt
        f[:, :, i] = ft
        a_pred[i] = at1
        p_pred[:, :, i] = pt1
        a_filt[i] = at
        p_filt[:, :, i] = pt

        log_l[i] = -(n / 2) * np.log(2 * np.pi) - 0.5 * np.log(det_ft) - 0.5 * vt @ inv_ft @ vt

    return KalmanFilterResult(
        y_pred=y_pred,
        v=v,
        f=f,
        a_pred=a_pred,
        p_pred=p_pred,
        a_filt=a_filt,
        p_filt=p_filt,
        log_l=log_l,
    )
