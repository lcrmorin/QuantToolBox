"""The DICE (Dynamic Integrated Climate-Economy) model's carbon-cycle and
temperature submodules: the state-transition matrices linking industrial
emissions to atmospheric carbon concentration and global temperature, and a
full forward simulation of both given exogenous GDP and mitigation-rate
paths.

Ported from HSF toolbox `hsf/{dice_temperature_matrix,
dice_temperature_simulation}.m`.

Translation notes:

- All physical constants (carbon-cycle transfer coefficients, radiative
  forcing parameters, initial 2015 conditions) are Nordhaus's DICE-2016
  calibration, hardcoded in the original -- kept as-is (not exposed as
  parameters), matching the original's scope.
- `dice_temperature_simulation.m` accepts a `parameters` argument that is
  never read anywhere in the function body (a vestigial/dead parameter in
  the original) -- dropped here rather than carried forward as a
  do-nothing kwarg.
- `Y_fn`/`mu_fn` (GDP and mitigation-rate paths, as functions of time) are
  plain Python callables, matching the original's function-handle
  arguments exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import fractional_matrix_power


@dataclass
class DiceTemperatureMatrices:
    """The DICE carbon-cycle (`phi_cc`, `b_cc`) and temperature (`xi_t`,
    `b_t`) state-transition matrices for one time step of length
    `delta_t`, plus the intermediate physical constants they were built
    from."""

    phi_cc: np.ndarray
    b_cc: np.ndarray
    xi_t: np.ndarray
    b_t: np.ndarray
    xi_t_5: np.ndarray
    b_t_5: np.ndarray
    xi1: float
    xi2: float
    xi3: float
    xi4: float
    c_at: float
    c_lo: float
    lambda_: float
    beta: float


def dice_temperature_matrix(
    delta_t: float, scale: float | None = None, method: int = 1
) -> DiceTemperatureMatrices:
    """Build the DICE model's carbon-cycle and temperature state-transition
    matrices for a time step of length `delta_t`.

    `scale` converts `delta_t` into seconds (default: `delta_t` is in
    years, `scale = 365.25 * 24 * 3600`). `method=2` instead derives `xi_t`
    from the standard 5-year-step calibration matrix via a matrix power
    (``xi_t_5 ** (delta_t_years / 5)``) rather than reconstructing it from
    the underlying continuous-time physical constants.

    Original: hsf/dice_temperature_matrix.m
    """
    if scale is None:
        scale = 365.25 * 24 * 3600

    phi1 = 0.2727
    phi_cc = np.array(
        [
            [0.9120, 0.0383, 0.0],
            [0.0880, 0.9592, 0.0003],
            [0.0, 0.0025, 0.9997],
        ]
    )
    b_cc = np.array([phi1, 0.0, 0.0])

    xi_t_5 = np.array([[86.30, 0.8624], [2.50, 97.50]]) / 100.0
    b_t_5 = np.array([0.098, 0.0])

    delta_5_seconds = 5.0 * scale
    delta_t_seconds = delta_t * scale

    xi1, xi2, xi3, xi4 = 0.098, 3.8 / 2.9, 0.088, 0.025
    c_at = delta_5_seconds / xi1
    lambda_ = xi2
    beta = xi3
    c_lo = delta_5_seconds * beta / xi4

    xi1_prime = 1.0 - (lambda_ + beta) * delta_t_seconds / c_at
    xi2_prime = beta * delta_t_seconds / c_at
    xi3_prime = beta * delta_t_seconds / c_lo
    xi4_prime = 1.0 - beta * delta_t_seconds / c_lo
    xi_t = np.array([[xi1_prime, xi2_prime], [xi3_prime, xi4_prime]])
    b_t = np.array([delta_t_seconds / c_at, 0.0])

    if method == 2:
        xi_t = fractional_matrix_power(xi_t_5, 1.0 / 5.0).real

    return DiceTemperatureMatrices(
        phi_cc=phi_cc,
        b_cc=b_cc,
        xi_t=xi_t,
        b_t=b_t,
        xi_t_5=xi_t_5,
        b_t_5=b_t_5,
        xi1=xi1,
        xi2=xi2,
        xi3=xi3,
        xi4=xi4,
        c_at=c_at,
        c_lo=c_lo,
        lambda_=lambda_,
        beta=beta,
    )


def dice_temperature_simulation(
    t0: float,
    t_end: float,
    delta_t: float,
    y_fn: Callable[[float], float],
    mu_fn: Callable[[float], float],
    numeric: bool = False,
) -> np.ndarray:
    """Forward-simulate the DICE model's industrial emissions, carbon-cycle
    concentrations, radiative forcing, and temperature, from `t0` to
    `t_end` in steps of `delta_t`, given a GDP path `y_fn(t)` and a
    mitigation-rate path `mu_fn(t)`.

    `numeric=True` uses the 5-year-calibration temperature matrices
    (`xi_t_5`, `b_t_5`) directly instead of the continuous-time
    reconstruction from `dice_temperature_matrix`.

    Returns an array of shape ``(n_iters + 1, 10)`` with columns
    ``[t, CE_t, sigma_t, CC_AT_t, CC_UP_t, CC_LO_t, F_EX_t, F_RAD_t,
    T_AT_t, T_LO_t]``.

    Original: hsf/dice_temperature_simulation.m (the unused `parameters`
    argument is dropped -- see module docstring)
    """
    n_iters = int(round((t_end - t0) / delta_t))

    ce_land_0 = 3.3
    delta_land = 0.20
    sigma_0 = 0.5491
    g_sigma_0 = 0.01
    delta_sigma = 0.001

    cc_0 = np.array([830.4, 1527.0, 10010.0])

    eta = 3.8
    cc_at_1750 = 588.0
    f_ex_0 = 0.25
    f_ex_2100 = 0.70
    delta_f_ex = 5.0 * (f_ex_2100 - f_ex_0) / 90.0
    f_rad_0 = (eta / np.log(2)) * np.log(cc_0[0] / cc_at_1750) + f_ex_0

    t_0_vec = np.array([0.8, 0.0068])

    matrices = dice_temperature_matrix(delta_t, scale=1.0)
    phi_cc, b_cc = matrices.phi_cc, matrices.b_cc
    if numeric:
        xi_t = np.array([[86.30, 0.8624], [2.50, 97.50]]) / 100.0
        b_t = np.array([0.098, 0.0])
    else:
        xi_t, b_t = matrices.xi_t, matrices.b_t

    y_0 = y_fn(t0)
    mu_0 = mu_fn(t0)
    ce_industry_0 = (1 - mu_0) * sigma_0 * y_0
    ce_0 = ce_industry_0 + ce_land_0

    t = t0
    sigma_t = sigma_0
    ce_land_t = ce_land_0
    ce_t = ce_0
    g_sigma_t = g_sigma_0
    cc_t = cc_0.copy()
    f_ex_t = f_ex_0
    f_rad_t = f_rad_0
    t_t = t_0_vec.copy()

    results = np.zeros((n_iters + 1, 10))
    results[0, :] = np.concatenate(([t, ce_t, sigma_t], cc_t, [f_ex_t, f_rad_t], t_t))

    for it in range(1, n_iters + 1):
        t = t + delta_t

        y_t = y_fn(t)
        mu_t = mu_fn(t)

        ce_industry_t = (1 - mu_t) * sigma_t * y_t
        ce_land_t = ce_land_t * (1 - delta_land)
        ce_t = ce_industry_t + ce_land_t

        g_sigma_t = 1.0 / (1 + delta_sigma) * g_sigma_t
        sigma_t = (1 + g_sigma_t) * sigma_t

        cc_t = phi_cc @ cc_t + b_cc * ce_t
        cc_at_t = cc_t[0]

        if t <= 2100:
            f_ex_t = f_ex_t + delta_f_ex
        f_rad_t = (eta / np.log(2)) * np.log(cc_at_t / cc_at_1750) + f_ex_t

        t_t = xi_t @ t_t + b_t * f_rad_t
        results[it, :] = np.concatenate(([t, ce_t, sigma_t], cc_t, [f_ex_t, f_rad_t], t_t))

    return results
