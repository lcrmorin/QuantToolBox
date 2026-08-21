"""Cumulative carbon-budget calculations: the total emissions ``integral of
CE(s) ds`` over ``[t0, t]`` under various assumed emissions trajectories
CE(s) (constant decline rates, GDP-adjusted compound decline, and
piecewise-linear historical/target paths).

Ported from HSF toolbox `hsf/{carbon_budget_linear,
carbon_budget_linear_trend,carbon_budget_linear_reduction,
carbon_budget_piecewise,carbon_budget_compound_reduction,
carbon_budget_Reduction}.m`.

Translation notes:

- `carbon_budget_linear.m` and `carbon_budget_linear_trend.m` are the exact
  same computation (``beta0 * (t - t0) + 0.5 * beta1 * (t^2 - t0^2)``,
  the closed-form integral of ``beta0 + beta1 * s``) -- the original files
  differ only in which of their two outputs (closed form vs. a
  `scipy.integrate.quad`-equivalent numerical cross-check) comes first.
  Merged into one function, `carbon_budget_linear`.
- `carbon_budget_linear_reduction.m` is a strict special case of
  `carbon_budget_Reduction.m`'s default ("linear rate") method, under the
  substitution ``r = reduction * ce_t0`` (verified algebraically and in
  tests) -- not ported as a separate function; use ``carbon_budget(t0, t,
  ce_t0, r=reduction * ce_t0, method=1)`` instead.
- Every original file returns both a closed-form value and a
  `integral()`-computed numerical cross-check of the same quantity (a
  self-verification pattern, not two different pieces of information).
  Only the closed-form value is exposed here; the numerical
  cross-check is instead used in this module's own test suite (via
  `scipy.integrate.quad`), which is a strictly better place for a
  redundant-by-construction correctness check than a return value every
  caller pays for.
- MATLAB's ``integral`` maps to `scipy.integrate.quad` where a numerical
  integral genuinely differs from a closed form (`carbon_budget_piecewise`'s
  piecewise-linear-interpolated integral, verified in tests against the
  closed-form sum).
"""

from __future__ import annotations

import numpy as np


def carbon_budget_linear(t0: float, t: float, beta0: float, beta1: float) -> float:
    """Cumulative emissions over [t0, t] under a linear emissions trend
    CE(s) = beta0 + beta1 * s.

    Original: hsf/carbon_budget_linear.m (identical to
    hsf/carbon_budget_linear_trend.m -- see module docstring)
    """
    return beta0 * (t - t0) + 0.5 * beta1 * (t**2 - t0**2)


def carbon_budget(t0: float, t: float, ce_t0: float, r: float, method: int = 1) -> float:
    """Cumulative emissions over [t0, t] starting from CE(t0), declining (or
    growing) at rate `r` under one of three conventions:

    - method=1 (default, "linear rate"): CE(s) = CE(t0) - r * (s - t0)
    - method=2 ("compound rate"): CE(s) = CE(t0) * (1 - r)^(s - t0)
    - method=3 ("growth rate"): CE(s) = CE(t0) * exp(-r * (s - t0))

    Original: hsf/carbon_budget_Reduction.m
    """
    dt = t - t0
    if method == 3:
        return (1.0 - np.exp(-r * dt)) / r * ce_t0
    if method == 2:
        return ((1.0 - r) ** dt - 1.0) / np.log(1.0 - r) * ce_t0
    return dt * ce_t0 - 0.5 * dt**2 * r


def carbon_budget_piecewise(t0: float, t: float, t_k: np.ndarray, ce_k: np.ndarray) -> float:
    """Cumulative emissions over [t0, t], for a piecewise-linear-interpolated
    emissions trajectory given at knots (`t_k`, `ce_k`).

    Original: hsf/carbon_budget_piecewise.m (this is `CB1`/`CB3`, the two
    closed-form sums the original computes -- verified algebraically
    identical, see this module's tests)
    """
    t_k = np.asarray(t_k, dtype=float)
    ce_k = np.asarray(ce_k, dtype=float)

    ce_t0 = float(np.interp(t0, t_k, ce_k))
    ce_t = float(np.interp(t, t_k, ce_k))

    mask = (t_k >= t0) & (t_k <= t)
    t_k = t_k[mask]
    ce_k = ce_k[mask]

    if t_k.shape[0] == 0 or t_k[0] != t0:
        t_k = np.concatenate(([t0], t_k))
        ce_k = np.concatenate(([ce_t0], ce_k))
    if t_k[-1] != t:
        t_k = np.concatenate((t_k, [t]))
        ce_k = np.concatenate((ce_k, [ce_t]))

    t_k1, t_k2 = t_k[:-1], t_k[1:]
    ce_k1, ce_k2 = ce_k[:-1], ce_k[1:]
    dt_k = t_k2 - t_k1

    beta0 = (t_k2 / dt_k) * ce_k1 - (t_k1 / dt_k) * ce_k2
    beta1 = (ce_k2 - ce_k1) / dt_k

    segments = beta0 * (t_k2 - t_k1) + 0.5 * beta1 * (t_k2**2 - t_k1**2)
    return float(np.sum(segments))


def carbon_budget_compound_reduction(
    t0: float,
    t: float,
    delta_r: float,
    r_minus: float,
    ce_t0: float,
    g_y: float | None = None,
) -> float:
    """Cumulative emissions over [t0, t], for CE(s) declining at compound
    rate `delta_r` from an initial level ``CE(t0) * (1 - r_minus)``,
    optionally compounded against a GDP growth rate `g_y` (CE(s) then
    grows/shrinks at the net rate of decarbonization vs. growth).

    Original: hsf/carbon_budget_compound_reduction.m
    """
    dt = t - t0
    if g_y is None:
        return ((1.0 - delta_r) ** dt - 1.0) / np.log(1.0 - delta_r) * (1.0 - r_minus) * ce_t0
    return (
        ((1.0 + g_y) ** dt * (1.0 - delta_r) ** dt - 1.0)
        / (np.log(1.0 + g_y) + np.log(1.0 - delta_r))
        * (1.0 - r_minus)
        * ce_t0
    )
