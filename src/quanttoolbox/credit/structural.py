"""Structural (asset-value) credit models: Black-Scholes option pricing, the
classic Merton (1974) firm-value default model (calibrated from observed
equity value/volatility), Blasberg (2024)'s extended Merton model with a
stochastic growth-adjustment factor, the Black-Cox (1976) first-passage
model, Merton (1976) jump-diffusion option/credit pricing, and the Reinders
et al. credit-transition-loss model.

Ported from HSF toolbox `credit/{Black_Scholes_Model,PD_Merton_Model,
B0_Extended_Merton_Model,E0_Extended_Merton_Model,
PD_Extended_Merton_Model,PD_Black_Cox_Model,Merton_Jump_Model,
Merton_Jump_Climate_Model,Reinders_Credit_Model}.m`.

Translation notes:

- `B0_Extended_Merton_Model.m`/`E0_Extended_Merton_Model.m` accept `mu_a`
  but never use it in the formula body -- kept in the Python signature
  anyway, matching the original's own documented reason ("not used here,
  kept for consistent signature with merton_PD"): all three
  `*_extended_merton_model` functions share one 10-argument call
  signature, and `pd_extended_merton_model` *does* use `mu_a`.
- `Reinders_Credit_Model.m` accepts `mu_A` but never uses it anywhere in
  the function body, with no such cross-function-consistency rationale
  documented (unlike the extended-Merton trio above) -- dropped from
  `reinders_credit_model`'s signature as genuinely vestigial, the same
  treatment given to `dice_temperature_simulation`'s unused `parameters`
  argument in `sustainable_finance/climate.py`.
- `Merton_Jump_Model.m` sums `n_max = max(50, ceil(...))` Poisson-weighted
  Black-Scholes terms with no early exit for `lambda=0`, while its sibling
  `Merton_Jump_Climate_Model.m` does `break` once `lambda=0` is detected.
  Both are mathematically equivalent either way -- once `lambda=0`, the
  Poisson weight `p_n` is exactly `0.0` for every `n >= 1` (verified: `p_n`
  recurses as ``p_n *= lambda*T/(n+1)``, which multiplies by `0.0` once
  `lambda=0`), so the terms for `n >= 1` contribute nothing regardless of
  whether the loop keeps running. The same early exit is added to both
  functions here as a pure performance improvement (skips the redundant
  `n_max` remaining iterations), not a behavior change.
- `pd_merton_model` (`PD_Merton_Model.m`) calibrates the unobserved asset
  value/volatility `(A0, sigma_A)` from observed equity value/volatility
  `(E0, sigma_E)` by minimizing a 2-equation least-squares objective --
  MATLAB's `fminunc` (unconstrained quasi-Newton) is `scipy.optimize.
  minimize(method="BFGS")` here. Positivity of `(A0, sigma_A)` is enforced
  via `abs()` inside the objective and on the final result, exactly as in
  the original (no bounds are passed to the optimizer either way). The
  original's manual "replicate every scalar/array input to a common length
  `n`" broadcasting (``e = ones(n,1); E0 = E0.*e; ...``) is replaced with
  `numpy.broadcast_arrays`, which does the same thing more directly; the
  per-scenario nonlinear solve itself still runs in a loop, since each
  scenario is an independent 2-parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from quanttoolbox.config import EstimationConfig


@dataclass
class BlackScholesResult:
    """European call/put prices under the generalized Black-Scholes
    model with cost-of-carry `b` (`b = r` for equities with no dividend,
    `b = r - q` for a dividend yield `q`, `b = 0` for futures, `b = r -
    r_f` for FX)."""

    call: np.ndarray
    put: np.ndarray


def black_scholes(
    s0: np.ndarray | float,
    k: np.ndarray | float,
    sigma: np.ndarray | float,
    t: np.ndarray | float,
    b: np.ndarray | float,
    r: np.ndarray | float,
) -> BlackScholesResult:
    """Generalized Black-Scholes European call/put prices with
    cost-of-carry `b` (spot `s0`, strike `k`, volatility `sigma`, maturity
    `t`, risk-free rate `r`).

    Original: credit/Black_Scholes_Model.m
    """
    d1 = (np.log(s0 / k) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    call = s0 * np.exp((b - r) * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    put = -s0 * np.exp((b - r) * t) * norm.cdf(-d1) + k * np.exp(-r * t) * norm.cdf(-d2)
    return BlackScholesResult(call=call, put=put)


@dataclass
class PdMertonModelResult:
    """Merton (1974) calibrated default probability: `a0`/`sigma_a` are
    the calibrated (unobserved) asset value/volatility, `dd_tau` the
    distance-to-default and `s_tau`/`pd_tau` the survival/default
    probability at horizon `tau`."""

    pd_tau: np.ndarray
    a0: np.ndarray
    sigma_a: np.ndarray
    s_tau: np.ndarray
    dd_tau: np.ndarray


def _pd_merton_model_objective(
    params: np.ndarray, e0: float, sigma_e: float, d: float, r: float, t: float
) -> float:
    a0, sigma_a = np.abs(params)
    d1 = (np.log(a0 / d) + (r + 0.5 * sigma_a**2) * t) / (sigma_a * np.sqrt(t))
    d2 = d1 - sigma_a * np.sqrt(t)

    obj1 = a0 * norm.cdf(d1) - np.exp(-r * t) * d * norm.cdf(d2) - e0
    obj2 = sigma_e * e0 - sigma_a * a0 * norm.cdf(d1)
    return float(obj1**2 + obj2**2)


def pd_merton_model(
    e0: np.ndarray | float,
    sigma_e: np.ndarray | float,
    d: np.ndarray | float,
    mu_a: np.ndarray | float,
    r: np.ndarray | float,
    t: np.ndarray | float,
    tau: np.ndarray | float,
    config: EstimationConfig | None = None,
) -> PdMertonModelResult:
    """Calibrate the Merton (1974) model's unobserved asset value/
    volatility `(A0, sigma_A)` from observed equity value/volatility
    `(E0, sigma_E)` and debt face value `D` at maturity `T` (via a
    2-equation least-squares fit), then compute the physical
    (real-world, drift `mu_a`) probability of default at horizon `tau`.

    Original: credit/PD_Merton_Model.m
    """
    if config is None:
        config = EstimationConfig()

    e0_arr, sigma_e_arr, d_arr, mu_a_arr, r_arr, t_arr, tau_arr = np.broadcast_arrays(
        np.asarray(e0, dtype=float),
        np.asarray(sigma_e, dtype=float),
        np.asarray(d, dtype=float),
        np.asarray(mu_a, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(tau, dtype=float),
    )
    shape = e0_arr.shape
    n = e0_arr.size

    a0_flat = np.empty(n)
    sigma_a_flat = np.empty(n)

    for i, (e0_i, sigma_e_i, d_i, r_i, t_i) in enumerate(
        zip(
            e0_arr.ravel(),
            sigma_e_arr.ravel(),
            d_arr.ravel(),
            r_arr.ravel(),
            t_arr.ravel(),
            strict=True,
        )
    ):
        result = minimize(
            _pd_merton_model_objective,
            x0=np.array([e0_i, sigma_e_i]),
            args=(e0_i, sigma_e_i, d_i, r_i, t_i),
            method="BFGS",
            options={"gtol": config.tol, "maxiter": config.max_iters},
        )
        a0_i, sigma_a_i = np.abs(result.x)
        a0_flat[i] = a0_i
        sigma_a_flat[i] = sigma_a_i

    a0 = a0_flat.reshape(shape)
    sigma_a = sigma_a_flat.reshape(shape)

    dd_tau = (np.log(a0 / d_arr) + (mu_a_arr - 0.5 * sigma_a**2) * tau_arr) / (
        sigma_a * np.sqrt(tau_arr)
    )
    s_tau = norm.cdf(dd_tau)
    pd_tau = 1.0 - s_tau

    return PdMertonModelResult(pd_tau=pd_tau, a0=a0, sigma_a=sigma_a, s_tau=s_tau, dd_tau=dd_tau)


def _extended_merton_reparametrize(
    sigma_a: np.ndarray | float,
    sigma_delta: np.ndarray | float,
    rho: np.ndarray | float,
    delta0: np.ndarray | float,
    mu_delta: np.ndarray | float,
    t: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    sigma_a_prime = np.sqrt(
        sigma_a**2 - rho * sigma_a * sigma_delta * t + (1.0 / 3.0) * sigma_delta**2 * t**2
    )
    delta0_prime = (
        delta0
        + 0.5 * mu_delta * t
        + 0.5 * rho * sigma_a * sigma_delta * t
        - (1.0 / 6.0) * sigma_delta**2 * t**2
    )
    return sigma_a_prime, delta0_prime


def b0_extended_merton_model(
    a0: np.ndarray | float,
    d: np.ndarray | float,
    r: np.ndarray | float,
    mu_a: np.ndarray | float,
    delta0: np.ndarray | float,
    mu_delta: np.ndarray | float,
    sigma_a: np.ndarray | float,
    sigma_delta: np.ndarray | float,
    rho: np.ndarray | float,
    t: np.ndarray | float,
) -> np.ndarray:
    """Bond (debt) value at t=0 in Blasberg (2024)'s extended Merton
    model, where the firm's growth-adjustment factor `delta(t)` follows
    its own Brownian motion correlated (`rho`) with the asset value.
    `mu_a` is accepted but unused -- see module docstring.

    Balance-sheet check: ``e0 + b0 == a0 * exp(-delta0_prime * t)``.

    Original: credit/B0_Extended_Merton_Model.m
    """
    sigma_a_prime, delta0_prime = _extended_merton_reparametrize(
        sigma_a, sigma_delta, rho, delta0, mu_delta, t
    )
    d1 = (np.log(a0 / d) + r * t - delta0_prime * t) / (
        sigma_a_prime * np.sqrt(t)
    ) + 0.5 * sigma_a_prime * np.sqrt(t)
    d2 = d1 - sigma_a_prime * np.sqrt(t)

    return a0 * np.exp(-delta0_prime * t) * norm.cdf(-d1) + d * np.exp(-r * t) * norm.cdf(d2)


def e0_extended_merton_model(
    a0: np.ndarray | float,
    d: np.ndarray | float,
    r: np.ndarray | float,
    mu_a: np.ndarray | float,
    delta0: np.ndarray | float,
    mu_delta: np.ndarray | float,
    sigma_a: np.ndarray | float,
    sigma_delta: np.ndarray | float,
    rho: np.ndarray | float,
    t: np.ndarray | float,
) -> np.ndarray:
    """Equity value at t=0 in Blasberg (2024)'s extended Merton model --
    see `b0_extended_merton_model` for the model description.

    Original: credit/E0_Extended_Merton_Model.m
    """
    sigma_a_prime, delta0_prime = _extended_merton_reparametrize(
        sigma_a, sigma_delta, rho, delta0, mu_delta, t
    )
    d1 = (np.log(a0 / d) + r * t - delta0_prime * t) / (
        sigma_a_prime * np.sqrt(t)
    ) + 0.5 * sigma_a_prime * np.sqrt(t)
    d2 = d1 - sigma_a_prime * np.sqrt(t)

    return a0 * np.exp(-delta0_prime * t) * norm.cdf(d1) - d * np.exp(-r * t) * norm.cdf(d2)


def pd_extended_merton_model(
    a0: np.ndarray | float,
    d: np.ndarray | float,
    r: np.ndarray | float,
    mu_a: np.ndarray | float,
    delta0: np.ndarray | float,
    mu_delta: np.ndarray | float,
    sigma_a: np.ndarray | float,
    sigma_delta: np.ndarray | float,
    rho: np.ndarray | float,
    t: np.ndarray | float,
) -> np.ndarray:
    """Physical (real-world, drift `mu_a`) probability of default at
    horizon `t` in Blasberg (2024)'s extended Merton model. `r` is
    accepted but unused (kept for signature consistency with
    `b0_extended_merton_model`/`e0_extended_merton_model`).

    Original: credit/PD_Extended_Merton_Model.m
    """
    sigma_a_prime, _ = _extended_merton_reparametrize(
        sigma_a, sigma_delta, rho, delta0, mu_delta, t
    )
    dd = (np.log(a0 / d) + (mu_a - delta0 - 0.5 * sigma_a**2) * t - 0.5 * mu_delta * t**2) / (
        sigma_a_prime * np.sqrt(t)
    )
    return norm.cdf(-dd)


@dataclass
class PdBlackCoxResult:
    """Black-Cox (1976) first-passage default probability at horizon
    `tau`, plus the intermediate `d1`/`d2`/`varphi` terms."""

    pd_tau: np.ndarray
    s_tau: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    varphi: np.ndarray


def pd_black_cox_model(
    a0: np.ndarray | float,
    mu_a: np.ndarray | float,
    sigma_a: np.ndarray | float,
    b: np.ndarray | float,
    tau: np.ndarray | float,
) -> PdBlackCoxResult:
    """Black-Cox (1976) first-passage-time default probability: the firm
    defaults as soon as its asset value `A(t)` (geometric Brownian motion
    with drift `mu_a`, volatility `sigma_a`, starting at `a0`) first
    crosses the constant barrier `b`, evaluated over horizon `tau`.

    Original: credit/PD_Black_Cox_Model.m
    """
    sigma_tau = sigma_a * np.sqrt(tau)
    sigma2_a = sigma_a * sigma_a
    nu_a = mu_a - 0.5 * sigma2_a
    varphi = (b / a0) ** (2.0 * nu_a / sigma2_a)

    d1 = (np.log(a0) - np.log(b) + mu_a * tau) / sigma_tau - 0.5 * sigma_tau
    d2 = (np.log(b) - np.log(a0) + mu_a * tau) / sigma_tau - 0.5 * sigma_tau

    s_tau = norm.cdf(d1) - varphi * norm.cdf(d2)
    pd_tau = 1.0 - s_tau

    return PdBlackCoxResult(pd_tau=pd_tau, s_tau=s_tau, d1=d1, d2=d2, varphi=varphi)


@dataclass
class MertonJumpResult:
    """Merton (1976) jump-diffusion European call/put prices, plus `k`
    (the expected relative jump size, used in the risk-neutral drift
    correction)."""

    call: np.ndarray
    put: np.ndarray
    k: np.ndarray


def merton_jump_model(
    s0: np.ndarray | float,
    k: np.ndarray | float,
    sigma: np.ndarray | float,
    t: np.ndarray | float,
    b: np.ndarray | float,
    r: np.ndarray | float,
    lambda_: float,
    mu_z: np.ndarray | float,
    sigma_z: np.ndarray | float,
) -> MertonJumpResult:
    """Merton (1976) jump-diffusion European option prices: a Poisson
    (rate `lambda_`) mixture of Black-Scholes prices, one per possible
    jump count `n`, with lognormal jump sizes (`mu_z`, `sigma_z`).

    Original: credit/Merton_Jump_Model.m
    """
    jump_k = np.exp(mu_z + 0.5 * sigma_z**2) - 1.0
    call = np.zeros_like(np.broadcast_arrays(s0, k, sigma, t, b, r)[0], dtype=float)
    put = np.zeros_like(call)

    n_max = int(max(50.0, float(np.ceil(lambda_ * t + 4.0 * np.sqrt(lambda_ * t)))))
    p_n = np.exp(-lambda_ * t)

    for n in range(n_max + 1):
        b_n = b - lambda_ * jump_k + n * np.log(1.0 + jump_k) / t
        sigma_n = np.sqrt(sigma**2 + (n * sigma_z**2) / t)
        bs_n = black_scholes(s0, k, sigma_n, t, b_n, r)

        call = call + p_n * bs_n.call
        put = put + p_n * bs_n.put
        p_n = p_n * (lambda_ * t) / (n + 1)
        if lambda_ == 0:
            break

    return MertonJumpResult(call=call, put=put, k=jump_k)


@dataclass
class MertonJumpClimateResult:
    """Merton (1976) jump-diffusion firm-value equity/bond values (a
    climate-risk application: sudden jumps represent transition-risk
    shocks to asset value), plus `k` (the expected relative jump size)."""

    e0: np.ndarray
    b0: np.ndarray
    k: np.ndarray


def merton_jump_climate_model(
    a0: np.ndarray | float,
    d: np.ndarray | float,
    sigma_a: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    lambda_: float,
    mu_z: np.ndarray | float,
    sigma_z: np.ndarray | float,
) -> MertonJumpClimateResult:
    """Merton (1976) jump-diffusion firm-value equity/bond values: the
    firm's asset value follows a jump-diffusion (Poisson rate `lambda_`,
    lognormal jump sizes `mu_z`/`sigma_z`) rather than plain geometric
    Brownian motion, otherwise the classic Merton (1974) structural
    setup (debt face value `d`, maturity `t`, risk-free rate `r`).

    Original: credit/Merton_Jump_Climate_Model.m
    """
    jump_k = np.exp(mu_z + 0.5 * sigma_z**2) - 1.0
    e0 = np.zeros_like(np.broadcast_arrays(a0, d, sigma_a, t, r)[0], dtype=float)
    b0 = np.zeros_like(e0)

    n_max = int(max(500.0, float(np.ceil(lambda_ * t + 4.0 * np.sqrt(lambda_ * t)))))
    p_n = np.exp(-lambda_ * t)

    for n in range(n_max + 1):
        b_n = r - lambda_ * jump_k + n * np.log(1.0 + jump_k) / t
        sigma_n = np.sqrt(sigma_a**2 + (n * sigma_z**2) / t)
        d1_n = (np.log(a0 / d) + (b_n + 0.5 * sigma_n**2) * t) / (sigma_n * np.sqrt(t))
        d2_n = d1_n - sigma_n * np.sqrt(t)

        e_n = a0 * np.exp((b_n - r) * t) * norm.cdf(d1_n) - d * np.exp(-r * t) * norm.cdf(d2_n)
        b_n_value = a0 * np.exp((b_n - r) * t) * norm.cdf(-d1_n) + d * np.exp(-r * t) * norm.cdf(
            d2_n
        )

        e0 = e0 + p_n * e_n
        b0 = b0 + p_n * b_n_value
        p_n = p_n * (lambda_ * t) / (n + 1)
        if lambda_ == 0:
            break

    return MertonJumpClimateResult(e0=e0, b0=b0, k=jump_k)


@dataclass
class ReindersCreditModelResult:
    """Reinders et al. credit-transition loss at asset-value shock `xi`:
    `loss` (the equity+debt mark-to-market loss), its first (`d_loss`)
    and second (`d2_loss`) derivatives with respect to `xi`, and the
    pre-/post-shock equity/debt values."""

    loss: np.ndarray
    d_loss: np.ndarray
    d2_loss: np.ndarray
    mv_e_t0: np.ndarray
    mv_d_t0: np.ndarray
    mv_e_t: np.ndarray
    mv_d_t: np.ndarray


def reinders_credit_model(
    xi: np.ndarray | float,
    a0: np.ndarray | float,
    d: np.ndarray | float,
    r: np.ndarray | float,
    sigma_a: np.ndarray | float,
    t: np.ndarray | float,
    omega_e: np.ndarray | float,
    omega_d: np.ndarray | float,
) -> ReindersCreditModelResult:
    """Reinders et al.'s credit-transition loss model: the equity+debt
    mark-to-market loss from an instantaneous fractional shock `xi` to
    the firm's asset value (`A(t) = A0 * (1 - xi)`), weighted by
    `omega_e`/`omega_d` (e.g. the investor's equity/debt holdings), plus
    the loss's first/second derivatives with respect to `xi`.

    Original: credit/Reinders_Credit_Model.m (the `mu_A` parameter is
    accepted but never used in the original's formula body, with no
    documented reason -- dropped here, see module docstring)
    """
    d1 = (np.log(a0 / d) + r * t) / (sigma_a * np.sqrt(t)) + 0.5 * sigma_a * np.sqrt(t)
    d2 = d1 - sigma_a * np.sqrt(t)
    mv_e_t0 = a0 * norm.cdf(d1) - d * np.exp(-r * t) * norm.cdf(d2)
    mv_d_t0 = a0 * norm.cdf(-d1) + d * np.exp(-r * t) * norm.cdf(d2)

    a_t = a0 * (1.0 - xi)
    d1 = (np.log(a_t / d) + r * t) / (sigma_a * np.sqrt(t)) + 0.5 * sigma_a * np.sqrt(t)
    d2 = d1 - sigma_a * np.sqrt(t)
    mv_e_t = a_t * norm.cdf(d1) - d * np.exp(-r * t) * norm.cdf(d2)
    mv_d_t = a_t * norm.cdf(-d1) + d * np.exp(-r * t) * norm.cdf(d2)

    loss = omega_e * (mv_e_t0 - mv_e_t) + omega_d * (mv_d_t0 - mv_d_t)
    d_loss = a0 * (omega_e * norm.cdf(d1) + omega_d * norm.cdf(-d1))
    d2_loss = a0 * (omega_d - omega_e) * norm.pdf(d1) / (1.0 - xi) / (sigma_a * np.sqrt(t))

    return ReindersCreditModelResult(
        loss=loss,
        d_loss=d_loss,
        d2_loss=d2_loss,
        mv_e_t0=mv_e_t0,
        mv_d_t0=mv_d_t0,
        mv_e_t=mv_e_t,
        mv_d_t=mv_d_t,
    )
