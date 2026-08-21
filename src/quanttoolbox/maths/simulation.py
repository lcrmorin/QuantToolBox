"""Geometric Brownian motion simulation, EWMA-based mean/vol estimation,
volatility targeting, and continuous algebraic Riccati / Lyapunov equation
solvers.

Ported from QuantToolBox/maths/{simulate_gbm,simulate_gbm2,
simulate_multi_gbm,compute_ewma,momentum_ewma,volatility_target,
algebraic_riccati_equation,lyapunov_equation}.m

Translation notes:

- ``algebraic_riccati_equation.m``/``lyapunov_equation.m`` hand-roll a
  Schur-decomposition solver and a Kronecker-product linear solve,
  respectively, for two classic control-theory equations that
  ``scipy.linalg`` already solves directly and robustly
  (``solve_continuous_are``, ``solve_lyapunov``) -- used here instead of
  porting the custom solvers.
- ``simulate_multi_gbm.m``'s body is byte-identical to
  ``simulate_gbm.m`` (single-asset simulation) despite taking a
  correlation parameter ``rho`` that is never used -- this looks like an
  incomplete/unfinished implementation in the original rather than an
  intentional simplification. ``simulate_multi_gbm`` here instead
  implements what the name and signature promise: a genuine N-asset
  correlated GBM simulation via Cholesky decomposition of the
  correlation matrix (generalizing ``simulate_gbm2``'s 2-asset case,
  which *is* correctly implemented in the original and is ported as-is).
- ``momentum_ewma`` simulates and analytically decomposes a simple
  EWMA-momentum trend-following strategy, matching the original's field
  names conceptually (renamed to snake_case, e.g. ``V_t`` ->
  ``v_t``/``result.v_t``) via a ``MomentumEWMAResult`` dataclass rather
  than a MATLAB-style long positional return tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are, solve_lyapunov, sqrtm


def simulate_gbm(
    x0: float,
    mu: float,
    sigma: float,
    t: np.ndarray,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths (exact scheme, no
    discretization bias) for a single asset.

    Original: maths/simulate_gbm.m

    Returns an (n_times, n_paths) array.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(t, dtype=float).flatten()
    n_t = t.shape[0]

    x = np.zeros((n_t, n_paths))
    x_prev = np.full(n_paths, x0)

    for i in range(n_t):
        dt = t[0] if i == 0 else t[i] - t[i - 1]
        k1 = (mu - 0.5 * sigma**2) * dt
        k2 = sigma * np.sqrt(dt)
        u = rng.standard_normal(n_paths)
        x_prev = x_prev * np.exp(k1 + k2 * u)
        x[i, :] = x_prev

    return x


def simulate_gbm2(
    x01: float,
    x02: float,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    t: np.ndarray,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate two correlated geometric Brownian motion processes (exact
    scheme).

    Original: maths/simulate_gbm2.m
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(t, dtype=float).flatten()
    n_t = t.shape[0]

    x1 = np.zeros((n_t, n_paths))
    x2 = np.zeros((n_t, n_paths))
    x1_prev = np.full(n_paths, x01)
    x2_prev = np.full(n_paths, x02)

    rho2 = rho**2

    for i in range(n_t):
        dt = t[0] if i == 0 else t[i] - t[i - 1]
        k1 = (mu1 - 0.5 * sigma1**2) * dt
        k2 = (mu2 - 0.5 * sigma2**2) * dt
        sqrt_dt = np.sqrt(dt)

        u1 = rng.standard_normal(n_paths)
        u2 = rho * u1 + np.sqrt(1 - rho2) * rng.standard_normal(n_paths)

        x1_prev = x1_prev * np.exp(k1 + sigma1 * sqrt_dt * u1)
        x2_prev = x2_prev * np.exp(k2 + sigma2 * sqrt_dt * u2)

        x1[i, :] = x1_prev
        x2[i, :] = x2_prev

    return x1, x2


def simulate_multi_gbm(
    x0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    t: np.ndarray,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate N correlated geometric Brownian motion processes (exact
    scheme), via Cholesky decomposition of the correlation matrix rho.

    Note: implemented as a genuine N-asset correlated simulator (see
    module docstring for why this differs from the original, whose
    ``simulate_multi_gbm.m`` body is identical to the uncorrelated
    single-asset ``simulate_gbm.m`` despite taking a correlation
    parameter).

    Original: maths/simulate_multi_gbm.m (reimplemented -- see docstring)

    Returns an (n_times, n_assets, n_paths) array.
    """
    rng = np.random.default_rng() if rng is None else rng
    x0 = np.asarray(x0, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float).flatten()
    rho = np.asarray(rho, dtype=float)
    t = np.asarray(t, dtype=float).flatten()
    n_assets = x0.shape[0]
    n_t = t.shape[0]

    chol = np.linalg.cholesky(rho)

    x = np.zeros((n_t, n_assets, n_paths))
    x_prev = np.tile(x0[:, None], (1, n_paths))

    for i in range(n_t):
        dt = t[0] if i == 0 else t[i] - t[i - 1]
        k1 = (mu - 0.5 * sigma**2) * dt
        sqrt_dt = np.sqrt(dt)

        z = rng.standard_normal((n_assets, n_paths))
        u = chol @ z

        x_prev = x_prev * np.exp(k1[:, None] + sigma[:, None] * sqrt_dt * u)
        x[i, :, :] = x_prev

    return x


def compute_ewma(
    prices: np.ndarray, lambda_mu: float, lambda_sigma: float | None = None, dt: float = 1.0 / 260
) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially-weighted moving average mean and volatility of a
    return series (in the mean-reversion-rate parameterization: larger
    lambda means faster decay toward the new observation).

    Original: maths/compute_ewma.m
    """
    from quanttoolbox.backtest.returns import price_to_return

    if lambda_sigma is None:
        lambda_sigma = lambda_mu

    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 1:
        prices = prices[:, None]
    n_rows, n_cols = prices.shape

    r_daily = price_to_return(prices, 1)

    mu_t = np.zeros((n_rows, n_cols))
    sigma0 = np.sqrt(1 / dt) * np.nanstd(r_daily, axis=0, ddof=1)
    sigma2_t = np.zeros((n_rows, n_cols))
    sigma2_t[0, :] = sigma0**2

    r_filled = np.where(np.isnan(r_daily), 0.0, r_daily)

    for i in range(1, n_rows):
        mu_t[i, :] = (1 - lambda_mu * dt) * mu_t[i - 1, :] + lambda_mu * r_filled[i, :]
        sigma2_t[i, :] = (1 - lambda_sigma * dt) * sigma2_t[i - 1, :] + lambda_sigma * r_filled[
            i, :
        ] ** 2

    sigma_t = np.sqrt(sigma2_t)
    missing = np.isnan(prices)
    mu_t = np.where(missing, np.nan, mu_t)
    sigma_t = np.where(missing, np.nan, sigma_t)

    return mu_t, sigma_t


@dataclass
class MomentumEWMAResult:
    r_realized: np.ndarray  # R_St: raw daily returns
    v_t: np.ndarray  # realized wealth index from actually trading the (lagged) exposure
    v_tilde_t: np.ndarray  # theoretical wealth index (gamma + theta components combined)
    g_t: np.ndarray  # theoretical wealth index, "gamma" component only
    g_small_t: np.ndarray  # theoretical wealth index, "theta"/variance-drag component only
    r_v_t: np.ndarray  # realized strategy return series (exposure * next-period return)
    r_tilde_v_t: np.ndarray  # theoretical strategy return series (gamma + theta)
    r_tilde_g_t: np.ndarray  # "gamma" component of the theoretical return
    r_tilde_g_small_t: np.ndarray  # "theta" component of the theoretical return
    e_t: np.ndarray  # applied exposure (position size) each period


def momentum_ewma(
    prices: np.ndarray,
    alpha: float,
    lambda_mu: float,
    lambda_sigma: float | None = None,
    dt: float = 1.0 / 260,
    multiplier: float = 1.0,
) -> MomentumEWMAResult:
    """Simulate and analytically decompose a simple EWMA-momentum
    trend-following strategy: exposure e_t = alpha * mu_t (position size
    proportional to the EWMA-estimated drift mu_t), applied with a
    one-period lag to next-period returns.

    Returns both the actually-realized strategy wealth index (``v_t``,
    from compounding the lagged-exposure-weighted returns) and a
    continuous-time-motivated analytical approximation to it, decomposed
    into a "gamma" component driven by changes in the squared EWMA drift
    (``g_t``) and a "theta"/variance-drag component (``g_small_t``); their
    product-equivalent sum is ``v_tilde_t``, which approximates ``v_t``.

    Original: maths/momentum_ewma.m
    """
    from quanttoolbox.backtest.returns import price_to_return

    if lambda_sigma is None:
        lambda_sigma = lambda_mu

    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 1:
        prices = prices[:, None]
    n_rows, n_cols = prices.shape

    mu_t, sigma_t = compute_ewma(prices, lambda_mu, lambda_sigma, dt)

    r_st = price_to_return(prices, 1)
    r_st = np.where(np.isnan(r_st), 0.0, r_st)

    e_lag = np.zeros(n_cols)
    r_vt = np.zeros((n_rows, n_cols))
    r_tilde_gt = np.zeros((n_rows, n_cols))
    r_tilde_g_small_t = np.zeros((n_rows, n_cols))
    e_t = np.zeros((n_rows, n_cols))
    mu_lag = mu_t[0, :]

    for k in range(1, n_rows):
        mu_k = mu_t[k, :]
        sigma_k = sigma_t[k, :]
        sr_k = mu_k / sigma_k

        r_vt[k, :] = e_lag * r_st[k, :]
        r_tilde_gt[k, :] = (0.5 * alpha / lambda_mu) * (mu_k**2 - mu_lag**2) * multiplier
        r_tilde_g_small_t[k, :] = (
            alpha
            * sigma_k**2
            * (sr_k**2 * (1 - 0.5 * alpha * sigma_k**2) - 0.5 * lambda_mu)
            * dt
            * multiplier
        )

        e_lag = alpha * mu_k * multiplier
        e_t[k, :] = e_lag
        mu_lag = mu_k

    r_tilde_vt = r_tilde_gt + r_tilde_g_small_t

    v_t = 100 * np.cumprod(1 + r_vt, axis=0)
    g_t = 100 * np.exp(np.cumsum(r_tilde_gt, axis=0))
    g_small_t = 100 * np.exp(np.cumsum(r_tilde_g_small_t, axis=0))
    v_tilde_t = 100 * np.exp(np.cumsum(r_tilde_vt, axis=0))

    return MomentumEWMAResult(
        r_realized=r_st,
        v_t=v_t,
        v_tilde_t=v_tilde_t,
        g_t=g_t,
        g_small_t=g_small_t,
        r_v_t=r_vt,
        r_tilde_v_t=r_tilde_vt,
        r_tilde_g_t=r_tilde_gt,
        r_tilde_g_small_t=r_tilde_g_small_t,
        e_t=e_t,
    )


@dataclass
class VolatilityTargetResult:
    y_t: np.ndarray  # rebased wealth index (base 100)
    sigma_t: np.ndarray  # estimated volatility
    leverage_t: np.ndarray  # applied leverage


def volatility_target(
    x_t: np.ndarray,
    lambda_: float,
    vol_target: float,
    min_leverage: float | None = 0.0,
    max_leverage: float | None = 1.0,
    dt: float = 1.0 / 260,
    multiplier: float = 1.0,
) -> VolatilityTargetResult:
    """Apply a volatility-targeting overlay to a price series: scale daily
    returns by leverage = vol_target / (EWMA volatility), clipped to
    [min_leverage, max_leverage] and lagged by one period (leverage is
    set using yesterday's volatility estimate).

    Original: maths/volatility_target.m
    """
    from quanttoolbox.backtest.returns import price_to_return

    x_t = np.asarray(x_t, dtype=float)
    if x_t.ndim == 1:
        x_t = x_t[:, None]

    _, sigma_t = compute_ewma(x_t, lambda_, lambda_, dt)
    sigma_t = multiplier * sigma_t

    leverage_t = vol_target / sigma_t
    if min_leverage is not None and max_leverage is not None:
        leverage_t = np.clip(leverage_t, min_leverage, max_leverage)
    elif max_leverage is not None:
        leverage_t = np.minimum(leverage_t, max_leverage)
    elif min_leverage is not None:
        leverage_t = np.maximum(leverage_t, min_leverage)

    leverage_lagged = np.full_like(leverage_t, np.nan)
    leverage_lagged[1:] = leverage_t[:-1]
    leverage_lagged[0] = leverage_lagged[1] if leverage_lagged.shape[0] > 1 else np.nan

    r_t = price_to_return(x_t, 1)
    r_filled = np.where(np.isnan(r_t), 0.0, r_t)
    r_levered = leverage_lagged * r_filled

    y_t = 100 * np.cumprod(1 + r_levered, axis=0)
    y_t = np.where(np.isnan(x_t), np.nan, y_t)

    return VolatilityTargetResult(y_t=y_t, sigma_t=sigma_t, leverage_t=leverage_lagged)


def algebraic_riccati_equation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solve the continuous algebraic Riccati equation A'X + XA - XBX + C = 0
    (B assumed symmetric positive semi-definite).

    Original: maths/algebraic_riccati_equation.m (reimplemented via
    scipy.linalg.solve_continuous_are -- see module docstring)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    # scipy solves A'X + XA - X @ B_s @ R^-1 @ B_s' @ X + Q = 0. To recover
    # the original's X @ B @ X term (not X @ B @ R^-1 @ B' @ X), factor
    # B = sqrt(B) @ sqrt(B)' (valid since B is symmetric PSD) and pass
    # B_s = sqrt(B), R = I, so B_s @ R^-1 @ B_s' == B exactly.
    b_sqrt = sqrtm(b).real
    return solve_continuous_are(a, b_sqrt, c, np.eye(a.shape[0]))


def lyapunov_equation(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solve the Lyapunov equation AX + XA' = C.

    Original: maths/lyapunov_equation.m (reimplemented via
    scipy.linalg.solve_lyapunov -- see module docstring)
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    return solve_lyapunov(a, c)
