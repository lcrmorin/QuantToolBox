"""Bond present value, yield to maturity, current yield, and quadratic-form
bond-portfolio risk against sector-level modified-duration / DTS targets.

Ported from HSF toolbox `bond/{compute_bond_price,compute_bond_ytm,
compute_coupon_yield,quadratic_form_bond_portfolio1,
quadratic_form_bond_portfolio2}.m`.

Translation notes:

- Yield inversion uses a bracketed SciPy solver with an expanding search
  interval, including negative yields. Positive future cash flows make the
  price strictly decreasing, so the supported yield is unique.
- `quadratic_form_bond_portfolio1.m`/`2.m` depend on `quadratic_form`/
  `quadratic_form_risk` -- defined in the HSF toolbox's `hsf/` folder, not
  `bond/` -- ported alongside as `quanttoolbox.sustainable_finance.risk`,
  since they're generic sector-based quadratic-risk building blocks, not
  bond-specific (see that module's docstring). `bond_portfolio_metrics.m`
  (portfolio-level modified duration / DTS, also in `hsf/`) is ported
  there too rather than duplicated here.
- `quadratic_form_bond_portfolio1.m`/`2.m` are duplicated verbatim between
  `bond/` and `hsf/` in the original source; ported once, here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from quanttoolbox.config import BisectionConfig
from quanttoolbox.sustainable_finance.risk import QuadraticFormResult, quadratic_form_risk


def _cash_flow_schedule(maturities, cash_flows):
    """Validate one schedule without accidentally broadcasting two vectors."""
    t = np.asarray(maturities, dtype=float)
    ct = np.asarray(cash_flows, dtype=float)
    if t.ndim > 1 or ct.ndim > 1 or t.size == 0 or ct.size == 0:
        raise ValueError("maturities and cash_flows must be nonempty scalars or 1-D arrays")
    ct = np.atleast_1d(ct)
    if t.ndim == 0:
        t = np.full(ct.shape, float(t))
    elif t.shape != ct.shape:
        raise ValueError("maturities and cash_flows must have matching lengths")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("maturities must be finite and nonnegative, in years")
    if not np.all(np.isfinite(ct)):
        raise ValueError("cash_flows must be finite")
    return t, ct


def _discount_rate(rate, method):
    """Return the equivalent continuous rate for a supported convention."""
    if method not in (1, 2):
        raise ValueError("method must be 1 (continuous) or 2 (annual compounding)")
    if np.ndim(rate) != 0 or not np.isfinite(rate):
        raise ValueError("rate must be a finite scalar")
    rate = float(rate)
    if method == 2:
        if rate <= -1:
            raise ValueError("annual-compounding rate must be greater than -1")
        return float(np.log1p(rate))
    return rate


def bond_price(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    rate: float,
    method: int = 1,
) -> float:
    """Present value of one cash-flow schedule at a flat annual rate.

    Times are finite, nonnegative year fractions; cash flows are a matching
    1-D sequence. A scalar maturity applies to every cash flow. Signed cash
    flows are allowed. No settlement date, accrued interest, or day-count
    adjustment is inferred: the result includes every supplied cash flow.

    ``method=1`` discounts by ``exp(-t * rate)`` (continuous compounding).
    ``method=2`` discounts by ``(1 + rate)**(-t)`` (annual compounding),
    requiring ``rate > -1``. Rates are decimals, e.g. 0.05 means 5%.

    Raises ValueError for invalid inputs and OverflowError if the present
    value cannot be represented as a finite float.

    Original: bond/compute_bond_price.m
    """
    t, ct = _cash_flow_schedule(maturities, cash_flows)
    q = _discount_rate(rate, method)
    nonzero = ct != 0
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.sum(ct[nonzero] * np.exp(-t[nonzero] * q))
    if not np.isfinite(value):
        raise OverflowError("bond present value exceeds floating-point range")
    return float(value)


def bond_ytm(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    price: float,
    config: BisectionConfig | None = None,
    method: int = 1,
) -> float:
    """Find the unique yield for a nonnegative cash-flow schedule.

    Uses the time and compounding conventions of :func:`bond_price`.
    Requires at least one positive future payment and a finite price greater
    than the sum of payments at time zero. Signed schedules are rejected:
    they can have multiple yields. Negative yields and yields above 100%
    are supported; annual-compounding yields must remain greater than -1.

    ``config.tol`` is the absolute yield tolerance (default 1e-5), subject
    to floating-point relative precision. ``config.max_iters`` limits the
    root solver iterations. Invalid inputs raise ValueError; an unbracketed
    or unconverged yield raises RuntimeError instead of returning NaN.

    Original: bond/compute_bond_ytm.m; the fixed [0, 1] bracket is extended.
    """
    t, ct = _cash_flow_schedule(maturities, cash_flows)
    _discount_rate(0.0, method)
    if np.any(ct < 0):
        raise ValueError("yield requires nonnegative cash_flows for a unique solution")
    future = (t > 0) & (ct > 0)
    if not np.any(future):
        raise ValueError("yield requires at least one positive future cash flow")
    immediate = float(np.sum(ct[t == 0]))
    if np.ndim(price) != 0 or not np.isfinite(price) or price <= immediate:
        raise ValueError("price must be finite and greater than the cash flows at time zero")
    if config is None:
        config = BisectionConfig(tol=1e-5)
    if not np.isfinite(config.tol) or config.tol <= 0:
        raise ValueError("config.tol must be finite and positive")
    if (
        isinstance(config.max_iters, bool)
        or not isinstance(config.max_iters, int | np.integer)
        or config.max_iters < 1
    ):
        raise ValueError("config.max_iters must be a positive integer")

    times = t[future]
    log_flows = np.log(ct[future])
    log_target = np.log(float(price) - immediate)

    def residual(rate):
        q = rate if method == 1 else np.log1p(rate)
        # Log prices avoid overflow when exploring deeply negative yields.
        with np.errstate(over="ignore"):
            return float(logsumexp(log_flows - times * q) - log_target)

    at_zero = residual(0.0)
    if at_zero == 0:
        return 0.0
    if at_zero > 0:
        low, high = 0.0, 1.0
        for _ in range(1024):
            if residual(high) <= 0:
                break
            high *= 2
            if not np.isfinite(high):
                raise RuntimeError("cannot bracket yield within floating-point range")
        else:
            raise RuntimeError("cannot bracket yield within floating-point range")
    else:
        low, high = (-1.0 if method == 1 else np.nextafter(-1.0, 0.0)), 0.0
        if method == 1:
            for _ in range(1024):
                if residual(low) >= 0:
                    break
                low *= 2
                if not np.isfinite(low):
                    raise RuntimeError("cannot bracket yield within floating-point range")
            else:
                raise RuntimeError("cannot bracket yield within floating-point range")
        elif residual(low) < 0:
            raise RuntimeError("yield is too close to -1 to represent with annual compounding")
    return float(brentq(residual, low, high, xtol=config.tol, maxiter=config.max_iters))


def _cash_flow_weights(maturities, cash_flows, rate, method):
    t, ct = _cash_flow_schedule(maturities, cash_flows)
    q = _discount_rate(rate, method)
    if np.any(ct < 0) or not np.any(ct > 0):
        raise ValueError(
            "sensitivity measures require nonnegative cash flows and a positive payment"
        )
    positive = ct > 0
    t = t[positive]
    logs = np.log(ct[positive]) - t * q
    if not np.all(np.isfinite(logs)):
        raise OverflowError("discounted cash-flow weights exceed floating-point range")
    return t, np.exp(logs - logsumexp(logs))


def bond_duration(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    rate: float,
    method: int = 1,
    *,
    modified: bool = True,
) -> float:
    """Duration in years, using the conventions of :func:`bond_price`.

    By default returns ``-P'(rate) / P(rate)``. For annual compounding this
    is Macaulay duration divided by ``1 + rate``; for continuous compounding
    it equals the present-value-weighted payment time. ``modified=False``
    returns the weighted payment time for either convention.
    Requires nonnegative cash flows with at least one positive payment.
    """
    t, weights = _cash_flow_weights(maturities, cash_flows, rate, method)
    duration = float(np.sum(weights * t))
    return duration / (1 + rate) if modified and method == 2 else duration


def bond_convexity(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    rate: float,
    method: int = 1,
) -> float:
    """Normalized price curvature ``P''(rate) / P(rate)``.

    Uses the conventions of :func:`bond_price` and nonnegative cash flows.
    Continuous compounding uses the weighted mean of ``t**2``; annual
    compounding uses ``t * (t + 1) / (1 + rate)**2``. For a small yield
    change ``dr``, ``dP/P ≈ -duration * dr + 0.5 * convexity * dr**2``.
    """
    t, weights = _cash_flow_weights(maturities, cash_flows, rate, method)
    curvature = t**2 if method == 1 else t * (t + 1) / (1 + rate) ** 2
    return float(np.sum(weights * curvature))


def coupon_yield(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    rate: float,
) -> float:
    """Legacy first-payment / continuously discounted present-value ratio.

    This is a current yield only when the first payment represents the
    annual coupon without principal. It does not annualize semiannual
    coupons or distinguish principal from coupons in a zero-coupon bond.

    Original: bond/compute_coupon_yield.m
    """
    _, ct = _cash_flow_schedule(maturities, cash_flows)
    price = bond_price(maturities, cash_flows, rate, method=1)
    if price <= 0:
        raise ValueError("coupon yield requires a positive present value")
    return float(ct[0] / price)


@dataclass
class BondPortfolioQuadraticForm:
    """Quadratic-form risk of a bond portfolio: ``qf(w) = 0.5 w'Qw - w'R +
    c``, combining modified-duration risk, DTS (duration-times-spread)
    risk, and a linear carry term.
    """

    qf: float
    q: np.ndarray
    r: np.ndarray
    c: float
    md: QuadraticFormResult
    dts: QuadraticFormResult


def bond_portfolio_quadratic_form(
    sector: np.ndarray,
    varphi_md: float,
    md: np.ndarray,
    md_star: np.ndarray,
    varphi_dts: float,
    dts: np.ndarray,
    dts_star: np.ndarray,
    gamma_carry: float,
    carry: np.ndarray,
    w: np.ndarray,
) -> BondPortfolioQuadraticForm:
    """Combine sector-level modified-duration and DTS quadratic-risk terms
    with a linear carry term into a single bond-portfolio quadratic form,
    evaluated at weights `w`.

    Original: bond/quadratic_form_bond_portfolio1.m
    """
    md_result = quadratic_form_risk(sector, md, md_star, w)
    dts_result = quadratic_form_risk(sector, dts, dts_star, w)

    carry = np.asarray(carry, dtype=float)
    w = np.asarray(w, dtype=float)

    q = varphi_md * md_result.q + varphi_dts * dts_result.q
    r = gamma_carry * carry + varphi_md * md_result.r + varphi_dts * dts_result.r
    c = varphi_md * md_result.c + varphi_dts * dts_result.c
    qf = 0.5 * w @ q @ w - w @ r + c

    return BondPortfolioQuadraticForm(
        qf=float(qf), q=q, r=r, c=float(c), md=md_result, dts=dts_result
    )


@dataclass
class BondPortfolioQuadraticFormVsBenchmark:
    """Same shape as `BondPortfolioQuadraticForm`, plus the individual
    active-share (AS)/MD/DTS terms the combined (q, r, c) was built from
    (each re-centered around the benchmark weights `b`).
    """

    qf: float
    q: np.ndarray
    r: np.ndarray
    c: float
    md: QuadraticFormResult
    dts: QuadraticFormResult
    q_as: np.ndarray
    r_as: np.ndarray
    c_as: float
    q_md: np.ndarray
    r_md: np.ndarray
    c_md: float
    q_dts: np.ndarray
    r_dts: np.ndarray
    c_dts: float


def bond_portfolio_quadratic_form_vs_benchmark(
    sector: np.ndarray,
    varphi_as: float,
    varphi_md: float,
    md: np.ndarray,
    md_star: np.ndarray | None,
    varphi_dts: float,
    dts: np.ndarray,
    dts_star: np.ndarray | None,
    gamma_carry: float,
    carry: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
) -> BondPortfolioQuadraticFormVsBenchmark:
    """Same as `bond_portfolio_quadratic_form`, but expressed relative to a
    benchmark weight vector `b`: adds an active-share quadratic penalty
    centered at `b` (``0.5(w-b)'(w-b)`` up to the constant, expanded into
    the (Q, R, c) form), and re-centers the MD/DTS quadratic forms around
    `b` too.

    `md_star`/`dts_star` default to per-sector zero targets when `None`
    (matching the original's `isempty(...)` fallback).

    Original: bond/quadratic_form_bond_portfolio2.m
    """
    n_sector = np.unique(np.asarray(sector)).shape[0]
    if md_star is None:
        md_star = np.zeros(n_sector)
    if dts_star is None:
        dts_star = np.zeros(n_sector)

    md_result = quadratic_form_risk(sector, md, md_star, w)
    dts_result = quadratic_form_risk(sector, dts, dts_star, w)

    w = np.asarray(w, dtype=float)
    b = np.asarray(b, dtype=float)
    carry = np.asarray(carry, dtype=float)
    n = w.shape[0]

    q_as = np.eye(n)
    r_as = b
    c_as = 0.5 * (b @ b)

    q_md = md_result.q
    r_md = md_result.r + md_result.q @ b
    c_md = 0.5 * b @ md_result.q @ b + b @ md_result.r + md_result.c

    q_dts = dts_result.q
    r_dts = dts_result.r + dts_result.q @ b
    c_dts = 0.5 * b @ dts_result.q @ b + b @ dts_result.r + dts_result.c

    q = varphi_as * q_as + varphi_md * q_md + varphi_dts * q_dts
    r = gamma_carry * carry + varphi_as * r_as + varphi_md * r_md + varphi_dts * r_dts
    c = gamma_carry * (b @ carry) + varphi_as * c_as + varphi_md * c_md + varphi_dts * c_dts
    qf = 0.5 * w @ q @ w - w @ r + c

    return BondPortfolioQuadraticFormVsBenchmark(
        qf=float(qf),
        q=q,
        r=r,
        c=float(c),
        md=md_result,
        dts=dts_result,
        q_as=q_as,
        r_as=r_as,
        c_as=float(c_as),
        q_md=q_md,
        r_md=r_md,
        c_md=float(c_md),
        q_dts=q_dts,
        r_dts=r_dts,
        c_dts=float(c_dts),
    )
