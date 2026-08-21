"""Bond present value, yield to maturity, current yield, and quadratic-form
bond-portfolio risk against sector-level modified-duration / DTS targets.

Ported from HSF toolbox `bond/{compute_bond_price,compute_bond_ytm,
compute_coupon_yield,quadratic_form_bond_portfolio1,
quadratic_form_bond_portfolio2}.m`.

Translation notes:

- `compute_bond_ytm.m` hand-rolls a bisection loop (stopping once
  ``b - a <= 1e-5``, bracketed in ``[0, 1]``) to find the rate that
  reprices a bond to a target price. Ported here via the package's
  existing `optim.bisection.bisection` rather than reimplementing the
  loop, since price(rate) is exactly the monotone bracket-and-root-find
  problem that already solves.
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

from quanttoolbox.config import BisectionConfig
from quanttoolbox.optim.bisection import bisection
from quanttoolbox.sustainable_finance.risk import QuadraticFormResult, quadratic_form_risk


def bond_price(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    rate: float,
    method: int = 1,
) -> float:
    """Present value of a bond's cash flows under a flat discount rate.

    method=1 (default): continuous discounting, ``exp(-t * rate)``.
    method=2: discrete (annually-compounded) discounting,
    ``1 / (1 + rate)**t``.

    Original: bond/compute_bond_price.m
    """
    t = np.asarray(maturities, dtype=float)
    ct = np.asarray(cash_flows, dtype=float)

    if method == 1:
        discount = np.exp(-t * rate)
    else:
        discount = 1.0 / (1.0 + rate) ** t

    return float(np.sum(ct * discount))


def bond_ytm(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    price: float,
    config: BisectionConfig | None = None,
) -> float:
    """Yield to maturity: the flat, continuously-compounded rate that
    reprices `cash_flows` to `price` (bisection search over the rate
    bracket [0, 1], i.e. 0%-100% annual yield).

    Original: bond/compute_bond_ytm.m (bisection loop replaced by
    `optim.bisection.bisection` -- see module docstring)
    """
    if config is None:
        config = BisectionConfig(tol=1e-5)  # matches the original's (b-a) > 1e-5 stopping test

    def f(rate: np.ndarray) -> np.ndarray:
        return bond_price(maturities, cash_flows, float(rate), method=1) - price

    return float(bisection(f, 0.0, 1.0, config=config))


def coupon_yield(
    maturities: np.ndarray | float,
    cash_flows: np.ndarray,
    rate: float,
) -> float:
    """Current yield: first cash flow / present value.

    Original: bond/compute_coupon_yield.m
    """
    ct = np.asarray(cash_flows, dtype=float)
    price = bond_price(maturities, cash_flows, rate, method=1)
    return float(ct[0] / price)


@dataclass
class BondPortfolioQuadraticForm:
    """Quadratic-form risk of a bond portfolio: ``qf(w) = 0.5 w'Qw - w'R +
    c``, combining modified-duration risk, DTS (duration-times-spread)
    risk, and a linear carry term."""

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
    (each re-centered around the benchmark weights `b`)."""

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
