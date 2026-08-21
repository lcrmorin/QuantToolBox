"""Generic quadratic-form risk building blocks used across sector-based
portfolio risk models -- e.g. bond-portfolio modified-duration/DTS risk in
`quanttoolbox.bond.pricing`, which these functions were factored out of.

Ported from HSF toolbox `hsf/{quadratic_form,quadratic_form_risk,
bond_portfolio_metrics}.m`.

Translation notes:

- `quadratic_form_risk.m` builds a quadratic-form penalty for deviating a
  sector-level risk measure (e.g. modified duration) from per-sector
  targets: for each sector j it contributes an outer-product term
  ``Q_j = (s_j * risk)(s_j * risk)'`` (``s_j`` the sector-j 0/1 indicator),
  summed across sectors into a single (Q, R, c) triple. Kept as a
  general-purpose helper rather than folded into `bond/pricing.py`, since
  the original file lives in `hsf/` (sustainable-finance-general), not
  `bond/`, and nothing here is bond-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quadratic_form(x: np.ndarray, q: np.ndarray, r: np.ndarray, c: float) -> float:
    """Evaluate the quadratic form qf(x) = 0.5 x'Qx - x'R + c.

    Original: hsf/quadratic_form.m
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    return float(0.5 * x @ q @ x - x @ r + c)


@dataclass
class QuadraticFormResult:
    """Sector-decomposed quadratic-risk form: qf = 0.5 w'Qw - w'R + c,
    with `q_j`/`r_j`/`c_j` holding each sector's own contribution to
    Q/R/c (summed into `q`/`r`/`c`)."""

    qf: float
    q: np.ndarray
    r: np.ndarray
    c: float
    n: int
    n_sector: int
    unique_sector: np.ndarray
    q_j: np.ndarray  # (n, n, n_sector)
    r_j: np.ndarray  # (n, n_sector)
    c_j: np.ndarray  # (n_sector,)


def quadratic_form_risk(
    sector: np.ndarray,
    risk: np.ndarray,
    risk_star: np.ndarray,
    w: np.ndarray | None = None,
) -> QuadraticFormResult:
    """Build a quadratic-form penalty for a sector-level risk measure (e.g.
    modified duration or DTS) deviating from per-sector targets
    `risk_star`, and (if `w` is given) evaluate it at portfolio weights w.

    Original: hsf/quadratic_form_risk.m
    """
    sector = np.asarray(sector)
    risk = np.asarray(risk, dtype=float).flatten()
    risk_star = np.asarray(risk_star, dtype=float).flatten()
    n = risk.shape[0]

    unique_sector = np.unique(sector)
    n_sector = unique_sector.shape[0]

    q = np.zeros((n, n))
    r = np.zeros(n)
    c = 0.0
    q_j = np.zeros((n, n, n_sector))
    r_j = np.zeros((n, n_sector))
    c_j = np.zeros(n_sector)

    for j, s in enumerate(unique_sector):
        s_j = (sector == s).astype(float)
        s_j_risk = s_j * risk

        q_j[:, :, j] = np.outer(s_j_risk, s_j_risk)
        r_j[:, j] = s_j_risk * risk_star[j]
        c_j[j] = 0.5 * risk_star[j] ** 2

        q += q_j[:, :, j]
        r += r_j[:, j]
        c += c_j[j]

    qf = quadratic_form(w, q, r, c) if w is not None else float("nan")

    return QuadraticFormResult(
        qf=qf,
        q=q,
        r=r,
        c=c,
        n=n,
        n_sector=n_sector,
        unique_sector=unique_sector,
        q_j=q_j,
        r_j=r_j,
        c_j=c_j,
    )


@dataclass
class BondPortfolioMetrics:
    """Portfolio- and sector-level modified duration (MD) and
    duration-times-spread (DTS)."""

    md_portfolio: float
    dts_portfolio: float
    md_by_sector: np.ndarray
    dts_by_sector: np.ndarray
    unique_sector: np.ndarray


def bond_portfolio_metrics(
    sector: np.ndarray,
    md: np.ndarray,
    dts: np.ndarray,
    w: np.ndarray,
) -> BondPortfolioMetrics:
    """Portfolio- and sector-level modified duration and DTS, weighted by
    portfolio weights w.

    Original: hsf/bond_portfolio_metrics.m
    """
    sector = np.asarray(sector)
    md = np.asarray(md, dtype=float).flatten()
    dts = np.asarray(dts, dtype=float).flatten()
    w = np.asarray(w, dtype=float).flatten()

    unique_sector = np.unique(sector)
    n_sector = unique_sector.shape[0]

    md_portfolio = float(np.sum(w * md))
    dts_portfolio = float(np.sum(w * dts))

    md_by_sector = np.zeros(n_sector)
    dts_by_sector = np.zeros(n_sector)
    for j, s in enumerate(unique_sector):
        s_j = (sector == s).astype(float)
        md_by_sector[j] = np.sum(s_j * w * md)
        dts_by_sector[j] = np.sum(s_j * w * dts)

    return BondPortfolioMetrics(
        md_portfolio=md_portfolio,
        dts_portfolio=dts_portfolio,
        md_by_sector=md_by_sector,
        dts_by_sector=dts_by_sector,
        unique_sector=unique_sector,
    )
