"""Cobb-Douglas consumer welfare under a relative-price shock: Marshallian
demand, indirect utility, and both the closed-form and independently
re-derived Compensating Variation (CV) and Equivalent Variation (EV)
welfare measures for a 2-good Cobb-Douglas consumer facing a price
change.

Not ported from the MATLAB HSF toolbox's general-purpose `hsf/` folder
(`reference/hsf_toolbox_matlab/hsf/`) -- no `.m` file there implements a
`cobb_douglas_welfare` function. The formulas do, however, match
`HSF/16. Exercise Solution/chap16_chap10_exercise1a.m` and
`chap16_chap10_exercise1b.m` in the `hfs-archive` checkout verbatim
(two inline exercise scripts, identical except `alpha` and `p1_t1`, that
this book's own Chapter-16 exercise solutions run once each rather than
defining as a reusable function). Promoted from HSF-Notebooks chapter
16e, which collapsed those two near-identical scripts into one
parametrized `cobb_douglas_welfare` helper.

Translation notes:

- The original notebook helper additionally builds a `x1_grid` plotting
  grid and the indifference-curve/budget-line series
  (`x2_utility_t0/t1`, `x2_budget_t0/t1/cv/ev`) consumed only by a
  companion `plot_welfare_diagram` function -- dropped here as
  visualization-only intermediates, matching this module's `label`
  parameter (print-formatting only, also dropped).
- A genuine, faithfully-reproduced asymmetry in the source: the "CV"
  numerical cross-check (`cv_numerical`) is computed from the *same*
  closed-form `y_cv` the primary `cv` value is built from, so it is
  tautologically equal to `cv` (not an independent check), whereas the
  "EV" numerical cross-check genuinely re-derives `y_ev` from a separate
  expenditure-minimization formula evaluated at `u_t1` and prices
  `(p1_t0, p2)` -- so `ev` and `ev_numerical` are independent
  computations that happen to agree to floating-point precision. Both
  are reproduced exactly as in the source rather than symmetrized.
- The task description accompanying this port asked to keep an
  internal `scipy.optimize.minimize`-based numerical cross-check, but no
  such call exists anywhere in the source notebook (confirmed by
  grepping the notebook for `scipy`/`optimize`) -- the notebook's own
  "numerical" cross-check is the alternate closed-form derivation
  described above, not a numerical optimizer. Faithfully porting the
  actual source therefore does not use `scipy.optimize.minimize`; this
  module's test suite instead adds an independent
  `scipy.optimize.minimize`-based utility-maximization check not present
  in the original, to verify the closed-form demand really is optimal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CobbDouglasWelfareResult:
    """Cobb-Douglas consumer welfare analysis for one before/after price
    shock: optimal demand and utility at the baseline prices (`_t0`,
    point A) and post-shock prices (`_t1`, point B), the Compensating
    Variation basket/income (`_cv`, point C) and welfare measure (`cv`),
    and the Equivalent Variation basket/income (`_ev`, point D) and
    welfare measure (`ev`), each alongside an independently-derived
    numerical cross-check (`u_numerical_t0`/`t1`, `cv_numerical`,
    `ev_numerical`) -- see the module docstring for the one genuine
    asymmetry between the CV and EV cross-checks."""

    alpha: float
    beta: float

    x1_star_t0: float
    x2_star_t0: float
    u_t0: float
    u_numerical_t0: float

    x1_star_t1: float
    x2_star_t1: float
    u_t1: float
    u_numerical_t1: float

    x1_star_cv: float
    x2_star_cv: float
    y_t0: float
    y_cv: float
    cv: float
    cv_numerical: float

    x1_star_ev: float
    x2_star_ev: float
    y_ev: float
    ev: float
    ev_numerical: float


def cobb_douglas_welfare(
    alpha: float, p1_t0: float, p1_t1: float, p2: float, y: float
) -> CobbDouglasWelfareResult:
    """Cobb-Douglas consumer welfare analysis for a consumer with
    preferences ``u(x1, x2) = x1**alpha * x2**beta`` (``beta = 1 -
    alpha``), income `y`, and good-2 price `p2` fixed, facing good-1's
    price move from `p1_t0` (baseline) to `p1_t1` (post-shock).

    Returns Marshallian demand and utility at both price points, plus the
    Compensating Variation (income that could be taken away, at the new
    prices, to hold the consumer at the *original* utility) and
    Equivalent Variation (income that must be given, at the old prices,
    to already put the consumer at the *post-shock* utility).

    Not ported from a single MATLAB HSF toolbox function -- see the
    module docstring.
    """
    beta = 1.0 - alpha
    alpha_beta = alpha + beta  # == 1.0; kept explicit, matching the source

    def demand_and_utility(p1: float, p2: float, y: float) -> tuple[float, float, float, float]:
        x1_star = (alpha / alpha_beta) * (y / p1)
        x2_star = (beta / alpha_beta) * (y / p2)
        u_numerical = x1_star**alpha * x2_star**beta
        u = (
            (alpha**alpha * beta**beta)
            / (alpha_beta**alpha_beta)
            * (y**alpha_beta)
            / (p1**alpha * p2**beta)
        )
        return x1_star, x2_star, u, u_numerical

    x1_star_t0, x2_star_t0, u_t0, u_numerical_t0 = demand_and_utility(p1_t0, p2, y)
    x1_star_t1, x2_star_t1, u_t1, u_numerical_t1 = demand_and_utility(p1_t1, p2, y)

    # CV: hold utility at u_t0, prices at (p1_t1, p2).
    x1_star_cv = (alpha / alpha_beta) * (y / p1_t0) * (p1_t0 / p1_t1) ** (beta / alpha_beta)
    x2_star_cv = (beta / alpha_beta) * (y / p2) * (p1_t1 / p1_t0) ** (alpha / alpha_beta)
    y_cv = y * (p1_t1 / p1_t0) ** (alpha / alpha_beta)
    cv = y * (1.0 - (p1_t1 / p1_t0) ** (alpha / alpha_beta))
    cv_numerical = y - y_cv

    # EV: hold utility at u_t1, prices at (p1_t0, p2).
    u_target = u_t1
    x1_star_num = u_target ** (1.0 / alpha_beta) * (alpha * p2 / (beta * p1_t0)) ** (
        beta / alpha_beta
    )
    x2_star_num = u_target ** (1.0 / alpha_beta) * (beta * p1_t0 / (alpha * p2)) ** (
        alpha / alpha_beta
    )
    y_num = p1_t0 * x1_star_num + p2 * x2_star_num
    x1_star_ev = (alpha / alpha_beta) * (y_num / p1_t0)
    x2_star_ev = (beta / alpha_beta) * (y_num / p2)
    y_ev = y_num
    ev = y * ((p1_t0 / p1_t1) ** (alpha / alpha_beta) - 1.0)
    ev_numerical = y_ev - y

    return CobbDouglasWelfareResult(
        alpha=alpha,
        beta=beta,
        x1_star_t0=x1_star_t0,
        x2_star_t0=x2_star_t0,
        u_t0=u_t0,
        u_numerical_t0=u_numerical_t0,
        x1_star_t1=x1_star_t1,
        x2_star_t1=x2_star_t1,
        u_t1=u_t1,
        u_numerical_t1=u_numerical_t1,
        x1_star_cv=x1_star_cv,
        x2_star_cv=x2_star_cv,
        y_t0=y,
        y_cv=y_cv,
        cv=cv,
        cv_numerical=cv_numerical,
        x1_star_ev=x1_star_ev,
        x2_star_ev=x2_star_ev,
        y_ev=y_ev,
        ev=ev,
        ev_numerical=ev_numerical,
    )
