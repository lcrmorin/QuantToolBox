"""Cobb-Douglas demand and welfare changes under a relative-price shock.

Adapted from HSF-Notebooks chapter 16e, which combines
``HSF/16. Exercise Solution/chap16_chap10_exercise1a.m`` and
``chap16_chap10_exercise1b.m`` into one parameterized calculation.
Plotting grids, budget-line series, and display labels are omitted.

The results include compensating variation (CV) and equivalent variation
(EV), together with the source's alternate calculations. ``cv_numerical``
uses the same closed-form expenditure as ``cv``; ``ev_numerical`` uses a
separate expenditure-minimization formula. Despite their names, neither
uses a numerical optimizer. The tests independently check demand with
numerical utility maximization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CobbDouglasWelfareResult:
    """Demand, utility, and welfare before and after a price shock.

    Suffixes ``_t0`` and ``_t1`` identify baseline and post-shock values.
    Suffixes ``_cv`` and ``_ev`` identify compensating- and
    equivalent-variation baskets and incomes. The ``*_numerical`` fields
    contain alternate closed-form calculations described in the module
    docstring; the CV calculation is not an independent check.
    """

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
