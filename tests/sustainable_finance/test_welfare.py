"""Tests for quanttoolbox.sustainable_finance.welfare."""

import numpy as np
from scipy.optimize import minimize

from quanttoolbox.sustainable_finance.welfare import cobb_douglas_welfare


def test_exercise_1a_matches_notebook_output():
    # Real regression data from HSF-Notebooks 16e_chap10_transition_risk_welfare.ipynb
    # (== chap16_chap10_exercise1a.m: alpha=0.6, p1: 1.00 -> 0.75, a price cut).
    result = cobb_douglas_welfare(alpha=0.6, p1_t0=1.0, p1_t1=0.75, p2=1.0, y=3.0)

    assert np.isclose(result.x1_star_t0, 1.8, atol=5e-6)
    assert np.isclose(result.x2_star_t0, 1.2, atol=5e-6)
    assert np.isclose(result.u_t0, 1.530509, atol=5e-6)
    assert np.isclose(result.u_numerical_t0, result.u_t0, atol=1e-9)

    assert np.isclose(result.x1_star_t1, 2.4, atol=5e-6)
    assert np.isclose(result.x2_star_t1, 1.2, atol=5e-6)
    assert np.isclose(result.u_t1, 1.818860, atol=5e-6)

    assert np.isclose(result.x1_star_cv, 2.019519, atol=5e-6)
    assert np.isclose(result.x2_star_cv, 1.009760, atol=5e-6)
    assert np.isclose(result.y_cv, 2.524399, atol=5e-6)
    assert np.isclose(result.cv, 0.475601, atol=5e-6)
    assert np.isclose(result.cv_numerical, 0.475601, atol=5e-6)

    assert np.isclose(result.x1_star_ev, 2.139123, atol=5e-6)
    assert np.isclose(result.x2_star_ev, 1.426082, atol=5e-6)
    assert np.isclose(result.y_ev, 3.565205, atol=5e-6)
    assert np.isclose(result.ev, 0.565205, atol=5e-6)
    assert np.isclose(result.ev_numerical, 0.565205, atol=5e-6)

    # Welfare-improving price cut: both CV and EV positive; |CV| <= |EV|
    # for a price decrease (the classical Willig-bounds ordering).
    assert result.cv > 0
    assert result.ev > 0
    assert abs(result.cv) <= abs(result.ev)


def test_exercise_1b_matches_notebook_output():
    # == chap16_chap10_exercise1b.m: alpha=0.5, p1: 1.00 -> 1.75, a price hike.
    result = cobb_douglas_welfare(alpha=0.5, p1_t0=1.0, p1_t1=1.75, p2=1.0, y=3.0)

    assert np.isclose(result.x1_star_t0, 1.5, atol=5e-6)
    assert np.isclose(result.x2_star_t0, 1.5, atol=5e-6)
    assert np.isclose(result.u_t0, 1.5, atol=5e-6)

    assert np.isclose(result.x1_star_t1, 0.857143, atol=5e-6)
    assert np.isclose(result.x2_star_t1, 1.5, atol=5e-6)
    assert np.isclose(result.u_t1, 1.133893, atol=5e-6)

    assert np.isclose(result.x1_star_cv, 1.133893, atol=5e-6)
    assert np.isclose(result.x2_star_cv, 1.984313, atol=5e-6)
    assert np.isclose(result.y_cv, 3.968627, atol=5e-6)
    assert np.isclose(result.cv, -0.968627, atol=5e-6)
    assert np.isclose(result.cv_numerical, -0.968627, atol=5e-6)

    assert np.isclose(result.x1_star_ev, 1.133893, atol=5e-6)
    assert np.isclose(result.x2_star_ev, 1.133893, atol=5e-6)
    assert np.isclose(result.y_ev, 2.267787, atol=5e-6)
    assert np.isclose(result.ev, -0.732213, atol=5e-6)
    assert np.isclose(result.ev_numerical, -0.732213, atol=5e-6)

    # Welfare-reducing price hike: both CV and EV negative; |EV| <= |CV|
    # for a price increase (the reverse Willig-bounds ordering).
    assert result.cv < 0
    assert result.ev < 0
    assert abs(result.ev) <= abs(result.cv)


def _maximize_utility_numerically(alpha, beta, p1, p2, y):
    """Independent scipy.optimize.minimize cross-check (not present in the
    source notebook -- see the module's translation notes) that the
    closed-form Marshallian demand really does maximize Cobb-Douglas
    utility subject to the budget constraint."""

    def neg_utility(x):
        x1, x2 = x
        if x1 <= 0 or x2 <= 0:
            return np.inf
        return -(x1**alpha * x2**beta)

    constraints = ({"type": "eq", "fun": lambda x: p1 * x[0] + p2 * x[1] - y},)
    x0 = np.array([y / (2 * p1), y / (2 * p2)])
    res = minimize(
        neg_utility,
        x0,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert res.success
    return res.x


def test_demand_maximizes_utility_numerically_via_scipy_optimize():
    alpha, p1_t0, p1_t1, p2, y = 0.6, 1.0, 0.75, 1.0, 3.0
    beta = 1.0 - alpha
    result = cobb_douglas_welfare(alpha, p1_t0, p1_t1, p2, y)

    x1_opt, x2_opt = _maximize_utility_numerically(alpha, beta, p1_t0, p2, y)
    assert np.isclose(x1_opt, result.x1_star_t0, atol=1e-4)
    assert np.isclose(x2_opt, result.x2_star_t0, atol=1e-4)

    x1_opt, x2_opt = _maximize_utility_numerically(alpha, beta, p1_t1, p2, y)
    assert np.isclose(x1_opt, result.x1_star_t1, atol=1e-4)
    assert np.isclose(x2_opt, result.x2_star_t1, atol=1e-4)


def test_cv_basket_maximizes_utility_at_new_prices_subject_to_y_cv():
    # The CV basket (point C) should itself be the utility-maximizing
    # bundle at the post-shock prices, spending exactly y_cv.
    alpha, p1_t0, p1_t1, p2, y = 0.6, 1.0, 0.75, 1.0, 3.0
    beta = 1.0 - alpha
    result = cobb_douglas_welfare(alpha, p1_t0, p1_t1, p2, y)

    x1_opt, x2_opt = _maximize_utility_numerically(alpha, beta, p1_t1, p2, result.y_cv)
    assert np.isclose(x1_opt, result.x1_star_cv, atol=1e-4)
    assert np.isclose(x2_opt, result.x2_star_cv, atol=1e-4)


def test_cv_and_ev_vanish_with_no_price_change():
    alpha, p1, p2, y = 0.4, 2.0, 1.0, 10.0
    result = cobb_douglas_welfare(alpha, p1, p1, p2, y)

    assert np.isclose(result.cv, 0.0, atol=1e-10)
    assert np.isclose(result.ev, 0.0, atol=1e-10)
    assert np.isclose(result.u_t0, result.u_t1, atol=1e-10)


def test_utility_closed_form_matches_direct_evaluation():
    alpha, p1_t0, p1_t1, p2, y = 0.35, 2.0, 3.0, 1.5, 12.0
    result = cobb_douglas_welfare(alpha, p1_t0, p1_t1, p2, y)

    assert np.isclose(result.u_t0, result.u_numerical_t0, atol=1e-9)
    assert np.isclose(result.u_t1, result.u_numerical_t1, atol=1e-9)
