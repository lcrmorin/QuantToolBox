"""Tests for quanttoolbox.portfolio.risk_budgeting."""

import numpy as np
import pytest

from quanttoolbox.portfolio.risk_budgeting import (
    erc_portfolio,
    risk_budgeting_frontier,
    risk_contribution,
    solve_box_constrained,
    solve_constrained,
    solve_unconstrained,
)


@pytest.fixture
def sample_cov(rng):
    n = 4
    a = rng.standard_normal((200, n))
    return a.T @ a / 200 + 0.01 * np.eye(n)


def test_risk_contribution_sums_to_total_risk(sample_cov):
    x = np.full(4, 0.25)
    rc = risk_contribution(x, sample_cov)
    assert np.isclose(np.sum(rc.risk_contribution), rc.risk, atol=1e-8)
    assert np.isclose(np.sum(rc.pct_risk_contribution), 1.0, atol=1e-8)


def test_erc_portfolio_equal_risk_contributions(sample_cov):
    result = erc_portfolio(sample_cov)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
    assert result.converged
    assert np.allclose(result.pct_risk_contribution, 0.25, atol=1e-3)


def test_erc_ccd_and_newton_agree(sample_cov):
    ccd = erc_portfolio(sample_cov, method="ccd")
    newton = erc_portfolio(sample_cov, method="newton")
    assert np.allclose(ccd.weights, newton.weights, atol=1e-3)


def test_solve_unconstrained_hits_target_budgets(sample_cov):
    b = np.array([0.4, 0.3, 0.2, 0.1])
    result = solve_unconstrained(sample_cov, b=b)
    assert np.allclose(result.pct_risk_contribution, b, atol=1e-3)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)


def test_solve_unconstrained_invalid_method_raises(sample_cov):
    with pytest.raises(ValueError):
        solve_unconstrained(sample_cov, method="bogus")


def test_solve_box_constrained_respects_bounds(sample_cov):
    result = solve_box_constrained(sample_cov, x_minus=0.15, x_plus=0.35, b=np.full(4, 0.25))
    assert np.all(result.weights >= 0.15 - 1e-3)
    assert np.all(result.weights <= 0.35 + 1e-3)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-3)


def test_solve_box_constrained_matches_unconstrained_when_bounds_slack(sample_cov):
    b = np.full(4, 0.25)
    unconstrained = solve_unconstrained(sample_cov, b=b)
    # bounds wide enough not to bind
    boxed = solve_box_constrained(sample_cov, x_minus=-1.0, x_plus=1.0, b=b)
    assert np.allclose(unconstrained.weights, boxed.weights, atol=1e-2)


def test_risk_budgeting_frontier_returns_list(sample_cov):
    mu = np.array([0.05, 0.03, 0.04, 0.02])
    results = risk_budgeting_frontier(sample_cov, b=None, mu=mu, c_values=np.array([0.5, 1.0, 2.0]))
    assert len(results) == 3
    for r in results:
        assert np.isclose(np.sum(r.weights), 1.0, atol=1e-6)


def test_solve_unconstrained_with_mu_reduces_risk_for_higher_expected_return(sample_cov):
    mu = np.array([0.10, 0.02, 0.02, 0.02])  # asset 0 has much higher expected return
    b = np.full(4, 0.25)
    result_no_mu = solve_unconstrained(sample_cov, b=b, mu=0.0, c=1.0)
    result_with_mu = solve_unconstrained(sample_cov, b=b, mu=mu, c=1.0)
    # with a return-adjusted risk measure, the high-mu asset should get more weight
    assert result_with_mu.weights[0] > result_no_mu.weights[0]


# ---------------------------------------------------------------------------
# VaR / ES risk-measure variants
# ---------------------------------------------------------------------------


def test_var_es_zero_mu_match_vol_based_erc(sample_cov):
    # with mu=0, the c multiplier cancels out of the FOC entirely, so
    # VaR/ES/vol-based risk budgeting should all give the same weights
    from quanttoolbox.portfolio.risk_budgeting import (
        solve_unconstrained_es,
        solve_unconstrained_var,
    )

    vol_result = solve_unconstrained(sample_cov, b=np.full(4, 0.25))
    var_result = solve_unconstrained_var(sample_cov, alpha=0.95, b=np.full(4, 0.25))
    es_result = solve_unconstrained_es(sample_cov, alpha=0.95, b=np.full(4, 0.25))

    assert np.allclose(vol_result.weights, var_result.weights, atol=1e-4)
    assert np.allclose(vol_result.weights, es_result.weights, atol=1e-4)


def test_var_multiplier_matches_normal_quantile():
    from scipy.stats import norm

    from quanttoolbox.portfolio.risk_budgeting import _var_multiplier

    assert np.isclose(_var_multiplier(0.95), norm.ppf(0.95))
    assert np.isclose(_var_multiplier(0.99), norm.ppf(0.99))


def test_es_multiplier_exceeds_var_multiplier():
    # ES multiplier is always >= VaR multiplier at the same confidence level
    # (expected shortfall is at least as extreme as VaR)
    from quanttoolbox.portfolio.risk_budgeting import _es_multiplier, _var_multiplier

    for alpha in [0.90, 0.95, 0.99]:
        assert _es_multiplier(alpha) >= _var_multiplier(alpha)


def test_risk_contribution_var_es_differ_with_nonzero_mu(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import risk_contribution_es, risk_contribution_var

    x = np.full(4, 0.25)
    mu = np.array([0.05, 0.03, 0.04, 0.02])
    rc_var = risk_contribution_var(x, sample_cov, mu, alpha=0.95)
    rc_es = risk_contribution_es(x, sample_cov, mu, alpha=0.95)
    # different multipliers -> different reported risk levels
    assert not np.isclose(rc_var.risk, rc_es.risk)


# ---------------------------------------------------------------------------
# General linear-constrained solver
# ---------------------------------------------------------------------------


def test_solve_constrained_box_only_matches_solve_box_constrained(sample_cov):
    b = np.full(4, 0.25)
    r1 = solve_box_constrained(sample_cov, x_minus=0.1, x_plus=0.4, b=b)
    r2 = solve_constrained(sample_cov, b=b, x_minus=0.1, x_plus=0.4)
    assert np.allclose(r1.weights, r2.weights, atol=1e-4)


def test_solve_constrained_inequality_respected(sample_cov):
    b = np.full(4, 0.25)
    c_ineq = np.array([[1.0, 1.0, 0.0, 0.0]])  # asset0 + asset1 <= 0.6
    d_ineq = np.array([0.6])
    result = solve_constrained(
        sample_cov, b=b, c_ineq=c_ineq, d_ineq=d_ineq, x_minus=0.0, x_plus=1.0
    )
    assert result.weights[0] + result.weights[1] <= 0.6 + 1e-3
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-3)
    assert result.converged


def test_solve_constrained_equality_respected(sample_cov):
    b = np.full(4, 0.25)
    a_eq = np.array([[1.0, -1.0, 0.0, 0.0]])  # force asset0 == asset1
    b_eq = np.array([0.0])
    result = solve_constrained(sample_cov, b=b, a_eq=a_eq, b_eq=b_eq, x_minus=0.0, x_plus=1.0)
    assert np.isclose(result.weights[0], result.weights[1], atol=1e-3)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-3)
    assert result.converged


def test_solve_constrained_auto_bracket_finds_shifted_root(sample_cov):
    # this configuration is known to need bracket expansion beyond the
    # naive (0.5, 2.0) default -- see the auto-bracket implementation
    b = np.full(4, 0.25)
    a_eq = np.array([[1.0, -1.0, 0.0, 0.0]])
    b_eq = np.array([0.0])
    result = solve_constrained(sample_cov, b=b, a_eq=a_eq, b_eq=b_eq, x_minus=0.0, x_plus=1.0)
    assert not np.any(np.isnan(result.weights))
    assert result.converged


# ---------------------------------------------------------------------------
# Target-matching frontier (mu-problem / sigma-problem)
# ---------------------------------------------------------------------------


def test_risk_budgeting_target_return_hits_target_exactly(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import _solve_rb_at_c, risk_budgeting_target

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    b = np.full(4, 0.25)
    _, mu_min, _, _ = _solve_rb_at_c(sample_cov, b, mu, 1.0, None, np.zeros(4), "unconstrained", {})
    _, mu_max, _, _ = _solve_rb_at_c(
        sample_cov, b, mu, 100.0, None, np.zeros(4), "unconstrained", {}
    )
    target = (mu_min + mu_max) / 2

    result = risk_budgeting_target(sample_cov, mu, target=target, target_type="return", b=b)
    assert result.converged
    assert np.isclose(result.active_return, target, atol=1e-6)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-4)


def test_risk_budgeting_target_volatility_hits_target_exactly(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import _solve_rb_at_c, risk_budgeting_target

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    b = np.full(4, 0.25)
    _, _, sig_min, _ = _solve_rb_at_c(
        sample_cov, b, mu, 1.0, None, np.zeros(4), "unconstrained", {}
    )
    _, _, sig_max, _ = _solve_rb_at_c(
        sample_cov, b, mu, 100.0, None, np.zeros(4), "unconstrained", {}
    )
    target = (sig_min + sig_max) / 2

    result = risk_budgeting_target(sample_cov, mu, target=target, target_type="volatility", b=b)
    assert result.converged
    assert np.isclose(result.active_volatility, target, atol=1e-6)


def test_risk_budgeting_target_out_of_range_returns_nan(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import risk_budgeting_target

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    result = risk_budgeting_target(sample_cov, mu, target=999.0, target_type="return")
    assert not result.converged
    assert np.all(np.isnan(result.weights))


def test_risk_budgeting_target_invalid_target_type_raises(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import risk_budgeting_target

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    with pytest.raises(ValueError):
        risk_budgeting_target(sample_cov, mu, target=0.05, target_type="bogus")


def test_risk_budgeting_target_frontier_returns_list(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import (
        _solve_rb_at_c,
        risk_budgeting_target_frontier,
    )

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    b = np.full(4, 0.25)
    _, mu_min, _, _ = _solve_rb_at_c(sample_cov, b, mu, 1.0, None, np.zeros(4), "unconstrained", {})
    _, mu_max, _, _ = _solve_rb_at_c(
        sample_cov, b, mu, 100.0, None, np.zeros(4), "unconstrained", {}
    )
    lo, hi = min(mu_min, mu_max), max(mu_min, mu_max)
    targets = np.linspace(lo + 1e-5, hi - 1e-5, 3)

    results = risk_budgeting_target_frontier(sample_cov, mu, targets, target_type="return", b=b)
    assert len(results) == 3
    for r, t in zip(results, targets, strict=True):
        assert r.converged
        assert np.isclose(r.active_return, t, atol=1e-5)


def test_risk_budgeting_target_with_benchmark(sample_cov):
    from quanttoolbox.portfolio.risk_budgeting import _solve_rb_at_c, risk_budgeting_target

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    b = np.full(4, 0.25)
    x_benchmark = np.full(4, 0.25)

    _, mu_min, _, _ = _solve_rb_at_c(sample_cov, b, mu, 1.0, None, x_benchmark, "unconstrained", {})
    _, mu_max, _, _ = _solve_rb_at_c(
        sample_cov, b, mu, 100.0, None, x_benchmark, "unconstrained", {}
    )
    target = (mu_min + mu_max) / 2

    result = risk_budgeting_target(
        sample_cov, mu, target=target, target_type="return", b=b, x_benchmark=x_benchmark
    )
    assert result.converged
    assert np.isclose(result.active_return, target, atol=1e-6)


def test_risk_budgeting_target_box_solver_respects_bounds(sample_cov):
    # slower (nested ADMM+bisection per trial c), so keep this single-case
    from quanttoolbox.portfolio.risk_budgeting import _solve_rb_at_c, risk_budgeting_target

    mu = np.array([0.08, 0.05, 0.06, 0.03])
    b = np.full(4, 0.25)
    box_kwargs = {"x_minus": 0.15, "x_plus": 0.35}

    _, mu_min, _, _ = _solve_rb_at_c(sample_cov, b, mu, 1.0, None, np.zeros(4), "box", box_kwargs)
    _, mu_max, _, _ = _solve_rb_at_c(sample_cov, b, mu, 100.0, None, np.zeros(4), "box", box_kwargs)
    target = (mu_min + mu_max) / 2

    result = risk_budgeting_target(
        sample_cov, mu, target=target, target_type="return", b=b, solver="box", **box_kwargs
    )
    assert result.converged
    assert np.all(result.weights >= 0.15 - 1e-3)
    assert np.all(result.weights <= 0.35 + 1e-3)
