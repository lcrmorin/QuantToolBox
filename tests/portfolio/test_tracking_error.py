"""Tests for quanttoolbox.portfolio.tracking_error."""

import numpy as np
import pytest

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.black_litterman import black_litterman_moments, implied_risk_premia
from quanttoolbox.portfolio.tracking_error import (
    minimum_te_portfolio,
    solve_mixed_norm_te_portfolio,
    te_frontier,
    te_portfolio,
    te_target_portfolio,
)
from quanttoolbox.stats.moments import corr_to_cov


@pytest.fixture
def sample_data(rng):
    n = 4
    a = rng.standard_normal((200, n))
    cov = a.T @ a / 200 + 0.01 * np.eye(n)
    mu = np.array([0.08, 0.05, 0.06, 0.03])
    x_benchmark = np.full(n, 0.25)
    return x_benchmark, mu, cov


def test_te_portfolio_sums_to_one(sample_data):
    x_b, mu, cov = sample_data
    result = te_portfolio(x_b, mu, cov, gamma=1.0)
    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-5)


def test_te_portfolio_zero_gamma_stays_at_benchmark(sample_data):
    x_b, mu, cov = sample_data
    result = te_portfolio(x_b, mu, cov, gamma=0.0)
    # with no return tilt and starting feasible, should stay at (or very near) benchmark
    assert np.allclose(result.weights, x_b, atol=1e-4)
    assert np.isclose(result.tracking_error, 0.0, atol=1e-4)


def test_te_portfolio_higher_gamma_increases_active_return(sample_data):
    x_b, mu, cov = sample_data
    low = te_portfolio(x_b, mu, cov, gamma=0.1)
    high = te_portfolio(x_b, mu, cov, gamma=5.0)
    assert high.active_return >= low.active_return


def test_te_portfolio_higher_gamma_increases_tracking_error(sample_data):
    x_b, mu, cov = sample_data
    low = te_portfolio(x_b, mu, cov, gamma=0.1)
    high = te_portfolio(x_b, mu, cov, gamma=5.0)
    assert high.tracking_error >= low.tracking_error - 1e-6


def test_minimum_te_portfolio_matches_te_portfolio_gamma_zero(sample_data):
    x_b, _, cov = sample_data
    min_te = minimum_te_portfolio(x_b, cov)
    zero_gamma = te_portfolio(x_b, np.zeros(4), cov, gamma=0.0)
    assert np.allclose(min_te.weights, zero_gamma.weights, atol=1e-6)


def test_te_frontier_returns_list(sample_data):
    x_b, mu, cov = sample_data
    results = te_frontier(x_b, mu, cov, gamma_values=np.array([0.0, 1.0, 3.0]))
    assert len(results) == 3
    tes = [r.tracking_error for r in results]
    assert tes[0] <= tes[1] <= tes[2] + 1e-6


def test_te_portfolio_respects_box_bounds(sample_data):
    x_b, mu, cov = sample_data
    result = te_portfolio(x_b, mu, cov, gamma=2.0, lb=0.1, ub=0.4)
    assert np.all(result.weights >= 0.1 - 1e-4)
    assert np.all(result.weights <= 0.4 + 1e-4)


def test_te_target_portfolio_matches_matlab_reference():
    # Examples/rpb/test_bl4.m: Black-Litterman posterior mean, sigma-problem
    # target-matching against x0 -- golden values cross-verified against
    # the original MATLAB source (compute_te_portfolio.m) run via Octave.
    sigma = np.array([0.15, 0.20, 0.25, 0.30])
    rho = xpnd(np.array([1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]), method=1)
    cov = corr_to_cov(sigma, rho)

    x0 = np.array([0.40, 0.30, 0.20, 0.10])
    irp = implied_risk_premia(x0, cov, 0.25)
    mu_tilde = 0.03 + irp.pi
    p_matrix = np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float)
    bl = black_litterman_moments(
        mu_tilde, cov, p_matrix, np.array([0.04, -0.01]), np.diag([0.10**2, 0.05**2])
    )

    te_targets = np.array([0.01, 0.02, 0.03])
    results = te_target_portfolio(x0, bl.mu_bar, cov, te_targets, problem="sigma", lb=0.0, ub=1.0)
    expected_weights = np.array(
        [
            [0.3553, 0.3024, 0.2294, 0.1129],
            [0.3107, 0.3048, 0.2589, 0.1257],
            [0.2660, 0.3071, 0.2883, 0.1386],
        ]
    )
    for r, target, expected in zip(results, te_targets, expected_weights, strict=True):
        assert np.isclose(r.tracking_error, target, atol=1e-4)
        assert np.allclose(r.weights, expected, atol=2e-3)
        # every scenario should land on the same information ratio (the
        # BL posterior mean's implied Sharpe ratio doesn't change with gamma)
        assert np.isclose(r.active_return / r.tracking_error, 0.143, atol=1e-3)


def test_te_target_portfolio_matches_frontier_at_recovered_gamma(sample_data):
    x_b, mu, cov = sample_data
    result = te_target_portfolio(x_b, mu, cov, 0.05, problem="sigma")[0]
    direct = te_portfolio(x_b, mu, cov, gamma=result.gamma)
    assert np.allclose(result.weights, direct.weights, atol=1e-4)


def test_solve_mixed_norm_te_portfolio_matches_notebook_reference():
    # HSF-Notebooks 11b_l1_l2_norm_reformulation.ipynb / 16f (cell defining
    # solve_mixed_norm_te), example call with Reduction=0.20 -- golden
    # values cross-verified by running that notebook's own
    # solve_mixed_norm_te function directly on the same inputs.
    b = np.array([0.35, 0.15, 0.20, 0.30])
    ci = np.array([117, 284, 162.5, 359])
    md = np.array([3.00, 5.00, 2.00, 6.00])
    dts = np.array([100, 150, 200, 250])
    sector = np.array([1, 1, 2, 2])

    result = solve_mixed_norm_te_portfolio(
        b, ci, md, dts, sector, reduction=0.20, varphi_as=100, varphi_md=25, varphi_dts=1
    )

    expected_w_slsqp = np.array([4.73396309e-01, 4.59909015e-08, 3.33018267e-01, 1.93585379e-01])
    expected_w_lp = np.array([0.47339635, 0.0, 0.33301823, 0.19358542])

    assert np.allclose(result.weights_slsqp, expected_w_slsqp, atol=1e-4)
    assert np.allclose(result.weights_lp, expected_w_lp, atol=1e-6)
    # both solve paths should agree closely with each other
    assert result.max_abs_diff < 1e-3
    assert np.isclose(np.sum(result.weights_slsqp), 1.0, atol=1e-5)
    assert np.isclose(np.sum(result.weights_lp), 1.0, atol=1e-6)
    # the CI-reduction target should bind
    assert np.isclose(result.metrics_lp.reduction, 0.20, atol=1e-4)
    assert np.isclose(result.metrics_slsqp.reduction, 0.20, atol=1e-4)
    assert np.isclose(result.metrics_slsqp.r_mix, result.metrics_lp.r_mix, atol=1e-3)


def test_solve_mixed_norm_te_portfolio_respects_box_bounds():
    b9 = np.array([0.21, 0.19, 0.16, 0.12, 0.11, 0.08, 0.06, 0.04, 0.03])
    ci9 = np.array([111, 52, 369, 157, 18, 415, 17, 253, 900])
    md9 = np.array([3.16, 6.48, 3.54, 9.23, 6.40, 2.30, 8.12, 7.96, 5.48])
    dts9 = np.array([107, 255, 75, 996, 289, 45, 620, 285, 125])
    sector9 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    lb9 = 0.25 * b9
    ub9 = 4 * b9

    result = solve_mixed_norm_te_portfolio(
        b9,
        ci9,
        md9,
        dts9,
        sector9,
        reduction=0.30,
        varphi_as=100,
        varphi_md=25,
        varphi_dts=0.001,
        lb=lb9,
        ub=ub9,
    )
    assert np.all(result.weights_slsqp >= lb9 - 1e-6)
    assert np.all(result.weights_slsqp <= ub9 + 1e-6)
    assert np.all(result.weights_lp >= lb9 - 1e-6)
    assert np.all(result.weights_lp <= ub9 + 1e-6)
    assert np.isclose(np.sum(result.weights_lp), 1.0, atol=1e-6)


def _build_infeasible_glide_path_scenario(target_index):
    # HSF-Notebooks 11h_net_zero_te_optimization.ipynb, section 3 (equity
    # glide path with green-intensity floor and carbon-momentum ceiling):
    # the 8-asset toy universe, with the CI-reduction glide path targets
    # at indices 6 (R=51.30%) and 7 (R=66.12%) documented in that
    # notebook's markdown as jointly infeasible with the CM ceiling, GI
    # floor, and momentum exclusion (max achievable reduction ~51%).
    b = np.array([0.20, 0.19, 0.17, 0.13, 0.12, 0.08, 0.06, 0.05])
    ci = np.array([100.5, 97.2, 250.4, 352.3, 27.1, 54.2, 78.6, 426.7])
    beta = np.array([0.30, 1.80, 0.85, 0.83, 1.47, 0.94, 1.67, 1.08])
    sigma_tilde = np.array([0.10, 0.05, 0.06, 0.12, 0.15, 0.04, 0.08, 0.07])
    sigma_m = 0.18
    cov = np.outer(beta, beta) * sigma_m**2 + np.diag(sigma_tilde**2)
    gi = np.array([10.2, 45.3, 7.5, 0.0, 0.0, 35.6, 17.8, 3.0]) / 100
    cm = np.array([-3.1, -1.2, -5.8, -1.4, 7.4, -2.6, 1.2, -8.0]) / 100
    cm_star, cm_plus, greenness = -0.03, 0.05, 1
    ci_b, gi_b = b @ ci, b @ gi
    ub = (cm <= cm_plus).astype(float)

    r_minus, delta_r, t0, offsets = 0.30, 0.07, 2020, [0, 1, 2, 3, 4, 5, 10]
    t = t0 + np.asarray(offsets)
    r_glide = np.concatenate([[0.0], 1 - (1 - r_minus) * (1 - delta_r) ** (t - t0)])
    target = r_glide[target_index]

    c_ineq = np.vstack([ci, cm, -gi])
    d_ineq = np.array([(1 - target) * ci_b, cm_star, -(1 + greenness) * gi_b])
    lb = np.zeros(len(b))
    return b, cov, c_ineq, d_ineq, lb, ub


def test_minimum_te_portfolio_raise_on_infeasible_glide_path():
    # on_infeasible="raise" (the default) must keep raising on these two
    # documented-infeasible 11h glide-path scenarios.
    for target_index in (6, 7):
        b, cov, c_ineq, d_ineq, lb, ub = _build_infeasible_glide_path_scenario(target_index)
        n = len(b)
        with pytest.raises(RuntimeError):
            minimum_te_portfolio(
                b,
                cov,
                a_eq=np.ones((1, n)),
                b_eq=np.array([1.0]),
                c_ineq=c_ineq,
                d_ineq=d_ineq,
                lb=lb,
                ub=ub,
            )


def test_minimum_te_portfolio_nearest_matches_11h_slsqp_reference():
    # on_infeasible="nearest" should return SLSQP's near-feasible point,
    # matching (to a reasonable tolerance) 11h's own local min_te_portfolio
    # SLSQP solver on the same two infeasible glide-path scenarios.
    expected = {
        6: (np.array([0.0, 0.0, 0.121913, 0.0, 0.0, 0.881887, 0.0, 0.0]), 0.05155184688722009),
        7: (np.array([0.0, 0.0, 0.125265, 0.0, 0.0, 0.874410, 0.0, 0.0]), 0.05169623482810603),
    }
    for target_index, (expected_w, expected_te) in expected.items():
        b, cov, c_ineq, d_ineq, lb, ub = _build_infeasible_glide_path_scenario(target_index)
        n = len(b)
        result = minimum_te_portfolio(
            b,
            cov,
            a_eq=np.ones((1, n)),
            b_eq=np.array([1.0]),
            c_ineq=c_ineq,
            d_ineq=d_ineq,
            lb=lb,
            ub=ub,
            on_infeasible="nearest",
        )
        assert np.all(np.isfinite(result.weights))
        assert np.isfinite(result.tracking_error)
        assert np.allclose(result.weights, expected_w, atol=1e-2)
        assert np.isclose(result.tracking_error, expected_te, atol=1e-2)


def test_minimum_te_portfolio_on_infeasible_nearest_does_not_change_feasible_case(sample_data):
    # on a feasible problem, "nearest" should agree with "raise" (both
    # reach the same optimum -- no solver ever needs to fall back).
    x_b, _, cov = sample_data
    raise_result = minimum_te_portfolio(x_b, cov, on_infeasible="raise")
    nearest_result = minimum_te_portfolio(x_b, cov, on_infeasible="nearest")
    assert np.allclose(raise_result.weights, nearest_result.weights, atol=1e-4)


def test_minimum_te_portfolio_invalid_on_infeasible_raises_value_error(sample_data):
    x_b, _, cov = sample_data
    with pytest.raises(ValueError):
        minimum_te_portfolio(x_b, cov, on_infeasible="bogus")
