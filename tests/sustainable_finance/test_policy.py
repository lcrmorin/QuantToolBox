"""Tests for quanttoolbox.sustainable_finance.policy."""

import numpy as np
from scipy.integrate import quad

from quanttoolbox.sustainable_finance.policy import weitzman_example


def test_scenario_5a_matches_notebook_output():
    # Real regression data from HSF-Notebooks 10e_weitzman_prices_vs_quantities.ipynb,
    # itself matching chap10_externality5a.m's own printed results.
    result = weitzman_example(
        alpha_b=100, beta_b=2, alpha_c=20, beta_c=8, epsilon_i=[30, -30], p_i=[0.50, 0.50]
    )

    assert np.allclose(result.q_star_i, [5.0, 11.0])
    assert np.allclose(result.q1_ast_i, [8.0, 8.0])
    assert np.allclose(result.q2_ast_i, [4.25, 11.75])
    assert np.allclose(result.w_q1_ast_i, [80.0, 560.0])
    assert np.allclose(result.w_q2_ast_i, [122.188, 602.188], atol=5e-3)
    assert np.isclose(result.mean_epsilon, 0.0)
    assert np.isclose(result.sigma_epsilon, 30.0)
    assert np.isclose(result.q_ast, 8.0)
    assert np.isclose(result.p_ast, 84.0)
    assert np.isclose(result.e_w_q1_ast, 320.0, atol=5e-3)
    assert np.isclose(result.e_w_q2_ast, 362.1875, atol=5e-3)
    assert np.isclose(result.delta_w, 42.1875, atol=5e-3)
    assert np.isclose(result.delta_w_analytical, 42.1875, atol=5e-3)


def test_scenario_5b_matches_notebook_output():
    result = weitzman_example(
        alpha_b=100,
        beta_b=2,
        alpha_c=20,
        beta_c=8,
        epsilon_i=[10, 30, 50],
        p_i=[0.40, 0.35, 0.25],
    )

    assert np.isclose(result.mean_epsilon, 27.0)
    assert np.isclose(result.sigma_epsilon, 15.843, atol=5e-3)
    assert np.isclose(result.q_ast, 5.300, atol=5e-3)
    assert np.isclose(result.p_ast, 89.400, atol=5e-3)
    assert np.isclose(result.e_w_q1_ast, 140.4500, atol=5e-3)
    assert np.isclose(result.e_w_q2_ast, 152.2156, atol=5e-3)
    assert np.isclose(result.delta_w, 11.7656, atol=5e-3)
    assert np.isclose(result.delta_w_analytical, 11.7656, atol=5e-3)


def test_scenario_5d_matches_notebook_output():
    result = weitzman_example(
        alpha_b=100,
        beta_b=4,
        alpha_c=20,
        beta_c=5,
        epsilon_i=[10, 25, 50],
        p_i=[0.40, 0.50, 0.10],
    )

    assert np.isclose(result.mean_epsilon, 21.500, atol=5e-3)
    assert np.isclose(result.sigma_epsilon, 11.843, atol=5e-3)
    assert np.isclose(result.q_ast, 6.500, atol=5e-3)
    assert np.isclose(result.p_ast, 74.000, atol=5e-3)
    assert np.isclose(result.e_w_q1_ast, 190.1250, atol=5e-3)
    assert np.isclose(result.e_w_q2_ast, 192.9300, atol=5e-3)
    assert np.isclose(result.delta_w, 2.8050, atol=5e-3)
    assert np.isclose(result.delta_w_analytical, 2.8050, atol=5e-3)


def test_welfare_matches_numerical_integral_of_marginal_welfare():
    # chap10_externality5a.m's own dW/integral() cross-check, reproduced via
    # scipy.integrate.quad (matching MATLAB's `integral` to ~1e-14 per the
    # HSF-Notebooks build notes).
    alpha_b, beta_b, alpha_c, beta_c = 100.0, 2.0, 20.0, 8.0
    epsilon_i = np.array([30.0, -30.0])

    def dw(q, epsilon):
        return (alpha_b - beta_b * q) - (alpha_c + beta_c * q + epsilon)

    for epsilon in epsilon_i:
        for q in [0, 1, 2, 3, 4]:
            numerical, _ = quad(dw, 0, q, args=(epsilon,))
            analytical = ((alpha_b - alpha_c - epsilon) - 0.5 * q * (beta_b + beta_c)) * q
            assert np.isclose(numerical, analytical, atol=1e-10)


def test_q_star_i_is_the_unconstrained_welfare_maximizer_per_state():
    # q_star_i should zero the marginal welfare dW/dq = SMB - SMC in each state.
    alpha_b, beta_b, alpha_c, beta_c = 100.0, 4.0, 20.0, 5.0
    epsilon_i = np.array([10.0, 25.0, 50.0])
    p_i = np.array([0.40, 0.50, 0.10])

    result = weitzman_example(alpha_b, beta_b, alpha_c, beta_c, epsilon_i, p_i)

    smb = alpha_b - beta_b * result.q_star_i
    smc = alpha_c + beta_c * result.q_star_i + epsilon_i
    assert np.allclose(smb, smc)


def test_price_instrument_dominates_when_beta_c_exceeds_beta_b():
    # Weitzman's classic result: the price instrument wins in expectation
    # iff beta_C > beta_B (slope of SMC exceeds slope of SMB).
    result = weitzman_example(
        alpha_b=100, beta_b=2, alpha_c=20, beta_c=8, epsilon_i=[30, -30], p_i=[0.5, 0.5]
    )
    assert result.delta_w > 0


def test_instruments_tie_when_beta_c_equals_beta_b():
    # chap10_externality5e.m sweeps beta_C and finds the two instruments'
    # expected welfare crosses exactly at beta_C == beta_B.
    result = weitzman_example(
        alpha_b=100, beta_b=4, alpha_c=20, beta_c=4, epsilon_i=[10, 25, 50], p_i=[0.4, 0.5, 0.1]
    )
    assert np.isclose(result.delta_w, 0.0, atol=1e-8)
    assert np.isclose(result.delta_w_analytical, 0.0, atol=1e-8)
