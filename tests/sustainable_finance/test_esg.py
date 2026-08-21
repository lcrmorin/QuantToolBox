"""Tests for quanttoolbox.sustainable_finance.esg."""

import numpy as np

from quanttoolbox.sustainable_finance.esg import (
    esg_beta_star,
    esg_minimum_variance,
    pedersen_portfolio,
)


def test_esg_beta_star_single_asset_matches_hand_derivation():
    # For n=1, the cross term (varphi_m * varphi_esg - varphi_m_esg^2) and
    # both raw omega1/omega2 correction terms vanish identically (single
    # asset is the equality case of Cauchy-Schwarz), collapsing the general
    # formula to a simple closed form -- derived by hand and cross-checked
    # here independently of the implementation's own algebraic path.
    b, be, st = 0.8, 0.3, 1.5
    sigma_m, sigma_esg = 0.2, 0.1

    omega0 = 1.0 + sigma_m**2 * b**2 / st**2 + sigma_esg**2 * be**2 / st**2
    expected_beta_star = omega0 / (sigma_m**2 * b / st**2)
    expected_beta_esg_star = omega0 / (sigma_esg**2 * be / st**2)

    beta_star, beta_esg_star = esg_beta_star(
        np.array([b]), sigma_m, np.array([be]), sigma_esg, np.array([st])
    )

    assert np.isclose(beta_star, expected_beta_star)
    assert np.isclose(beta_esg_star, expected_beta_esg_star)


def test_esg_minimum_variance_weights_sum_to_one():
    beta = np.array([0.9, 1.1, 1.0, 0.8])
    beta_esg = np.array([0.2, -0.1, 0.3, 0.0])
    sigma_tilde = np.array([0.15, 0.20, 0.18, 0.22])

    result = esg_minimum_variance(beta, 0.18, beta_esg, 0.05, sigma_tilde)

    assert np.isclose(np.sum(result.x), 1.0)
    assert np.isclose(np.sum(result.x_tilde), 1.0)
    assert result.sigma_x > 0.0


def test_esg_minimum_variance_default_e_matches_all_ones_mask():
    beta = np.array([0.9, 1.1, 1.0])
    beta_esg = np.array([0.2, -0.1, 0.3])
    sigma_tilde = np.array([0.15, 0.20, 0.18])

    default = esg_minimum_variance(beta, 0.18, beta_esg, 0.05, sigma_tilde)
    explicit = esg_minimum_variance(beta, 0.18, beta_esg, 0.05, sigma_tilde, e=np.ones(3))

    assert np.allclose(default.x, explicit.x)
    assert np.isclose(default.sigma_x, explicit.sigma_x)


def test_esg_minimum_variance_mask_restricts_universe():
    beta = np.array([0.9, 1.1, 1.0, 0.8])
    beta_esg = np.array([0.2, -0.1, 0.3, 0.0])
    sigma_tilde = np.array([0.15, 0.20, 0.18, 0.22])
    e = np.array([1.0, 1.0, 0.0, 1.0])

    result = esg_minimum_variance(beta, 0.18, beta_esg, 0.05, sigma_tilde, e=e)

    assert result.x[2] == 0.0
    assert result.sigma_tilde_matrix.shape == (3, 3)
    assert np.isclose(np.sum(result.x_tilde), 1.0)


def test_pedersen_portfolio_residual_weight_matches_one_minus_sum():
    mu = np.array([0.06, 0.08, 0.05])
    r = 0.02
    sigma = np.array(
        [
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.004],
            [0.002, 0.004, 0.03],
        ]
    )
    s = np.array([0.3, 0.7, 0.5])

    result = pedersen_portfolio(mu, r, sigma, s, sigma_bar=0.15, s_bar=0.4)

    assert np.allclose(result.w_r, 1.0 - np.sum(result.w, axis=0))


def test_pedersen_portfolio_internal_consistency_of_returned_diagnostics():
    mu = np.array([0.06, 0.08, 0.05])
    r = 0.02
    sigma = np.array(
        [
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.004],
            [0.002, 0.004, 0.03],
        ]
    )
    s = np.array([0.3, 0.7, 0.5])
    pi_ = mu - r

    result = pedersen_portfolio(mu, r, sigma, s, sigma_bar=0.15, s_bar=0.4)
    w = result.w[:, 0]

    assert np.isclose(result.pi_w[0], w @ pi_)
    assert np.isclose(result.sigma_w[0], np.sqrt(w @ sigma @ w))
    assert np.isclose(result.s_w[0], (w @ s) / np.sum(w))


def test_pedersen_portfolio_broadcasts_scalar_sigma_bar_over_s_bar_array():
    mu = np.array([0.06, 0.08])
    r = 0.02
    sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
    s = np.array([0.3, 0.7])

    result = pedersen_portfolio(mu, r, sigma, s, sigma_bar=0.15, s_bar=np.array([0.3, 0.4, 0.5]))

    assert result.w.shape == (2, 3)
    assert np.allclose(result.sigma_bar, [0.15, 0.15, 0.15])
    assert np.allclose(result.s_bar, [0.3, 0.4, 0.5])
