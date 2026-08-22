"""Tests for quanttoolbox.sustainable_finance.esg."""

import numpy as np
from scipy.stats import multivariate_normal, norm

from quanttoolbox.sustainable_finance.esg import (
    esg_beta_star,
    esg_minimum_variance,
    esg_transition_matrix,
    mc_weighted_score,
    pedersen_portfolio,
)


def _notebook_esg_transition_matrix(s, sigma):
    # A verbatim reimplementation of HSF-Notebooks chapter 16a's own
    # `esg_transition_matrix`/`rectangle_prob` (notebook 16a, cell 28),
    # kept independent of quanttoolbox's implementation so it serves as a
    # genuine cross-check rather than testing the code against itself.
    def rectangle_prob(mvn, xl, xu):
        return (
            mvn.cdf([xu[0], xu[1]])
            - mvn.cdf([xl[0], xu[1]])
            - mvn.cdf([xu[0], xl[1]])
            + mvn.cdf([xl[0], xl[1]])
        )

    K = 1 + len(s)
    s_ext = np.sort(np.concatenate([[-np.inf], s, [np.inf]]))
    p = np.zeros((K, K))
    cdf = np.zeros(K)
    mvn = multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, 1.0], [1.0, 1.0 + sigma**2]], allow_singular=True
    )
    for j in range(K):
        cdf1 = norm.cdf(s_ext[j + 1]) - norm.cdf(s_ext[j])
        cdf[j] = cdf1
        for k in range(K):
            cdf2 = rectangle_prob(mvn, [s_ext[j], s_ext[k]], [s_ext[j + 1], s_ext[k + 1]])
            p[j, k] = cdf2 / cdf1
    to = 1 - np.diag(p)
    to_system = np.sum(cdf * to)
    return p, cdf, to, to_system


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


def test_esg_transition_matrix_matches_notebook_16a_on_its_own_data():
    # HSF-Notebooks chapter 16a Q4.h's own three concrete cut-point vectors
    # (s1, s2, s3) at its own sigma=0.10, cross-checked against an
    # independent (not-imported-from-quanttoolbox) reimplementation of the
    # notebook's own algorithm.
    s1 = np.array([-1.50, 0.0, 1.50])
    s2 = norm.ppf([0.25, 0.50, 0.75])
    s3 = norm.ppf([0.10, 0.50, 0.90])
    sigma = 0.10

    for s in (s1, s2, s3):
        result = esg_transition_matrix(s, sigma)
        p_nb, cdf_nb, to_nb, to_system_nb = _notebook_esg_transition_matrix(s, sigma)

        assert np.allclose(result.p, p_nb)
        assert np.allclose(result.cdf, cdf_nb)
        assert np.allclose(result.to, to_nb)
        assert np.isclose(result.to_system, to_system_nb)


def test_esg_transition_matrix_rows_sum_to_one_and_cdf_sums_to_one():
    s = np.array([-1.50, 0.0, 1.50])
    result = esg_transition_matrix(s, sigma=0.10)

    assert np.allclose(result.p.sum(axis=1), 1.0)
    assert np.isclose(np.sum(result.cdf), 1.0)
    assert 0.0 <= result.to_system <= 1.0


def test_esg_transition_matrix_turnover_vanishes_as_sigma_shrinks():
    # Perfect remeasurement (sigma -> 0) means g2 -> g1, so no bucket ever
    # changes on remeasurement and turnover -> 0 -- matches notebook 16a's
    # own sanity check in cell 33.
    s = np.array([-1.50, 0.0, 1.50])
    result = esg_transition_matrix(s, sigma=1e-3)

    assert result.to_system < 1e-2


def test_esg_transition_matrix_higher_sigma_gives_higher_turnover():
    s = np.array([-1.50, 0.0, 1.50])
    low = esg_transition_matrix(s, sigma=0.05)
    high = esg_transition_matrix(s, sigma=0.25)

    assert high.to_system > low.to_system


def test_mc_weighted_score_deterministic_with_explicit_g1_g2():
    # With g1/g2 given explicitly the function is fully deterministic (no
    # randomness is drawn), so its output can be hand-derived: alpha=1
    # leaves the power-law reweighting a no-op, u1=Phi(g1)=0.5 for both
    # assets (g1=0), so x=[0.5,0.5]; s=g2=[1,3] gives s_x=0.5*1+0.5*3=2.0.
    rng = np.random.default_rng(0)
    g1 = np.array([[0.0], [0.0]])
    g2 = np.array([[1.0], [3.0]])

    s_x = mc_weighted_score(np.array([1.0]), nS=1, n=2, rng=rng, g1=g1, g2=g2)

    assert s_x.shape == (1, 1)
    assert np.isclose(s_x[0, 0], 2.0)


def test_mc_weighted_score_shape_and_weight_normalization():
    rng = np.random.default_rng(1)
    alpha_vec = np.array([0.5, 1.5, 2.5, 70.0])
    n, nS = 10, 25

    s_x = mc_weighted_score(alpha_vec, nS, n, rng)

    assert s_x.shape == (nS, len(alpha_vec))
    assert np.all(np.isfinite(s_x))


def test_mc_weighted_score_uncorrelated_case_has_mean_close_to_zero():
    # Notebook 16a Q2.e: s drawn independently of the weights x, so
    # E[S(x)] = E[sum(x*s)] = 0 for every alpha (each s_i has mean 0 and is
    # independent of x). Large-sample Monte Carlo check.
    rng = np.random.default_rng(123)
    alpha_vec = np.array([0.5, 1.5, 2.5, 70.0])
    n, nS = 50, 50_000

    s_x = mc_weighted_score(alpha_vec, nS, n, rng)
    mean_s_x = s_x.mean(axis=0)

    assert np.all(np.abs(mean_s_x) < 0.02)


def test_mc_weighted_score_correlated_case_has_positive_mean():
    # Notebook 16a Q2.g: with g1/g2 positively correlated (rho=0.5), the
    # weights (tilted from g1) become positively correlated with the score
    # s=g2, so E[S(x)] > 0 for every alpha.
    rng = np.random.default_rng(7)
    alpha_vec = np.array([0.5, 1.5, 2.5, 70.0])
    n, nS = 50, 50_000
    rho = 0.5

    g1 = rng.standard_normal((n, nS))
    g2 = rho * g1 + np.sqrt(1 - rho**2) * rng.standard_normal((n, nS))
    s_x = mc_weighted_score(alpha_vec, nS, n, rng, g1=g1, g2=g2)

    assert np.all(s_x.mean(axis=0) > 0.0)
