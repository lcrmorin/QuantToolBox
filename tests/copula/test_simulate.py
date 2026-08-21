"""Tests for quanttoolbox.copula.simulate."""

import numpy as np
from scipy.stats import kendalltau
from scipy.stats import norm as normdist
from scipy.stats import t as tdist

from quanttoolbox.copula.dependence import frank_tau, gumbel_tau
from quanttoolbox.copula.families import amh_conditional_cdf, gumbel_cdf
from quanttoolbox.copula.simulate import (
    empirical_quantile_transform,
    simulate_amh,
    simulate_frank,
    simulate_from_conditional_cdf,
    simulate_gaussian_copula,
    simulate_gumbel,
    simulate_student_copula,
)


def test_simulate_gumbel_reproducible_and_in_unit_square():
    theta = 2.0
    u1a, u2a = simulate_gumbel(theta, 500, random_state=42)
    u1b, u2b = simulate_gumbel(theta, 500, random_state=42)
    assert np.array_equal(u1a, u1b) and np.array_equal(u2a, u2b)
    assert np.all(u1a > 0.0) and np.all(u1a < 1.0)
    assert np.all(u2a > 0.0) and np.all(u2a < 1.0)


def test_simulate_gumbel_empirical_kendall_tau_matches_theory():
    theta = 2.0
    u1, u2 = simulate_gumbel(theta, 20000, random_state=0)
    tau_emp = kendalltau(u1, u2).statistic
    assert np.isclose(tau_emp, gumbel_tau(theta), atol=0.02)


def test_simulate_gumbel_samples_satisfy_own_conditional_cdf():
    # Sanity check that the samples are consistent with the Gumbel CDF:
    # empirical C(u1, u2) should track the analytic Gumbel CDF.
    theta = 1.5
    u1, u2 = simulate_gumbel(theta, 20000, random_state=1)
    for u1_cut, u2_cut in [(0.3, 0.3), (0.5, 0.7), (0.8, 0.2)]:
        empirical = np.mean((u1 <= u1_cut) & (u2 <= u2_cut))
        analytic = gumbel_cdf(u1_cut, u2_cut, theta)
        assert np.isclose(empirical, analytic, atol=0.02)


def test_simulate_from_conditional_cdf_matches_simulate_gumbel_engine():
    # simulate_gumbel is documented as simulate_from_conditional_cdf with
    # Gumbel's conditional CDF plugged in -- verify that's literally true
    # for identical random draws.
    theta = 2.0

    def conditional_cdf(u1, u2):
        u1t = -np.log(u1)
        u2t = -np.log(u2)
        w = u1t**theta + u2t**theta
        beta = 1.0 / theta
        return np.exp(-(w**beta)) * (1.0 + (u2t / u1t) ** theta) ** (beta - 1.0) / u1

    u1a, u2a = simulate_from_conditional_cdf(conditional_cdf, 300, random_state=7)
    u1b, u2b = simulate_gumbel(theta, 300, random_state=7)
    assert np.allclose(u1a, u1b)
    assert np.allclose(u2a, u2b)


def test_simulate_from_conditional_cdf_works_for_amh_via_generic_engine():
    theta = 0.6
    u1, u2 = simulate_from_conditional_cdf(
        lambda u1, u2: amh_conditional_cdf(u1, u2, theta), 5000, random_state=3
    )
    assert np.all(np.isfinite(u2))
    assert np.all(u2 > 0.0) and np.all(u2 < 1.0)
    # Cross-check against the closed-form AMH simulator's dependence level.
    u1_cf, u2_cf = simulate_amh(theta, 5000, random_state=3)
    tau_generic = kendalltau(u1, u2).statistic
    tau_closed_form = kendalltau(u1_cf, u2_cf).statistic
    assert np.isclose(tau_generic, tau_closed_form, atol=0.03)


def test_simulate_amh_in_unit_square_and_reproducible():
    theta = 0.5
    u1a, u2a = simulate_amh(theta, 2000, random_state=5)
    u1b, u2b = simulate_amh(theta, 2000, random_state=5)
    assert np.array_equal(u1a, u1b) and np.array_equal(u2a, u2b)
    assert np.all(u2a >= 0.0) and np.all(u2a <= 1.0)
    assert np.all(np.isfinite(u2a))


def test_simulate_frank_empirical_kendall_tau_matches_theory():
    theta = 3.0
    u1, u2 = simulate_frank(theta, 20000, random_state=0)
    tau_emp = kendalltau(u1, u2).statistic
    assert np.isclose(tau_emp, frank_tau(theta), atol=0.02)
    assert np.all(u2 > 0.0) and np.all(u2 < 1.0)


def test_simulate_gaussian_copula_recovers_correlation_and_uniform_margins():
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    u = simulate_gaussian_copula(corr, 50000, random_state=0)
    assert u.shape == (50000, 3)
    assert np.all(u > 0.0) and np.all(u < 1.0)

    z = normdist.ppf(u)
    empirical_corr = np.corrcoef(z.T)
    assert np.allclose(empirical_corr, corr, atol=0.03)

    # Each column, mapped back through Phi^-1, should look standard normal:
    # a loose Kolmogorov-Smirnov-style check via mean/std.
    for col in range(3):
        assert np.isclose(z[:, col].mean(), 0.0, atol=0.05)
        assert np.isclose(z[:, col].std(), 1.0, atol=0.05)


def test_simulate_student_copula_recovers_correlation():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    nu = 5.0
    u = simulate_student_copula(corr, nu, 50000, random_state=0)
    assert u.shape == (50000, 2)
    assert np.all(u > 0.0) and np.all(u < 1.0)

    x = tdist.ppf(u, nu)
    empirical_corr = np.corrcoef(x.T)
    assert np.allclose(empirical_corr, corr, atol=0.05)


def test_simulate_student_copula_has_heavier_tail_dependence_than_gaussian():
    # A well-known qualitative property distinguishing the two elliptical
    # copulas: for the same linear correlation, the Student-t copula
    # exhibits (asymptotic) tail dependence and the Gaussian does not --
    # so joint extreme-corner co-occurrence should be higher for Student.
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    n = 40000
    u_gauss = simulate_gaussian_copula(corr, n, random_state=0)
    u_student = simulate_student_copula(corr, 4.0, n, random_state=0)

    cutoff = 0.01
    joint_gauss = np.mean((u_gauss[:, 0] < cutoff) & (u_gauss[:, 1] < cutoff))
    joint_student = np.mean((u_student[:, 0] < cutoff) & (u_student[:, 1] < cutoff))
    assert joint_student > joint_gauss


def test_empirical_quantile_transform_output_within_reference_range():
    rng = np.random.default_rng(2)
    x_ref = rng.normal(size=2000)
    u = rng.random((1000, 1))
    y = empirical_quantile_transform(u, x_ref)
    assert y.shape == (1000, 1)
    assert y.min() >= x_ref.min() - 1e-8
    assert y.max() <= x_ref.max() + 1e-8


def test_empirical_quantile_transform_is_monotone_in_u():
    rng = np.random.default_rng(3)
    x_ref = np.sort(rng.normal(size=500))
    u = np.linspace(0.01, 0.99, 20)[:, None]
    y = empirical_quantile_transform(u, x_ref)
    assert np.all(np.diff(y[:, 0]) >= -1e-12)


def test_empirical_quantile_transform_endpoints_match_reference_extremes():
    x_ref = np.sort(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    u = np.array([[1e-9], [1.0 - 1e-9]])
    y = empirical_quantile_transform(u, x_ref)
    assert np.isclose(y[0, 0], x_ref[0], atol=1e-2)
    assert np.isclose(y[1, 0], x_ref[-1], atol=1e-2)


def test_empirical_quantile_transform_per_column_reference():
    rng = np.random.default_rng(4)
    x_ref = np.column_stack([rng.normal(size=1000), rng.exponential(size=1000)])
    u = rng.random((200, 2))
    y = empirical_quantile_transform(u, x_ref)
    assert y.shape == (200, 2)
    assert y[:, 0].min() >= x_ref[:, 0].min() - 1e-8
    assert y[:, 1].min() >= x_ref[:, 1].min() - 1e-8
