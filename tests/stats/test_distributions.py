"""Tests for quanttoolbox.stats.distributions."""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, f, invgauss, lognorm, norm, skewnorm, t

from quanttoolbox.stats.distributions import (
    bates_cdf,
    bates_pdf,
    beta_cdf,
    beta_pdf,
    chi2_cdf,
    chi2_ppf,
    chi2_sf,
    constant_correlation_matrix,
    f_cdf,
    f_sf,
    gqf1_cdf,
    gqf1_moments,
    gqf1_to_gqf2,
    gqf2_cdf,
    gqf2_moments,
    gqf2_to_gqf1,
    inverse_gaussian_cdf,
    inverse_gaussian_pdf,
    lognormal_cdf,
    lognormal_pdf,
    mvn_cdf,
    mvn_pdf,
    mvn_rvs,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    normal_ratio_cdf,
    normal_ratio_pdf,
    order_statistic_cdf,
    order_statistic_ppf,
    poisson_binomial_pmf,
    skew_normal_cdf,
    skew_normal_moments,
    skew_normal_pdf,
    skew_normal_ppf,
    skew_normal_rvs,
    skew_t_cdf,
    skew_t_moments,
    skew_t_pdf,
    skew_t_ppf,
    skew_t_rvs,
    student_t_cdf,
    student_t_pdf,
    student_t_ppf,
    student_t_sf,
)

# ---------------------------------------------------------------------------
# Simple wrappers: just confirm they agree with direct scipy.stats calls
# ---------------------------------------------------------------------------


def test_normal_cdf_matches_scipy():
    x = np.array([-1.0, 0.0, 1.5])
    assert np.allclose(normal_cdf(x, mu=1.0, sigma=2.0), norm.cdf(x, loc=1.0, scale=2.0))


def test_normal_ppf_matches_scipy():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(normal_ppf(p), norm.ppf(p))


def test_normal_pdf_matches_scipy():
    x = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(normal_pdf(x), norm.pdf(x))


def test_student_t_roundtrip():
    p = np.array([0.1, 0.5, 0.9])
    x = student_t_ppf(p, nu=5)
    assert np.allclose(student_t_cdf(x, nu=5), p)


def test_student_t_sf_complements_cdf():
    x = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(student_t_cdf(x, nu=5) + student_t_sf(x, nu=5), 1.0)


def test_chi2_sf_complements_cdf():
    x = np.array([1.0, 3.0, 5.0])
    assert np.allclose(chi2_cdf(x, nu=4) + chi2_sf(x, nu=4), 1.0)


def test_chi2_cdf_matches_scipy():
    x = np.array([1.0, 3.0])
    assert np.allclose(chi2_cdf(x, nu=4), chi2.cdf(x, df=4))


def test_f_sf_complements_cdf():
    x = np.array([0.5, 1.0, 2.0])
    assert np.allclose(f_cdf(x, nu1=3, nu2=10) + f_sf(x, nu1=3, nu2=10), 1.0)


def test_f_cdf_matches_scipy():
    x = np.array([1.0, 2.0])
    assert np.allclose(f_cdf(x, nu1=3, nu2=10), f.cdf(x, dfn=3, dfd=10))


def test_mvn_cdf_and_pdf_2d(rng):
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    x = np.array([0.5, -0.5])
    cdf = mvn_cdf(x, mu, sigma)
    pdf = mvn_pdf(x, mu, sigma)
    assert 0.0 <= cdf <= 1.0
    assert pdf > 0.0


def test_mvn_rvs_shape_and_moments(rng):
    mu = np.array([1.0, -1.0])
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    samples = mvn_rvs(mu, sigma, n_samples=20000, random_state=42)
    assert samples.shape == (20000, 2)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples.T), sigma, atol=0.15)


# ---------------------------------------------------------------------------
# GQF #1: sum(a_i * (Z_i + b_i)^2)  -- validate against Monte Carlo
# ---------------------------------------------------------------------------


def _simulate_gqf1(a, b, n_samples, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    z = rng.standard_normal((n_samples, a.shape[0]))
    return np.sum(a * (z + b) ** 2, axis=1)


def test_gqf1_moments_match_monte_carlo(rng):
    a = np.array([1.0, 2.0, 0.5])
    b = np.array([0.5, -0.5, 1.0])
    samples = _simulate_gqf1(a, b, 500_000, rng)

    mean, sigma, _, _ = gqf1_moments(a, b)
    assert abs(mean - samples.mean()) < 0.05
    assert abs(sigma - samples.std()) < 0.05


def test_gqf1_cdf_matches_monte_carlo(rng):
    a = np.array([1.0, 2.0, 0.5])
    b = np.array([0.5, -0.5, 1.0])
    samples = _simulate_gqf1(a, b, 500_000, rng)

    for x in [1.0, 3.0, 6.0]:
        empirical = np.mean(samples <= x)
        analytic = gqf1_cdf(x, a, b)
        assert abs(empirical - analytic) < 0.02


def test_gqf1_cdf_is_monotonic():
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 0.0])
    xs = np.linspace(0.1, 10, 20)
    cdfs = np.array([gqf1_cdf(x, a, b) for x in xs])
    assert np.all(np.diff(cdfs) >= -1e-6)


def test_gqf1_special_case_matches_chi2():
    # a=[1,1,1], b=[0,0,0] -> sum of 3 iid squared standard normals = chi2(3)
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0])
    x = np.array([1.0, 3.0, 6.0])
    assert np.allclose(gqf1_cdf(x, a, b), chi2.cdf(x, df=3), atol=1e-3)


# ---------------------------------------------------------------------------
# GQF #1 <-> GQF #2 round-trip and cross-check against Monte Carlo
# ---------------------------------------------------------------------------


def test_gqf1_to_gqf2_and_back_roundtrip():
    a = np.array([1.0, 2.0, 0.5])
    b = np.array([0.5, -0.5, 1.0])
    mu, sigma, q = gqf1_to_gqf2(a, b)
    a_back, b_back = gqf2_to_gqf1(mu, sigma, q)

    # order may differ (eigh doesn't preserve MATLAB's eig ordering) -> sort both
    order_orig = np.argsort(a)
    order_back = np.argsort(a_back)
    assert np.allclose(a[order_orig], a_back[order_back], atol=1e-8)
    assert np.allclose(np.abs(b[order_orig]), np.abs(b_back[order_back]), atol=1e-6)


def test_gqf2_moments_match_gqf1_moments():
    a = np.array([1.0, 2.0, 0.5])
    b = np.array([0.5, -0.5, 1.0])
    mu, sigma, q = gqf1_to_gqf2(a, b)

    m1, s1, _, _ = gqf1_moments(a, b)
    m2, s2, _, _, _, _ = gqf2_moments(mu, sigma, q)

    assert abs(m1 - m2) < 1e-8
    assert abs(s1 - s2) < 1e-8


def test_gqf2_cdf_matches_monte_carlo(rng):
    a = np.array([1.0, 2.0, 0.5])
    b = np.array([0.5, -0.5, 1.0])
    mu, sigma, q = gqf1_to_gqf2(a, b)
    samples = _simulate_gqf1(a, b, 500_000, rng)

    # GQF2's cdf is a coarser cumulant-matching approximation (by design,
    # same as the original MATLAB algorithm) so it's less tight than GQF1's
    # exact series expansion -- allow a slightly wider tolerance.
    for x in [1.0, 3.0, 6.0]:
        empirical = np.mean(samples <= x)
        analytic = gqf2_cdf(x, mu, sigma, q)
        assert abs(empirical - analytic) < 0.03


# ---------------------------------------------------------------------------
# HSF toolbox extension: simple new wrappers
# ---------------------------------------------------------------------------


def test_student_t_pdf_matches_scipy():
    x = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(student_t_pdf(x, nu=5), t.pdf(x, df=5))


def test_chi2_ppf_matches_scipy():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(chi2_ppf(p, nu=4), chi2.ppf(p, df=4))


# ---------------------------------------------------------------------------
# Beta / lognormal / inverse-Gaussian / Bates
# ---------------------------------------------------------------------------


def test_beta_cdf_pdf_match_scipy():
    x = np.array([0.2, 0.5, 0.8])
    assert np.allclose(beta_cdf(x, 2.0, 3.0), beta_dist.cdf(x, 2.0, 3.0))
    assert np.allclose(beta_pdf(x, 2.0, 3.0), beta_dist.pdf(x, 2.0, 3.0))


def test_lognormal_cdf_pdf_match_scipy_reparameterization():
    mu, sigma = 0.5, 0.8
    x = np.array([0.5, 1.5, 3.0])
    expected_cdf = lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    expected_pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    assert np.allclose(lognormal_cdf(x, mu, sigma), expected_cdf)
    assert np.allclose(lognormal_pdf(x, mu, sigma), expected_pdf)


def test_inverse_gaussian_cdf_pdf_match_scipy_reparameterization():
    mu, lam = 2.0, 3.0
    x = np.array([1.0, 2.5, 5.0])
    expected_cdf = invgauss.cdf(x, mu / lam, scale=lam)
    expected_pdf = invgauss.pdf(x, mu / lam, scale=lam)
    assert np.allclose(inverse_gaussian_cdf(x, mu, lam), expected_cdf)
    assert np.allclose(inverse_gaussian_pdf(x, mu, lam), expected_pdf)


def test_bates_cdf_n1_is_uniform():
    # Bates(n=1) is just Uniform(0, 1): cdf(x) = x.
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(bates_cdf(x, 1), x)


def test_bates_pdf_n1_is_uniform():
    x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(bates_pdf(x, 1), np.ones_like(x))


def test_bates_cdf_endpoints():
    assert bates_cdf(np.array([0.0]), 4)[0] == 0.0
    assert bates_cdf(np.array([1.0]), 4)[0] == 1.0


def test_bates_pdf_integrates_to_one_numerically():
    x = np.linspace(1e-6, 1 - 1e-6, 20001)
    pdf = bates_pdf(x, 3)
    integral = np.trapezoid(pdf, x)
    assert np.isclose(integral, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Poisson-binomial
# ---------------------------------------------------------------------------


def test_poisson_binomial_pmf_sums_to_one_and_matches_manual_dp():
    p = np.array([0.2, 0.5, 0.7, 0.3])
    n = p.shape[0]

    k, pmf = poisson_binomial_pmf(p)
    assert np.allclose(k, np.arange(n + 1))
    assert np.isclose(pmf.sum(), 1.0)

    # manual direct-recursion cross-check (1-based MATLAB semantics
    # transliterated to 0-based Python)
    manual = np.zeros(n + 1)
    manual[0] = 1.0
    for i in range(1, n + 1):
        nxt = np.zeros(n + 1)
        nxt[0] = (1 - p[i - 1]) * manual[0]
        nxt[i] = p[i - 1] * manual[i - 1]
        for j in range(1, i):
            nxt[j] = p[i - 1] * manual[j - 1] + (1 - p[i - 1]) * manual[j]
        manual = nxt
    assert np.allclose(pmf, manual)


def test_poisson_binomial_reduces_to_binomial_for_equal_p():
    from scipy.stats import binom

    p = np.full(6, 0.4)
    k, pmf = poisson_binomial_pmf(p)
    assert np.allclose(pmf, binom.pmf(k, 6, 0.4))


# ---------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------


def test_order_statistic_cdf_max_matches_f_x_power_n():
    # The CDF of the maximum (i = n) of n iid draws is F(x)^n.
    f_x = np.array([0.2, 0.5, 0.8])
    n = 5
    result = order_statistic_cdf(f_x, n, i_select=np.array([n]))
    assert np.allclose(result[:, 0], f_x**n)


def test_order_statistic_cdf_min_matches_one_minus_survival_power_n():
    # The CDF of the minimum (i = 1) of n iid draws is 1 - (1 - F(x))^n.
    f_x = np.array([0.2, 0.5, 0.8])
    n = 5
    result = order_statistic_cdf(f_x, n, i_select=np.array([1]))
    assert np.allclose(result[:, 0], 1 - (1 - f_x) ** n)


def test_order_statistic_ppf_round_trips_cdf_on_a_grid():
    n = 4
    x_grid = np.linspace(0.01, 0.99, 500)
    f_x = x_grid  # Uniform(0,1): F(x) = x
    i_select = np.array([2])

    alpha = np.array([0.3, 0.6])
    q = order_statistic_ppf(alpha, x_grid, f_x, n, i_select=i_select)

    # q should be close to where order_statistic_cdf crosses alpha
    cdf_at_q = order_statistic_cdf(np.array(q[:, 0]), n, i_select=i_select)
    assert np.all(cdf_at_q[:, 0] >= alpha - 1e-2)


def test_constant_correlation_matrix_shape_and_values():
    c = constant_correlation_matrix(4, 0.3)
    assert c.shape == (4, 4)
    assert np.allclose(np.diag(c), 1.0)
    off_diag = c[~np.eye(4, dtype=bool)]
    assert np.allclose(off_diag, 0.3)


# ---------------------------------------------------------------------------
# Normal-ratio distribution
# ---------------------------------------------------------------------------


def test_normal_ratio_cdf_is_between_zero_and_one():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    result = normal_ratio_cdf(z, mu_x=0.0, sigma_x=1.0, mu_y=2.0, sigma_y=0.5)
    assert np.all(result.p >= 0.0)
    assert np.all(result.p <= 1.0)
    assert np.all(np.diff(result.p) >= -1e-8)  # monotone non-decreasing CDF


def test_normal_ratio_pdf_matches_numerical_derivative_of_cdf():
    mu_x, sigma_x, mu_y, sigma_y = 0.5, 1.0, 3.0, 0.4
    z0 = 0.2
    h = 1e-4
    cdf_hi = normal_ratio_cdf(np.array([z0 + h]), mu_x, sigma_x, mu_y, sigma_y).p[0]
    cdf_lo = normal_ratio_cdf(np.array([z0 - h]), mu_x, sigma_x, mu_y, sigma_y).p[0]
    numerical_pdf = (cdf_hi - cdf_lo) / (2 * h)

    pdf = normal_ratio_pdf(np.array([z0]), mu_x, sigma_x, mu_y, sigma_y).p[0]
    assert np.isclose(pdf, numerical_pdf, atol=1e-3)


# ---------------------------------------------------------------------------
# Skew-normal (verified equivalent to scipy.stats.skewnorm)
# ---------------------------------------------------------------------------


def test_skew_normal_cdf_pdf_match_scipy_skewnorm():
    xi, omega, eta = 0.5, 1.3, 2.0
    x = np.array([-1.0, 0.0, 0.8, 2.0])
    assert np.allclose(
        skew_normal_cdf(x, xi, omega, eta), skewnorm.cdf(x, eta, loc=xi, scale=omega)
    )
    assert np.allclose(
        skew_normal_pdf(x, xi, omega, eta), skewnorm.pdf(x, eta, loc=xi, scale=omega)
    )


def test_skew_normal_ppf_round_trips_cdf():
    xi, omega, eta = 0.5, 1.3, 2.0
    p = np.array([0.1, 0.4, 0.6, 0.9])
    x = skew_normal_ppf(p, xi, omega, eta)
    assert np.allclose(skew_normal_cdf(x, xi, omega, eta), p)


def test_skew_normal_moments_match_scipy_stats_mvsk():
    xi, omega, eta = 0.5, 1.3, 2.0
    mean, sigma, gamma1, gamma2 = skew_normal_moments(xi, omega, eta)
    exp_mean, exp_var, exp_skew, exp_kurt = skewnorm.stats(eta, loc=xi, scale=omega, moments="mvsk")
    assert np.isclose(mean, exp_mean)
    assert np.isclose(sigma, np.sqrt(exp_var))
    assert np.isclose(gamma1, exp_skew)
    assert np.isclose(gamma2, exp_kurt)


def test_skew_normal_reduces_to_normal_at_eta_zero():
    xi, omega = 1.0, 2.0
    x = np.array([-1.0, 0.5, 3.0])
    assert np.allclose(skew_normal_pdf(x, xi, omega, 0.0), norm.pdf(x, loc=xi, scale=omega))


def test_skew_normal_rvs_sample_mean_close_to_theoretical(rng):
    xi, omega, eta = 0.0, 1.0, 3.0
    samples = skew_normal_rvs(xi, omega, eta, size=200_000, random_state=rng)
    mean, sigma, _, _ = skew_normal_moments(xi, omega, eta)
    assert np.isclose(samples.mean(), mean, atol=0.02)
    assert np.isclose(samples.std(), sigma, atol=0.02)


# ---------------------------------------------------------------------------
# Skew-t (Azzalini) -- no scipy equivalent, hand-ported
# ---------------------------------------------------------------------------


def test_skew_t_pdf_reduces_to_scaled_student_t_at_eta_zero():
    xi, omega, nu = 1.0, 2.0, 6.0
    x = np.array([-1.0, 0.5, 3.0])
    xc = (x - xi) / omega
    expected = t.pdf(xc, df=nu) / omega
    assert np.allclose(skew_t_pdf(x, xi, omega, 0.0, nu), expected)


def test_skew_t_cdf_reduces_to_scaled_student_t_at_eta_zero():
    xi, omega, nu = 0.0, 1.0, 8.0
    x = np.array([-1.0, 0.5, 2.0])
    # eta=0 must be handled via the e>=0 branch since eta_abs is clamped
    # away from exactly zero (see module docstring / cdfST.m's 1e-8 floor)
    result = skew_t_cdf(x, xi, omega, 0.0, nu)
    expected = t.cdf(x, df=nu)
    assert np.allclose(result, expected, atol=1e-2)


def test_skew_t_ppf_round_trips_cdf():
    xi, omega, eta, nu = 0.0, 1.0, 1.5, 8.0
    p = np.array([0.25, 0.5, 0.75])
    x = skew_t_ppf(p, xi, omega, eta, nu)
    recovered = skew_t_cdf(x, xi, omega, eta, nu)
    assert np.allclose(recovered, p, atol=5e-3)


def test_skew_t_rvs_sample_moments_close_to_theoretical(rng):
    xi, omega, eta, nu = 0.0, 1.0, 2.0, 10.0
    samples = skew_t_rvs(xi, omega, eta, nu, size=300_000, random_state=rng)
    mean, sigma, _, _ = skew_t_moments(xi, omega, eta, nu)
    assert np.isclose(samples.mean(), mean, atol=0.03)
    assert np.isclose(samples.std(), sigma, atol=0.05)
