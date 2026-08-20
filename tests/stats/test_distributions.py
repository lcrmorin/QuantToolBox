"""Tests for quanttoolbox.stats.distributions."""

import numpy as np
from scipy.stats import chi2, f, norm

from quanttoolbox.stats.distributions import (
    chi2_cdf,
    chi2_sf,
    f_cdf,
    f_sf,
    gqf1_cdf,
    gqf1_moments,
    gqf1_to_gqf2,
    gqf2_cdf,
    gqf2_moments,
    gqf2_to_gqf1,
    mvn_cdf,
    mvn_pdf,
    mvn_rvs,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    student_t_cdf,
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
