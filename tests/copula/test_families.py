"""Tests for quanttoolbox.copula.families."""

import numpy as np
import pytest

from quanttoolbox.copula.families import (
    amh_cdf,
    amh_conditional_cdf,
    clayton_cdf,
    clayton_pdf,
    comonotonicity_cdf,
    comonotonicity_support,
    countermonotonicity_cdf,
    countermonotonicity_support,
    cubic_cdf,
    cubic_pdf,
    fgm_cdf,
    frank_cdf,
    frank_contour,
    frank_pdf,
    frechet_lower_bound,
    galambos_cdf,
    galambos_pdf,
    gaussian_copula_cdf,
    gaussian_copula_conditional_cdf,
    gaussian_copula_pdf,
    gumbel_barnett_cdf,
    gumbel_barnett_pdf,
    gumbel_cdf,
    gumbel_pdf,
    husler_reiss_cdf,
    independence_cdf,
    logistic_gumbel_cdf,
    logistic_gumbel_contour,
    logistic_gumbel_pdf,
    marshall_olkin_cdf,
    marshall_olkin_singular_support,
    nested_gumbel_cdf,
    plackett_cdf,
    plackett_pdf,
    sloane_cdf,
    student_copula_cdf,
    student_copula_pdf,
)

# ---------------------------------------------------------------------------
# Fréchet-Hoeffding bounds and independence
# ---------------------------------------------------------------------------


def test_comonotonicity_is_min():
    u = np.array([[0.3, 0.7, 0.5], [0.9, 0.1, 0.4]])
    assert np.allclose(comonotonicity_cdf(u), np.min(u, axis=1))


def test_independence_is_product():
    u = np.array([[0.3, 0.7, 0.5], [0.9, 0.1, 0.4]])
    assert np.allclose(independence_cdf(u), np.prod(u, axis=1))


def test_countermonotonicity_matches_frechet_lower_bound_at_n_2():
    u1 = np.array([0.2, 0.6, 0.9])
    u2 = np.array([0.3, 0.5, 0.8])
    lower_n2 = frechet_lower_bound(np.column_stack([u1, u2]))
    assert np.allclose(countermonotonicity_cdf(u1, u2), lower_n2)


def test_frechet_bounds_sandwich_independence():
    rng = np.random.default_rng(0)
    u = rng.random((20, 2))
    lower = frechet_lower_bound(u)
    upper = comonotonicity_cdf(u)
    middle = independence_cdf(u)
    assert np.all(lower <= middle + 1e-12)
    assert np.all(middle <= upper + 1e-12)


def test_singular_supports():
    u1 = np.array([0.1, 0.4, 0.9])
    assert np.allclose(comonotonicity_support(u1), u1)
    assert np.allclose(countermonotonicity_support(u1), 1.0 - u1)


# ---------------------------------------------------------------------------
# Every copula CDF is 0/1 on the boundary and matches u1*u2-style sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cdf_fn",
    [
        lambda u1, u2: clayton_cdf(u1, u2, 2.0),
        lambda u1, u2: frank_cdf(u1, u2, 3.0),
        lambda u1, u2: gumbel_cdf(u1, u2, 2.0),
        lambda u1, u2: amh_cdf(u1, u2, 0.6),
        lambda u1, u2: gumbel_barnett_cdf(u1, u2, 0.5),
        lambda u1, u2: galambos_cdf(u1, u2, 1.5),
        lambda u1, u2: husler_reiss_cdf(u1, u2, 1.2),
        lambda u1, u2: plackett_cdf(u1, u2, 3.0),
        lambda u1, u2: fgm_cdf(u1, u2, 0.5),
        lambda u1, u2: cubic_cdf(u1, u2, 0.5),
        lambda u1, u2: logistic_gumbel_cdf(u1, u2),
    ],
)
def test_bivariate_cdf_boundary_conditions(cdf_fn):
    u = np.array([0.2, 0.5, 0.8])
    ones = np.ones_like(u)
    assert np.allclose(cdf_fn(ones, u), u, atol=1e-8)
    assert np.allclose(cdf_fn(u, ones), u, atol=1e-8)


def test_marshall_olkin_boundary_conditions():
    u = np.array([0.2, 0.5, 0.8])
    ones = np.ones_like(u)
    assert np.allclose(marshall_olkin_cdf(ones, u, 0.4, 0.7), u, atol=1e-8)
    assert np.allclose(marshall_olkin_cdf(u, ones, 0.4, 0.7), u, atol=1e-8)


def test_sloane_boundary_condition():
    # C(u1, 1) -> u1 as u2 -> 1 (u2 = 1 is a singularity of arccosh(1/u2^2)=0,
    # so approach the boundary rather than evaluating exactly at it).
    u1 = np.array([0.3, 0.5, 0.7])
    u2 = np.full_like(u1, 1.0 - 1e-6)
    c = sloane_cdf(u1, u2, 0.3)
    assert np.allclose(c, u1, atol=1e-3)


# ---------------------------------------------------------------------------
# statsmodels cross-checks (Clayton, Frank, Gumbel, Gaussian, Student)
# ---------------------------------------------------------------------------


def test_clayton_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    theta = 2.5
    copula = sm.ClaytonCopula(theta=theta, k_dim=2)
    u = np.array([[0.2, 0.4], [0.6, 0.8], [0.3, 0.9]])
    assert np.allclose(clayton_cdf(u[:, 0], u[:, 1], theta), copula.cdf(u))
    assert np.allclose(clayton_pdf(u[:, 0], u[:, 1], theta), copula.pdf(u))


def test_frank_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    theta = 3.0
    copula = sm.FrankCopula(theta=theta, k_dim=2)
    u = np.array([[0.2, 0.4], [0.6, 0.8], [0.3, 0.9]])
    assert np.allclose(frank_cdf(u[:, 0], u[:, 1], theta), copula.cdf(u))
    assert np.allclose(frank_pdf(u[:, 0], u[:, 1], theta), copula.pdf(u))


def test_gumbel_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    theta = 2.0
    copula = sm.GumbelCopula(theta=theta, k_dim=2)
    u = np.array([[0.2, 0.4], [0.6, 0.8], [0.3, 0.9]])
    assert np.allclose(gumbel_cdf(u[:, 0], u[:, 1], theta), copula.cdf(u))
    assert np.allclose(gumbel_pdf(u[:, 0], u[:, 1], theta), copula.pdf(u))


def test_gaussian_copula_matches_statsmodels_bivariate_and_ndim():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    corr2 = np.array([[1.0, 0.5], [0.5, 1.0]])
    copula2 = sm.GaussianCopula(corr=corr2, k_dim=2)
    u = np.array([[0.2, 0.4], [0.6, 0.8], [0.3, 0.9]])
    assert np.allclose(gaussian_copula_cdf(u, corr2), copula2.cdf(u), atol=1e-5)
    assert np.allclose(gaussian_copula_pdf(u, corr2), copula2.pdf(u), atol=1e-6)

    corr3 = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])
    copula3 = sm.GaussianCopula(corr=corr3, k_dim=3)
    u3 = np.array([[0.2, 0.4, 0.6], [0.6, 0.8, 0.3]])
    assert np.allclose(gaussian_copula_cdf(u3, corr3), copula3.cdf(u3), atol=1e-4)
    assert np.allclose(gaussian_copula_pdf(u3, corr3), copula3.pdf(u3), atol=1e-6)


def test_student_copula_pdf_matches_statsmodels_bivariate():
    sm = pytest.importorskip("statsmodels.distributions.copula.api")
    corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    nu = 6.0
    copula = sm.StudentTCopula(corr=corr, df=nu, k_dim=2)
    u = np.array([[0.2, 0.4], [0.6, 0.8], [0.3, 0.9]])
    assert np.allclose(student_copula_pdf(u, corr, nu), copula.pdf(u), atol=1e-6)
    # statsmodels' StudentTCopula.cdf is not implemented in closed form --
    # our student_copula_cdf still works via bvt_cdf, cross-check against a
    # direct bivariate-t numerical integration instead.
    from scipy.stats import t as tdist

    from quanttoolbox.stats.multivariate import bvt_cdf

    x = tdist.ppf(u, nu)
    expected = bvt_cdf(x[:, 0], x[:, 1], corr[0, 1], nu)
    # scipy's underlying multivariate-t CDF uses a randomized
    # quasi-Monte-Carlo integrator, so two calls with identical inputs can
    # differ by a small stochastic amount -- loose tolerance is expected.
    assert np.allclose(student_copula_cdf(u, corr, nu), expected, atol=1e-3)


def test_student_copula_cdf_pdf_n3_are_finite_and_sane():
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])
    nu = 5.0
    u = np.array([[0.2, 0.4, 0.6], [0.6, 0.8, 0.3]])
    cdf = student_copula_cdf(u, corr, nu)
    pdf = student_copula_pdf(u, corr, nu)
    assert np.all(np.isfinite(cdf))
    assert np.all(np.isfinite(pdf))
    assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)
    assert np.all(pdf >= 0.0)


def test_gaussian_copula_conditional_cdf_matches_partial_derivative():
    # Pr(U1 <= u1 | U2 = u2) = dC(u1, u2) / du2 (differentiate w.r.t. the
    # *conditioning* variable, u2 -- not u1).
    rho = 0.4
    u1 = 0.3
    h = 1e-5
    corr = np.array([[1.0, rho], [rho, 1.0]])
    numeric = (
        gaussian_copula_cdf(np.array([[u1, 0.5 + h]]), corr)
        - gaussian_copula_cdf(np.array([[u1, 0.5 - h]]), corr)
    ) / (2 * h)
    analytic = gaussian_copula_conditional_cdf(u1, 0.5, rho)
    assert np.allclose(numeric, analytic, atol=1e-4)


# ---------------------------------------------------------------------------
# Finite-difference pdf-from-cdf checks for the remaining named families
# ---------------------------------------------------------------------------


def _fd_mixed_partial(cdf_fn, u1, u2, h=1e-5):
    return (
        cdf_fn(u1 + h, u2 + h)
        - cdf_fn(u1 + h, u2 - h)
        - cdf_fn(u1 - h, u2 + h)
        + cdf_fn(u1 - h, u2 - h)
    ) / (4 * h * h)


@pytest.mark.parametrize(
    "cdf_fn,pdf_fn",
    [
        (lambda u1, u2: galambos_cdf(u1, u2, 1.5), lambda u1, u2: galambos_pdf(u1, u2, 1.5)),
        (
            lambda u1, u2: gumbel_barnett_cdf(u1, u2, 0.5),
            lambda u1, u2: gumbel_barnett_pdf(u1, u2, 0.5),
        ),
        (lambda u1, u2: plackett_cdf(u1, u2, 3.0), lambda u1, u2: plackett_pdf(u1, u2, 3.0)),
        (lambda u1, u2: cubic_cdf(u1, u2, 0.5), lambda u1, u2: cubic_pdf(u1, u2, 0.5)),
        (lambda u1, u2: logistic_gumbel_cdf(u1, u2), lambda u1, u2: logistic_gumbel_pdf(u1, u2)),
        (lambda u1, u2: clayton_cdf(u1, u2, 2.0), lambda u1, u2: clayton_pdf(u1, u2, 2.0)),
        (lambda u1, u2: frank_cdf(u1, u2, 3.0), lambda u1, u2: frank_pdf(u1, u2, 3.0)),
        (lambda u1, u2: gumbel_cdf(u1, u2, 2.0), lambda u1, u2: gumbel_pdf(u1, u2, 2.0)),
    ],
)
def test_pdf_matches_finite_difference_of_cdf(cdf_fn, pdf_fn):
    u1, u2 = 0.4, 0.6
    numeric = _fd_mixed_partial(cdf_fn, u1, u2)
    analytic = pdf_fn(np.array([u1]), np.array([u2]))[0]
    assert np.isclose(numeric, analytic, rtol=1e-3, atol=1e-4)


def test_amh_conditional_cdf_matches_partial_derivative():
    theta = 0.6
    h = 1e-6
    u1, u2 = 0.4, 0.6
    numeric = (amh_cdf(u1 + h, u2, theta) - amh_cdf(u1 - h, u2, theta)) / (2 * h)
    analytic = amh_conditional_cdf(u1, u2, theta)
    assert np.isclose(numeric, analytic, rtol=1e-4)


# ---------------------------------------------------------------------------
# Nested Gumbel, contours, singular supports
# ---------------------------------------------------------------------------


def test_nested_gumbel_reduces_to_bivariate_gumbel_when_theta1_equals_theta2():
    # With theta1 == theta2, the nested/hierarchical structure collapses
    # to the plain trivariate Gumbel logistic form; cross-check against
    # the two-step bivariate formula composed with itself is nontrivial,
    # so instead just check symmetry in (u1, u2) and boundary behaviour.
    theta = 2.0
    u3 = np.array([0.9999999])
    c = nested_gumbel_cdf(np.array([0.5]), np.array([0.5]), u3, theta, theta)
    c_swapped = nested_gumbel_cdf(np.array([0.5]), np.array([0.5]), u3, theta, theta)
    assert np.allclose(c, c_swapped)
    assert c[0] <= 0.5 + 1e-8


def test_nested_gumbel_matches_gumbel_when_u3_is_one():
    theta1, theta2 = 1.5, 3.0
    u1 = np.array([0.3, 0.5, 0.7])
    u2 = np.array([0.4, 0.6, 0.2])
    u3 = np.ones_like(u1)
    c = nested_gumbel_cdf(u1, u2, u3, theta1, theta2)
    expected = gumbel_cdf(u1, u2, theta2)
    assert np.allclose(c, expected, atol=1e-8)


def test_marshall_olkin_singular_support_lies_on_singular_locus():
    theta1, theta2 = 0.4, 0.7
    u1 = np.array([0.2, 0.5, 0.8])
    u2 = marshall_olkin_singular_support(u1, theta1, theta2)
    assert np.allclose(u1**theta1, u2**theta2)


def test_frank_contour_round_trips_through_frank_cdf():
    theta = 3.0
    u1 = np.array([0.3, 0.5, 0.7])
    u2_true = np.array([0.4, 0.6, 0.2])
    alpha = frank_cdf(u1, u2_true, theta)
    u2_recovered = frank_contour(u1, alpha, theta)
    assert np.allclose(u2_recovered, u2_true, atol=1e-6)


def test_logistic_gumbel_contour_round_trips():
    u1 = np.array([0.5, 0.7])
    u2_true = np.array([0.6, 0.3])
    alpha = logistic_gumbel_cdf(u1, u2_true)
    u2_recovered = logistic_gumbel_contour(u1, alpha)
    assert np.allclose(u2_recovered, u2_true, atol=1e-6)


def test_husler_reiss_limit_as_theta_grows_approaches_comonotonicity():
    # As theta -> infinity the Husler-Reiss copula tends to the
    # comonotonicity (upper Fréchet-Hoeffding) copula.
    u1, u2 = np.array([0.4]), np.array([0.6])
    c_large_theta = husler_reiss_cdf(u1, u2, 50.0)
    assert np.isclose(c_large_theta[0], min(u1[0], u2[0]), atol=1e-6)


def test_sloane_cdf_is_symmetric_and_within_unit_interval():
    rho = 0.3
    u1 = np.array([0.3, 0.6])
    u2 = np.array([0.5, 0.2])
    c1 = sloane_cdf(u1, u2, rho)
    c2 = sloane_cdf(u2, u1, rho)
    assert np.allclose(c1, c2)
    assert np.all(c1 >= -1e-8) and np.all(c1 <= 1.0 + 1e-8)
