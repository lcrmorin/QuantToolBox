"""Tests for quanttoolbox.stats.multivariate."""

import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

from quanttoolbox.stats.distributions import mvn_cdf, mvn_pdf
from quanttoolbox.stats.multivariate import bvn_cdf, bvn_pdf, bvt_cdf


def test_bvn_cdf_matches_scipy_multivariate_normal_directly():
    rho = 0.4
    cov = np.array([[1.0, rho], [rho, 1.0]])
    x, y = 0.7, -0.3

    expected = multivariate_normal.cdf([x, y], mean=[0.0, 0.0], cov=cov)
    assert np.isclose(bvn_cdf(x, y, rho), expected)


def test_bvn_cdf_matches_the_general_n_dimensional_mvn_cdf():
    # bvn_cdf is a 2D-specialized call signature for exactly the same
    # computation already ported (for arbitrary dimension) as
    # stats.distributions.mvn_cdf -- they must agree.
    rho = -0.6
    x, y = 1.2, 0.4
    sigma = np.array([[1.0, rho], [rho, 1.0]])

    general = mvn_cdf(np.array([x, y]), mu=np.array([0.0, 0.0]), sigma=sigma)
    specialized = bvn_cdf(x, y, rho)
    assert np.isclose(general, specialized)


def test_bvn_cdf_broadcasts_elementwise_over_arrays():
    x = np.array([0.5, -1.0, 0.0])
    y = np.array([0.2, 0.3, -0.5])
    rho = np.array([0.1, 0.5, -0.3])

    result = bvn_cdf(x, y, rho)
    assert result.shape == (3,)

    for i in range(3):
        expected = multivariate_normal.cdf(
            [x[i], y[i]], mean=[0.0, 0.0], cov=[[1.0, rho[i]], [rho[i], 1.0]]
        )
        assert np.isclose(result[i], expected)


def test_bvn_cdf_scalar_output_is_a_plain_float_like_value():
    result = bvn_cdf(0.5, 0.5, 0.2)
    assert np.isscalar(result) or result.shape == ()


def test_bvn_pdf_matches_scipy_multivariate_normal_pdf():
    mu1, mu2 = 1.0, -0.5
    sigma1, sigma2 = 1.5, 0.8
    rho = 0.3
    x1, x2 = 1.2, -0.1

    cov = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ]
    )
    expected = multivariate_normal.pdf([x1, x2], mean=[mu1, mu2], cov=cov)
    assert np.isclose(bvn_pdf(x1, x2, mu1, mu2, sigma1, sigma2, rho), expected)


def test_bvn_pdf_matches_the_general_n_dimensional_mvn_pdf_at_standard_params():
    rho = 0.25
    x1, x2 = 0.4, -0.7
    sigma = np.array([[1.0, rho], [rho, 1.0]])

    general = mvn_pdf(np.array([x1, x2]), mu=np.array([0.0, 0.0]), sigma=sigma)
    specialized = bvn_pdf(x1, x2, 0.0, 0.0, 1.0, 1.0, rho)
    assert np.isclose(general, specialized)


def test_bvn_pdf_integrates_to_bvn_cdf_shape_sanity():
    # Not a full integration check -- just confirms the PDF is positive and
    # symmetric under (x1, x2, rho) -> (-x1, -x2, rho) for the zero-mean,
    # unit-variance case (a property of the bivariate normal density).
    rho = 0.4
    p1 = bvn_pdf(0.6, -0.3, 0.0, 0.0, 1.0, 1.0, rho)
    p2 = bvn_pdf(-0.6, 0.3, 0.0, 0.0, 1.0, 1.0, rho)
    assert p1 > 0
    assert np.isclose(p1, p2)


def test_bvt_cdf_matches_scipy_multivariate_t_directly():
    # scipy's multivariate_t.cdf is itself a randomized quasi-Monte-Carlo
    # estimator (same Genz algorithm family -- see module docstring), so
    # repeated calls with identical arguments differ by ~1e-4/1e-5; compare
    # with a tolerance that reflects that simulation noise rather than exact
    # equality.
    rho, nu = 0.3, 6.0
    x, y = 0.8, -0.4
    shape = np.array([[1.0, rho], [rho, 1.0]])

    expected = multivariate_t.cdf([x, y], loc=[0.0, 0.0], shape=shape, df=nu)
    assert np.isclose(bvt_cdf(x, y, rho, nu), expected, atol=5e-3)


def test_bvt_cdf_broadcasts_elementwise_over_arrays():
    x = np.array([0.5, -1.0])
    y = np.array([0.2, 0.3])
    rho = np.array([0.1, 0.5])
    nu = np.array([4.0, 10.0])

    result = bvt_cdf(x, y, rho, nu)
    assert result.shape == (2,)
    for i in range(2):
        expected = multivariate_t.cdf(
            [x[i], y[i]], loc=[0.0, 0.0], shape=[[1.0, rho[i]], [rho[i], 1.0]], df=nu[i]
        )
        assert np.isclose(result[i], expected, atol=5e-3)


def test_bvt_cdf_approaches_bvn_cdf_for_large_degrees_of_freedom():
    # Student-t with large nu converges to the normal distribution.
    rho = 0.35
    x, y = 0.5, -0.2
    assert np.isclose(bvt_cdf(x, y, rho, 500.0), bvn_cdf(x, y, rho), atol=1e-3)
