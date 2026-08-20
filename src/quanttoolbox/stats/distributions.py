"""Probability distribution functions: CDF/PDF/quantile wrappers and the
GQF (generalized quadratic form) distribution family.

Ported from QuantToolbox/stats/{cdfn,cdfni,cdft,cdfti,cdftc,cdfchi2,
cdfchi2c,cdff,cdffc,cdfmvn,pdfmvn,pdfn,rndmvn,gqf1_*,gqf2_*}.m

Translation notes:

- The simple distribution wrappers (normal/Student-t/chi-square/F/MVN) are
  thin passthroughs to ``scipy.stats`` -- MATLAB's Statistics Toolbox
  functions (``tcdf``, ``chi2cdf``, ``mvncdf``, ...) map directly onto
  ``scipy.stats.{t,chi2,f,multivariate_normal}``, so there's no need to
  hand-roll any of this.
- GQF #1 is the distribution of ``sum(a_i * (Z_i + b_i)^2)`` for
  independent standard normals Z_i (a "weighted noncentral chi-square
  mixture"); GQF #2 is the distribution of the general quadratic form
  ``(X - mu)' Q (X - mu)`` (or similar) for ``X ~ N(mu, Sigma)``. Both use
  cumulant-based approximations (a Laguerre-type series for GQF1, a
  three-cumulant noncentral-chi-square matching for GQF2 -- see Solomon &
  Stephens (1978) / Buckley & Eagleson (1988) for the underlying theory).
  These have no scipy equivalent and are ported algorithm-for-algorithm.
- MATLAB's ``missex(cdf, cond)`` (replace-with-missing where cond is true)
  is inlined as ``np.where(cond, np.nan, cdf)``.
- MATLAB's ``eig`` is replaced with ``numpy.linalg.eigh`` in
  ``gqf2_to_gqf1`` since the matrix involved is symmetric by construction;
  the eigenvalue *order* may differ from MATLAB's, but the resulting
  (a, b) pairs describe the same distribution regardless of order.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy import linalg
from scipy.stats import chi2, f, multivariate_normal, ncx2, norm, t

# ---------------------------------------------------------------------------
# Simple distributions (thin scipy.stats wrappers)
# ---------------------------------------------------------------------------


def normal_cdf(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    """Original: stats/cdfn.m"""
    return norm.cdf(x, loc=mu, scale=sigma)


def normal_ppf(p: float | np.ndarray) -> float | np.ndarray:
    """Quantile function of N(0,1). Original: stats/cdfni.m"""
    return norm.ppf(p)


def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Original: stats/pdfn.m"""
    return norm.pdf(x, loc=mu, scale=sigma)


def student_t_cdf(x: float | np.ndarray, nu: float) -> float | np.ndarray:
    """Original: stats/cdft.m"""
    return t.cdf(x, df=nu)


def student_t_ppf(p: float | np.ndarray, nu: float) -> float | np.ndarray:
    """Original: stats/cdfti.m"""
    return t.ppf(p, df=nu)


def student_t_sf(x: np.ndarray, nu: float) -> np.ndarray:
    """Upper-tail (survival) Student's t. Original: stats/cdftc.m"""
    return t.sf(x, df=nu)


def chi2_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    """Original: stats/cdfchi2.m"""
    return chi2.cdf(x, df=nu)


def chi2_sf(x: np.ndarray, nu: float) -> np.ndarray:
    """Upper-tail (survival) chi-square. Original: stats/cdfchi2c.m"""
    return chi2.sf(x, df=nu)


def f_cdf(x: np.ndarray, nu1: float, nu2: float) -> np.ndarray:
    """Original: stats/cdff.m"""
    return f.cdf(x, dfn=nu1, dfd=nu2)


def f_sf(x: np.ndarray, nu1: float, nu2: float) -> np.ndarray:
    """Upper-tail (survival) F distribution. Original: stats/cdffc.m"""
    return f.sf(x, dfn=nu1, dfd=nu2)


def mvn_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CDF of the multivariate normal N(mu, Sigma). Original: stats/cdfmvn.m"""
    return multivariate_normal.cdf(x, mean=np.asarray(mu).flatten(), cov=sigma)


def mvn_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """PDF of the multivariate normal N(mu, Sigma). Original: stats/pdfmvn.m"""
    return multivariate_normal.pdf(x, mean=np.asarray(mu).flatten(), cov=sigma)


def mvn_rvs(mu: np.ndarray, sigma: np.ndarray, n_samples: int, random_state=None) -> np.ndarray:
    """Draw samples from the multivariate normal N(mu, Sigma).

    Original: stats/rndmvn.m
    """
    return multivariate_normal.rvs(
        mean=np.asarray(mu).flatten(), cov=sigma, size=n_samples, random_state=random_state
    )


# ---------------------------------------------------------------------------
# GQF #1: distribution of sum(a_i * (Z_i + b_i)^2), Z_i ~ iid N(0,1)
# ---------------------------------------------------------------------------


def gqf1_moments(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, std dev, skewness, and excess kurtosis of the GQF #1 distribution.

    Original: stats/gqf1_moments.m
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    b2, a2, a3, a4 = b**2, a**2, a**3, a**4

    mean = np.sum(a * (1 + b2))
    denom = np.sum(a2 * (1 + 2 * b2))
    sigma = np.sqrt(2 * denom)
    gamma1 = 2 * np.sqrt(2) * np.sum(a3 * (1 + 3 * b2)) / denom**1.5
    gamma2 = 12 * np.sum(a4 * (1 + 4 * b2)) / denom**2
    return mean, sigma, gamma1, gamma2


def gqf1_coeffs(a: np.ndarray, b: np.ndarray, beta: float, order: int) -> np.ndarray:
    """Laguerre-series mixing coefficients for the GQF #1 CDF/PDF.

    Original: stats/gqf1_coeffs.m
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(a.shape[0], b.shape[0])
    a = np.broadcast_to(a, (n,))
    b = np.broadcast_to(b, (n,))

    beta_a = beta / a
    beta_a_c = 1 - beta_a
    b2_a = b**2 / a
    zeta = np.sum(b**2)

    g = np.zeros(order)
    for m in range(1, order + 1):
        g[m - 1] = np.sum(beta_a_c**m) + m * beta * np.sum(b2_a * beta_a_c ** (m - 1))

    coeffs = np.zeros(order + 1)
    coeffs[0] = np.exp(-zeta / 2) * np.prod(np.sqrt(beta_a))
    for j in range(1, order + 1):
        coeffs[j] = np.sum(g[0:j][::-1] * coeffs[0:j]) / (2 * j)

    return coeffs


def gqf1_cdf(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, beta: float | None = None, order: int = 200
) -> np.ndarray:
    """CDF of the GQF #1 distribution. Original: stats/gqf1_cdf.m"""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    if beta is None or beta == 0:
        beta = 0.8 * a.min()

    coeffs = gqf1_coeffs(a, b, beta, order)
    x_star = np.asarray(x, dtype=float) / beta

    cdf = np.zeros_like(x_star, dtype=float)
    for j in range(order + 1):
        cdf = cdf + coeffs[j] * chi2.cdf(x_star, df=n + 2 * j)

    return np.where((cdf < -1e-4) | (cdf > 1.0001), np.nan, cdf)


def gqf1_pdf(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, beta: float | None = None, order: int = 100
) -> np.ndarray:
    """PDF of the GQF #1 distribution. Original: stats/gqf1_pdf.m"""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    if beta is None or beta == 0:
        beta = 0.8 * a.min()

    coeffs = gqf1_coeffs(a, b, beta, order)
    x_star = np.asarray(x, dtype=float) / beta

    pdf = np.zeros_like(x_star, dtype=float)
    for j in range(order + 1):
        pdf = pdf + coeffs[j] * chi2.pdf(x_star, df=n + 2 * j) / beta

    return pdf


def gqf1_to_gqf2(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert GQF #1 parameters (a, b) to GQF #2 parameters (mu, Sigma, Q).

    Original: stats/gqf1_to_gqf2.m
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(a.shape[0], b.shape[0])
    a = np.broadcast_to(a, (n,))
    b = np.broadcast_to(b, (n,))

    mu = b
    sigma = np.eye(n)
    q = np.diag(a)
    return mu, sigma, q


# ---------------------------------------------------------------------------
# GQF #2: distribution of the quadratic form X'QX for X ~ N(mu, Sigma)
# ---------------------------------------------------------------------------


def gqf2_moments(
    mu: np.ndarray, sigma: np.ndarray, q: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Mean, std dev, skewness, excess kurtosis, and the two Pearson-type
    shape statistics (s1, s2) used by gqf2_cdf/gqf2_pdf.

    Original: stats/gqf2_moments.m
    """
    mu = np.asarray(mu, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)
    n = sigma.shape[0]
    n_kappa = 4

    kappa = np.zeros(n_kappa)
    p = q @ sigma
    p_old = np.eye(n)
    for k in range(1, n_kappa + 1):
        p_new = p_old @ p
        kappa[k - 1] = (
            (2 ** (k - 1)) * factorial(k - 1) * (np.trace(p_new) + k * mu.T @ p_old @ q @ mu)
        )
        p_old = p_new

    mean = kappa[0]
    sigma_out = np.sqrt(kappa[1])
    gamma1 = kappa[2] / sigma_out**3
    gamma2 = kappa[3] / sigma_out**4

    s1 = gamma1 / np.sqrt(8)
    s2 = gamma2 / 12
    return mean, sigma_out, gamma1, gamma2, s1, s2


def _gqf2_noncentral_chi2_params(s1: float, s2: float) -> tuple[float, float]:
    """Shared nu/zeta (df, noncentrality) matching logic for gqf2_cdf/pdf."""
    if s1**2 > s2:
        omega = 1 / (s1 - np.sqrt(s1**2 - s2))
        zeta = s1 * omega**3 - omega**2
        nu = omega**2 - 2 * zeta
    else:
        zeta = 0.0
        nu = 1 / s1**2
    return nu, zeta


def gqf2_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    """CDF of the GQF #2 distribution, via noncentral-chi-square matching.

    Original: stats/gqf2_cdf.m
    """
    m, sig, _, _, s1, s2 = gqf2_moments(mu, sigma, q)
    nu, zeta = _gqf2_noncentral_chi2_params(s1, s2)

    mu_star = nu + zeta
    sigma_star = np.sqrt(2 * nu + 4 * zeta)

    x_star = (np.asarray(x, dtype=float) - m) / sig
    x_star = mu_star + sigma_star * x_star
    return ncx2.cdf(x_star, df=nu, nc=zeta)


def gqf2_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    """PDF of the GQF #2 distribution, via noncentral-chi-square matching.

    Original: stats/gqf2_pdf.m
    """
    m, sig, _, _, s1, s2 = gqf2_moments(mu, sigma, q)
    nu, zeta = _gqf2_noncentral_chi2_params(s1, s2)

    mu_star = nu + zeta
    sigma_star = np.sqrt(2 * nu + 4 * zeta)

    x_star = (np.asarray(x, dtype=float) - m) / sig
    x_star = mu_star + sigma_star * x_star
    return ncx2.pdf(x_star, df=nu, nc=zeta) * (sigma_star / sig)


def gqf2_to_gqf1(mu: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert GQF #2 parameters (mu, Sigma, Q) to GQF #1 parameters (a, b).

    Original: stats/gqf2_to_gqf1.m

    Note: eigenvalue ordering may differ from MATLAB's ``eig`` (which does
    not sort), but the resulting (a, b) pairs describe the same
    distribution regardless of order.
    """
    mu = np.asarray(mu, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    sigma_sqrt = linalg.sqrtm(sigma).real
    b_mat = sigma_sqrt @ q @ sigma_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(b_mat)

    m_vec = eigenvectors.T @ np.linalg.inv(sigma_sqrt) @ mu

    a = eigenvalues
    b = m_vec
    return a, b
