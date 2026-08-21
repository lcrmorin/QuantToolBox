"""Probability distribution functions: CDF/PDF/quantile wrappers and the
GQF (generalized quadratic form) distribution family.

Ported from QuantToolbox/stats/{cdfn,cdfni,cdft,cdfti,cdftc,cdfchi2,
cdfchi2c,cdff,cdffc,cdfmvn,pdfmvn,pdfn,rndmvn,gqf1_*,gqf2_*}.m, extended
with HSF toolbox `stats/{cdfSN,cdfSNi,pdfSN,momSN,rndSN,cdfST,cdfSTi,pdfST,
momST,rndST,cdfBates,pdfBates,cdfbeta,pdfbeta,cdfig,pdfig,cdfln,pdfln,
cdfNormalRatio,pdfNormalRatio,pdfPoissonBinomial,cdfchi2i,pdft,
compute_cdf_order_statistics,compute_inv_cdf_order_statistics,
constant_correlation_matrix}.m` (HSF toolbox port -- see
docs/migration_map.md).

Translation notes:

- The simple distribution wrappers (normal/Student-t/chi-square/F/MVN) are
  thin passthroughs to ``scipy.stats`` -- MATLAB's Statistics Toolbox
  functions (``tcdf``, ``chi2cdf``, ``mvncdf``, ...) map directly onto
  ``scipy.stats.{t,chi2,f,multivariate_normal}``, so there's no need to
  hand-roll any of this. ``student_t_pdf``/``chi2_ppf`` (from the HSF
  toolbox's ``pdft.m``/``cdfchi2i.m``) fill out this same family.
- GQF #1 is the distribution of ``sum(a_i * (Z_i + b_i)^2)`` for
  independent standard normals Z_i (a "weighted noncentral chi-square
  mixture"); GQF #2 is the distribution of the general quadratic form
  ``(X - mu)' Q (X - mu)`` (or similar) for ``X ~ N(mu, Sigma)``. Both use
  cumulant-based approximations (a Laguerre-type series for GQF1, a
  three-cumulant noncentral-chi-square matching for GQF2 -- see Solomon &
  Stephens (1978) / Buckley & Eagleson (1988) for the underlying theory).
  These have no scipy equivalent and are ported algorithm-for-algorithm.
- Beta/lognormal/inverse-Gaussian (``cdfbeta``/``pdfbeta``, ``cdfln``/
  ``pdfln``, ``cdfig``/``pdfig``): thin wrappers with the original's call
  signature, same "call-site compatibility" reasoning as the simple
  wrappers above -- ``beta_cdf``/``beta_pdf`` go straight to
  ``scipy.stats.beta``; ``lognormal_cdf``/``lognormal_pdf`` and
  ``inverse_gaussian_cdf``/``inverse_gaussian_pdf`` are evaluated directly
  from the closed-form formulas the originals use (verified algebraically
  equivalent to ``scipy.stats.lognorm``/``invgauss`` under those
  distributions' own shape/scale reparameterization, but ported directly
  rather than requiring callers to compute that reparameterization
  themselves).
- Bates distribution (``bates_cdf``/``bates_pdf``): no scipy equivalent
  (scipy has no named Bates distribution) -- ported algorithm-for-algorithm
  (``scipy.special.comb`` replaces MATLAB's ``nchoosek``).
- Poisson-binomial PMF (``poisson_binomial_pmf``): matches
  ``scipy.stats.poisson_binom`` exactly (verified numerically against both
  the original's FFT and direct-recursion branches) -- switched to the
  scipy-backed distribution rather than hand-rolling either branch.
- Order statistics (``order_statistic_cdf``/``order_statistic_ppf``): the
  order-statistic CDF formula ``F_{i:n}(x) = P(Binom(n, F_x) >= i)`` is
  evaluated via ``scipy.stats.binom.sf`` rather than hand-computing
  binomial coefficients in a loop -- same formula, more numerically stable
  for large n. The quantile version is a grid search (given sample points
  `x` and their CDF values `F_x`, find the first point where the
  order-statistic CDF crosses `alpha`) with no scipy equivalent -- ported
  directly.
- Normal-ratio distribution (``normal_ratio_cdf``/``normal_ratio_pdf``,
  Hinkley's distribution for the ratio of two independent normals): no
  scipy equivalent -- ported directly, using this module's own
  ``quanttoolbox.stats.multivariate.bvn_cdf`` in place of the original's
  ``cdfbvn`` calls, matching the original's ``(p, a_z, b_z, c, rho_z)``
  multi-output signature.
- Skew-normal (``skew_normal_{cdf,ppf,pdf,moments,rvs}``): switched
  entirely to ``scipy.stats.skewnorm`` -- verified numerically that
  ``pdfSN.m``'s formula, ``momSN.m``'s moment formulas, and
  ``scipy.stats.skewnorm``'s pdf/cdf/``stats(moments="mvsk")`` all agree
  exactly under the direct parameter mapping (``eta`` -> ``a``, ``xi`` ->
  ``loc``, ``omega`` -> ``scale``). The original's two alternate numerical
  CDF branches (``mtd`` 0 vs 1, both bivariate-normal-orthant identities
  for the same value) and its hand-rolled Newton-iteration quantile
  function (``cdfSNi.m``) are therefore unnecessary -- ``scipy`` computes
  the exact analytic answer directly via ``.cdf``/``.ppf``.
- Skew-t (``skew_t_{cdf,ppf,pdf,moments,rvs}``): **not** the same family as
  ``scipy.stats.jf_skew_t`` (Jones & Faddy's skew-t, a different two-shape-
  parameter family) -- this is Azzalini's skew-t (one skewness parameter
  `eta` plus degrees of freedom `nu`), which has no scipy equivalent.
  Ported algorithm-for-algorithm, including the original's Newton-iteration
  quantile function (``cdfSTi.m``, via ``quanttoolbox.config.NewtonConfig``)
  and its bivariate-Student-t-based CDF (via
  ``quanttoolbox.stats.multivariate.bvt_cdf``).
- ``max_size.m`` (a MATLAB shape-broadcasting helper) is not ported as a
  function -- superseded entirely by numpy's native broadcasting
  (``np.broadcast_arrays``), used throughout this module and
  ``stats.multivariate`` instead.
- MATLAB's ``missex(cdf, cond)`` (replace-with-missing where cond is true)
  is inlined as ``np.where(cond, np.nan, cdf)``.
- MATLAB's ``eig`` is replaced with ``numpy.linalg.eigh`` in
  ``gqf2_to_gqf1`` since the matrix involved is symmetric by construction;
  the eigenvalue *order* may differ from MATLAB's, but the resulting
  (a, b) pairs describe the same distribution regardless of order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy import linalg
from scipy.special import comb, gammaln
from scipy.stats import beta as beta_dist
from scipy.stats import binom, chi2, f, multivariate_normal, ncx2, norm, poisson_binom, skewnorm, t

from quanttoolbox.config import NewtonConfig
from quanttoolbox.stats.multivariate import bvn_cdf, bvt_cdf

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


def student_t_pdf(x: np.ndarray, nu: float) -> np.ndarray:
    """Original: stats/pdft.m (HSF toolbox)"""
    return t.pdf(x, df=nu)


def chi2_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    """Original: stats/cdfchi2.m"""
    return chi2.cdf(x, df=nu)


def chi2_ppf(p: np.ndarray, nu: float) -> np.ndarray:
    """Original: stats/cdfchi2i.m (HSF toolbox)"""
    return chi2.ppf(p, df=nu)


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


# ---------------------------------------------------------------------------
# Beta / lognormal / inverse-Gaussian / Bates (HSF toolbox stats/)
# ---------------------------------------------------------------------------


def beta_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Original: stats/cdfbeta.m (HSF toolbox)"""
    return beta_dist.cdf(x, a, b)


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Original: stats/pdfbeta.m (HSF toolbox)"""
    return beta_dist.pdf(x, a, b)


def lognormal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """CDF of a lognormal variable, i.e. log(X) ~ N(mu, sigma^2).

    Original: stats/cdfln.m (HSF toolbox)
    """
    x = np.asarray(x, dtype=float)
    return np.asarray(normal_cdf((np.log(x) - mu) / sigma))


def lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """PDF of a lognormal variable, i.e. log(X) ~ N(mu, sigma^2).

    Original: stats/pdfln.m (HSF toolbox)
    """
    x = np.asarray(x, dtype=float)
    y = (np.log(x) - mu) / sigma
    return 1.0 / (x * np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * y**2)


def inverse_gaussian_cdf(x: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """CDF of the inverse Gaussian (Wald) distribution, mean `mu`, shape
    `lam`.

    Original: stats/cdfig.m (HSF toolbox; algebraically equivalent to
    ``scipy.stats.invgauss.cdf(x, mu / lam, scale=lam)``, ported directly
    to keep the original's ``(mu, lam)`` call signature)
    """
    x = np.asarray(x, dtype=float)
    alpha = (x - mu) / mu * np.sqrt(lam / x)
    alpha_bar = -(x + mu) / mu * np.sqrt(lam / x)
    return normal_cdf(alpha) + np.exp(2 * lam / mu) * normal_cdf(alpha_bar)


def inverse_gaussian_pdf(x: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """PDF of the inverse Gaussian (Wald) distribution, mean `mu`, shape
    `lam`.

    Original: stats/pdfig.m (HSF toolbox)
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(lam / (2 * np.pi * x**3)) * np.exp(-0.5 * lam / mu**2 * (x - mu) ** 2 / x)


def bates_cdf(x: np.ndarray, n: int) -> np.ndarray:
    """CDF of the Bates distribution: the mean of `n` iid Uniform(0, 1)
    variables. No scipy equivalent.

    Original: stats/cdfBates.m (HSF toolbox)
    """
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x)
    for k in range(n + 1):
        s = (n * x > k).astype(float)
        cdf = cdf + ((-1.0) ** k) * comb(n, k) * (n * x - k) ** n * s
    cdf = cdf / factorial(n)
    cdf = np.where(x == 0, 0.0, cdf)
    cdf = np.where(x == 1, 1.0, cdf)
    return cdf


def bates_pdf(x: np.ndarray, n: int) -> np.ndarray:
    """PDF of the Bates distribution: the mean of `n` iid Uniform(0, 1)
    variables. No scipy equivalent.

    Original: stats/pdfBates.m (HSF toolbox)
    """
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    for k in range(n + 1):
        pdf = pdf + ((-1.0) ** k) * comb(n, k) * (n * x - k) ** (n - 1) * np.sign(n * x - k)
    pdf = 0.5 * pdf * n / factorial(n - 1)
    if n > 1:
        pdf = np.where(x == 0, 0.0, pdf)
        pdf = np.where(x == 1, 0.0, pdf)
    else:
        pdf = np.where(x == 0, 1.0, pdf)
        pdf = np.where(x == 1, 1.0, pdf)
    return pdf


# ---------------------------------------------------------------------------
# Poisson-binomial distribution (sum of independent, non-identical Bernoullis)
# ---------------------------------------------------------------------------


def poisson_binomial_pmf(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PMF of the sum of `n` independent Bernoulli(p_i) variables, for
    support k = 0..n. Matches ``scipy.stats.poisson_binom`` exactly
    (verified numerically against the original's own FFT and
    direct-recursion branches) -- neither branch is hand-rolled here.

    Original: stats/pdfPoissonBinomial.m (HSF toolbox)
    """
    p = np.asarray(p, dtype=float).flatten()
    n = p.shape[0]
    k = np.arange(n + 1)
    return k, poisson_binom.pmf(k, p)


# ---------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------


def order_statistic_cdf(f_x: np.ndarray, n: int, i_select: np.ndarray | None = None) -> np.ndarray:
    """CDF of the i-th order statistic (i = 1..n) of n iid draws from a
    distribution with CDF value(s) `f_x`: ``F_{i:n} = P(Binom(n, F_x) >=
    i)``, evaluated via ``scipy.stats.binom.sf`` rather than hand-computing
    binomial coefficients (same formula, more numerically stable for large
    n).

    `f_x` may be an array (one row per evaluation point); returns an array
    of shape (len(f_x), len(i_select)) with one column per selected order
    statistic i (default: all of 1..n).

    Original: stats/compute_cdf_order_statistics.m (HSF toolbox)
    """
    f_x = np.atleast_1d(np.asarray(f_x, dtype=float))
    if i_select is None:
        i_select = np.arange(1, n + 1)
    i_select = np.asarray(i_select, dtype=int)

    # F_{i:n}(x) = P(Binom(n, F_x) >= i) = sf(i - 1, n, F_x)
    out = np.stack([binom.sf(i - 1, n, f_x) for i in i_select], axis=-1)
    return out


def order_statistic_ppf(
    alpha: np.ndarray,
    x: np.ndarray,
    f_x: np.ndarray,
    n: int,
    i_select: np.ndarray | None = None,
) -> np.ndarray:
    """Quantile of the i-th order statistic, found by grid search: given
    sample points `x` with CDF values `f_x`, returns (for each `alpha` and
    each selected order statistic i) the first `x` value where
    `order_statistic_cdf` crosses `alpha`. No scipy equivalent -- ported
    directly.

    Original: stats/compute_inv_cdf_order_statistics.m (HSF toolbox)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    x = np.asarray(x, dtype=float)
    if i_select is None:
        i_select = np.arange(1, n + 1)
    i_select = np.asarray(i_select, dtype=int)

    f_i_n = order_statistic_cdf(f_x, n, i_select)  # shape (len(x), len(i_select))

    q = np.full((alpha.shape[0], i_select.shape[0]), np.nan)
    for a_idx, a in enumerate(alpha):
        for k in range(i_select.shape[0]):
            candidates = np.flatnonzero(f_i_n[:, k] >= a)
            if candidates.size > 0:
                q[a_idx, k] = x[candidates[0]]
    return q


def constant_correlation_matrix(n: int, rho: float) -> np.ndarray:
    """An n x n correlation matrix with constant off-diagonal correlation
    `rho` and unit diagonal.

    Original: stats/constant_correlation_matrix.m (HSF toolbox)
    """
    c = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


# ---------------------------------------------------------------------------
# Normal-ratio distribution (Hinkley): Z = X / Y for independent normals
# ---------------------------------------------------------------------------


@dataclass
class NormalRatioResult:
    """CDF or PDF of Z = X / Y (X, Y independent normals), plus the
    intermediate quantities (``a_z``, ``b_z``, ``c``, ``rho_z``) the
    original returns alongside the probability."""

    p: np.ndarray
    a_z: np.ndarray
    b_z: np.ndarray
    c: np.ndarray
    rho_z: np.ndarray


def _normal_ratio_terms(
    z: np.ndarray, mu_x: float, sigma_x: float, mu_y: float, sigma_y: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    z = np.asarray(z, dtype=float)
    mu2_x, mu2_y = mu_x**2, mu_y**2
    sigma2_x, sigma2_y = sigma_x**2, sigma_y**2
    z2 = z**2

    a_z = np.sqrt(z2 / sigma2_x + 1.0 / sigma2_y)
    b_z = (mu_x / sigma2_x) * z + (mu_y / sigma2_y)
    c = mu2_x / sigma2_x + mu2_y / sigma2_y
    rho_z = z / (sigma_x * a_z)
    return z, a_z, b_z, c, rho_z


def normal_ratio_cdf(
    z: np.ndarray, mu_x: float, sigma_x: float, mu_y: float, sigma_y: float
) -> NormalRatioResult:
    """CDF of Z = X / Y, for independent X ~ N(mu_x, sigma_x^2) and
    Y ~ N(mu_y, sigma_y^2) (Hinkley's ratio distribution). No scipy
    equivalent.

    Original: stats/cdfNormalRatio.m (HSF toolbox)
    """
    z, a_z, b_z, c, rho_z = _normal_ratio_terms(z, mu_x, sigma_x, mu_y, sigma_y)

    x1 = (mu_x - mu_y * z) / (sigma_x * sigma_y * a_z)
    y1 = -mu_y / sigma_y
    x2, y2 = -x1, -y1

    p = bvn_cdf(x1, y1, rho_z) + bvn_cdf(x2, y2, rho_z)
    return NormalRatioResult(p=p, a_z=a_z, b_z=b_z, c=np.full_like(a_z, c), rho_z=rho_z)


def normal_ratio_pdf(
    z: np.ndarray, mu_x: float, sigma_x: float, mu_y: float, sigma_y: float
) -> NormalRatioResult:
    """PDF of Z = X / Y, for independent X ~ N(mu_x, sigma_x^2) and
    Y ~ N(mu_y, sigma_y^2) (Hinkley's ratio distribution). No scipy
    equivalent.

    Original: stats/pdfNormalRatio.m (HSF toolbox)
    """
    z, a_z, b_z, c, rho_z = _normal_ratio_terms(z, mu_x, sigma_x, mu_y, sigma_y)
    a2_z, a3_z, b2_z = a_z**2, a_z**3, b_z**2

    p1 = b_z / (sigma_x * sigma_y * np.sqrt(2 * np.pi) * a3_z)
    p2 = normal_cdf(b_z / a_z) - normal_cdf(-b_z / a_z)
    p3 = np.exp((b2_z - c * a2_z) / (2 * a2_z))
    p4 = np.exp(-c / 2) / (sigma_x * sigma_y * a2_z * np.pi)
    p = p1 * p2 * p3 + p4
    return NormalRatioResult(p=p, a_z=a_z, b_z=b_z, c=np.full_like(a_z, c), rho_z=rho_z)


# ---------------------------------------------------------------------------
# Skew-normal (fully switched to scipy.stats.skewnorm -- see module
# docstring for the numerical verification)
# ---------------------------------------------------------------------------


def skew_normal_cdf(x: np.ndarray, xi: float, omega: float, eta: float) -> np.ndarray:
    """CDF of Azzalini's skew-normal distribution (location `xi`, scale
    `omega`, shape `eta`).

    Original: stats/cdfSN.m (HSF toolbox; both of the original's numerical
    branches are exact identities for this same value -- ``scipy`` computes
    it directly, see module docstring)
    """
    return skewnorm.cdf(x, eta, loc=xi, scale=omega)


def skew_normal_ppf(p: np.ndarray, xi: float, omega: float, eta: float) -> np.ndarray:
    """Quantile function of Azzalini's skew-normal distribution.

    Original: stats/cdfSNi.m (HSF toolbox; the original's Newton iteration
    is unnecessary -- ``scipy.stats.skewnorm.ppf`` is exact, see module
    docstring)
    """
    return skewnorm.ppf(p, eta, loc=xi, scale=omega)


def skew_normal_pdf(x: np.ndarray, xi: float, omega: float, eta: float) -> np.ndarray:
    """PDF of Azzalini's skew-normal distribution.

    Original: stats/pdfSN.m (HSF toolbox)
    """
    return skewnorm.pdf(x, eta, loc=xi, scale=omega)


def skew_normal_moments(xi: float, omega: float, eta: float) -> tuple[float, float, float, float]:
    """Mean, std dev, skewness, and excess kurtosis of Azzalini's
    skew-normal distribution.

    Original: stats/momSN.m (HSF toolbox; matches
    ``scipy.stats.skewnorm.stats(eta, loc=xi, scale=omega,
    moments="mvsk")`` exactly, see module docstring)
    """
    mean, var, skew, kurt = skewnorm.stats(eta, loc=xi, scale=omega, moments="mvsk")
    return float(mean), float(np.sqrt(var)), float(skew), float(kurt)


def skew_normal_rvs(
    xi: float, omega: float, eta: float, size: int = 1, random_state=None
) -> np.ndarray:
    """Draw samples from Azzalini's skew-normal distribution.

    Original: stats/rndSN.m (HSF toolbox)
    """
    return skewnorm.rvs(eta, loc=xi, scale=omega, size=size, random_state=random_state)


# ---------------------------------------------------------------------------
# Skew-t (Azzalini): no scipy equivalent -- ``scipy.stats.jf_skew_t`` is a
# different family (Jones & Faddy), see module docstring
# ---------------------------------------------------------------------------


def skew_t_cdf(
    x: np.ndarray, xi: float, omega: float, eta: float, nu: float, method: int = 0
) -> np.ndarray:
    """CDF of Azzalini's skew-t distribution (location `xi`, scale `omega`,
    shape `eta`, degrees of freedom `nu`).

    Original: stats/cdfST.m (HSF toolbox)
    """
    xc = (np.asarray(x, dtype=float) - xi) / omega
    if method == 1:
        delta = eta / np.sqrt(1 + eta**2)
        return 2.0 * bvt_cdf(xc, 0.0, -delta, nu)

    e = float(eta >= 0)
    eta_abs = max(abs(eta), 1e-8)
    delta = (1 - eta_abs**2) / (1 + eta_abs**2)
    cdf = bvt_cdf(xc, xc, delta, nu)
    return cdf * e + (2 * student_t_cdf(xc, nu) - cdf) * (1 - e)


def skew_t_pdf(x: np.ndarray, xi: float, omega: float, eta: float, nu: float) -> np.ndarray:
    """PDF of Azzalini's skew-t distribution.

    Original: stats/pdfST.m (HSF toolbox)
    """
    xc = (np.asarray(x, dtype=float) - xi) / omega
    cdf = student_t_cdf(eta * xc * np.sqrt((nu + 1) / (xc**2 + nu)), nu + 1)
    pdf = student_t_pdf(xc, nu) / omega
    return 2 * pdf * cdf


def skew_t_ppf(
    p: np.ndarray,
    xi: float,
    omega: float,
    eta: float,
    nu: float,
    config: NewtonConfig | None = None,
) -> np.ndarray:
    """Quantile function of Azzalini's skew-t distribution, via Newton
    iteration (no closed form / no scipy equivalent).

    The default tolerance is looser than `NewtonConfig`'s own default
    (`1e-4` rather than `1e-10`): `skew_t_cdf` is itself backed by
    `bvt_cdf`, whose underlying `scipy.stats.multivariate_t.cdf` uses a
    randomized quasi-Monte-Carlo integrator with irreducible call-to-call
    noise on the order of `1e-4` (verified empirically -- repeated calls
    with identical inputs vary by ~7e-5 to ~1e-4). A tighter tolerance
    (e.g. the `1e-8` this function used before) can never actually be
    satisfied, so the loop always burns its full `max_iters` budget
    without improving on the noise floor -- roughly a 4x slowdown for no
    accuracy gain, confirmed by direct timing. `1e-4` lets the loop exit
    as soon as it reaches that floor, matching the precision the
    underlying CDF can actually deliver.

    Original: stats/cdfSTi.m (HSF toolbox)
    """
    if config is None:
        config = NewtonConfig(tol=1e-4, max_iters=50)

    p = np.atleast_1d(np.asarray(p, dtype=float))
    x = student_t_ppf(p, nu)
    q = skew_t_cdf(x, 0.0, 1.0, eta, nu)
    e_max = q > 0.90
    e_min = q < 0.10
    if eta >= 0.0:
        x = 0.02 * e_min + (1 - e_min) * x
    else:
        x = -0.02 * e_max + (1 - e_max) * x

    for _ in range(config.max_iters):
        cdf = skew_t_cdf(x, 0.0, 1.0, eta, nu)
        pdf = skew_t_pdf(x, 0.0, 1.0, eta, nu)
        dx = (cdf - p) / pdf
        x = x - dx
        if np.max(np.abs(dx)) <= config.tol:
            break

    x = np.where(np.abs(p - skew_t_cdf(x, 0.0, 1.0, eta, nu)) >= 0.01, np.nan, x)
    return xi + omega * x


def skew_t_moments(
    xi: float, omega: float, eta: float, nu: float
) -> tuple[float, float, float, float]:
    """Mean, std dev, skewness, and excess kurtosis of Azzalini's skew-t
    distribution (requires nu > 4 for the kurtosis to be finite).

    Original: stats/momST.m (HSF toolbox)
    """
    delta = eta / np.sqrt(1 + eta**2)
    m0 = delta * np.sqrt(nu / np.pi) * np.exp(gammaln(0.5 * (nu - 1)) - gammaln(0.5 * nu))
    mean = xi + omega * m0
    sigma = omega * np.sqrt(nu / (nu - 2) - m0**2)

    gamma1 = (
        m0
        * (nu * (3 - delta**2) / (nu - 3) - 3 * nu / (nu - 2) + 2 * m0**2)
        * (nu / (nu - 2) - m0**2) ** (-1.5)
    )
    gamma2 = (
        3 * nu**2 / (nu - 2) / (nu - 4)
        - 4 * m0**2 * nu * (3 - delta**2) / (nu - 3)
        + 6 * m0**2 * nu / (nu - 2)
        - 3 * m0**4
    ) * (nu / (nu - 2) - m0**2) ** (-2) - 3.0

    return float(mean), float(sigma), float(gamma1), float(gamma2)


def skew_t_rvs(
    xi: float, omega: float, eta: float, nu: float, size: int = 1, random_state=None
) -> np.ndarray:
    """Draw samples from Azzalini's skew-t distribution: a skew-normal
    variate divided by sqrt(chi-square(nu) / nu).

    Original: stats/rndST.m (HSF toolbox)
    """
    rng = np.random.default_rng(random_state)
    n = skew_normal_rvs(0.0, omega, eta, size=size, random_state=rng)
    chi = rng.chisquare(nu, size=size)
    return xi + n / np.sqrt(chi / nu)
