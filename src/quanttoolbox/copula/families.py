"""CDFs and PDFs for the HSF toolbox's copula families: the Fréchet-Hoeffding
bounds and independence copula, the Gaussian and Student-t (elliptical)
copulas, five one-parameter Archimedean copulas (Clayton, Frank, Gumbel,
AMH, Gumbel-Barnett), two extreme-value copulas (Galambos, Husler-Reiss),
and five other named families (Plackett, FGM, Cubic, logistic-Gumbel,
Marshall-Olkin, Sloane, nested/hierarchical Gumbel).

Ported from HSF toolbox `copula/{cdfCopula*,pdfCopula*,
cdfConditionalCopula*,cdfmvn,cdfSloaneCopula,contourCopula*,
singularCopula*}.m` (23 families across ~50 files).

Architecture -- avoiding the original's per-family boilerplate:

- The four Fréchet-Hoeffding-bound / independence functions each existed
  twice in the original (`cdfCopulaUpper.m` for n dimensions vs.
  `cdfCopulaUpper2.m` for 2; same for `Lower`/`Product`) purely because
  MATLAB has no convenient "2-column array or two scalars" polymorphism.
  `comonotonicity_cdf`/`independence_cdf` here take a single ``u`` array
  of shape ``(n_obs, n_dim)`` and work for any `n_dim` including 2,
  collapsing each pair of `.m` files into one function.
  `countermonotonicity_cdf` stays bivariate-only (it's the only one of
  the four that is a genuine copula, i.e. has uniform margins, exactly
  when `n_dim = 2`; `frechet_lower_bound` is the general-`n` non-copula
  bound `cdfCopulaLower.m` computes).
- `gaussian_copula_cdf`/`_pdf` and `student_copula_cdf`/`_pdf` likewise
  collapse the n-dimensional (`cdfCopulaNormal.m`/`cdfCopulaStudent.m`)
  and bivariate (`cdfCopulaNormal2.m`/`cdfCopulaStudent2.m`) originals
  into one function each, special-casing `n_dim = 2` to call
  `quanttoolbox.stats.multivariate.bvn_cdf`/`bvt_cdf` -- already a
  tested, `scipy`-backed, near-singular-correlation-robust
  implementation -- rather than rederiving bivariate-normal/Student CDF
  logic a second time inside this module. The general-`n` path calls
  `quanttoolbox.stats.distributions.mvn_cdf` (Gaussian) or
  `scipy.stats.multivariate_t.cdf` (Student, no existing n-dimensional
  wrapper to reuse).
- Archimedean/extreme-value/other named families (Clayton, Frank, Gumbel,
  AMH, Gumbel-Barnett, Galambos, Husler-Reiss, Plackett, FGM, Cubic,
  logistic-Gumbel, Marshall-Olkin, Sloane) are each only a few lines of
  closed-form `numpy`, so are transliterated directly rather than forced
  through a generic Archimedean-generator abstraction that would save
  little code while obscuring each family's actual (and structurally
  quite different) closed form. Clayton/Frank/Gumbel's CDF, PDF, and
  Kendall's tau were verified to match `statsmodels.distributions.copula`
  exactly (`statsmodels` is already a dependency of this package); they
  are kept as plain vectorized `numpy` functions here rather than thin
  wrappers around the `statsmodels` class-based API because the original
  (and the rest of this port) vectorizes `theta` itself per observation,
  which `statsmodels`' one-`theta`-per-instance design does not support.
- Every family the original ported a CDF for but never ported a PDF for
  (AMH, Husler-Reiss, Marshall-Olkin, FGM, Sloane) stays that way here --
  no PDF is fabricated for families the original author evidently chose
  not to derive one for.
- `cdfCopulaGumbel3.m` (a 3-variable nested/hierarchical Gumbel copula)
  is ported as `nested_gumbel_cdf`, but its companion `pdfCopulaGumbel3.m`
  is **not** ported: independently re-deriving ``d^3 C / du1 du2 du3``
  both symbolically (via `sympy`) and via a 3-D central finite difference
  disagreed with the original formula (verified at `theta1=1.5,
  theta2=3.0, u=(0.3, 0.5, 0.7)`: symbolic/numeric both give ~1.2096,
  the original formula gives ~1.0064) -- the original's PDF has a
  genuine bug. Given how easy it would be to introduce a *different*
  subtle error in a hand-derived replacement for this rarely-used
  trivariate extension, the safer choice is to ship the (verified-correct)
  CDF only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_t
from scipy.stats import norm as normdist

from quanttoolbox.stats.distributions import mvn_cdf
from quanttoolbox.stats.multivariate import bvn_cdf, bvn_pdf, bvt_cdf

# ---------------------------------------------------------------------------
# Fréchet-Hoeffding bounds and independence
# ---------------------------------------------------------------------------


def comonotonicity_cdf(u: np.ndarray) -> np.ndarray:
    """The comonotonicity (Fréchet-Hoeffding upper-bound, "M") copula:
    ``C(u) = min(u_1, ..., u_n)``. `u` has shape ``(n_obs, n_dim)``.

    Original: copula/cdfCopulaUpper.m (n-dim) and cdfCopulaUpper2.m
    (bivariate) -- merged, see module docstring.
    """
    u = np.asarray(u, dtype=float)
    return np.min(u, axis=1)


def countermonotonicity_cdf(u1: np.ndarray | float, u2: np.ndarray | float) -> np.ndarray:
    """The countermonotonicity (Fréchet-Hoeffding lower-bound, "W") copula:
    ``C(u1, u2) = max(u1 + u2 - 1, 0)``. Only a genuine copula (uniform
    margins) in 2 dimensions -- see `frechet_lower_bound` for the general
    -`n` bound.

    Original: copula/cdfCopulaLower2.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    return np.maximum(u1 + u2 - 1.0, 0.0)


def frechet_lower_bound(u: np.ndarray) -> np.ndarray:
    """The Fréchet-Hoeffding lower bound for `n_dim` dimensions:
    ``max(u_1 + ... + u_n - n_dim + 1, 0)``. Not itself a copula for
    `n_dim > 2` (no distribution attains it as its copula), but every
    copula is bounded below by it. `u` has shape ``(n_obs, n_dim)``.

    Original: copula/cdfCopulaLower.m
    """
    u = np.asarray(u, dtype=float)
    n_dim = u.shape[1]
    return np.maximum(np.sum(u, axis=1) - n_dim + 1.0, 0.0)


def independence_cdf(u: np.ndarray) -> np.ndarray:
    """The independence (product) copula: ``C(u) = u_1 * u_2 * ... *
    u_n``. `u` has shape ``(n_obs, n_dim)``.

    Original: copula/cdfCopulaProduct.m (n-dim) and cdfCopulaProduct2.m
    (bivariate) -- merged, see module docstring.
    """
    u = np.asarray(u, dtype=float)
    return np.prod(u, axis=1)


def comonotonicity_support(u1: np.ndarray | float) -> np.ndarray:
    """The comonotonicity copula's singular support: ``u2 = u1``.

    Original: copula/singularCopulaUpper2.m
    """
    return np.asarray(u1, dtype=float)


def countermonotonicity_support(u1: np.ndarray | float) -> np.ndarray:
    """The countermonotonicity copula's singular support: ``u2 = 1 - u1``.

    Original: copula/singularCopulaLower2.m
    """
    return 1.0 - np.asarray(u1, dtype=float)


# ---------------------------------------------------------------------------
# Elliptical copulas: Gaussian, Student-t
# ---------------------------------------------------------------------------


def gaussian_copula_cdf(u: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """The Gaussian copula CDF: ``C(u) = Phi_corr(Phi^-1(u_1), ...,
    Phi^-1(u_n))``, `Phi_corr` the correlation-`corr` multivariate normal
    CDF. `u` has shape ``(n_obs, n_dim)``; for `n_dim = 2`, delegates to
    `quanttoolbox.stats.multivariate.bvn_cdf` (see module docstring).

    Original: copula/cdfCopulaNormal.m (n-dim) and cdfCopulaNormal2.m
    (bivariate) -- merged.
    """
    u = np.asarray(u, dtype=float)
    corr = np.asarray(corr, dtype=float)
    x = normdist.ppf(u)
    if u.shape[1] == 2:
        return bvn_cdf(x[:, 0], x[:, 1], corr[0, 1])
    return mvn_cdf(x, np.zeros(u.shape[1]), corr)


def gaussian_copula_pdf(u: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """The Gaussian copula density.

    Original: copula/pdfCopulaNormal.m (n-dim) and pdfCopulaNormal2.m
    (bivariate) -- merged.
    """
    u = np.asarray(u, dtype=float)
    corr = np.asarray(corr, dtype=float)
    x = normdist.ppf(u)
    if u.shape[1] == 2:
        rho = corr[0, 1]
        return bvn_pdf(x[:, 0], x[:, 1], 0.0, 0.0, 1.0, 1.0, rho) / (
            normdist.pdf(x[:, 0]) * normdist.pdf(x[:, 1])
        )
    corr_inv = np.linalg.inv(corr) - np.eye(u.shape[1])
    quad_form = np.einsum("ij,jk,ik->i", x, corr_inv, x)
    return np.exp(-0.5 * quad_form) / np.sqrt(np.linalg.det(corr))


def gaussian_copula_conditional_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray:
    """The bivariate Gaussian copula's conditional CDF ``Pr(U1 <= u1 |
    U2 = u2)``, used for conditional simulation.

    Original: copula/cdfConditionalCopulaNormal2.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return normdist.cdf((normdist.ppf(u1) - rho * normdist.ppf(u2)) / np.sqrt(1.0 - rho**2))


def student_copula_cdf(u: np.ndarray, corr: np.ndarray, nu: float) -> np.ndarray:
    """The Student-t copula CDF with `nu` degrees of freedom. `u` has
    shape ``(n_obs, n_dim)``; for `n_dim = 2`, delegates to
    `quanttoolbox.stats.multivariate.bvt_cdf`.

    Original: copula/cdfCopulaStudent.m (n-dim) and cdfCopulaStudent2.m
    (bivariate, looping over a scenario grid of `rho`/`nu` pairs) --
    merged: sweep scenarios with a plain Python loop over `corr`/`nu` at
    the call site instead of baking scenario-looping into the function.
    """
    u = np.asarray(u, dtype=float)
    corr = np.asarray(corr, dtype=float)
    from scipy.stats import t as tdist

    x = tdist.ppf(u, nu)
    if u.shape[1] == 2:
        return bvt_cdf(x[:, 0], x[:, 1], corr[0, 1], nu)
    return multivariate_t.cdf(x, loc=np.zeros(u.shape[1]), shape=corr, df=float(nu))


def student_copula_pdf(u: np.ndarray, corr: np.ndarray, nu: float) -> np.ndarray:
    """The Student-t copula density with `nu` degrees of freedom.

    Original: copula/pdfCopulaStudent.m (n-dim) and pdfCopulaStudent2.m
    (bivariate) -- merged.
    """
    from scipy.special import gammaln
    from scipy.stats import t as tdist

    u = np.asarray(u, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n_dim = u.shape[1]
    x = tdist.ppf(u, nu)

    corr_inv = np.linalg.inv(corr)
    corr_det = np.linalg.det(corr)
    quad_form = np.einsum("ij,jk,ik->i", x, corr_inv, x)

    log_num = gammaln((nu + n_dim) / 2.0) + (n_dim - 1) * gammaln(nu / 2.0)
    log_den = n_dim * gammaln((nu + 1.0) / 2.0) + 0.5 * np.log(corr_det)
    log_pdf = (
        -0.5 * (nu + n_dim) * np.log1p(quad_form / nu)
        + 0.5 * (nu + 1.0) * np.sum(np.log1p(x**2 / nu), axis=1)
        + log_num
        - log_den
    )
    pdf = np.exp(log_pdf)
    return np.where(np.isnan(pdf), 0.0, pdf)


# ---------------------------------------------------------------------------
# Archimedean copulas
# ---------------------------------------------------------------------------


def clayton_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Clayton copula CDF (verified to match
    `statsmodels.distributions.copula.archimedean.ClaytonCopula`).

    Original: copula/cdfCopulaClayton.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return np.maximum(u1 ** (-theta) + u2 ** (-theta) - 1.0, 0.0) ** (-1.0 / theta)


def clayton_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Clayton copula density (verified to match `statsmodels`).

    Original: copula/pdfCopulaClayton.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    base = np.maximum(u1 ** (-theta) + u2 ** (-theta) - 1.0, 0.0)
    pdf = (1.0 + theta) * ((u1 * u2) ** (-theta - 1.0)) * base ** (-(2.0 * theta + 1.0) / theta)
    return np.where(np.isnan(pdf), 0.0, pdf)


def frank_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Frank copula CDF (verified to match `statsmodels`).

    Original: copula/cdfCopulaFrank.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a = np.exp(-theta * u1) - 1.0
    b = np.exp(-theta * u2) - 1.0
    return -np.log(1.0 + a * b / (np.exp(-theta) - 1.0)) / theta


def frank_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Frank copula density (verified to match `statsmodels`).

    Original: copula/pdfCopulaFrank.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    eta = 1.0 - np.exp(-theta)
    v1 = 1.0 - np.exp(-theta * u1)
    v2 = 1.0 - np.exp(-theta * u2)
    return np.exp(-theta * (u1 + u2)) * theta * eta / (eta - v1 * v2) ** 2


def gumbel_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Gumbel copula CDF (verified to match `statsmodels`).

    Original: copula/cdfCopulaGumbel.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return np.exp(-(((-np.log(u1)) ** theta + (-np.log(u2)) ** theta) ** (1.0 / theta)))


def gumbel_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Gumbel copula density (verified to match `statsmodels`).

    Original: copula/pdfCopulaGumbel.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    u1t = -np.log(u1)
    u2t = -np.log(u2)
    w = u1t**theta + u2t**theta
    pdf = (
        ((u1t * u2t) ** (theta - 1.0))
        * (w ** (1.0 / theta) + theta - 1.0)
        / (w ** (2.0 - 1.0 / theta))
        / (u1 * u2)
    )
    return pdf * gumbel_cdf(u1, u2, theta)


def nested_gumbel_cdf(
    u1: np.ndarray | float,
    u2: np.ndarray | float,
    u3: np.ndarray | float,
    theta1: np.ndarray | float,
    theta2: np.ndarray | float,
) -> np.ndarray:
    """A 3-variable nested (hierarchical) Gumbel copula: (u1, u2) are
    coupled at strength `theta2` first, then combined with `u3` at
    strength `theta1` (requires ``theta2 >= theta1 >= 1`` for a valid
    copula). No PDF is provided -- see module docstring.

    Original: copula/cdfCopulaGumbel3.m
    """
    u1t = -np.log(np.asarray(u1, dtype=float))
    u2t = -np.log(np.asarray(u2, dtype=float))
    u3t = -np.log(np.asarray(u3, dtype=float))
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)

    inner = u1t**theta2 + u2t**theta2
    return np.exp(-((inner ** (theta1 / theta2) + u3t**theta1) ** (1.0 / theta1)))


def amh_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Ali-Mikhail-Haq (AMH) copula CDF. No PDF is provided in the
    original -- see module docstring; `amh_conditional_cdf` (the partial
    derivative w.r.t. `u1`) is available for simulation.

    Original: copula/cdfCopulaAMH.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return u1 * u2 / (1.0 - theta * (1.0 - u1) * (1.0 - u2))


def amh_conditional_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The AMH copula's conditional CDF ``Pr(U2 <= u2 | U1 = u1)``
    (``dC/du1``), used for conditional simulation.

    Original: copula/cdfConditionalCopulaAMH.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    numerator = (1.0 - theta) * u2 + theta * u2**2
    denominator = (1.0 - theta * (1.0 - u1) * (1.0 - u2)) ** 2
    return numerator / denominator


def gumbel_barnett_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Gumbel-Barnett copula CDF.

    Original: copula/cdfCopulaGumbelBarnett.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return u1 * u2 * np.exp(-theta * np.log(u1) * np.log(u2))


def gumbel_barnett_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Gumbel-Barnett copula density.

    Original: copula/pdfCopulaGumbelBarnett.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    log_u1 = np.log(u1)
    log_u2 = np.log(u2)
    return (1.0 - theta - theta * (log_u1 + log_u2) + theta**2 * log_u1 * log_u2) * np.exp(
        -theta * log_u1 * log_u2
    )


# ---------------------------------------------------------------------------
# Extreme-value copulas
# ---------------------------------------------------------------------------


def galambos_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Galambos copula CDF.

    Original: copula/cdfCopulaGalambos.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    u1t = -np.log(u1)
    u2t = -np.log(u2)
    return u1 * u2 * np.exp((u1t ** (-theta) + u2t ** (-theta)) ** (-1.0 / theta))


def galambos_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Galambos copula density.

    Original: copula/pdfCopulaGalambos.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    u1t = -np.log(u1)
    u2t = -np.log(u2)
    cdf = galambos_cdf(u1, u2, theta)
    p = (
        1.0
        - (u1t ** (-theta) + u2t ** (-theta)) ** (-1.0 / theta - 1.0)
        * (u1t ** (-theta - 1.0) + u2t ** (-theta - 1.0))
        + (u1t ** (-theta) + u2t ** (-theta)) ** (-1.0 / theta - 2.0)
        * (u1t * u2t) ** (-1.0 - theta)
        * (1.0 + theta + (u1t ** (-theta) + u2t ** (-theta)) ** (-1.0 / theta))
    )
    return cdf / (u1 * u2) * p


def husler_reiss_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Husler-Reiss copula CDF. No PDF is provided in the original --
    see module docstring.

    Original: copula/cdfCopulaHuslerReiss.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    # `+ 0.0` normalizes the -0.0 that `-np.log(1.0)` produces (IEEE 754
    # negation of +0.0) back to +0.0 -- left as -0.0, `u2t / u1t` below
    # would evaluate to -inf instead of +inf at the u1 == 1 boundary, and
    # `log(-inf)` is NaN instead of the correct +inf.
    u1t = -np.log(u1) + 0.0
    u2t = -np.log(u2) + 0.0
    phi1 = 1.0 / theta + 0.5 * theta * np.log(u1t / u2t)
    phi2 = 1.0 / theta + 0.5 * theta * np.log(u2t / u1t)
    return np.exp(-u1t * normdist.cdf(phi1) - u2t * normdist.cdf(phi2))


# ---------------------------------------------------------------------------
# Other named families
# ---------------------------------------------------------------------------


def plackett_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Plackett copula CDF.

    Original: copula/cdfCopulaPlackett.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    eta = theta - 1.0
    w = 1.0 + eta * (u1 + u2)
    return 0.5 * (w - np.sqrt(w**2 - 4.0 * theta * eta * u1 * u2)) / eta


def plackett_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Plackett copula density.

    Original: copula/pdfCopulaPlackett.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    eta = theta - 1.0
    w = 1.0 + eta * (u1 + u2)
    return theta * (w - 2.0 * eta * u1 * u2) / (w**2 - 4.0 * theta * eta * u1 * u2) ** 1.5


def fgm_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The Farlie-Gumbel-Morgenstern (FGM) copula CDF. No PDF is provided
    in the original -- see module docstring.

    Original: copula/cdfCopulaFGM.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return u1 * u2 * (1.0 + theta * (1.0 - u1) * (1.0 - u2))


def cubic_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The cubic copula CDF.

    Original: copula/cdfCopulaCubic.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return u1 * u2 + theta * u1 * (u1 - 1.0) * (2.0 * u1 - 1.0) * u2 * (u2 - 1.0) * (2.0 * u2 - 1.0)


def cubic_pdf(
    u1: np.ndarray | float, u2: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """The cubic copula density.

    Original: copula/pdfCopulaCubic.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return 1.0 + theta * (6.0 * u1**2 - 6.0 * u1 + 1.0) * (6.0 * u2**2 - 6.0 * u2 + 1.0)


def logistic_gumbel_cdf(u1: np.ndarray | float, u2: np.ndarray | float) -> np.ndarray:
    """The (parameter-free) logistic-Gumbel copula CDF.

    Original: copula/cdfCopulaLogisticGumbel.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    return u1 * u2 / (u1 + u2 - u1 * u2)


def logistic_gumbel_pdf(u1: np.ndarray | float, u2: np.ndarray | float) -> np.ndarray:
    """The (parameter-free) logistic-Gumbel copula density.

    Original: copula/pdfCopulaLogisticGumbel.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    return 2.0 * u1 * u2 / (u1 + u2 - u1 * u2) ** 3


def logistic_gumbel_contour(u1: np.ndarray | float, alpha: np.ndarray | float) -> np.ndarray:
    """`u2` such that the logistic-Gumbel copula's CDF equals `alpha`
    along the curve through `u1`, i.e. a level-set/contour curve used for
    plotting: values of `u1 < alpha` fall outside the valid contour and
    return `nan`.

    Original: copula/contourCopulaLogisticGumbel.m
    """
    u1 = np.asarray(u1, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    u1 = np.where(u1 < alpha, np.nan, u1)
    return alpha * u1 / (u1 + alpha * u1 - alpha)


def marshall_olkin_cdf(
    u1: np.ndarray | float,
    u2: np.ndarray | float,
    theta1: np.ndarray | float,
    theta2: np.ndarray | float,
) -> np.ndarray:
    """The Marshall-Olkin copula CDF. Has a singular component along
    ``u1**theta1 == u2**theta2`` (see `marshall_olkin_singular_support`);
    no PDF is provided in the original.

    Original: copula/cdfCopulaMarshallOlkin.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    cond = u1**theta1 > u2**theta2
    return (
        u1 ** (1.0 - theta1)
        * u2 ** (1.0 - theta2)
        * (u1**theta1 + np.where(cond, u2**theta2 - u1**theta1, 0.0))
    )


def marshall_olkin_singular_support(
    u1: np.ndarray | float, theta1: float, theta2: float
) -> np.ndarray:
    """The Marshall-Olkin copula's singular support: ``u2 = u1 **
    (theta1 / theta2)``.

    Original: copula/singularCopulaMarshallOlkin.m
    """
    u1 = np.asarray(u1, dtype=float)
    return u1 ** (theta1 / theta2)


def sloane_cdf(
    u1: np.ndarray | float, u2: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray:
    """The Sloane copula CDF. No PDF is provided in the original.

    Original: copula/cdfSloaneCopula.m
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    rho = np.asarray(rho, dtype=float)

    u1v = np.arccosh(1.0 / u1**2)
    u2v = np.arccosh(1.0 / u2**2)
    is_u1_smaller = u1v <= u2v
    zeta = np.where(is_u1_smaller, u1v, u2v)
    xi = np.abs(u1v - u2v)

    c1 = np.cosh(xi) * np.cosh(zeta * np.sqrt(1.0 + rho)) * np.cosh(zeta * np.sqrt(1.0 - rho))
    c2 = np.sinh(xi) * np.cosh(zeta * np.sqrt(1.0 - rho)) * np.sinh(zeta * np.sqrt(1.0 + rho))
    c3 = np.sinh(xi) * np.cosh(zeta * np.sqrt(1.0 + rho)) * np.sinh(zeta * np.sqrt(1.0 - rho))

    c = c1 + 0.5 * np.sqrt(1.0 + rho) * c2 + 0.5 * np.sqrt(1.0 - rho) * c3
    c = 1.0 / np.sqrt(c)
    return np.where(np.isreal(c), np.real(c), 0.0)


def frank_contour(
    u1: np.ndarray | float, alpha: np.ndarray | float, theta: np.ndarray | float
) -> np.ndarray:
    """`u2` such that the Frank copula's CDF equals `alpha` along the
    curve through `u1` -- a level-set/contour curve used for plotting.
    Returns `nan` where the contour has no (real-valued, in-``[0,1]``)
    solution.

    Original: copula/contourCopulaFrank.m
    """
    u1 = np.asarray(u1, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    theta = np.asarray(theta, dtype=float)

    u2 = (
        -np.log(
            1.0
            + (np.exp(-alpha * theta) - 1.0) * (np.exp(-theta) - 1.0) / (np.exp(-theta * u1) - 1.0)
        )
        / theta
    )
    u2 = np.where(np.abs(np.imag(u2)) > 0.01, np.nan, np.real(u2))
    return np.where((u2 < 0.0) | (u2 > 1.0), np.nan, u2)
