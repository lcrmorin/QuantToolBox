"""Jump-diffusion risk measures: thin parameter-transform wrappers around
the Gaussian mixture machinery in ``mixtures.gaussian_mixture``, plus
lognormal moment/skewness formulas.

Ported from QuantToolBox/mixture/{jump_compute_var,jump_compute_es,
jump_compute_rc_var,jump_compute_rc_es,jump_compute_rb_var,
jump_compute_rb_es,jump_pdf_assets,jump_pdf_portfolio,jump_simulate,
jump_skewness,jump_skewness_portfolio,jump_univariate_thresholding,
jump_probability_filtering,lognormal_moments,lognormal_skewness,
bivariate_lognormal_skewness}.m

Model: over a short time step dt, returns follow a diffusion (mean
mu_bar, covariance Sigma_bar) with a Poisson-arrival jump component
(intensity lambda, jump mean mu_tilde, jump covariance Sigma_tilde). This
is exactly a 2-component Gaussian mixture with

    pi1 = 1 - lambda*dt,  mu1 = mu_bar*dt,           Sigma1 = Sigma_bar*dt
    pi2 = lambda*dt,      mu2 = mu_bar*dt + mu_tilde, Sigma2 = Sigma_bar*dt + Sigma_tilde

so every ``jump_*.m`` function in the original is just this parameter
transform followed by a call into the corresponding ``mixture_*.m``
function -- ported here as ``jump_to_mixture_params`` plus one-line
wrappers around ``gaussian_mixture``'s functions, rather than
independently re-implementing the same math.
"""

from __future__ import annotations

import numpy as np

from quanttoolbox.mixtures.gaussian_mixture import (
    MixtureParams,
    RiskBudgetingResult,
    RiskContributionResult,
    mixture_compute_es,
    mixture_compute_rb_es,
    mixture_compute_rb_var,
    mixture_compute_rc_es,
    mixture_compute_rc_var,
    mixture_compute_var,
    mixture_pdf_assets,
    mixture_pdf_portfolio,
    mixture_probability_filtering,
    mixture_simulate,
    mixture_skewness,
    mixture_skewness_portfolio,
    mixture_univariate_thresholding,
)


def jump_to_mixture_params(
    mu_bar: np.ndarray | float,
    sigma_bar: np.ndarray | float,
    mu_tilde: np.ndarray | float,
    sigma_tilde: np.ndarray | float,
    lambda_: float,
    dt: float,
) -> MixtureParams:
    """Convert jump-diffusion parameters (diffusion mean/cov, jump
    intensity/mean/cov, time step) into the equivalent 2-component
    Gaussian mixture parameterization.

    sigma_bar/sigma_tilde may be given as covariance matrices (for the
    multivariate case) or as scalar standard deviations (for the
    univariate skewness/thresholding helpers, which take sigma1/sigma2
    directly rather than full covariance matrices).

    Original: the parameter-transform preamble shared by every
    jump_*.m function (see module docstring)
    """
    mu_bar = np.asarray(mu_bar, dtype=float) if not np.isscalar(mu_bar) else mu_bar
    sigma_bar = np.asarray(sigma_bar, dtype=float) if not np.isscalar(sigma_bar) else sigma_bar
    mu_tilde = np.asarray(mu_tilde, dtype=float) if not np.isscalar(mu_tilde) else mu_tilde
    sigma_tilde = (
        np.asarray(sigma_tilde, dtype=float) if not np.isscalar(sigma_tilde) else sigma_tilde
    )

    pi1 = 1 - lambda_ * dt
    mu1 = mu_bar * dt
    sigma1 = sigma_bar * dt
    pi2 = lambda_ * dt
    mu2 = mu_bar * dt + mu_tilde
    sigma2 = sigma_bar * dt + sigma_tilde

    return MixtureParams(pi1=pi1, mu1=mu1, sigma1=sigma1, pi2=pi2, mu2=mu2, sigma2=sigma2)


def _jump_univariate_params(
    mu_bar: float, sigma_bar: float, mu_tilde: float, sigma_tilde: float, lambda_: float, dt: float
) -> tuple[float, float, float, float, float, float]:
    """Univariate jump->mixture transform matching jump_skewness.m /
    jump_univariate_thresholding.m's convention (sigma1/sigma2 as std
    devs, with sigma2 combining variances in quadrature rather than
    Sigma1*dt + Sigma_tilde -- see those originals)."""
    pi1 = 1 - lambda_ * dt
    mu1 = mu_bar * dt
    sigma1 = sigma_bar * np.sqrt(dt)
    pi2 = lambda_ * dt
    mu2 = mu_bar * dt + mu_tilde
    sigma2 = np.sqrt(sigma1**2 + sigma_tilde**2)
    return pi1, mu1, sigma1, pi2, mu2, sigma2


def jump_compute_var(
    x: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    alpha: float,
) -> tuple[float, float]:
    """Value-at-Risk under the jump-diffusion model. Original: jump_compute_var.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_compute_var(x, params, alpha)


def jump_compute_es(
    x: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Expected Shortfall under the jump-diffusion model. Original: jump_compute_es.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_compute_es(x, params, alpha)


def jump_compute_rc_var(
    x: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    alpha: float,
) -> RiskContributionResult:
    """VaR risk contribution under the jump-diffusion model. Original: jump_compute_rc_var.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_compute_rc_var(x, params, alpha)


def jump_compute_rc_es(
    x: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    alpha: float,
) -> RiskContributionResult:
    """ES risk contribution under the jump-diffusion model. Original: jump_compute_rc_es.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_compute_rc_es(x, params, alpha)


def jump_compute_rb_var(
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    alpha: float,
    b: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    x_minus: float = 0.0,
    x_plus: float = 1.0,
) -> RiskBudgetingResult:
    """VaR risk budgeting under the jump-diffusion model. Original: jump_compute_rb_var.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_compute_rb_var(params, alpha, b=b, x0=x0, x_minus=x_minus, x_plus=x_plus)


def jump_compute_rb_es(
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    alpha: float,
    b: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    x_minus: float = 0.0,
    x_plus: float = 1.0,
) -> RiskBudgetingResult:
    """ES risk budgeting under the jump-diffusion model. Original: jump_compute_rb_es.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_compute_rb_es(params, alpha, b=b, x0=x0, x_minus=x_minus, x_plus=x_plus)


def jump_pdf_assets(
    y: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal asset PDFs under the jump-diffusion model. Original: jump_pdf_assets.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_pdf_assets(y, params)


def jump_pdf_portfolio(
    y: np.ndarray,
    x: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Portfolio return PDF under the jump-diffusion model. Original: jump_pdf_portfolio.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_pdf_portfolio(y, x, params)


def jump_simulate(
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns under the jump-diffusion model. Original: jump_simulate.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_simulate(params, n_samples, rng=rng)


def jump_probability_filtering(
    r_t: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior jump-regime probability given an observation. Original: jump_probability_filtering.m"""
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_probability_filtering(r_t, params)


def jump_skewness(
    mu_bar: float, sigma_bar: float, mu_tilde: float, sigma_tilde: float, lambda_: float, dt: float
) -> tuple[float, float, float]:
    """Mean/std/skewness of univariate returns under the jump-diffusion model.

    Original: jump_skewness.m
    """
    pi1, mu1, sigma1, pi2, mu2, sigma2 = _jump_univariate_params(
        mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt
    )
    return mixture_skewness(pi1, mu1, sigma1, pi2, mu2, sigma2)


def jump_skewness_portfolio(
    x: np.ndarray,
    mu_bar: np.ndarray,
    sigma_bar: np.ndarray,
    mu_tilde: np.ndarray,
    sigma_tilde: np.ndarray,
    lambda_: float,
    dt: float,
) -> tuple[float, float, float]:
    """Mean/std/skewness of a portfolio's return under the jump-diffusion model.

    Original: jump_skewness_portfolio.m
    """
    params = jump_to_mixture_params(mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt)
    return mixture_skewness_portfolio(x, params)


def jump_univariate_thresholding(
    mu_bar: float,
    sigma_bar: float,
    mu_tilde: float,
    sigma_tilde: float,
    lambda_: float,
    dt: float,
    pi2_star: float,
) -> tuple[float, float]:
    """Univariate jump-regime classification thresholds. Original: jump_univariate_thresholding.m"""
    pi1, mu1, sigma1, pi2, mu2, sigma2 = _jump_univariate_params(
        mu_bar, sigma_bar, mu_tilde, sigma_tilde, lambda_, dt
    )
    return mixture_univariate_thresholding(pi1, mu1, sigma1, pi2, mu2, sigma2, pi2_star)


def lognormal_moments(
    mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean, std dev, skewness, and excess kurtosis of exp(N(mu, sigma^2))
    (a lognormal random variable), via its raw (uncentered) moments.

    Original: mixture/lognormal_moments.m
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    m = [np.exp(k * mu + (k**2) * sigma**2 / 2) for k in range(1, 5)]

    mu_x = m[0]
    var_x = m[1] - m[0] ** 2
    sigma_x = np.sqrt(var_x)

    gamma1_x = (m[2] - 3 * m[0] * m[1] + 2 * m[0] ** 3) / sigma_x**3
    gamma2_x = (m[3] - 4 * m[0] * m[2] + 6 * m[0] ** 2 * m[1] - 3 * m[0] ** 4) / sigma_x**4

    return mu_x, sigma_x, gamma1_x, gamma2_x


def lognormal_skewness(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Skewness of a lognormal random variable (closed form, in terms of
    sigma only).

    Original: mixture/lognormal_skewness.m
    """
    sigma2 = np.asarray(sigma, dtype=float) ** 2
    return (np.exp(3 * sigma2) - 3 * np.exp(sigma2) + 2) / (np.exp(sigma2) - 1) ** 1.5


def bivariate_lognormal_skewness(
    mu_x: float, sigma_x: float, mu_y: float, sigma_y: float, rho: float
) -> float:
    """Skewness of the sum of two correlated lognormal random variables
    X=exp(N(mu_x,sigma_x^2)), Y=exp(N(mu_y,sigma_y^2)) with correlation
    rho between the underlying normals.

    Original: mixture/bivariate_lognormal_skewness.m
    """
    sigma2_x, sigma2_y = sigma_x**2, sigma_y**2
    mu1_x = np.exp(mu_x + 0.5 * sigma2_x)
    mu1_y = np.exp(mu_y + 0.5 * sigma2_y)

    mu2_x = np.exp(2 * mu_x + sigma2_x) * (np.exp(sigma2_x) - 1)
    mu2_y = np.exp(2 * mu_y + sigma2_y) * (np.exp(sigma2_y) - 1)
    cov_xy = (
        np.exp(mu_x + mu_y + 0.5 * sigma2_x + 0.5 * sigma2_y + rho * sigma_x * sigma_y)
        - mu1_x * mu1_y
    )
    var_xy = mu2_x + mu2_y + 2 * cov_xy

    mu3_x = (np.exp(3 * sigma2_x) - 3 * np.exp(sigma2_x) + 2) * np.exp(3 * mu_x + 1.5 * sigma2_x)
    mu3_y = (np.exp(3 * sigma2_y) - 3 * np.exp(sigma2_y) + 2) * np.exp(3 * mu_y + 1.5 * sigma2_y)

    cov_xxy = (
        np.exp(2 * mu_x + sigma2_x + mu_y + 0.5 * sigma2_y)
        * (np.exp(rho * sigma_x * sigma_y) - 1)
        * (np.exp(sigma2_x + rho * sigma_x * sigma_y) + np.exp(sigma2_x) - 2)
    )
    cov_xyy = (
        np.exp(2 * mu_y + sigma2_y + mu_x + 0.5 * sigma2_x)
        * (np.exp(rho * sigma_x * sigma_y) - 1)
        * (np.exp(sigma2_y + rho * sigma_x * sigma_y) + np.exp(sigma2_y) - 2)
    )

    mu3_xy = mu3_x + mu3_y + 3 * (cov_xxy + cov_xyy)
    return float(mu3_xy / var_xy**1.5)
