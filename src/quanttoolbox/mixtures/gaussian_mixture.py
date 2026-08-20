"""Two-component Gaussian mixture models: moments, PDF, simulation, EM
estimation, VaR/ES risk measures, and risk budgeting.

Ported from QuantToolbox/mixture/{mixture_moments,mixture_pdf_assets,
mixture_pdf_portfolio,mixture_simulate,mixture_skewness,
mixture_skewness_portfolio,mixture_univariate_thresholding,
mixture_probability_filtering,mixture_compute_var,mixture_compute_es,
mixture_compute_rc_var,mixture_compute_rc_es,mixture_compute_rb_var,
mixture_compute_rb_es,estimate_em_mixture,logl_em_mixture}.m

Model: a random vector R is pi1-probability drawn from N(mu1, Sigma1) and
pi2=1-pi1-probability drawn from N(mu2, Sigma2) -- e.g. a "normal regime"
and a "stress regime" for asset returns.

Translation notes:

- ``mixture_compute_rb_var``/``mixture_compute_rb_es`` originally support
  two solver algorithms: (1) ``fmincon`` minimizing the sum of squared
  deviations between (risk contribution / budget) ratios across assets,
  and (2) an ``fminunc`` log-barrier variant referencing a
  ``RB_lagrangian`` global that's never actually set anywhere in the
  original codebase (a global exists but nothing ever assigns to it before
  use -- effectively dead/broken code). Only algorithm (1) is ported here,
  via ``scipy.optimize.minimize`` (SLSQP, budget-constrained).
- MATLAB's ``global MIXTURE_*``/``RB_*`` state-passing to nested
  objective/constraint functions is replaced by closures capturing the
  mixture parameters directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.stats import multivariate_normal, norm

from quanttoolbox.optim.bisection import bisection


@dataclass
class MixtureParams:
    pi1: float
    mu1: np.ndarray
    sigma1: np.ndarray  # covariance matrix
    pi2: float
    mu2: np.ndarray
    sigma2: np.ndarray


def mixture_moments(params: MixtureParams) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of the mixture distribution.

    Original: mixture/mixture_moments.m
    """
    p = params
    mu_bar = p.pi1 * p.mu1 + p.pi2 * p.mu2
    d_mu = p.mu1 - p.mu2
    sigma_bar = p.pi1 * p.sigma1 + p.pi2 * p.sigma2 + p.pi1 * p.pi2 * np.outer(d_mu, d_mu)
    return mu_bar, sigma_bar


def mixture_skewness(
    pi1: float, mu1: float, sigma1: float, pi2: float, mu2: float, sigma2: float
) -> tuple[float, float, float]:
    """Mean, standard deviation, and skewness of a scalar 2-component
    Gaussian mixture.

    Original: mixture/mixture_skewness.m
    """
    var1, var2 = sigma1**2, sigma2**2
    d_mu = mu1 - mu2

    mu = pi1 * mu1 + pi2 * mu2
    sigma = np.sqrt(pi1 * var1 + pi2 * var2 + pi1 * pi2 * d_mu**2)
    gamma1 = pi1 * pi2 * ((pi2 - pi1) * d_mu**3 + 3 * d_mu * (var1 - var2))
    gamma1 = gamma1 / sigma**3

    return mu, sigma, gamma1


def mixture_skewness_portfolio(x: np.ndarray, params: MixtureParams) -> tuple[float, float, float]:
    """Mean, standard deviation, and skewness of a portfolio x's return
    under the mixture distribution.

    Original: mixture/mixture_skewness_portfolio.m
    """
    x = np.asarray(x, dtype=float).flatten()
    p = params
    mu1_x = float(x @ p.mu1)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    mu2_x = float(x @ p.mu2)
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))
    return mixture_skewness(p.pi1, mu1_x, sigma1_x, p.pi2, mu2_x, sigma2_x)


def mixture_pdf_assets(
    y: np.ndarray, params: MixtureParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal PDF of each asset under the mixture distribution, evaluated
    at y (one column per asset, matching mu1/mu2's dimension).

    Original: mixture/mixture_pdf_assets.m
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    p = params
    sigma1 = np.sqrt(np.diag(p.sigma1))
    sigma2 = np.sqrt(np.diag(p.sigma2))

    pdf1 = norm.pdf(y, loc=p.mu1, scale=sigma1)
    pdf2 = norm.pdf(y, loc=p.mu2, scale=sigma2)
    pdf = p.pi1 * pdf1 + p.pi2 * pdf2
    return pdf, pdf1, pdf2


def mixture_pdf_portfolio(
    y: np.ndarray, x: np.ndarray, params: MixtureParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PDF of a portfolio x's return under the mixture distribution,
    evaluated at y.

    Original: mixture/mixture_pdf_portfolio.m
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    p = params

    mu1_x = float(x @ p.mu1)
    mu2_x = float(x @ p.mu2)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))

    pdf1 = norm.pdf(y, loc=mu1_x, scale=sigma1_x)
    pdf2 = norm.pdf(y, loc=mu2_x, scale=sigma2_x)
    pdf = p.pi1 * pdf1 + p.pi2 * pdf2
    return pdf, pdf1, pdf2


def mixture_simulate(
    params: MixtureParams, n_samples: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n_samples draws from the mixture distribution.

    Original: mixture/mixture_simulate.m

    Returns (samples, regime) where regime[i]=1 if sample i was drawn from
    component 1, else 2.
    """
    rng = np.random.default_rng() if rng is None else rng
    p = params
    n = np.asarray(p.mu1).shape[0]

    is_regime1 = rng.uniform(size=n_samples) <= p.pi1
    u1 = multivariate_normal.rvs(mean=p.mu1, cov=p.sigma1, size=n_samples, random_state=rng)
    u2 = multivariate_normal.rvs(mean=p.mu2, cov=p.sigma2, size=n_samples, random_state=rng)
    if n == 1:
        # scipy's rvs collapses to shape (n_samples,) for a 1-D distribution;
        # reshape to (n_samples, 1) for consistency with the n>1 case
        # (np.atleast_2d would incorrectly prepend a dimension instead).
        u1 = u1.reshape(n_samples, 1)
        u2 = u2.reshape(n_samples, 1)

    u = np.where(is_regime1[:, None], u1, u2)
    regime = np.where(is_regime1, 1, 2)
    return u, regime


def mixture_probability_filtering(
    r_t: np.ndarray, params: MixtureParams
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior regime probabilities given an observation r_t (Bayes'
    rule applied to the mixture likelihood).

    Original: mixture/mixture_probability_filtering.m
    """
    p = params
    pdf1 = multivariate_normal.pdf(r_t, mean=p.mu1, cov=p.sigma1)
    pdf2 = multivariate_normal.pdf(r_t, mean=p.mu2, cov=p.sigma2)

    c1 = p.pi1 * pdf1
    c2 = p.pi2 * pdf2
    pi1_t = c1 / (c1 + c2)
    pi2_t = 1 - pi1_t
    return pi1_t, pi2_t


def mixture_univariate_thresholding(
    pi1: float, mu1: float, sigma1: float, pi2: float, mu2: float, sigma2: float, pi2_star: float
) -> tuple[float, float]:
    """Find the threshold values [y_minus, y_plus] outside of which the
    posterior probability of being in regime 2 exceeds pi2_star (a
    univariate regime-classification boundary).

    Original: mixture/mixture_univariate_thresholding.m
    """
    a = (pi2 * (1 - pi2_star) * sigma1) / (pi1 * pi2_star * sigma2)

    alpha = sigma2**2 - sigma1**2
    beta = mu2 * sigma1**2 - mu1 * sigma2**2
    gamma = mu1**2 * sigma2**2 - mu2**2 * sigma1**2 + 2 * sigma2**2 * sigma1**2 * np.log(a)

    delta = beta**2 - alpha * gamma
    y_minus = -(beta + np.sqrt(delta)) / alpha
    y_plus = (-beta + np.sqrt(delta)) / alpha
    return float(y_minus), float(y_plus)


def mixture_compute_var(x: np.ndarray, params: MixtureParams, alpha: float) -> tuple[float, float]:
    """Value-at-Risk of a portfolio x's return under the mixture
    distribution, at confidence level alpha (found via bisection on the
    mixture CDF).

    Original: mixture/mixture_compute_var.m

    Returns (VaR_mixture, VaR_Gaussian) -- the latter using only the
    first (normal-regime) component, for comparison.
    """
    x = np.asarray(x, dtype=float).flatten()
    p = params

    mu1_x = float(x @ p.mu1)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    mu2_x = float(x @ p.mu2)
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))

    var_gaussian = -mu1_x + norm.ppf(alpha) * sigma1_x
    min_var, max_var = var_gaussian / 5, 5 * var_gaussian

    def objective(var: np.ndarray) -> np.ndarray:
        v = float(var)
        h1 = (v + mu1_x) / sigma1_x
        h2 = (v + mu2_x) / sigma2_x
        return np.array(p.pi1 * norm.cdf(h1) + p.pi2 * norm.cdf(h2) - alpha)

    var_mixture = float(bisection(objective, min_var, max_var))
    return var_mixture, var_gaussian


def _psi_function(a: float, b: float, c: float, alpha: float) -> float:
    h = (a + b) / c
    return (c * norm.pdf(h) - b * norm.cdf(-h)) / (1 - alpha)


def mixture_compute_es(
    x: np.ndarray, params: MixtureParams, alpha: float
) -> tuple[float, float, float, float]:
    """Expected Shortfall of a portfolio x's return under the mixture
    distribution, at confidence level alpha.

    Original: mixture/mixture_compute_es.m

    Returns (ES_mixture, VaR_mixture, ES_Gaussian, VaR_Gaussian).
    """
    x = np.asarray(x, dtype=float).flatten()
    p = params

    mu1_x = float(x @ p.mu1)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    mu2_x = float(x @ p.mu2)
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))

    var_mixture, var_gaussian = mixture_compute_var(x, params, alpha)
    es_gaussian = -mu1_x + sigma1_x * norm.pdf(norm.ppf(alpha)) / (1 - alpha)
    es_mixture = p.pi1 * _psi_function(var_mixture, mu1_x, sigma1_x, alpha) + p.pi2 * _psi_function(
        var_mixture, mu2_x, sigma2_x, alpha
    )

    return es_mixture, var_mixture, es_gaussian, var_gaussian


@dataclass
class RiskContributionResult:
    risk: float
    marginal_risk: np.ndarray
    risk_contribution: np.ndarray
    pct_risk_contribution: np.ndarray


def mixture_compute_rc_var(
    x: np.ndarray, params: MixtureParams, alpha: float
) -> RiskContributionResult:
    """VaR risk contribution decomposition under the mixture distribution.

    Original: mixture/mixture_compute_rc_var.m
    """
    x = np.asarray(x, dtype=float).flatten()
    p = params
    var_mixture, _ = mixture_compute_var(x, params, alpha)

    mu1_x = float(x @ p.mu1)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    mu2_x = float(x @ p.mu2)
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))

    h1_x = (var_mixture + mu1_x) / sigma1_x
    h2_x = (var_mixture + mu2_x) / sigma2_x

    w1 = p.pi1 * norm.pdf(h1_x) / sigma1_x
    w2 = p.pi2 * norm.pdf(h2_x) / sigma2_x

    mr = w1 * ((h1_x / sigma1_x) * (p.sigma1 @ x) - p.mu1) + w2 * (
        (h2_x / sigma2_x) * (p.sigma2 @ x) - p.mu2
    )
    mr = mr / (w1 + w2)
    rc = x * mr
    prc = rc / np.sum(rc)

    return RiskContributionResult(
        risk=var_mixture, marginal_risk=mr, risk_contribution=rc, pct_risk_contribution=prc
    )


def mixture_compute_rc_es(
    x: np.ndarray, params: MixtureParams, alpha: float
) -> RiskContributionResult:
    """ES risk contribution decomposition under the mixture distribution.

    Original: mixture/mixture_compute_rc_es.m
    """
    x = np.asarray(x, dtype=float).flatten()
    p = params
    var_result = mixture_compute_rc_var(x, params, alpha)
    var_mixture = var_result.risk

    mu_tilde = p.mu2 - p.mu1
    mu1_x = float(x @ p.mu1)
    sigma1_x = float(np.sqrt(x @ p.sigma1 @ x))
    mu2_x = float(x @ p.mu2)
    sigma2_x = float(np.sqrt(x @ p.sigma2 @ x))

    h1_x = (var_mixture + mu1_x) / sigma1_x
    h2_x = (var_mixture + mu2_x) / sigma2_x

    es = p.pi1 * _psi_function(var_mixture, mu1_x, sigma1_x, alpha) + p.pi2 * _psi_function(
        var_mixture, mu2_x, sigma2_x, alpha
    )

    w1 = p.pi1 * norm.pdf(h1_x) / sigma1_x
    w2 = p.pi2 * norm.pdf(h2_x) / sigma2_x

    delta1_x = (1 + (h1_x / sigma1_x) * var_mixture) * (p.sigma1 @ x) - var_mixture / (w1 + w2) * (
        w1 * (h1_x / sigma1_x) * (p.sigma1 @ x)
        + w2 * ((h2_x / sigma2_x) * (p.sigma2 @ x) - mu_tilde)
    )
    delta2_x = (1 + (h2_x / sigma2_x) * var_mixture) * (p.sigma2 @ x) - var_mixture / (w1 + w2) * (
        w2 * (h2_x / sigma2_x) * (p.sigma2 @ x)
        + w1 * ((h1_x / sigma1_x) * (p.sigma1 @ x) + mu_tilde)
    )

    mr = (
        w1 * delta1_x
        + w2 * delta2_x
        - (p.pi1 * p.mu1 * norm.cdf(-h1_x) + p.pi2 * p.mu2 * norm.cdf(-h2_x))
    )
    mr = mr / (1 - alpha)
    rc = x * mr
    prc = rc / np.sum(rc)

    return RiskContributionResult(
        risk=es, marginal_risk=mr, risk_contribution=rc, pct_risk_contribution=prc
    )


@dataclass
class RiskBudgetingResult:
    weights: np.ndarray
    risk: float
    marginal_risk: np.ndarray
    risk_contribution: np.ndarray
    pct_risk_contribution: np.ndarray
    converged: bool


def _solve_mixture_rb(
    rc_fn,
    params: MixtureParams,
    alpha: float,
    b: np.ndarray | None,
    x0: np.ndarray | None,
    x_minus: float,
    x_plus: float,
) -> RiskBudgetingResult:
    n = np.asarray(params.mu1).shape[0]
    b_arr = np.full(n, 1.0 / n) if b is None else np.asarray(b, dtype=float).flatten()
    b_arr = b_arr / np.sum(b_arr)
    x0_arr = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float).flatten()

    def objective(x: np.ndarray) -> float:
        rc = rc_fn(x, params, alpha).risk_contribution
        ratio = rc / b_arr
        diff = ratio[:, None] - ratio[None, :]
        return float(np.sum(diff**2))

    constraints = [LinearConstraint(np.ones((1, n)), 1.0, 1.0)]
    bounds = [(x_minus, x_plus)] * n

    result = minimize(objective, x0_arr, method="SLSQP", bounds=bounds, constraints=constraints)
    x = result.x
    rc_result = rc_fn(x, params, alpha)

    return RiskBudgetingResult(
        weights=x,
        risk=rc_result.risk,
        marginal_risk=rc_result.marginal_risk,
        risk_contribution=rc_result.risk_contribution,
        pct_risk_contribution=rc_result.pct_risk_contribution,
        converged=result.success,
    )


def mixture_compute_rb_var(
    params: MixtureParams,
    alpha: float,
    b: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    x_minus: float = 0.0,
    x_plus: float = 1.0,
) -> RiskBudgetingResult:
    """VaR risk budgeting portfolio under the mixture distribution: find
    weights x (in [x_minus, x_plus], summing to 1) whose VaR risk
    contributions best match the target budgets b.

    Original: mixture/mixture_compute_rb_var.m (algorithm 1 -- see module
    docstring for the unported algorithm 2 branch)
    """
    return _solve_mixture_rb(mixture_compute_rc_var, params, alpha, b, x0, x_minus, x_plus)


def mixture_compute_rb_es(
    params: MixtureParams,
    alpha: float,
    b: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    x_minus: float = 0.0,
    x_plus: float = 1.0,
) -> RiskBudgetingResult:
    """ES risk budgeting portfolio under the mixture distribution.

    Original: mixture/mixture_compute_rb_es.m (algorithm 1 -- see module
    docstring for the unported algorithm 2 branch)
    """
    return _solve_mixture_rb(mixture_compute_rc_es, params, alpha, b, x0, x_minus, x_plus)


@dataclass
class EMMixtureResult:
    params: MixtureParams
    log_l: float
    pi1_t: np.ndarray
    pi2_t: np.ndarray
    n_iters: int
    converged: bool


def estimate_em_mixture(
    data: np.ndarray,
    init: MixtureParams,
    estimate_mixing_weights: bool = True,
    tol: float = 1e-6,
    max_iters: int = 1000,
) -> EMMixtureResult:
    """Fit a 2-component Gaussian mixture to data via Expectation-Maximization.

    estimate_mixing_weights=True (default): re-estimate pi1/pi2 each
    iteration (standard EM). False: hold pi1/pi2 fixed at their initial
    values (e.g. when they're set exogenously, as in the jump-diffusion
    parameterization).

    Original: mixture/{estimate_em_mixture,logl_em_mixture}.m
    """
    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim == 1:
        y = data_arr[:, None]
    else:
        y = data_arr
    y = y[~np.isnan(y).any(axis=1)]
    n_obs = y.shape[0]

    pi1, pi2 = init.pi1, init.pi2
    mu1, mu2 = np.asarray(init.mu1, dtype=float), np.asarray(init.mu2, dtype=float)
    sigma1, sigma2 = np.asarray(init.sigma1, dtype=float), np.asarray(init.sigma2, dtype=float)

    converged = False
    n_iter = 0
    pi1_t = np.full(n_obs, pi1)
    pi2_t = np.full(n_obs, pi2)

    for n_iter in range(1, max_iters + 1):  # noqa: B007 (used after loop)
        old = (pi1, pi2, mu1.copy(), mu2.copy(), sigma1.copy(), sigma2.copy())

        pdf1 = multivariate_normal.pdf(y, mean=mu1, cov=sigma1)
        pdf2 = multivariate_normal.pdf(y, mean=mu2, cov=sigma2)
        sum1, sum2 = pi1 * pdf1, pi2 * pdf2

        pi1_t = sum1 / (sum1 + sum2)
        pi2_t = 1 - pi1_t

        if estimate_mixing_weights:
            pi1, pi2 = float(np.mean(pi1_t)), float(np.mean(pi2_t))

        mu1 = (pi1_t[:, None] * y).sum(axis=0) / pi1_t.sum()
        mu2 = (pi2_t[:, None] * y).sum(axis=0) / pi2_t.sum()

        y1c, y2c = y - mu1, y - mu2
        sigma1 = (pi1_t[:, None, None] * (y1c[:, :, None] * y1c[:, None, :])).sum(
            axis=0
        ) / pi1_t.sum()
        sigma2 = (pi2_t[:, None, None] * (y2c[:, :, None] * y2c[:, None, :])).sum(
            axis=0
        ) / pi2_t.sum()

        diff = np.concatenate(
            [
                [pi1 - old[0], pi2 - old[1]],
                (mu1 - old[2]).flatten(),
                (mu2 - old[3]).flatten(),
                (sigma1 - old[4]).flatten(),
                (sigma2 - old[5]).flatten(),
            ]
        )
        err = float(np.max(np.abs(diff)))
        if err <= tol:
            converged = True
            break

    log_l = float(
        np.sum(
            np.log(
                pi1 * multivariate_normal.pdf(y, mean=mu1, cov=sigma1)
                + pi2 * multivariate_normal.pdf(y, mean=mu2, cov=sigma2)
            )
        )
    )

    fitted = MixtureParams(pi1=pi1, mu1=mu1, sigma1=sigma1, pi2=pi2, mu2=mu2, sigma2=sigma2)
    return EMMixtureResult(
        params=fitted, log_l=log_l, pi1_t=pi1_t, pi2_t=pi2_t, n_iters=n_iter, converged=converged
    )
