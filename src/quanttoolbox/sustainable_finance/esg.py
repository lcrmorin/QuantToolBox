"""ESG-tilted portfolio construction: the "implied ESG beta" a
minimum-variance investor effectively targets (Roncalli's ESG beta-star
model), the resulting minimum-variance-plus-ESG-tilt portfolio, and
Pedersen, Fitzgibbons & Pomorski (2021)'s ESG-efficient-frontier portfolio.

Ported from HSF toolbox `hsf/{compute_esg_beta_star,
compute_esg_minimum_variance,compute_pedersen_portfolio}.m`.

Translation notes:

- `hsf/cdp_filter.m` (CDP -- Carbon Disclosure Project -- data loading and
  regional/sectoral filtering, plus trend estimation) is **not** ported: it
  loads a specific `data/chap9_cdp3.mat` dataset this package does not
  ship, and its region/sector categories are hardcoded to that one
  textbook chapter's data -- a data-loading script tied to one dataset,
  not a general-purpose library function (same category as the untranslated
  `tools/` display helpers noted in docs/migration_map.md).
- `compute_esg_beta_star.m`'s local `e = ones(n,1)` is unrelated to
  `compute_esg_minimum_variance.m`'s `e` parameter (a universe-selection
  mask) despite the shared MATLAB variable name -- kept as an internal
  local (`ones_n`) here to avoid the naming collision.
- MATLAB's ``logical(e)`` (boolean-mask submatrix selection) is
  ``Sigma[mask][:, mask]`` in numpy, where `mask = e.astype(bool)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def esg_beta_star(
    beta: np.ndarray,
    sigma_m: float,
    beta_esg: np.ndarray,
    sigma_esg: float,
    sigma_tilde: np.ndarray,
) -> tuple[float, float]:
    """The "implied" market beta and ESG beta (``beta_star``,
    ``beta_esg_star``) at which a minimum-variance investor's residual risk
    is fully diversified, given per-asset market/ESG factor loadings
    (`beta`, `beta_esg`), factor volatilities (`sigma_m`, `sigma_esg`), and
    idiosyncratic volatilities (`sigma_tilde`).

    Original: hsf/compute_esg_beta_star.m
    """
    beta = np.asarray(beta, dtype=float)
    beta_esg = np.asarray(beta_esg, dtype=float)
    sigma_tilde = np.asarray(sigma_tilde, dtype=float)
    ones_n = np.ones_like(beta)

    sigma_m_sqr = sigma_m**2
    sigma_esg_sqr = sigma_esg**2
    sigma_m_esg_sqr = (sigma_m * sigma_esg) ** 2

    beta_tilde = beta / sigma_tilde**2
    varphi_m = beta @ beta_tilde

    beta_esg_tilde = beta_esg / sigma_tilde**2
    varphi_esg = beta_esg @ beta_esg_tilde

    varphi_m_esg = beta @ beta_esg_tilde

    omega0 = (
        1.0
        + sigma_m_sqr * varphi_m
        + sigma_esg_sqr * varphi_esg
        + sigma_m_esg_sqr * (varphi_m * varphi_esg - varphi_m_esg**2)
    )
    omega1 = varphi_esg * (beta_tilde @ ones_n) - varphi_m_esg * (beta_esg_tilde @ ones_n)
    omega1 = sigma_m_sqr * ((beta_tilde @ ones_n) + sigma_esg_sqr * omega1)
    omega2 = varphi_m * (beta_esg_tilde @ ones_n) - varphi_m_esg * (beta_tilde @ ones_n)
    omega2 = sigma_esg_sqr * ((beta_esg_tilde @ ones_n) + sigma_m_sqr * omega2)

    beta_star = omega0 / omega1
    beta_esg_star = omega0 / omega2
    return float(beta_star), float(beta_esg_star)


@dataclass
class EsgMinimumVarianceResult:
    """Minimum-variance-plus-ESG-tilt portfolio weights `x`, the implied
    covariance matrix `sigma`, its portfolio volatility `sigma_x`, the
    beta-star pair the tilt was built from, and (`sigma_tilde_matrix`,
    `x_tilde`) the plain (untilted) minimum-variance portfolio restricted
    to the selected universe, for comparison."""

    x: np.ndarray
    sigma: np.ndarray
    sigma_x: float
    beta_star: float
    beta_esg_star: float
    sigma_tilde_matrix: np.ndarray
    x_tilde: np.ndarray


def esg_minimum_variance(
    beta: np.ndarray,
    sigma_m: float,
    beta_esg: np.ndarray,
    sigma_esg: float,
    sigma_tilde: np.ndarray,
    e: np.ndarray | None = None,
) -> EsgMinimumVarianceResult:
    """The minimum-variance portfolio tilted toward the ESG beta-star of
    `esg_beta_star`, restricted to an optional universe-selection mask `e`
    (default: the full universe).

    Original: hsf/compute_esg_minimum_variance.m
    """
    beta = np.asarray(beta, dtype=float)
    beta_esg = np.asarray(beta_esg, dtype=float)
    sigma_tilde = np.asarray(sigma_tilde, dtype=float)
    n = beta.shape[0]

    d = np.diag(sigma_tilde**2)
    sigma_m_sqr = sigma_m**2
    sigma_esg_sqr = sigma_esg**2
    sigma = np.outer(beta, beta) * sigma_m_sqr + np.outer(beta_esg, beta_esg) * sigma_esg_sqr + d

    if e is None:
        e = np.ones(n)
    e = np.asarray(e, dtype=float)

    beta_masked = e * beta
    beta_esg_masked = e * beta_esg
    beta_star, beta_esg_star = esg_beta_star(
        beta_masked, sigma_m, beta_esg_masked, sigma_esg, sigma_tilde
    )

    x = 1.0 / sigma_tilde**2 * (1.0 - beta_masked / beta_star - beta_esg_masked / beta_esg_star)
    x = e * x
    x = x / np.sum(x)
    sigma_x = float(np.sqrt(x @ sigma @ x))

    mask = e.astype(bool)
    sigma_tilde_matrix = sigma[np.ix_(mask, mask)]
    n_tilde = sigma_tilde_matrix.shape[0]
    e_tilde = np.ones(n_tilde)
    inv_sigma_tilde = np.linalg.inv(sigma_tilde_matrix)
    x_tilde = (inv_sigma_tilde @ e_tilde) / (e_tilde @ inv_sigma_tilde @ e_tilde)

    return EsgMinimumVarianceResult(
        x=x,
        sigma=sigma,
        sigma_x=sigma_x,
        beta_star=beta_star,
        beta_esg_star=beta_esg_star,
        sigma_tilde_matrix=sigma_tilde_matrix,
        x_tilde=x_tilde,
    )


@dataclass
class PedersenPortfolioResult:
    """Pedersen-Fitzgibbons-Pomorski ESG-efficient portfolios: `w` (shape
    (n_assets, n_scenarios)) holds one portfolio per (`sigma_bar`, `s_bar`)
    scenario, with `w_r` its residual (cash/risk-free) weight and the rest
    the per-scenario risk/return/ESG-score/Sharpe-ratio diagnostics."""

    w: np.ndarray
    w_r: np.ndarray
    sigma_bar: np.ndarray
    s_bar: np.ndarray
    lambda1: np.ndarray
    lambda2: np.ndarray
    pi_w: np.ndarray
    sigma_w: np.ndarray
    s_w: np.ndarray
    sr_w: np.ndarray
    c_x_y: np.ndarray


def pedersen_portfolio(
    mu: np.ndarray,
    r: float,
    sigma: np.ndarray,
    s: np.ndarray,
    sigma_bar: np.ndarray | float,
    s_bar: np.ndarray | float,
) -> PedersenPortfolioResult:
    """Pedersen, Fitzgibbons & Pomorski (2021)'s ESG-efficient-frontier
    portfolio: mean-variance-optimal subject to a target volatility
    `sigma_bar` and a target portfolio ESG score `s_bar`, swept over one
    portfolio per (`sigma_bar`, `s_bar`) pair (broadcast to a common
    length).

    Original: hsf/compute_pedersen_portfolio.m
    """
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(s, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.shape[0]

    pi_ = mu - r
    inv_sigma = np.linalg.inv(sigma)
    ones_n = np.ones(n)

    c_1_pi = ones_n @ inv_sigma @ pi_
    c_1_s = ones_n @ inv_sigma @ s
    c_s_pi = s @ inv_sigma @ pi_
    c_s_s = s @ inv_sigma @ s
    c_1_1 = ones_n @ inv_sigma @ ones_n
    c_pi_pi = pi_ @ inv_sigma @ pi_
    c_x_y = np.array([c_1_pi, c_s_pi, c_s_s, c_1_s, c_1_1, c_pi_pi])

    sigma_bar_arr = np.atleast_1d(np.asarray(sigma_bar, dtype=float))
    s_bar_arr = np.atleast_1d(np.asarray(s_bar, dtype=float))
    n_iters = max(sigma_bar_arr.shape[0], s_bar_arr.shape[0])
    sigma_bar_arr = np.broadcast_to(sigma_bar_arr, (n_iters,))
    s_bar_arr = np.broadcast_to(s_bar_arr, (n_iters,))

    all_lambda1 = np.zeros(n_iters)
    all_lambda2 = np.zeros(n_iters)
    all_w = np.zeros((n, n_iters))
    all_pi_w = np.zeros(n_iters)
    all_sigma_w = np.zeros(n_iters)
    all_s_w = np.zeros(n_iters)
    all_sr_w = np.zeros((n_iters, 2))

    for it in range(n_iters):
        sb, tb = sigma_bar_arr[it], s_bar_arr[it]
        denom = c_s_s - 2 * c_1_s * tb + c_1_1 * tb**2
        lambda2 = (c_1_pi * tb - c_s_pi) / denom
        aux = c_pi_pi - (c_1_pi * tb - c_s_pi) ** 2 / denom
        lambda1 = -1.0 / (2 * sb) * np.sqrt(aux)
        w = (-1.0 / (2 * lambda1)) * inv_sigma @ (pi_ + lambda2 * (s - tb))

        all_lambda1[it] = lambda1
        all_lambda2[it] = lambda2
        all_w[:, it] = w
        all_pi_w[it] = w @ pi_
        all_sigma_w[it] = np.sqrt(w @ sigma @ w)
        all_s_w[it] = (w @ s) / np.sum(w)
        all_sr_w[it, 0] = (w @ pi_) / sb
        all_sr_w[it, 1] = np.sqrt(aux)

    w_r = 1.0 - np.sum(all_w, axis=0)

    return PedersenPortfolioResult(
        w=all_w,
        w_r=w_r,
        sigma_bar=sigma_bar_arr,
        s_bar=s_bar_arr,
        lambda1=all_lambda1,
        lambda2=all_lambda2,
        pi_w=all_pi_w,
        sigma_w=all_sigma_w,
        s_w=all_s_w,
        sr_w=all_sr_w,
        c_x_y=c_x_y,
    )
