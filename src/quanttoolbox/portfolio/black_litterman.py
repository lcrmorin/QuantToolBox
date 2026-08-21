"""Black-Litterman implied returns and posterior (view-updated) moments.

Ported from QuantToolBox/rpb/{implied_risk_premia,
compute_Black_Litterman_moments}.m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImpliedRiskPremia:
    pi: np.ndarray  # implied excess returns
    phi: float  # market price of risk
    gamma: float  # implied risk-aversion coefficient


def implied_risk_premia(
    x: np.ndarray, cov_matrix: np.ndarray, sharpe_ratio: float
) -> ImpliedRiskPremia:
    """Back out the implied excess-return vector (and risk-aversion
    parameters) consistent with a given market-cap-weighted portfolio x
    achieving Sharpe ratio sharpe_ratio -- the standard first step of the
    Black-Litterman framework's equilibrium prior.

    Original: rpb/implied_risk_premia.m
    """
    x = np.asarray(x, dtype=float).flatten()
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    sigma_x = float(np.sqrt(x @ cov_matrix @ x))
    phi = sharpe_ratio / sigma_x
    gamma = sigma_x / sharpe_ratio
    pi = sharpe_ratio * (cov_matrix @ x) / sigma_x

    return ImpliedRiskPremia(pi=pi, phi=phi, gamma=gamma)


@dataclass
class BlackLittermanMoments:
    mu_bar: np.ndarray  # posterior expected returns
    sigma_bar: np.ndarray  # posterior covariance


def black_litterman_moments(
    mu_tilde: np.ndarray,
    gamma_matrix: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
) -> BlackLittermanMoments:
    """Combine an equilibrium prior (mu_tilde, gamma_matrix) with investor
    views (P @ mu = Q, view uncertainty Omega) into posterior expected
    returns and covariance, via the standard Black-Litterman conditional
    normal update.

    Original: rpb/compute_Black_Litterman_moments.m

    Parameters
    ----------
    mu_tilde : (n,) prior (equilibrium) expected returns.
    gamma_matrix : (n, n) prior covariance of expected returns (often
        tau * Sigma, where Sigma is the asset covariance and tau is a
        small scalar).
    p : (k, n) view matrix, one row per view.
    q : (k,) view target values.
    omega : (k, k) view uncertainty covariance.
    """
    mu_tilde = np.asarray(mu_tilde, dtype=float).flatten()
    gamma_matrix = np.asarray(gamma_matrix, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float).flatten()
    omega = np.asarray(omega, dtype=float)

    pg = p @ gamma_matrix
    inner = np.linalg.inv(omega + pg @ p.T)

    mu_bar = mu_tilde + gamma_matrix @ p.T @ inner @ (q - p @ mu_tilde)
    sigma_bar = gamma_matrix - gamma_matrix @ p.T @ inner @ pg

    return BlackLittermanMoments(mu_bar=mu_bar, sigma_bar=sigma_bar)
