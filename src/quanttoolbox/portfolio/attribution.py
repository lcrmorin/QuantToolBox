"""CAPM-style beta/priced-risk/alpha attribution of every asset against a
given portfolio used as the pricing benchmark (Pastor & Pedersen (2022)-
style decomposition, as used for ESG-tilted-portfolio pricing).

Not ported from the MATLAB HSF toolbox -- no `.m` file in `hfs-archive`
implements this. It's the standard CAPM beta/priced-risk/residual
decomposition against an arbitrary reference portfolio (rather than the
market portfolio specifically), matching the Python formula already used
in this book's own notebooks (HSF-Notebooks chapter 3b).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CapmAttributionResult:
    """CAPM beta/priced-risk/alpha decomposition of every asset against a
    reference portfolio `x`.

    - `mu_x`/`sigma_x`: the reference portfolio's own expected return and
      volatility.
    - `beta_x`: each asset's beta against `x` (``Cov(asset, x) /
      Var(x)``).
    - `pi_x`: each asset's priced (beta-compensated) excess-return
      component, ``beta_x * (mu_x - r)``.
    - `alpha_x`: each asset's residual excess return not explained by its
      beta against `x`, ``(mu - r) - pi_x``.
    """

    mu_x: float
    sigma_x: float
    beta_x: np.ndarray
    pi_x: np.ndarray
    alpha_x: np.ndarray


def beta_pi_alpha(
    x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, r: float
) -> CapmAttributionResult:
    """CAPM beta/priced-risk/alpha decomposition of every asset (expected
    returns `mu`, covariance matrix `sigma`, risk-free rate `r`) against
    the reference portfolio `x`.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mu_x = float(x @ mu)
    sigma_x = float(np.sqrt(x @ sigma @ x))
    beta_x = (sigma @ x) / sigma_x**2
    pi_x = beta_x * (mu_x - r)
    alpha_x = (mu - r) - pi_x

    return CapmAttributionResult(
        mu_x=mu_x, sigma_x=sigma_x, beta_x=beta_x, pi_x=pi_x, alpha_x=alpha_x
    )
