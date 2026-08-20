"""Tests for quanttoolbox.portfolio.erc_mdp (re-export convenience module)."""

import numpy as np

from quanttoolbox.portfolio.erc_mdp import erc_portfolio, mdp_portfolio
from quanttoolbox.portfolio.mean_variance import mdp_portfolio as mdp_direct
from quanttoolbox.portfolio.risk_budgeting import erc_portfolio as erc_direct


def test_erc_mdp_reexports_match_direct_imports(rng):
    n = 4
    a = rng.standard_normal((100, n))
    cov = a.T @ a / 100 + 0.01 * np.eye(n)

    erc_result_a = erc_portfolio(cov)
    erc_result_b = erc_direct(cov)
    assert np.allclose(erc_result_a.weights, erc_result_b.weights)

    mdp_result_a = mdp_portfolio(cov)
    mdp_result_b = mdp_direct(cov)
    assert np.allclose(mdp_result_a.weights, mdp_result_b.weights)
