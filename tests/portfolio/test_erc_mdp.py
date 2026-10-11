"""Tests for quanttoolbox.portfolio.erc_mdp (re-export convenience module)."""

from quanttoolbox.portfolio.erc_mdp import erc_portfolio, mdp_portfolio
from quanttoolbox.portfolio.mean_variance import mdp_portfolio as mdp_direct
from quanttoolbox.portfolio.risk_budgeting import erc_portfolio as erc_direct


def test_erc_mdp_reexports_match_direct_imports():
    assert erc_portfolio is erc_direct
    assert mdp_portfolio is mdp_direct
