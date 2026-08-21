"""Convenience re-exports for Equal Risk Contribution and Most Diversified
Portfolio construction.

Ported from QuantToolBox/rpb/compute_erc_portfolio.m,
QuantToolBox/mloapa/compute_{ERC_ADMM,ERC_CCD,MDP_ADMM}.m.

These are both already implemented in full elsewhere in this package --
ERC in ``portfolio.risk_budgeting`` (it's the b=1/n special case of risk
budgeting) and MDP in ``portfolio.mean_variance`` (it needs its own
nonlinear solve, unrelated to risk budgeting's machinery). This module
just re-exports both under the names matching the original MATLAB
toolbox's dedicated ``rpb``/``mloapa`` entry points, so callers used to
those names don't need to know the underlying module split.
"""

from __future__ import annotations

from quanttoolbox.portfolio.mean_variance import MDPResult, mdp_portfolio
from quanttoolbox.portfolio.risk_budgeting import RiskBudgetingResult, erc_portfolio

__all__ = ["MDPResult", "RiskBudgetingResult", "erc_portfolio", "mdp_portfolio"]
