"""Convenience exports for equal-risk-contribution and diversified portfolios.

``erc_portfolio`` is implemented in ``portfolio.risk_budgeting`` as the
equal-budget case; ``mdp_portfolio`` is implemented in
``portfolio.mean_variance`` using nonlinear optimization.

Original entry points: QuantToolBox/rpb/compute_erc_portfolio.m and
QuantToolBox/mloapa/compute_{ERC_ADMM,ERC_CCD,MDP_ADMM}.m.
"""

from __future__ import annotations

from quanttoolbox.portfolio.mean_variance import MDPResult, mdp_portfolio
from quanttoolbox.portfolio.risk_budgeting import RiskBudgetingResult, erc_portfolio

__all__ = ["MDPResult", "RiskBudgetingResult", "erc_portfolio", "mdp_portfolio"]
