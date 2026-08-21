# `quanttoolbox.portfolio`

## `portfolio.risk_budgeting`

!!! info "Python alternatives"
    **Keep** — the single largest and most-tested piece of custom work in the whole port (75+ original MATLAB files consolidated). [`riskparityportfolio`](https://github.com/mirca/riskparityportfolio) covers basic risk parity but not the box-constrained/general-linear-constrained/VaR-ES/target-matching breadth here. Genuinely the strongest "keep" case in the whole library.

::: quanttoolbox.portfolio.risk_budgeting

## `portfolio.mean_variance`, `portfolio.black_litterman`, `portfolio.tracking_error`

!!! info "Python alternatives"
    **Hybrid**: [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt) is mature and actively maintained, covering mean-variance, Black-Litterman, CVaR, and discrete allocation — genuinely worth using directly for standard portfolio construction. Keep these modules for tighter integration with the rest of this codebase (shared `solve_qp`, direct ridge/lasso penalty composability).

::: quanttoolbox.portfolio.mean_variance

::: quanttoolbox.portfolio.black_litterman

::: quanttoolbox.portfolio.tracking_error

## `portfolio.erc_mdp`

Thin re-export convenience module — see `risk_budgeting` (ERC) and
`mean_variance` (MDP) for the actual implementations and alternatives
discussion.

::: quanttoolbox.portfolio.erc_mdp
