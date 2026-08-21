# `quanttoolbox.portfolio`

## `portfolio.risk_budgeting`

!!! info "Python alternatives"
    **Keep** — the single largest and most-tested piece of custom work in the whole port (75+ original MATLAB files consolidated). [`riskparityportfolio`](https://github.com/mirca/riskparityportfolio) covers basic risk parity but not the box-constrained/general-linear-constrained/VaR-ES/target-matching breadth here. Genuinely the strongest "keep" case in the whole library.

::: quanttoolbox.portfolio.risk_budgeting

### Examples

??? example "Equal risk contribution and box-constrained risk budgeting — rpb/test_box1.py"
    ```python
    --8<-- "examples/rpb/test_box1.py"
    ```

??? example "Risk budgeting toward unequal target budgets — rpb/test_erc3.py"
    ```python
    --8<-- "examples/rpb/test_erc3.py"
    ```

??? example "Risk contribution decomposition and equal-budget risk budgeting — rpb/test_erc2.py"
    ```python
    --8<-- "examples/rpb/test_erc2.py"
    ```

## `portfolio.mean_variance`, `portfolio.black_litterman`, `portfolio.tracking_error`

!!! info "Python alternatives"
    **Hybrid**: [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt) is mature and actively maintained, covering mean-variance, Black-Litterman, CVaR, and discrete allocation — genuinely worth using directly for standard portfolio construction. Keep these modules for tighter integration with the rest of this codebase (shared `solve_qp`, direct ridge/lasso penalty composability).

::: quanttoolbox.portfolio.mean_variance

::: quanttoolbox.portfolio.black_litterman

::: quanttoolbox.portfolio.tracking_error

### Examples

??? example "Black-Litterman view matched to six tracking-error targets — rpb/test_bl4.py"
    ```python
    --8<-- "examples/rpb/test_bl4.py"
    ```

??? example "Black-Litterman view sensitivity across five scenarios — rpb/test_bl2.py"
    ```python
    --8<-- "examples/rpb/test_bl2.py"
    ```

??? example "Black-Litterman views matched to a benchmark's volatility — rpb/test_bl3.py"
    ```python
    --8<-- "examples/rpb/test_bl3.py"
    ```

??? example "Full-view Black-Litterman with tracking-error target-matching — rpb/test_bl5.py"
    ```python
    --8<-- "examples/rpb/test_bl5.py"
    ```

??? example "Mean-variance frontier: risk-aversion, return, and volatility targets — rpb/test_mvo2.py"
    ```python
    --8<-- "examples/rpb/test_mvo2.py"
    ```

??? example "Mean-variance optimization plus ridge/lasso-penalized portfolios — rpb/test_lasso1.py"
    ```python
    --8<-- "examples/rpb/test_lasso1.py"
    ```

??? example "Minimum-variance portfolio under general linear constraints — rpb/test_minvar2.py"
    ```python
    --8<-- "examples/rpb/test_minvar2.py"
    ```

??? example "Mixed ridge+lasso penalties toward two different target vectors — rpb/test_lasso5.py"
    ```python
    --8<-- "examples/rpb/test_lasso5.py"
    ```

??? example "Raw QP solve vs. te_portfolio, plus a gamma-recovery check — rpb/test_bl6.py"
    ```python
    --8<-- "examples/rpb/test_bl6.py"
    ```

??? example "Ridge, lasso, and mixed-norm penalized portfolios — rpb/test_lasso3.py"
    ```python
    --8<-- "examples/rpb/test_lasso3.py"
    ```

??? example "Ridge/lasso penalty sweep, points from a 250-point scan — rpb/test_lasso2.py"
    ```python
    --8<-- "examples/rpb/test_lasso2.py"
    ```

??? example "Volatility-target mean-variance under three weight-bound configurations — rpb/test_mvo3.py"
    ```python
    --8<-- "examples/rpb/test_mvo3.py"
    ```

## `portfolio.erc_mdp`

Thin re-export convenience module — see `risk_budgeting` (ERC) and
`mean_variance` (MDP) for the actual implementations and alternatives
discussion.

::: quanttoolbox.portfolio.erc_mdp
