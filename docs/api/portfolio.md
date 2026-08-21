# `quanttoolbox.portfolio`

## `portfolio.risk_budgeting`

!!! info "Python alternatives"
    **Keep** — the single largest and most-tested piece of custom work in the whole port (75+ original MATLAB files consolidated). [`riskparityportfolio`](https://github.com/mirca/riskparityportfolio) covers basic risk parity but not the box-constrained/general-linear-constrained/VaR-ES/target-matching breadth here. Genuinely the strongest "keep" case in the whole library.

::: quanttoolbox.portfolio.risk_budgeting

### Examples

??? example "rpb/test_box1.py"
    ```python
    --8<-- "examples/rpb/test_box1.py"
    ```

??? example "rpb/test_erc2.py"
    ```python
    --8<-- "examples/rpb/test_erc2.py"
    ```

??? example "rpb/test_erc3.py"
    ```python
    --8<-- "examples/rpb/test_erc3.py"
    ```

## `portfolio.mean_variance`, `portfolio.black_litterman`, `portfolio.tracking_error`

!!! info "Python alternatives"
    **Hybrid**: [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt) is mature and actively maintained, covering mean-variance, Black-Litterman, CVaR, and discrete allocation — genuinely worth using directly for standard portfolio construction. Keep these modules for tighter integration with the rest of this codebase (shared `solve_qp`, direct ridge/lasso penalty composability).

::: quanttoolbox.portfolio.mean_variance

::: quanttoolbox.portfolio.black_litterman

::: quanttoolbox.portfolio.tracking_error

### Examples

??? example "rpb/test_bl2.py"
    ```python
    --8<-- "examples/rpb/test_bl2.py"
    ```

??? example "rpb/test_bl3.py"
    ```python
    --8<-- "examples/rpb/test_bl3.py"
    ```

??? example "rpb/test_bl4.py"
    ```python
    --8<-- "examples/rpb/test_bl4.py"
    ```

??? example "rpb/test_bl5.py"
    ```python
    --8<-- "examples/rpb/test_bl5.py"
    ```

??? example "rpb/test_bl6.py"
    ```python
    --8<-- "examples/rpb/test_bl6.py"
    ```

??? example "rpb/test_lasso1.py"
    ```python
    --8<-- "examples/rpb/test_lasso1.py"
    ```

??? example "rpb/test_lasso2.py"
    ```python
    --8<-- "examples/rpb/test_lasso2.py"
    ```

??? example "rpb/test_lasso3.py"
    ```python
    --8<-- "examples/rpb/test_lasso3.py"
    ```

??? example "rpb/test_lasso5.py"
    ```python
    --8<-- "examples/rpb/test_lasso5.py"
    ```

??? example "rpb/test_minvar2.py"
    ```python
    --8<-- "examples/rpb/test_minvar2.py"
    ```

??? example "rpb/test_mvo2.py"
    ```python
    --8<-- "examples/rpb/test_mvo2.py"
    ```

??? example "rpb/test_mvo3.py"
    ```python
    --8<-- "examples/rpb/test_mvo3.py"
    ```

## `portfolio.erc_mdp`

Thin re-export convenience module — see `risk_budgeting` (ERC) and
`mean_variance` (MDP) for the actual implementations and alternatives
discussion.

::: quanttoolbox.portfolio.erc_mdp
