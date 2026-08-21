# `quanttoolbox.linalg`

!!! info "Python alternatives"
    No standalone public utility module does this. `vec`/`vech`/`xpnd`/commutation/duplication/elimination matrices show up as private internals scattered inside packages like `linearmodels`, but nothing exposes them cleanly. **Keep** — genuinely fills a gap.

## `linalg.special_matrices`

::: quanttoolbox.linalg.special_matrices

### Examples

??? example "Black-Litterman view sensitivity across five scenarios — rpb/test_bl2.py"
    ```python
    --8<-- "examples/rpb/test_bl2.py"
    ```

??? example "Elimination/duplication/commutation matrix identities across sizes — matrix/matrix2.py"
    ```python
    --8<-- "examples/matrix/matrix2.py"
    ```

??? example "Equal risk contribution and box-constrained risk budgeting — rpb/test_box1.py"
    ```python
    --8<-- "examples/rpb/test_box1.py"
    ```

??? example "Explicit/implicit constraint round-trip, plus a design() demo — optim/explicit2.py"
    ```python
    --8<-- "examples/optim/explicit2.py"
    ```

??? example "Larger restricted SUR system, two covariance variants compared — ects/varx4b.py"
    ```python
    --8<-- "examples/ects/varx4b.py"
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

??? example "Principal component analysis of a 3-asset correlation matrix — stats/pca1.py"
    ```python
    --8<-- "examples/stats/pca1.py"
    ```

??? example "Restricted seemingly-unrelated-regressions system, two-step GLS — ects/varx4a.py"
    ```python
    --8<-- "examples/ects/varx4a.py"
    ```

??? example "Restricted VAR via concentrated ML, vs. restricted least squares — ects/varx2c.py"
    ```python
    --8<-- "examples/ects/varx2c.py"
    ```

??? example "Restricted VAR(2) via least squares and concentrated ML — ects/varx1d.py"
    ```python
    --8<-- "examples/ects/varx1d.py"
    ```

??? example "Restricted-coefficient VAR via restricted least squares — ects/varx2b.py"
    ```python
    --8<-- "examples/ects/varx2b.py"
    ```

??? example "Ridge, lasso, and mixed-norm penalized portfolios — rpb/test_lasso3.py"
    ```python
    --8<-- "examples/rpb/test_lasso3.py"
    ```

??? example "Risk budgeting toward unequal target budgets — rpb/test_erc3.py"
    ```python
    --8<-- "examples/rpb/test_erc3.py"
    ```

??? example "Risk contribution decomposition and equal-budget risk budgeting — rpb/test_erc2.py"
    ```python
    --8<-- "examples/rpb/test_erc2.py"
    ```

??? example "Simultaneous-equations system: OLS, GLS, and LIML compared — ects/varx5a.py"
    ```python
    --8<-- "examples/ects/varx5a.py"
    ```

??? example "vech/xpnd (both orderings) and reshaper/reshapec compared — matrix/reshape1.py"
    ```python
    --8<-- "examples/matrix/reshape1.py"
    ```
