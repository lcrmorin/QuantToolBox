# `quanttoolbox.optim`

## `optim.proximal` / `optim.projection`

!!! info "Python alternatives"
    **Keep** — [`pyproximal`](https://github.com/PyLops/pyproximal) covers some overlapping norm/constraint proximal operators, but is oriented toward signal-processing/inverse-problems use cases, not portfolio constraints (turnover, combined Dykstra-projected linear+box systems). No strong off-the-shelf equivalent for this exact operator set.

::: quanttoolbox.optim.proximal

::: quanttoolbox.optim.projection

### Examples

??? example "Proximal projection under bounds, (in)equalities, and combined constraints — optim/proximal1.py"
    ```python
    --8<-- "examples/optim/proximal1.py"
    ```

??? example "Proximal-L1 vs. two L1-ball-projection turnover algorithms — optim/prox_turnover1.py"
    ```python
    --8<-- "examples/optim/prox_turnover1.py"
    ```

## `optim.quadprog` (`solve_qp`)

!!! info "Python alternatives"
    **Hybrid**: for hot loops calling `solve_qp` many times (e.g. risk-budgeting's inner ADMM loop), calling `qpsolvers.solve_qp` directly — bypassing cvxpy's DSL-parsing overhead — would likely be faster. Worth profiling if performance matters; keep the cvxpy-based version for its more expressive constraint composition (ridge/lasso penalties, arbitrary constraints).

::: quanttoolbox.optim.quadprog

### Examples

??? example "Mean-variance optimization plus ridge/lasso-penalized portfolios — rpb/test_lasso1.py"
    ```python
    --8<-- "examples/rpb/test_lasso1.py"
    ```

??? example "Mixed ridge+lasso penalties toward two different target vectors — rpb/test_lasso5.py"
    ```python
    --8<-- "examples/rpb/test_lasso5.py"
    ```

??? example "Ridge, lasso, and mixed-norm penalized portfolios — rpb/test_lasso3.py"
    ```python
    --8<-- "examples/rpb/test_lasso3.py"
    ```

## `optim.bisection`

!!! info "Python alternatives"
    **Hybrid**: `scipy.optimize.brentq`/`bisect` converge faster (superlinear) for scalar root-finding and are compiled. **Keep** our vectorized (array-broadcast, many-roots-at-once) version — scipy's root finders are scalar-only.

::: quanttoolbox.optim.bisection

### Examples

??? example "Explicit-to-implicit conversion for three simultaneous restrictions — optim/explicit3.py"
    ```python
    --8<-- "examples/optim/explicit3.py"
    ```

??? example "Explicit/implicit constraint round-trip, plus a design() demo — optim/explicit2.py"
    ```python
    --8<-- "examples/optim/explicit2.py"
    ```
