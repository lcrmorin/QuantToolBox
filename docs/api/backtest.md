# `quanttoolbox.backtest`

!!! info "Python alternatives"
    Full backtesting frameworks like [`vectorbt`](https://github.com/polakowo/vectorbt), `bt`, `backtrader`, and `zipline` exist, but are a different paradigm (event-driven or much heavier) than this module's lightweight, transparent, vectorized style. `vectorbt` specifically is Numba-accelerated and worth knowing about if performance on very large universes becomes a bottleneck — but it's not a drop-in replacement. **Keep** this module for its scope.

## `backtest.returns`

::: quanttoolbox.backtest.returns

### Examples

??? example "backtest/backtest2.py"
    ```python
    --8<-- "examples/backtest/backtest2.py"
    ```

??? example "backtest/backtest4.py"
    ```python
    --8<-- "examples/backtest/backtest4.py"
    ```

??? example "backtest/backtest5.py"
    ```python
    --8<-- "examples/backtest/backtest5.py"
    ```

??? example "backtest/mdd1.py"
    ```python
    --8<-- "examples/backtest/mdd1.py"
    ```

??? example "backtest/unfunded1.py"
    ```python
    --8<-- "examples/backtest/unfunded1.py"
    ```

??? example "backtest/unfunded2.py"
    ```python
    --8<-- "examples/backtest/unfunded2.py"
    ```

## `backtest.stats`

::: quanttoolbox.backtest.stats

### Examples

??? example "backtest/backtest4.py"
    ```python
    --8<-- "examples/backtest/backtest4.py"
    ```

??? example "backtest/mdd1.py"
    ```python
    --8<-- "examples/backtest/mdd1.py"
    ```

## `backtest.reporting`

::: quanttoolbox.backtest.reporting

### Examples

??? example "backtest/backtest2.py"
    ```python
    --8<-- "examples/backtest/backtest2.py"
    ```

??? example "backtest/backtest3.py"
    ```python
    --8<-- "examples/backtest/backtest3.py"
    ```

??? example "backtest/backtest4.py"
    ```python
    --8<-- "examples/backtest/backtest4.py"
    ```

??? example "backtest/backtest5.py"
    ```python
    --8<-- "examples/backtest/backtest5.py"
    ```

??? example "backtest/unfunded1.py"
    ```python
    --8<-- "examples/backtest/unfunded1.py"
    ```

??? example "backtest/unfunded2.py"
    ```python
    --8<-- "examples/backtest/unfunded2.py"
    ```
