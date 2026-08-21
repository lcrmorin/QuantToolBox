# `quanttoolbox.dates`

!!! info "Python alternatives"
    - `convert.py`: pandas has no built-in Excel-serial-date conversion — **keep** this wrapper.
    - `rebalancing.py`: [`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) gives real exchange holiday calendars (ours is weekday-only) — worth wiring in as an optional calendar source for production use; keep our nearest-available-date snapping logic either way.

## `dates.convert`

::: quanttoolbox.dates.convert

### Examples

??? example "Flat transaction cost under a realistic rebalancing schedule — backtest/backtest5.py"
    ```python
    --8<-- "examples/backtest/backtest5.py"
    ```

??? example "Maximum drawdown in relative mode — backtest/mdd1.py"
    ```python
    --8<-- "examples/backtest/mdd1.py"
    ```

??? example "Per-asset bid/ask transaction costs vs. turnover — backtest/backtest4.py"
    ```python
    --8<-- "examples/backtest/backtest4.py"
    ```

??? example "Three rebalancing schedules on a small backtest — backtest/backtest2.py"
    ```python
    --8<-- "examples/backtest/backtest2.py"
    ```

## `dates.rebalancing`

::: quanttoolbox.dates.rebalancing

### Examples

??? example "Flat transaction cost under a realistic rebalancing schedule — backtest/backtest5.py"
    ```python
    --8<-- "examples/backtest/backtest5.py"
    ```

??? example "Maximum drawdown in relative mode — backtest/mdd1.py"
    ```python
    --8<-- "examples/backtest/mdd1.py"
    ```

??? example "Per-asset bid/ask transaction costs vs. turnover — backtest/backtest4.py"
    ```python
    --8<-- "examples/backtest/backtest4.py"
    ```

??? example "Three rebalancing schedules on a small backtest — backtest/backtest2.py"
    ```python
    --8<-- "examples/backtest/backtest2.py"
    ```
