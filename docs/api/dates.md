# `quanttoolbox.dates`

!!! info "Python alternatives"
    - `convert.py`: pandas has no built-in Excel-serial-date conversion — **keep** this wrapper.
    - `rebalancing.py`: [`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) gives real exchange holiday calendars (ours is weekday-only) — worth wiring in as an optional calendar source for production use; keep our nearest-available-date snapping logic either way.

## `dates.convert`

::: quanttoolbox.dates.convert

## `dates.rebalancing`

::: quanttoolbox.dates.rebalancing
