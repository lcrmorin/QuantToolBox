# `quanttoolbox.bond`

## `bond.pricing`

!!! info "Python alternatives"
    **Hybrid**: [`QuantLib-Python`](https://github.com/lballabio/QuantLib-SWIG) has far more complete bond conventions (day counts, calendars, callable/amortizing structures) than this flat-rate, cash-flow-list version — reach for it once real-world conventions matter. **Keep** this module for quick, dependency-free pricing/YTM against an explicit cash-flow schedule, and for `bond_portfolio_quadratic_form(_vs_benchmark)`: sector-level MD/DTS-targeting quadratic risk forms with no equivalent found elsewhere. See [Library alternatives](../library_alternatives.md) for the full reasoning.

::: quanttoolbox.bond.pricing
