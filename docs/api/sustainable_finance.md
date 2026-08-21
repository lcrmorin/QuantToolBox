# `quanttoolbox.sustainable_finance`

## `sustainable_finance.risk`

!!! info "Python alternatives"
    **Keep** — sector-decomposed quadratic-form risk building blocks (`quadratic_form`, `quadratic_form_risk`) and portfolio/sector modified-duration & DTS aggregation (`bond_portfolio_metrics`), generic to sector-based portfolio risk models. No general-purpose equivalent found; see [Library alternatives](../library_alternatives.md) for the full reasoning.

::: quanttoolbox.sustainable_finance.risk

## `sustainable_finance.carbon`

!!! info "Python alternatives"
    **Keep** — closed-form cumulative-emissions ("carbon budget") integrals under linear, compound-decline, piecewise-linear, and GDP-adjusted trajectories. No general-purpose equivalent found; the piecewise closed-form sum and each other formula are cross-checked against `scipy.integrate.quad` in this module's tests. Two near-duplicate originals (`carbon_budget_linear.m`/`carbon_budget_linear_trend.m`) were merged, and one (`carbon_budget_linear_reduction.m`) was recognized as a strict special case of `carbon_budget_Reduction.m` and not ported separately — see the module docstring.

::: quanttoolbox.sustainable_finance.carbon

## `sustainable_finance.esg`

!!! info "Python alternatives"
    **Keep** — Roncalli's implied-ESG-beta minimum-variance tilt (`esg_beta_star`, `esg_minimum_variance`) and Pedersen-Fitzgibbons-Pomorski (2021)'s ESG-efficient-frontier portfolio (`pedersen_portfolio`). No general-purpose equivalent found. `hsf/cdp_filter.m` (a CDP-dataset-specific data-loading/filtering script) was not ported — see the module docstring.

::: quanttoolbox.sustainable_finance.esg

## `sustainable_finance.climate`

!!! info "Python alternatives"
    **Keep** — the DICE (Dynamic Integrated Climate-Economy) model's carbon-cycle/temperature state-transition matrices and forward simulation. No general-purpose equivalent found; physical constants are Nordhaus's DICE-2016 calibration, hardcoded as in the original.

::: quanttoolbox.sustainable_finance.climate

## `sustainable_finance.ecology`

!!! info "Python alternatives"
    **Keep** — species-area/endemics-area relationships, species-abundance-distribution histogram binning (including Preston's log2 "octave" classes), and Hurlbert's rarefaction estimator. No general-purpose equivalent found (these are ecology-specific biodiversity measures, not general statistics).

::: quanttoolbox.sustainable_finance.ecology

## `sustainable_finance.entropy`

!!! info "Python alternatives"
    **Keep** — Shannon entropy/mutual-information decomposition (`shannon_entropy`, `shannon_entropy_markov_chain`) and the Israel-Rosenthal-Wei (2001) Markov-generator-matrix repair technique (`estimate_markov_generator`), used here for rating-migration-generator estimation. No general-purpose equivalent found.

::: quanttoolbox.sustainable_finance.entropy
