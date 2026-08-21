# `quanttoolbox.credit`

## `credit.structural`

!!! info "Python alternatives"
    **Hybrid**: `black_scholes`'s generalized cost-of-carry `b` parameterization has no single-function equivalent in `py_vollib`/`mibian` (both split by asset class instead) -- keep for that convenience, or call `scipy.stats.norm.cdf` directly for a one-off. The Merton (1974/1976), Black-Cox (1976), Blasberg (2024) extended-Merton, and Reinders et al. structural credit models are niche enough (each a specific published model, not a general option-pricing primitive) that no general credit-risk or derivatives-pricing library surveyed implements them as public utilities -- **keep**. `pd_merton_model`'s asset-value/volatility calibration uses `scipy.optimize.minimize(method="BFGS")` in place of MATLAB's `fminunc`, one line doing what a hand-rolled Newton loop would otherwise require.

::: quanttoolbox.credit.structural

## `credit.reduced_form`

!!! info "Python alternatives"
    **Keep** -- default-time survival/density/hazard functions implied by a continuous-time Markov generator matrix (via `scipy.linalg.expm`), and the (piecewise-)exponential default-time model (survival/CDF/PDF/quantile/simulation). No general-purpose equivalent found: `lifelines` and `scikit-survival` model *estimation* from observed survival data, not simulation/inversion from an assumed hazard specification given up front.

::: quanttoolbox.credit.reduced_form
