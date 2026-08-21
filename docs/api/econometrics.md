# `quanttoolbox.econometrics`

## `econometrics.estimation` (OLS/GMM/ML, Wald test)

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.sandbox.regression.gmm.GMM`, `statsmodels.base.model.GenericLikelihoodModel`, or the [`linearmodels`](https://github.com/bashtage/linearmodels) package (more modern GMM/IV support) are more mature for standard use cases. Keep this module for the explicit `theta = RR @ gamma + r` linear-restriction interface, which none of the alternatives expose as directly.

::: quanttoolbox.econometrics.estimation

## `econometrics.var`

!!! info "Python alternatives"
    **Switch** to `statsmodels.tsa.api.VAR` for the unrestricted case — comprehensive lag-order selection, impulse response functions, forecast error variance decomposition, forecasting. **Keep** `varx_estimate`'s linear-restriction support (`a_eq`/`b_eq`) — statsmodels' VAR doesn't support arbitrary parameter restrictions.

::: quanttoolbox.econometrics.var

## `econometrics.kalman`

!!! info "Python alternatives"
    **Switch** to `statsmodels.tsa.statespace.MLEModel`/`kalman_filter` for anything beyond simple filtering — smoothing, built-in MLE fitting, diffuse initialization, a compiled Cython backend. Keep this module for simple, transparent filtering or minimal-dependency use.

::: quanttoolbox.econometrics.kalman

## `econometrics.whittle`

!!! info "Python alternatives"
    **Keep** — Whittle (frequency-domain) estimation isn't implemented in statsmodels, `arch`, or other common packages. Genuinely fills a gap.

::: quanttoolbox.econometrics.whittle

## `econometrics.tests` (ADF)

!!! info "Python alternatives"
    Already **switched** to `statsmodels.tsa.stattools.adfuller`. The [`arch`](https://github.com/bashtage/arch) package's `arch.unitroot` module has a wider family of unit-root tests (Phillips-Perron, DFGLS, KPSS, Zivot-Andrews) if more than ADF is ever needed.

::: quanttoolbox.econometrics.tests
