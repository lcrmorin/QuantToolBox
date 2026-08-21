# `quanttoolbox.econometrics`

## `econometrics.estimation` (OLS/GMM/ML, Wald test)

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.sandbox.regression.gmm.GMM`, `statsmodels.base.model.GenericLikelihoodModel`, or the [`linearmodels`](https://github.com/bashtage/linearmodels) package (more modern GMM/IV support) are more mature for standard use cases. Keep this module for the explicit `theta = RR @ gamma + r` linear-restriction interface, which none of the alternatives expose as directly.

::: quanttoolbox.econometrics.estimation

### Examples

??? example "Beta-distribution MLE for LGD modeling — ects/ml3.py"
    ```python
    --8<-- "examples/ects/ml3.py"
    ```

??? example "Local Linear Trend model fit by time-domain ML, vs. Whittle — ects/kalman2c.py"
    ```python
    --8<-- "examples/ects/kalman2c.py"
    ```

??? example "ML recovery of all free state-space model parameters — ects/kalman3d.py"
    ```python
    --8<-- "examples/ects/kalman3d.py"
    ```

??? example "Numerical vs. analytical Jacobian/Hessian across covariance estimators — ects/ml4.py"
    ```python
    --8<-- "examples/ects/ml4.py"
    ```

??? example "OLS vs. Gaussian-MLE covariance: Hessian, OPG, and HC estimators — ects/ml2.py"
    ```python
    --8<-- "examples/ects/ml2.py"
    ```

??? example "OLS vs. Gaussian-MLE covariance: Hessian, OPG, and HC standard errors — stats/cov1.py"
    ```python
    --8<-- "examples/stats/cov1.py"
    ```

??? example "OLS, LAD, quantile, and SVM regression compared — svm/svm6.py"
    ```python
    --8<-- "examples/svm/svm6.py"
    ```

??? example "OLS, median, LAD, and Huber regression compared — ects/robust2.py"
    ```python
    --8<-- "examples/ects/robust2.py"
    ```

??? example "OLS, MLE, and GMM under an optional linear restriction — ects/gmm1.py"
    ```python
    --8<-- "examples/ects/gmm1.py"
    ```

??? example "OLS/median/LAD plus quantile regression via IRLS and exact LP — ects/robust3.py"
    ```python
    --8<-- "examples/ects/robust3.py"
    ```

??? example "OLS/SVM-LS and quantile/SVM-epsilon regression on synthetic data — svm/svm8.py"
    ```python
    --8<-- "examples/svm/svm8.py"
    ```

??? example "Time-domain ML of Kalman noise variances, two starting states — ects/kalman1b.py"
    ```python
    --8<-- "examples/ects/kalman1b.py"
    ```

??? example "Time-domain vs. frequency-domain (Whittle) Kalman ML — ects/kalman1c.py"
    ```python
    --8<-- "examples/ects/kalman1c.py"
    ```

??? example "Time-varying-coefficient model with variances estimated by ML — ects/kalman4b.py"
    ```python
    --8<-- "examples/ects/kalman4b.py"
    ```

??? example "VAR(0) equivalence to plain OLS — ects/varx3.py"
    ```python
    --8<-- "examples/ects/varx3.py"
    ```

??? example "Wald test for no Granger-causality between VAR variables — ects/varx1b.py"
    ```python
    --8<-- "examples/ects/varx1b.py"
    ```

??? example "Wald test for no instantaneous causality in a VAR — ects/varx1c.py"
    ```python
    --8<-- "examples/ects/varx1c.py"
    ```

## `econometrics.var`

!!! info "Python alternatives"
    **Switch** to `statsmodels.tsa.api.VAR` for the unrestricted case — comprehensive lag-order selection, impulse response functions, forecast error variance decomposition, forecasting. **Keep** `varx_estimate`'s linear-restriction support (`a_eq`/`b_eq`) — statsmodels' VAR doesn't support arbitrary parameter restrictions.

::: quanttoolbox.econometrics.var

### Examples

??? example "Larger restricted SUR system, two covariance variants compared — ects/varx4b.py"
    ```python
    --8<-- "examples/ects/varx4b.py"
    ```

??? example "Reduced form of a dynamic simultaneous-equations system — ects/varx2a.py"
    ```python
    --8<-- "examples/ects/varx2a.py"
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

??? example "Simultaneous-equations system: OLS, GLS, and LIML compared — ects/varx5a.py"
    ```python
    --8<-- "examples/ects/varx5a.py"
    ```

??? example "VAR lag-order selection — ects/varx1e.py"
    ```python
    --8<-- "examples/ects/varx1e.py"
    ```

??? example "VAR(0) equivalence to plain OLS — ects/varx3.py"
    ```python
    --8<-- "examples/ects/varx3.py"
    ```

??? example "VAR(2) on log-differenced macroeconomic data — ects/varx1a.py"
    ```python
    --8<-- "examples/ects/varx1a.py"
    ```

??? example "Wald test for no Granger-causality between VAR variables — ects/varx1b.py"
    ```python
    --8<-- "examples/ects/varx1b.py"
    ```

??? example "Wald test for no instantaneous causality in a VAR — ects/varx1c.py"
    ```python
    --8<-- "examples/ects/varx1c.py"
    ```

## `econometrics.kalman`

!!! info "Python alternatives"
    **Switch** to `statsmodels.tsa.statespace.MLEModel`/`kalman_filter` for anything beyond simple filtering — smoothing, built-in MLE fitting, diffuse initialization, a compiled Cython backend. Keep this module for simple, transparent filtering or minimal-dependency use.

::: quanttoolbox.econometrics.kalman

### Examples

??? example "Filtered state with a 95% confidence band — ects/kalman3c.py"
    ```python
    --8<-- "examples/ects/kalman3c.py"
    ```

??? example "General state-space filter from a steady-state start — ects/kalman3a.py"
    ```python
    --8<-- "examples/ects/kalman3a.py"
    ```

??? example "Local Linear Trend Kalman filter, fixed variance parameters — ects/kalman2a.py"
    ```python
    --8<-- "examples/ects/kalman2a.py"
    ```

??? example "Local Linear Trend model fit by time-domain ML, vs. Whittle — ects/kalman2c.py"
    ```python
    --8<-- "examples/ects/kalman2c.py"
    ```

??? example "Local Linear Trend model fit by Whittle (frequency-domain) ML — ects/kalman2b.py"
    ```python
    --8<-- "examples/ects/kalman2b.py"
    ```

??? example "Local-level Kalman filter (Harvey 1990) — ects/panel1.py"
    ```python
    --8<-- "examples/ects/panel1.py"
    ```

??? example "ML recovery of all free state-space model parameters — ects/kalman3d.py"
    ```python
    --8<-- "examples/ects/kalman3d.py"
    ```

??? example "Same state-space filter via the time-varying code path — ects/kalman3b.py"
    ```python
    --8<-- "examples/ects/kalman3b.py"
    ```

??? example "Time-domain ML of Kalman noise variances, two starting states — ects/kalman1b.py"
    ```python
    --8<-- "examples/ects/kalman1b.py"
    ```

??? example "Time-domain vs. frequency-domain (Whittle) Kalman ML — ects/kalman1c.py"
    ```python
    --8<-- "examples/ects/kalman1c.py"
    ```

??? example "Time-varying-beta model from a deliberately bad starting point — stats/kalman2.py"
    ```python
    --8<-- "examples/stats/kalman2.py"
    ```

??? example "Time-varying-beta regression via bounded maximum likelihood — stats/kalman1.py"
    ```python
    --8<-- "examples/stats/kalman1.py"
    ```

??? example "Time-varying-coefficient model with variances estimated by ML — ects/kalman4b.py"
    ```python
    --8<-- "examples/ects/kalman4b.py"
    ```

??? example "Time-varying-coefficient regression recovered by a Kalman filter — ects/kalman4a.py"
    ```python
    --8<-- "examples/ects/kalman4a.py"
    ```

## `econometrics.whittle`

!!! info "Python alternatives"
    **Keep** — Whittle (frequency-domain) estimation isn't implemented in statsmodels, `arch`, or other common packages. Genuinely fills a gap.

::: quanttoolbox.econometrics.whittle

### Examples

??? example "Local Linear Trend model fit by Whittle (frequency-domain) ML — ects/kalman2b.py"
    ```python
    --8<-- "examples/ects/kalman2b.py"
    ```

??? example "Raw periodogram of a 4-observation series — maths/pdgm1.py"
    ```python
    --8<-- "examples/maths/pdgm1.py"
    ```

??? example "Raw periodogram of an 8-observation series — maths/pdgm2.py"
    ```python
    --8<-- "examples/maths/pdgm2.py"
    ```

??? example "Time-domain vs. frequency-domain (Whittle) Kalman ML — ects/kalman1c.py"
    ```python
    --8<-- "examples/ects/kalman1c.py"
    ```

??? example "Whittle estimation with a custom Bloomfield spectral density — ects/whittle2.py"
    ```python
    --8<-- "examples/ects/whittle2.py"
    ```

??? example "Whittle local-level MLE on real macroeconomic data — ects/whittle1.py"
    ```python
    --8<-- "examples/ects/whittle1.py"
    ```

## `econometrics.tests` (ADF)

!!! info "Python alternatives"
    Already **switched** to `statsmodels.tsa.stattools.adfuller`. The [`arch`](https://github.com/bashtage/arch) package's `arch.unitroot` module has a wider family of unit-root tests (Phillips-Perron, DFGLS, KPSS, Zivot-Andrews) if more than ADF is ever needed.

::: quanttoolbox.econometrics.tests
