# `quanttoolbox.econometrics`

## `econometrics.estimation` (OLS/GMM/ML, Wald test)

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.sandbox.regression.gmm.GMM`, `statsmodels.base.model.GenericLikelihoodModel`, or the [`linearmodels`](https://github.com/bashtage/linearmodels) package (more modern GMM/IV support) are more mature for standard use cases. Keep this module for the explicit `theta = RR @ gamma + r` linear-restriction interface, which none of the alternatives expose as directly.

::: quanttoolbox.econometrics.estimation

### Examples

??? example "ects/gmm1.py"
    ```python
    --8<-- "examples/ects/gmm1.py"
    ```

??? example "ects/kalman1b.py"
    ```python
    --8<-- "examples/ects/kalman1b.py"
    ```

??? example "ects/kalman1c.py"
    ```python
    --8<-- "examples/ects/kalman1c.py"
    ```

??? example "ects/kalman2c.py"
    ```python
    --8<-- "examples/ects/kalman2c.py"
    ```

??? example "ects/kalman3d.py"
    ```python
    --8<-- "examples/ects/kalman3d.py"
    ```

??? example "ects/kalman4b.py"
    ```python
    --8<-- "examples/ects/kalman4b.py"
    ```

??? example "ects/ml2.py"
    ```python
    --8<-- "examples/ects/ml2.py"
    ```

??? example "ects/ml3.py"
    ```python
    --8<-- "examples/ects/ml3.py"
    ```

??? example "ects/ml4.py"
    ```python
    --8<-- "examples/ects/ml4.py"
    ```

??? example "ects/robust2.py"
    ```python
    --8<-- "examples/ects/robust2.py"
    ```

??? example "ects/robust3.py"
    ```python
    --8<-- "examples/ects/robust3.py"
    ```

??? example "ects/varx1b.py"
    ```python
    --8<-- "examples/ects/varx1b.py"
    ```

??? example "ects/varx1c.py"
    ```python
    --8<-- "examples/ects/varx1c.py"
    ```

??? example "ects/varx3.py"
    ```python
    --8<-- "examples/ects/varx3.py"
    ```

??? example "stats/cov1.py"
    ```python
    --8<-- "examples/stats/cov1.py"
    ```

??? example "svm/svm6.py"
    ```python
    --8<-- "examples/svm/svm6.py"
    ```

??? example "svm/svm8.py"
    ```python
    --8<-- "examples/svm/svm8.py"
    ```

## `econometrics.var`

!!! info "Python alternatives"
    **Switch** to `statsmodels.tsa.api.VAR` for the unrestricted case — comprehensive lag-order selection, impulse response functions, forecast error variance decomposition, forecasting. **Keep** `varx_estimate`'s linear-restriction support (`a_eq`/`b_eq`) — statsmodels' VAR doesn't support arbitrary parameter restrictions.

::: quanttoolbox.econometrics.var

### Examples

??? example "ects/varx1a.py"
    ```python
    --8<-- "examples/ects/varx1a.py"
    ```

??? example "ects/varx1b.py"
    ```python
    --8<-- "examples/ects/varx1b.py"
    ```

??? example "ects/varx1c.py"
    ```python
    --8<-- "examples/ects/varx1c.py"
    ```

??? example "ects/varx1d.py"
    ```python
    --8<-- "examples/ects/varx1d.py"
    ```

??? example "ects/varx1e.py"
    ```python
    --8<-- "examples/ects/varx1e.py"
    ```

??? example "ects/varx2a.py"
    ```python
    --8<-- "examples/ects/varx2a.py"
    ```

??? example "ects/varx2b.py"
    ```python
    --8<-- "examples/ects/varx2b.py"
    ```

??? example "ects/varx2c.py"
    ```python
    --8<-- "examples/ects/varx2c.py"
    ```

??? example "ects/varx3.py"
    ```python
    --8<-- "examples/ects/varx3.py"
    ```

??? example "ects/varx4a.py"
    ```python
    --8<-- "examples/ects/varx4a.py"
    ```

??? example "ects/varx4b.py"
    ```python
    --8<-- "examples/ects/varx4b.py"
    ```

??? example "ects/varx5a.py"
    ```python
    --8<-- "examples/ects/varx5a.py"
    ```

## `econometrics.kalman`

!!! info "Python alternatives"
    **Switch** to `statsmodels.tsa.statespace.MLEModel`/`kalman_filter` for anything beyond simple filtering — smoothing, built-in MLE fitting, diffuse initialization, a compiled Cython backend. Keep this module for simple, transparent filtering or minimal-dependency use.

::: quanttoolbox.econometrics.kalman

### Examples

??? example "ects/kalman1b.py"
    ```python
    --8<-- "examples/ects/kalman1b.py"
    ```

??? example "ects/kalman1c.py"
    ```python
    --8<-- "examples/ects/kalman1c.py"
    ```

??? example "ects/kalman2a.py"
    ```python
    --8<-- "examples/ects/kalman2a.py"
    ```

??? example "ects/kalman2b.py"
    ```python
    --8<-- "examples/ects/kalman2b.py"
    ```

??? example "ects/kalman2c.py"
    ```python
    --8<-- "examples/ects/kalman2c.py"
    ```

??? example "ects/kalman3a.py"
    ```python
    --8<-- "examples/ects/kalman3a.py"
    ```

??? example "ects/kalman3b.py"
    ```python
    --8<-- "examples/ects/kalman3b.py"
    ```

??? example "ects/kalman3c.py"
    ```python
    --8<-- "examples/ects/kalman3c.py"
    ```

??? example "ects/kalman3d.py"
    ```python
    --8<-- "examples/ects/kalman3d.py"
    ```

??? example "ects/kalman4a.py"
    ```python
    --8<-- "examples/ects/kalman4a.py"
    ```

??? example "ects/kalman4b.py"
    ```python
    --8<-- "examples/ects/kalman4b.py"
    ```

??? example "ects/panel1.py"
    ```python
    --8<-- "examples/ects/panel1.py"
    ```

??? example "stats/kalman1.py"
    ```python
    --8<-- "examples/stats/kalman1.py"
    ```

??? example "stats/kalman2.py"
    ```python
    --8<-- "examples/stats/kalman2.py"
    ```

## `econometrics.whittle`

!!! info "Python alternatives"
    **Keep** — Whittle (frequency-domain) estimation isn't implemented in statsmodels, `arch`, or other common packages. Genuinely fills a gap.

::: quanttoolbox.econometrics.whittle

### Examples

??? example "ects/kalman1c.py"
    ```python
    --8<-- "examples/ects/kalman1c.py"
    ```

??? example "ects/kalman2b.py"
    ```python
    --8<-- "examples/ects/kalman2b.py"
    ```

??? example "ects/whittle1.py"
    ```python
    --8<-- "examples/ects/whittle1.py"
    ```

??? example "ects/whittle2.py"
    ```python
    --8<-- "examples/ects/whittle2.py"
    ```

??? example "maths/pdgm1.py"
    ```python
    --8<-- "examples/maths/pdgm1.py"
    ```

??? example "maths/pdgm2.py"
    ```python
    --8<-- "examples/maths/pdgm2.py"
    ```

## `econometrics.tests` (ADF)

!!! info "Python alternatives"
    Already **switched** to `statsmodels.tsa.stattools.adfuller`. The [`arch`](https://github.com/bashtage/arch) package's `arch.unitroot` module has a wider family of unit-root tests (Phillips-Perron, DFGLS, KPSS, Zivot-Andrews) if more than ADF is ever needed.

::: quanttoolbox.econometrics.tests
