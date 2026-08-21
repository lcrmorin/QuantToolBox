# `quanttoolbox.stats`

## `stats.distributions`

!!! info "Python alternatives"
    Simple wrappers (normal/t/chi2/F/MVN) are one-liners around `scipy.stats` — **keep**, but no reason to avoid calling `scipy.stats` directly either. The GQF1/GQF2 (generalized quadratic-form) distributions are genuinely niche — nothing in scipy, statsmodels, or elsewhere implements these. **Keep.**

::: quanttoolbox.stats.distributions

### Examples

??? example "ects/quantile1.py"
    ```python
    --8<-- "examples/ects/quantile1.py"
    ```

## `stats.moments`

!!! info "Python alternatives"
    `rolling_correlation`/`rolling_volatility`: **switch** to `pandas.DataFrame.rolling().corr()`/`.std()` for the standard case — Cython-backed, meaningfully faster. Keep ours only for the `method=2` "returns computed within each window" variant pandas doesn't offer. `active_share`, `herfindahl_index`, `asynchronous_cov`, `weekly_cov`: **keep** — no general-purpose equivalent exists.

::: quanttoolbox.stats.moments

### Examples

??? example "rpb/test_bl2.py"
    ```python
    --8<-- "examples/rpb/test_bl2.py"
    ```

??? example "rpb/test_box1.py"
    ```python
    --8<-- "examples/rpb/test_box1.py"
    ```

??? example "rpb/test_erc2.py"
    ```python
    --8<-- "examples/rpb/test_erc2.py"
    ```

??? example "rpb/test_erc3.py"
    ```python
    --8<-- "examples/rpb/test_erc3.py"
    ```

??? example "rpb/test_lasso1.py"
    ```python
    --8<-- "examples/rpb/test_lasso1.py"
    ```

??? example "rpb/test_lasso3.py"
    ```python
    --8<-- "examples/rpb/test_lasso3.py"
    ```

??? example "rpb/test_lasso5.py"
    ```python
    --8<-- "examples/rpb/test_lasso5.py"
    ```

??? example "rpb/test_minvar2.py"
    ```python
    --8<-- "examples/rpb/test_minvar2.py"
    ```

??? example "rpb/test_mvo2.py"
    ```python
    --8<-- "examples/rpb/test_mvo2.py"
    ```

## `stats.regression.ols`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.OLS`/`WLS` is more complete for plain unrestricted regression. Keep this module for the `restriction=(RR, r)` linear-restriction parameterization, which statsmodels doesn't support as directly.

::: quanttoolbox.stats.regression.ols

### Examples

??? example "stats/elasticnet1.py"
    ```python
    --8<-- "examples/stats/elasticnet1.py"
    ```

??? example "stats/lasso1.py"
    ```python
    --8<-- "examples/stats/lasso1.py"
    ```

??? example "stats/lasso2.py"
    ```python
    --8<-- "examples/stats/lasso2.py"
    ```

??? example "stats/pca1.py"
    ```python
    --8<-- "examples/stats/pca1.py"
    ```

??? example "stats/ridge2.py"
    ```python
    --8<-- "examples/stats/ridge2.py"
    ```

## `stats.regression.ridge`

!!! info "Python alternatives"
    **Hybrid**: `sklearn.linear_model.Ridge`/`RidgeCV` is better optimized for large/sparse problems. Keep `ridge_tau_targeted` — its L2-norm-*budget* (not penalty) parameterization has no sklearn equivalent.

::: quanttoolbox.stats.regression.ridge

### Examples

??? example "stats/ridge2.py"
    ```python
    --8<-- "examples/stats/ridge2.py"
    ```

## `stats.regression.lasso`

!!! info "Python alternatives"
    **Switch** the penalized-form solvers (`lasso_ccd`, `lasso_admm`) to `sklearn.linear_model.Lasso`/`ElasticNet` — Cython-compiled, extensively battle-tested. **Keep** `lasso_tau_constrained` — sklearn has no L1-*budget* interface.

::: quanttoolbox.stats.regression.lasso

### Examples

??? example "stats/elasticnet1.py"
    ```python
    --8<-- "examples/stats/elasticnet1.py"
    ```

??? example "stats/lasso1.py"
    ```python
    --8<-- "examples/stats/lasso1.py"
    ```

??? example "stats/lasso2.py"
    ```python
    --8<-- "examples/stats/lasso2.py"
    ```

## `stats.regression.kernel`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.nonparametric.KernelReg` supports automatic bandwidth selection via cross-validation and both local-constant/local-linear estimators. Switch for general use; keep ours where the exact original bandwidth formula must match existing MATLAB-based results.

::: quanttoolbox.stats.regression.kernel

### Examples

??? example "stats/kernel1.py"
    ```python
    --8<-- "examples/stats/kernel1.py"
    ```

??? example "stats/qreg2.py"
    ```python
    --8<-- "examples/stats/qreg2.py"
    ```

## `stats.regression.quantile`

!!! info "Python alternatives"
    **Switch** to `statsmodels.regression.quantile_regression.QuantReg` or `sklearn.linear_model.QuantileRegressor` — both more battle-tested than this module's `scipy.optimize.linprog`-based implementation. Keep ours only if the LP slack-variable (`u`, `v`) outputs are needed downstream.

::: quanttoolbox.stats.regression.quantile

### Examples

??? example "ects/quantile2.py"
    ```python
    --8<-- "examples/ects/quantile2.py"
    ```

??? example "ects/robust2.py"
    ```python
    --8<-- "examples/ects/robust2.py"
    ```

??? example "ects/robust3.py"
    ```python
    --8<-- "examples/ects/robust3.py"
    ```

??? example "stats/qreg1.py"
    ```python
    --8<-- "examples/stats/qreg1.py"
    ```

??? example "svm/svm6.py"
    ```python
    --8<-- "examples/svm/svm6.py"
    ```

??? example "svm/svm8.py"
    ```python
    --8<-- "examples/svm/svm8.py"
    ```

## `stats.regression.robust`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.robust.robust_linear_model.RLM` covers Huber, Tukey biweight, Andrew's wave, Hampel, and trimmed-mean M-estimators via IRLS — a superset of this module's Huber implementation. It does *not* cover LAD or quantile M-estimation directly (though `QuantReg(q=0.5)` is exactly LAD). Use RLM for general M-estimation; keep `lad_regression`/`quantile_m_regression`/`inverse_quantile_m_regression` for those specific losses.

::: quanttoolbox.stats.regression.robust

### Examples

??? example "ects/robust2.py"
    ```python
    --8<-- "examples/ects/robust2.py"
    ```

??? example "ects/robust3.py"
    ```python
    --8<-- "examples/ects/robust3.py"
    ```

??? example "svm/svm6.py"
    ```python
    --8<-- "examples/svm/svm6.py"
    ```

??? example "svm/svm8.py"
    ```python
    --8<-- "examples/svm/svm8.py"
    ```
