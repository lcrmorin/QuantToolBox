# `quanttoolbox.stats`

## `stats.distributions`

!!! info "Python alternatives"
    Simple wrappers (normal/t/chi2/F/MVN) are one-liners around `scipy.stats` — **keep**, but no reason to avoid calling `scipy.stats` directly either. The GQF1/GQF2 (generalized quadratic-form) distributions are genuinely niche — nothing in scipy, statsmodels, or elsewhere implements these. **Keep.**

::: quanttoolbox.stats.distributions

## `stats.moments`

!!! info "Python alternatives"
    `rolling_correlation`/`rolling_volatility`: **switch** to `pandas.DataFrame.rolling().corr()`/`.std()` for the standard case — Cython-backed, meaningfully faster. Keep ours only for the `method=2` "returns computed within each window" variant pandas doesn't offer. `active_share`, `herfindahl_index`, `asynchronous_cov`, `weekly_cov`: **keep** — no general-purpose equivalent exists.

::: quanttoolbox.stats.moments

## `stats.regression.ols`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.OLS`/`WLS` is more complete for plain unrestricted regression. Keep this module for the `restriction=(RR, r)` linear-restriction parameterization, which statsmodels doesn't support as directly.

::: quanttoolbox.stats.regression.ols

## `stats.regression.ridge`

!!! info "Python alternatives"
    **Hybrid**: `sklearn.linear_model.Ridge`/`RidgeCV` is better optimized for large/sparse problems. Keep `ridge_tau_targeted` — its L2-norm-*budget* (not penalty) parameterization has no sklearn equivalent.

::: quanttoolbox.stats.regression.ridge

## `stats.regression.lasso`

!!! info "Python alternatives"
    **Switch** the penalized-form solvers (`lasso_ccd`, `lasso_admm`) to `sklearn.linear_model.Lasso`/`ElasticNet` — Cython-compiled, extensively battle-tested. **Keep** `lasso_tau_constrained` — sklearn has no L1-*budget* interface.

::: quanttoolbox.stats.regression.lasso

## `stats.regression.kernel`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.nonparametric.KernelReg` supports automatic bandwidth selection via cross-validation and both local-constant/local-linear estimators. Switch for general use; keep ours where the exact original bandwidth formula must match existing MATLAB-based results.

::: quanttoolbox.stats.regression.kernel

## `stats.regression.quantile`

!!! info "Python alternatives"
    **Switch** to `statsmodels.regression.quantile_regression.QuantReg` or `sklearn.linear_model.QuantileRegressor` — both more battle-tested than this module's `scipy.optimize.linprog`-based implementation. Keep ours only if the LP slack-variable (`u`, `v`) outputs are needed downstream.

::: quanttoolbox.stats.regression.quantile

## `stats.regression.robust`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.robust.robust_linear_model.RLM` covers Huber, Tukey biweight, Andrew's wave, Hampel, and trimmed-mean M-estimators via IRLS — a superset of this module's Huber implementation. It does *not* cover LAD or quantile M-estimation directly (though `QuantReg(q=0.5)` is exactly LAD). Use RLM for general M-estimation; keep `lad_regression`/`quantile_m_regression`/`inverse_quantile_m_regression` for those specific losses.

::: quanttoolbox.stats.regression.robust
