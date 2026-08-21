# Migration Map: MATLAB QuantToolbox → Python quanttoolbox

Status legend: ⬜ not started · 🟨 in progress · ✅ ported & tested

See also: [`matlab_bugs_found.md`](./matlab_bugs_found.md) for genuine
defects found in the *original* MATLAB source during porting, and
[`library_alternatives.md`](./library_alternatives.md) for an assessment
of which ported modules should be replaced with mature Python libraries
versus which genuinely justify staying custom.

| Python module | Original MATLAB files | Status |
|---|---|---|
| `dates/convert.py` | `dates/Excel2Matlab_Dates.m`, `Matlab2Excel_Dates.m`, `is_yyyymmdd.m`, `numdate.m`, `datenum2.m`, `excel_column.m` | ✅ |
| `dates/rebalancing.py` | `dates/generic_rebalancing.m` + `annual/monthly/quarterly/semi_annual/weekly_rebalancing.m`, `generate_trading_dates.m` | ✅ |
| `backtest/returns.py` | `backtest/price2return.m`, `price2return2.m`, `return2price.m`, `price2unfunded.m`, `unfunded2price.m`, `capitalized_libor*.m` | ✅ |
| `backtest/stats.py` | `backtest/maximum_drawdown.m`, `static_turnover.m`, `annualized_turnover.m`, `average_return.m`, `monthly_statistics.m`, `yearly_statistics.m`, `index_repeated_data.m` | ✅ |
| `backtest/reporting.py` | `backtest/generate_backtest.m`, `generate_backtest2.m`, `backtest_reporting.m` | ✅ |
| `stats/distributions.py` | `stats/cdfn.m`, `cdfni.m`, `cdft.m`, `cdfti.m`, `cdftc.m`, `cdfchi2.m`, `cdfchi2c.m`, `cdff.m`, `cdffc.m`, `cdfmvn.m`, `pdfmvn.m`, `pdfn.m`, `rndmvn.m`, `gqf1_*.m`, `gqf2_*.m` | ✅ |
| `stats/moments.py` | `stats/skewness_coefficient.m`, `kurtosis_coefficient.m`, `herfindahl_index.m`, `mean_absolute_difference.m`, `cov2cor.m`, `cor2cov.m`, `corrx.m`, `pearson_correlation.m`, `active_share*.m`, `asynchronous_cov.m`, `weekly_cov.m`, `rolling_correlation.m`, `rolling_volatility.m` | ✅ |
| `stats/regression/ols.py` | `stats/regOLS.m`, `regCenter.m`, `regStandardize.m`, `regCND.m`, `regPCA.m` | ✅ |
| `stats/regression/ridge.py` | `stats/regRidge.m`, `regRidge2.m` (+ `theo/` duplicates) | ✅ |
| `stats/regression/lasso.py` | `stats/regLassoCCD.m`, `regLassoADMM.m`, `regLasso.m`, `selectLasso.m`, `regElasticNet.m` (+ `theo/` duplicates) | ✅ |
| `stats/regression/kernel.py` | `stats/regKernelDensity.m`, `regKernelMean.m`, `regKernelQuantile.m`, `regKernelPayoff.m` | ✅ |
| `stats/regression/quantile.py` (new, not in original module map) | `stats/quantile_regression.m`, `qrCopulaNormal.m`, `qrCopulaStudent.m` | ✅ |
| `stats/regression/robust.py` | `ects/robust_regression.m`, `robust_huber_regression.m`, `robust_lad_regression.m`, `robust_quantile_regression.m`, `robust_inverse_quantile_regression.m` | ✅ |
| `econometrics/estimation.py` | `ects/ols_estimation.m`, `ols_constrained_estimation.m`, `gmm_estimation.m`, `gmm_constrained_estimation.m`, `ml_estimation.m`, `ml_constrained_estimation.m`, `wald_test.m` | ✅ |
| `econometrics/var.py` | `ects/varx_cls.m`, `varx_cml.m`, `varx_ls.m`, `varx_ml.m`, `varx_order.m`, `varx_constrained_estimation_onestep.m`, `var_constrained_estimation_onestep.m` | ✅ |
| `econometrics/kalman.py` | `ects/state_space_model.m`, `ssm_set.m`, `ssm_steady_state.m`, `Kalman_filtering.m` | ✅ |
| `econometrics/whittle.py` | `ects/whittle_estimation.m`, `whittle_constrained_estimation.m`, `whittle_local_level.m`, `whittle_local_linear_trend.m`, `maths/pdgm.m`, `maths/periodogram.m` | ✅ |
| `econometrics/tests.py` | `ects/adf_test.m` (via `statsmodels.tsa.stattools.adfuller` -- see module docstring) | ✅ |
| `optim/proximal.py` | `optim/proximal_L1.m`, `proximal_L2.m`, `proximal_Linfinity.m`, `proximal_bounds.m`, `proximal_equality.m`, `proximal_inequality.m`, `proximal_linear_constraints.m`, `proximal_max.m`, `proximal_turnover.m`, `soft_thresholding.m` | ✅ |
| `optim/projection.py` | `optim/projection_L1.m`, `projection_L2.m`, `projection_Linfinity.m`, `projection_box_L2.m` | ✅ |
| `optim/quadprog.py` | `optim/quadprog_bc_ccd.m`, `quadprog_lasso.m`, `quadprog_mixed_norm.m`, `quadprog_mixed2_norm.m`, `quadprog_ridge.m`, `quadprog_turnover.m`, `qp_hyperplane.m` | ✅ |
| `optim/bisection.py` | `optim/bisection.m`, `bisection2.m`, `explicit2implicit.m`, `implicit2explicit.m` | ✅ |
| `portfolio/risk_budgeting.py` | `rpb/compute_rb_*.m`, `crb/compute_rb_sd_*.m` (~55 files), `mloapa/compute_ERC_*.m`, `lagrange_rb_sd.m` | ✅ |
| `portfolio/mean_variance.py` | `rpb/compute_mvo_portfolio*.m`, `compute_minvar_portfolio.m`, `mloapa/compute_MinVar_ADMM*.m`, `compute_MDP_ADMM.m`, `compute_mdp_*.m` | ✅ |
| `portfolio/tracking_error.py` | `rpb/compute_te_portfolio*.m`, `compute_minimum_te_portfolio.m`, `compute_te_portfolio_mixed_norm.m` | ✅ |
| `portfolio/black_litterman.py` | `rpb/compute_Black_Litterman_moments.m`, `implied_risk_premia.m` | ✅ |
| `portfolio/erc_mdp.py` | `rpb/compute_erc_portfolio.m`, `mloapa/compute_ERC_*.m`, `compute_MDP_ADMM.m` | ✅ |
| `mixtures/gaussian_mixture.py` | `mixture/mixture_*.m`, `estimate_em_mixture.m`, `logl_em_mixture.m` | ✅ |
| `mixtures/jump_diffusion.py` | `mixture/jump_*.m`, `bivariate_lognormal_skewness.m`, `lognormal_moments.m`, `lognormal_skewness.m` | ✅ |
| `svm/svm.py` | `svm/svm_classification_dual.m`, `svm_classification_primal.m`, `svm_regression_dual.m`, `svm_regression_primal.m` (+ `theo/` duplicates) | ✅ |
| `spline/spline.py` | `spline/csspline.m`, `dspline.m`, `fspline.m`, `intspline.m`, `invspline.m`, `band.m`, `bandrv.m`, `bandsolpd.m`, `rotater.m` | ✅ |
| `maths/numerical_diff.py` | `maths/numerical_gradient.m`, `numerical_hessian.m`, `numerical_jacobian.m`, `sign_operator.m` (`stats/robust_vcv.m`/`ml_robust_vcv.m` are superseded by `econometrics.estimation.ml_estimation(..., cov="hc")`, not separately ported) | ✅ |
| `maths/simulation.py` | `maths/simulate_gbm.m`, `simulate_gbm2.m`, `simulate_multi_gbm.m`, `compute_ewma.m`, `momentum_ewma.m`, `volatility_target.m`, `algebraic_riccati_equation.m`, `lyapunov_equation.m` | ✅ |
| `linalg/special_matrices.py` | `matrix/vec.m`, `vech.m`, `vecr.m`, `xpnd.m`, `commutation_matrix.m`, `duplication_matrix.m`, `elimination_matrix.m`, `reshapec.m`, `reshaper.m`, `diagrv.m`, `lowmat.m`, `upmat.m`, `design.m` | ✅ |
| `viz/export.py` | `tools/save_graphic.m`, `save_graphic2.m` | ✅ |

## Not ported (intentionally dropped)

| Original | Reason |
|---|---|
| `matrix/rows.m`, `cols.m`, `sumc.m`, `meanc.m`, `stdc.m`, `maxc.m`, `minc.m`, `prodc.m` | Replaced by native NumPy (`.shape`, `.sum()`, `.mean()`, `.std()`, ...) everywhere they're used. |
| `tools/packr.m`, `selif.m`, `seqa.m`, `lag1.m`, `lagn.m`, `missrv.m`, `miss.m`, `rev.m`, `trimr.m` | Replaced by pandas/NumPy idioms (`.dropna()`, boolean indexing, `np.arange`, `.shift()`, `.fillna()`, slicing). |
| `export/*.m` (16-file third-party `export_fig` package) | matplotlib's native `savefig(dpi=..., bbox_inches="tight")` covers the same need. |
| `latex/*.m`, `tools/latex_sa.m`, `latex_tabular.m` | Use `pandas.DataFrame.to_latex()` / Jinja2 templates instead. |
| `init_color.m` | MATLAB plot-color globals; not needed with matplotlib style sheets. |
| `init_global.m` | Superseded by `config.py` dataclasses (defaults live with each config class instead of one monolithic init script). |

## Notes for translators

General guidance for anyone extending or re-verifying this port (see
[`matlab_bugs_found.md`](./matlab_bugs_found.md) for the specific,
already-resolved bugs found in the original source along the way):

- Every module still using MATLAB `global` state (ADMM/CCD tolerances, MVO
  problem context, Proximal_Algorithm, GMM/ML/Whittle settings) should take
  the corresponding dataclass from `quanttoolbox.config` as an explicit
  argument with a default instance.
- `crb/` and `rpb/` together account for ~95 MATLAB files, many of them
  near-identical solver variants (ADMM/CCD/Newton/fmincon × unconstrained/
  box/linear constraints). These collapse into `portfolio/risk_budgeting.py`
  as a single class; check the original file names above only to confirm
  behavioral parity during testing, not to preserve a 1:1 file structure.
- MATLAB is 1-indexed and column-major; watch for off-by-one loop bounds and
  `reshape`/`vec` ordering differences when porting arithmetic directly.
- `quadprog(...)` calls (MATLAB Optimization Toolbox) map to `qpsolvers.solve_qp`
  or `cvxpy` — solver choice affects numerical tolerance, so regression-test
  against the MATLAB output where precision matters (e.g. SVM support vectors).
- When translating a formula that depends on a specific scaling/normalization
  convention (e.g. a `2π` factor, a `1/n` normalization), verify it against
  a known closed-form result or a numerical derivative rather than trusting
  the transcription — several of the found bugs were exactly this kind of
  silent scaling-factor mismatch.
