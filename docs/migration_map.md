# Migration Map: MATLAB QuantToolbox → Python quanttoolbox

Status legend: ⬜ not started · 🟨 in progress · ✅ ported & tested

This page covers the whole picture: which original MATLAB module each
Python module replaces, which files were intentionally dropped, a
file-by-file tracker for every original *example* script, and general
notes for anyone extending or re-verifying the port. See also
[`library_alternatives.md`](./library_alternatives.md) for an assessment
of which ported modules should be replaced with mature Python libraries
versus which genuinely justify staying custom (including, for the ones
worth keeping, whether the gap is worth contributing back upstream).

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
| `stats/regression/lasso.py` | `stats/regLassoCCD.m`, `regLassoADMM.m`, `regLasso.m`, `selectLasso.m`, `regElasticNet.m` (+ `theo/` duplicates) | 🟨 |
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
| `maths/arccosh.m`, `arcsinh.m` (HSF toolbox, see below) | Directly superseded by `numpy.arccosh`/`numpy.arcsinh`. |

## HSF toolbox port (planned)

A second MATLAB library -- the function library behind Roncalli's
*Handbook of Sustainable Finance*, from
[`hfs-archive`](https://github.com/lcrmorin/hfs-archive) -- is being
ported into `quanttoolbox` directly as new modules, rather than
maintained as a separate package. Its source (167 files) lives at
[`reference/hsf_toolbox_matlab/`](../reference/hsf_toolbox_matlab) as
reference material for the port; that folder is excluded from the
installable package (not listed in `pyproject.toml`'s package scoping).

None of this is started yet -- the table below is the planned module
breakdown, not a status report. See
[HSF-Notebooks](https://github.com/lcrmorin/HSF-Notebooks)'s
`CHAPTERS.md` (chapter 0) for the matching entry and the per-chapter
notebooks that will exercise these modules once ported.

| Python module (planned) | Original MATLAB files (`0. Toolbox/`) | Status |
|---|---|---|
| `bond/pricing.py` (new module) | `bond/compute_bond_price.m`, `compute_bond_ytm.m`, `compute_coupon_yield.m`, `quadratic_form_bond_portfolio1.m`, `quadratic_form_bond_portfolio2.m` | ⬜ |
| `copula/families.py` (new module) | `copula/cdfCopula*.m`, `pdfCopula*.m` (23 families: Normal, Student, Clayton, Frank, Gumbel (+2 variants), Plackett, FGM, AMH, Galambos, Husler-Reiss, Marshall-Olkin, Gumbel-Barnett, logistic-Gumbel, Sloane, cubic, upper/lower Fréchet, product) | ⬜ |
| `copula/dependence.py` | `copula/KendallCopula*.m`, `SpearmanCopula*.m` (5 families each), `dependogram.m`, `DebyeFunction.m`, `diLogFunction.m` | ⬜ |
| `copula/simulate.py` | `copula/rndCopula*.m`, `rndnCopula.m` (8 families) | ⬜ |
| `credit/structural.py` (new module) | `credit/Black_Scholes_Model.m`, `PD_Merton_Model.m`, `B0_Extended_Merton_Model.m`, `E0_Extended_Merton_Model.m`, `PD_Extended_Merton_Model.m`, `PD_Black_Cox_Model.m`, `Merton_Jump_Model.m`, `Merton_Jump_Climate_Model.m`, `Reinders_Credit_Model.m` | ⬜ |
| `credit/reduced_form.py` | `credit/Density_Markov_Generator.m`, `Hazard_Markov_Generator.m`, `Survival_Markov_Generator.m`, `cdfExponential.m`, `pdfExponential.m`, `invExponential.m`, `rndExponential.m`, `survivalExponential.m` | ⬜ |
| `stats/multivariate.py` (new module) | `genz/*.m` (10 files, Genz-Bretz quadrature for MVN/MVT CDFs) + `stats/cdfbvn.m`, `pdfbvn.m`, `cdfbvt.m` (bivariate normal/Student) | ⬜ |
| `stats/distributions.py` (extend existing) | `stats/cdfSN*.m`/`pdfSN.m`/`momSN.m`/`rndSN.m` (skew-normal), `cdfST*.m`/`pdfST.m`/`momST.m`/`rndST.m` (skew-t), `cdfBates.m`/`pdfBates.m`, `cdfbeta.m`/`pdfbeta.m`, `cdfig.m`/`pdfig.m` (inverse Gaussian), `cdfln.m`/`pdfln.m` (lognormal), `cdfNormalRatio.m`/`pdfNormalRatio.m`, `pdfPoissonBinomial.m`, `cdfchi2i.m`, `pdft.m`, `compute_cdf_order_statistics.m`, `compute_inv_cdf_order_statistics.m`, `constant_correlation_matrix.m`, `max_size.m` | ⬜ |
| `stats/dose_response.py` (new module) | `stats/drcHormetic1.m`, `drcHormetic2.m`, `drcLogLogistic.m`, `drcLogNormal.m`, `drcWeibull1.m`, `drcWeibull2.m` | ⬜ |
| `sustainable_finance/carbon.py` (new module) | `hsf/carbon_budget_linear.m`, `carbon_budget_linear_reduction.m`, `carbon_budget_linear_trend.m`, `carbon_budget_piecewise.m`, `carbon_budget_compound_reduction.m`, `carbon_budget_Reduction.m` | ⬜ |
| `sustainable_finance/esg.py` | `hsf/compute_esg_beta_star.m`, `compute_esg_minimum_variance.m`, `compute_pedersen_portfolio.m`, `cdp_filter.m` | ⬜ |
| `sustainable_finance/climate.py` | `hsf/dice_temperature_matrix.m`, `dice_temperature_simulation.m` | ⬜ |
| `sustainable_finance/ecology.py` | `hsf/species_area_relationship.m`, `endemics_area_relationship.m`, `species_abundance_distribution.m`, `hurlbert.m` | ⬜ |
| `sustainable_finance/entropy.py` | `hsf/shannon_entropy.m`, `shannon_entropy_markov_chain.m`, `estimate_markov_generator.m` | ⬜ |
| `sustainable_finance/risk.py` | `hsf/quadratic_form.m`, `quadratic_form_risk.m`, `bond_portfolio_metrics.m` | ⬜ |

Notes on the table above:

- `copula1.m`-`copula4.m` sit loose at the toolbox root (not inside
  `copula/`) -- worked examples simulating correlated Poisson/
  exponential/log-normal marginals through a Gaussian copula, in
  French. Not a module to port; more likely future material for a
  `copula` notebook in HSF-Notebooks once `copula/simulate.py` exists.
- `quadratic_form_bond_portfolio1.m`/`2.m` exist identically in both
  `bond/` and `hsf/` -- ported once, under `bond/pricing.py`.
- `genz/qsimvn*.m`/`qsilatmvnv.m`/`bvn.m`/`bvnu.m` and `copula/cdfmvn.m`,
  `stats/cdfmvn.m`, `stats/pdfmvn.m` all look like variants of the same
  multivariate-normal CDF/quadrature machinery already ported once as
  `stats/distributions.py`'s `cdfmvn`/`pdfmvn` (see the module table
  above) -- reconcile these against each other before porting rather
  than assuming each is a distinct algorithm.
- `copula/` (67 files) is the largest single block and the most likely
  to want splitting further once work starts (e.g. one file per family
  instead of the 3-file grouping above); treat the module names here as
  a starting proposal, not a commitment.

## Example translation tracker

Tracks every script in the original MATLAB toolbox's `Examples/` folder
(138 files across 10 subfolders) against its Python translation status.
The `tutorial/` folder (11 generic MATLAB-101 lessons, unrelated to any
quanttoolbox function) is out of scope for this tracker -- see
[HSF-Notebooks](https://github.com/lcrmorin/HSF-Notebooks)'s
`notebooks/tutorials/` for the quanttoolbox-specific walkthroughs that
replace it.

Status legend: ⬜ not translated · ✅ translated

The **Notes** column flags examples that call `plot`/`figure` or
`xlsread`/`xlswrite`/`readtable`/`writetable`. These are translated for
their *numeric* logic (the actual computation), with the plotting/Excel
I/O portions dropped rather than translated — a doc page showing a
`matplotlib` chart or a pandas DataFrame export is more useful than a
literal port of MATLAB's plotting calls.

---

### backtest/ (9 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `backtest/` | `backtest1.m` | ⬜ | Exercises `fillmiss`/`findnomiss`, which became private helpers (`_fill_missing_ffill_bfill`/`_first_last_valid`) rather than public functions during porting -- not translated as a standalone example for now |
| `backtest/` | `backtest2.m` | ✅ | Translated (numeric core; plot dropped) — 3 rebalancing schedules (single, 4 fixed positions, date-subset with 2 out-of-range placeholders + a NaN price) |
| `backtest/` | `backtest3.m` | ✅ | Translated — generate_backtest at 4 rebalancing frequencies |
| `backtest/` | `backtest4.m` | ✅ | Translated — per-asset bid/ask transaction costs; turnover cross-checked against `static_turnover` |
| `backtest/` | `backtest5.m` | ✅ | Translated — flat transaction cost under the original's actually-exercised (monthly/every-date) rebalancing schedule |
| `backtest/` | `fillmiss1.m` | ⬜ | Same reason as `backtest1.m` -- exercises `fillmiss`/`findnomiss`, now private helpers |
| `backtest/` | `mdd1.m` | ✅ | Translated (numeric core; plot dropped) — maximum_drawdown, relative mode |
| `backtest/` | `unfunded1.m` | ✅ | Translated (numeric core; plot dropped) — funded vs. two unfunded-formulation backtests, cross-checked against each other |
| `backtest/` | `unfunded2.m` | ✅ | Translated (numeric core; plot dropped) — small hand-traceable n=8 version of unfunded1.py's round-trip |

### dates/ (5 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `dates/` | `date1.m` | ⬜ | Excel I/O (reads/writes `date1.xlsx`) |
| `dates/` | `date2.m` | ⬜ | Excel I/O |
| `dates/` | `date3.m` | ✅ | Self-contained — see `examples/building_blocks.md` |
| `dates/` | `date4.m` | ✅ | Identical `generate_trading_dates` call already covered by `date3.m` |
| `dates/` | `rebalancing_dates.m` | ✅ | Uses a `.mat` file with MATLAB's newer `datetime` object type (MCOS class) that's awkward to reproduce byte-for-byte from Python; the underlying `weekly_rebalancing`/`monthly_rebalancing`/`quarterly_rebalancing` functions are already covered by the package's own test suite with synthetic data. |

### ects/ (39 `.m` files; 9 `.asc` data files are not examples)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `ects/` | `gmm1.m` | ✅ | Translated — OLS/MLE/GMM, unconstrained and beta[3]=1-restricted, on a dataset with one missing y; `theta7` (a second, redundant encoding of the same restriction) not re-translated separately (see gmm1.py docstring) |
| `ects/` | `kalman1a.m` | ✅ | Byte-identical to `panel1.m` (verified via diff) -- already translated as `panel1.py`, not duplicated |
| `ects/` | `kalman1b.m` | ✅ | Translated (numeric core; plot dropped) — time-domain ML of (sigma_epsilon, sigma_eta) under two Kalman a0 choices during estimation, compared post-hoc |
| `ects/` | `kalman1c.m` | ✅ | Translated (numeric core; plot dropped) — time-domain vs. frequency-domain (Whittle) ML compared; Whittle estimate cross-checked against `whittle1.py`'s output |
| `ects/` | `kalman2a.m` | ✅ | Translated (numeric core; plot dropped) — Local Linear Trend Kalman filter on Gnp.asc (uses `Gnp.asc`, not `Kalman3.asc` as the tracker previously noted), fixed variance parameters |
| `ects/` | `kalman2b.m` | ✅ | Translated (numeric core; plot dropped) — LLT model, Whittle (frequency-domain) ML on Gnp.asc |
| `ects/` | `kalman2c.m` | ✅ | Translated (numeric core; plot dropped) — LLT model, time-domain ML on Gnp.asc; cross-checked against kalman2b.py's Whittle estimate |
| `ects/` | `kalman3a.m` | ✅ | Translated — general 2-state/2-obs SSM filter on `Kalman3.asc`, from steady-state initial mean/covariance |
| `ects/` | `kalman3b.m` | ✅ | Translated — same model/data as kalman3a.py, run through the time-varying (3-D matrix) code path; verified to reproduce kalman3a.py's output exactly |
| `ects/` | `kalman3c.m` | ✅ | Translated (numeric core; plot dropped) — filtered state + 95% confidence band |
| `ects/` | `kalman3d.m` | ✅ | Translated — ML recovery of all 11 free SSM parameters, starting from and compared against their true values |
| `ects/` | `kalman4a.m` | ✅ | Translated (numeric core; plot dropped) — simulated random-walk-coefficient regression, time-varying-Z Kalman filter recovers the beta_t path; seeded RNG |
| `ects/` | `kalman4b.m` | ✅ | Translated (numeric core; plot dropped) — same simulation as kalman4a.py, but (sigma_epsilon, sigma_beta1, sigma_beta2) estimated by ML rather than assumed known |
| `ects/` | `ml1.m` | ⬜ | Near-duplicate of `ml2.m` (same data/model), whose distinctive feature is exercising MATLAB's string-name function dispatch (`ml_estimation('ml1_fn',sv)`) -- not meaningful to translate; lower priority |
| `ects/` | `ml1_fn.m` | ⬜ | helper function for `ml1.m`, not standalone |
| `ects/` | `ml2.m` | ✅ | Translated — OLS vs. Gaussian-MLE covariance matrix, Hessian/OPG/HC estimators |
| `ects/` | `ml3.m` | ✅ | Translated — Beta-distribution MLE (LGD modeling), Hessian/OPG/HC covariance |
| `ects/` | `ml4.m` | ✅ | Translated — numerical vs. analytical Jacobian/Hessian, all 3 covariance estimators under both |
| `ects/` | `ols1.m` | ✅ | Self-contained — see `examples/regression.md` |
| `ects/` | `panel1.m` | ✅ | Translated — **note:** despite its filename/old tracker note, this is a local-level Kalman filter example (Harvey 1990), not panel data; corrected and translated using `econometrics.kalman` (numeric core; plot dropped) |
| `ects/` | `quantile1.m` | ✅ | Translated (numeric core; plot dropped) — pinball-loss minimization vs. sorted-sample quantile vs. true Normal quantile |
| `ects/` | `quantile2.m` | ✅ | Translated (numeric core; plot dropped) — Monte Carlo (nS=2000) OLS vs. LAD under heteroskedastic noise |
| `ects/` | `robust1.m` | ✅ | Self-contained — see `examples/regression.md` |
| `ects/` | `robust2.m` | ✅ | Translated — OLS/median/LAD/Huber regression |
| `ects/` | `robust3.m` | ✅ | Translated — OLS/median/LAD, plus quantile regression via both IRLS M-estimation and the exact LP formulation (cross-checked against each other) |
| `ects/` | `varx1a.m` | ✅ | Translated — VAR(2) on log-differenced Lutkepohl.asc investment/income/consumption (embedded), via `varx_estimate(..., method="ls")` |
| `ects/` | `varx1b.m` | ✅ | Translated — Wald test for no Granger-causality (income/consumption -> investment) via `wald_test` |
| `ects/` | `varx1c.m` | ✅ | Translated — Wald test for no instantaneous causality, on the ML fit's Cholesky-augmented theta/vcv |
| `ects/` | `varx1d.m` | ✅ | Translated — restricted VAR(2) (7 of 21 coefficients free, via `design(w)`) estimated by both LS and concentrated ML |
| `ects/` | `varx1e.m` | ✅ | Translated — lag-order selection (`varx_order`, p=1..5) on same data as varx1a.m |
| `ects/` | `varx2a.m` | ✅ | Translated — reduced form of a dynamic simultaneous-equations system (Lutkepohl ch.10) via `varx_estimate(..., p=1)` |
| `ects/` | `varx2b.m` | ✅ | Translated — varx2a.py's model with 4 coefficients restricted, via restricted LS |
| `ects/` | `varx2c.m` | ✅ | Translated — same restricted model as varx2b.py, via concentrated ML; cross-checked against it |
| `ects/` | `varx3.m` | ✅ | Translated — trivial 5-obs OLS vs. `varx_estimate(..., p=0)` equivalence check |
| `ects/` | `varx4a.m` | ✅ | Translated — JHGLL restricted SUR (2-eq, embedded 20-obs data), two-step GLS |
| `ects/` | `varx4b.m` | ✅ | Translated — JHGLL restricted SUR (3-eq, embedded 30-obs data), two Sigma variants compared |
| `ects/` | `varx5a.m` | ✅ | Translated — JHGLL simultaneous-equations system, OLS/GLS-2S/GLS-3S/LIML compared (all four converge to consistent Gamma/B/PI) |
| `ects/` | `whittle1.m` | ✅ | Translated — Whittle local-level MLE on the same Purse.asc data as panel1.m (single BFGS path; original's 3-optimizer comparison collapses to one) |
| `ects/` | `whittle2.m` | ✅ | Translated — Bloomfield exponential spectral density (order 4), custom `sdf_fn` passed to `whittle_estimation` |

### maths/ (9 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `maths/` | `grad1.m` | ✅ | See `examples/building_blocks.md` |
| `maths/` | `grad2.m` | ✅ | Translated — scalar function of 2 variables (poly + log + exp) |
| `maths/` | `grad3.m` | ✅ | Translated — elementwise fun summed first (same trick as grad1.m) so the scalar-valued `numerical_gradient` can be used |
| `maths/` | `grad4.m` | ✅ | Translated — same elementwise-via-sum trick as grad3.m |
| `maths/` | `grad5.m` | ✅ | Translated — both the already-scalar `fun` and the explicit-sum `fun2` formulations (same gradient either way) |
| `maths/` | `hess1.m` | ✅ | See `examples/building_blocks.md` |
| `maths/` | `hess2.m` | ✅ | Translated — numerical gradient + Hessian vs. analytical, at a point with a very small x2 |
| `maths/` | `pdgm1.m` | ✅ | Translated — not actually a plot despite the earlier note; raw periodogram of a 4-obs series via `periodogram`, output verified against the MATLAB source's own comment |
| `maths/` | `pdgm2.m` | ✅ | Translated — same as pdgm1.py, 8-obs series, output verified against the MATLAB source's own comment |

### matrix/ (9 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `matrix/` | `design1.m` | ⬜ | Not translated as a standalone example -- `design()`'s rounding behavior is exercised directly by the package's own test suite instead |
| `matrix/` | `matrix1.m` | ✅ | Elimination/duplication/commutation matrix shapes and sums checked against expected values |
| `matrix/` | `matrix2.m` | ✅ | Translated — 7 elimination/duplication/commutation-matrix identities checked for M=1..10; uses `np.isclose`/`allclose` rather than exact `==` since `inv(D'*D)` isn't bit-exact |
| `matrix/` | `reshape1.m` | ✅ | Translated — vech/xpnd (both orderings) and reshaper/reshapec; the `shiftr` portion is dropped (not ported, see `shiftr1.m` below) |
| `matrix/` | `shiftc1.m` | ⬜ | Exercises `shiftc` — not ported (GAUSS shift primitive, superseded by `numpy.roll`/slicing idioms) |
| `matrix/` | `shiftr1.m` | ⬜ | Exercises `shiftr` — not ported, same reason |
| `matrix/` | `submat1.m` | ⬜ | Exercises `submat` — not ported (superseded by NumPy fancy indexing) |
| `matrix/` | `vec1.m` | ✅ | See `examples/building_blocks.md` |
| `matrix/` | `vech1.m` | ✅ | See `examples/building_blocks.md` |

### optim/ (11 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `optim/` | `bisection1.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `explicit1.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `explicit2.m` | ✅ | Translated — explicit/implicit round-trip for one constraint, plus a `design()` matrix demo |
| `optim/` | `explicit3.m` | ✅ | Translated — explicit-to-implicit for 3 simultaneous zero-restrictions |
| `optim/` | `prox_L1.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `prox_L2.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `prox_Linfinity.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `prox_turnover1.m` | ✅ | Translated — proximal-L1 vs. both projection-L1 algorithms; fixed-seed substitution for unseeded `rand` |
| `optim/` | `prox_turnover2.m` | ⬜ | Not translated as a standalone example -- exercises `proximal_turnover` via MATLAB's `fminunc`/`optimoptions`, which has no direct Python equivalent path; the function itself is already covered by the package's own test suite |
| `optim/` | `proximal1.m` | ✅ | Translated — bounds/1-4 inequalities/1-2 equalities/combined linear constraints; the original's `Proximal_Algorithm` 1-vs-2 comparison is moot post-port (see proximal1.py's docstring — the redundant closed-form-vs-QP branches for bounds/equality weren't kept) |
| `optim/` | `turnover1.m` | ✅ | See `examples/building_blocks.md` |

### rpb/ (21 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `rpb/` | `test_bl1.m` | ✅ | See `examples/black_litterman.md` |
| `rpb/` | `test_bl2.m` | ✅ | Translated — sensitivity analysis across 5 view scenarios, fixed risk-aversion MVO |
| `rpb/` | `test_bl3.m` | ✅ | Sigma-target MVO mode -- `mean_variance.mvo_target_portfolio`; 5 BL view scenarios matched to the x0 benchmark's volatility |
| `rpb/` | `test_bl4.m` | ✅ | TE-target-matching mode -- `tracking_error.te_target_portfolio`; 6 TE targets |
| `rpb/` | `test_bl5.m` | ✅ | Same TE-target-matching mode as test_bl4.m, plus a second Black-Litterman worked example (full-view, `P=I`); 5 blocks |
| `rpb/` | `test_bl6.m` | ✅ | Raw `quadprog` call translated via `te_portfolio` (shown to solve the identical QP), plus `te_target_portfolio` and a gamma-recovery self-consistency check |
| `rpb/` | `test_box1.m` | ✅ | Translated — ERC + box-constrained RB at progressively wider bounds |
| `rpb/` | `test_erc1.m` | ✅ | See `examples/risk_budgeting.md` |
| `rpb/` | `test_erc2.m` | ✅ | Translated |
| `rpb/` | `test_erc3.m` | ✅ | Translated |
| `rpb/` | `test_lasso1.m` | ✅ | Translated — unconstrained/budget-constrained MVO (identical, since `mvo_portfolio` auto-applies the budget constraint when `a_eq` is omitted, matching `compute_mvo_portfolio.m`) plus ridge/lasso via `solve_qp` (no auto budget constraint there, matching the raw `quadprog_ridge`/`quadprog_lasso`) |
| `rpb/` | `test_lasso2.m` | ✅ | Translated — numeric core (representative points from the 250-point ridge/lasso static/dynamic sweep `test_lasso3.m` demonstrates directly), plot dropped |
| `rpb/` | `test_lasso3.m` | ✅ | Translated via `solve_qp` ridge/lasso penalties (replaces original's `quadprog_ridge`/`quadprog_lasso`/`quadprog_mixed`) |
| `rpb/` | `test_lasso4.m` | ✅ | Byte-identical to `test_lasso3.m` -- same translation covers both |
| `rpb/` | `test_lasso5.m` | ✅ | Translated — mixed ridge+lasso penalties toward two different target vectors |
| `rpb/` | `test_minvar1.m` | ✅ | See `examples/mean_variance.md` |
| `rpb/` | `test_minvar2.m` | ✅ | Translated — MinVar with general linear equality/inequality constraints |
| `rpb/` | `test_mvo1.m` | ✅ | See `examples/mean_variance.md` |
| `rpb/` | `test_mvo2.m` | ✅ | All 3 sections translated -- gamma-problem (`mvo_frontier`), mu-problem and sigma-problem (`mvo_target_portfolio`) |
| `rpb/` | `test_mvo3.m` | ✅ | Sigma-problem under 3 weight-bound configurations |
| `rpb/` | `test_te1.m` | ✅ | See `examples/mean_variance.md` |

### stats/ (19 files; `ridge.inc` is a shared data snippet, not a standalone example)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `stats/` | `count1.m` | ⬜ | Exercises `counts`/`countmiss` (GAUSS tabulation utilities) — not ported, no quanttoolbox equivalent to demonstrate (superseded by `np.histogram`/`pd.value_counts`) |
| `stats/` | `cov1.m` | ✅ | Translated — OLS in closed form vs. Gaussian MLE (`ml_estimation`) comparing Hessian/OPG/HC standard errors; `ml_robust_vcv`'s 5 partially-redundant covariance variants condensed to `ml_estimation`'s 3 `cov=` options |
| `stats/` | `elasticnet1.m` | ✅ | Translated — elastic net path (alpha=0.5) via `elastic_net_ccd` |
| `stats/` | `elasticnet2.m` | ✅ | Same translation as `elasticnet1.m` (alpha=0.25 case) — covered together |
| `stats/` | `kalman1.m` | ✅ | Translated — time-varying-beta random-walk-coefficient model, bounded MLE via `scipy.optimize.minimize(..., method="L-BFGS-B")` (`ml_estimation` has no box-constraint support) since the `ects/` Kalman pass is now done |
| `stats/` | `kalman2.m` | ✅ | Translated — same model as kalman1.py, a0 fixed at [-20,-20] (unrecovered from a deliberately bad start) rather than estimated |
| `stats/` | `kernel1.m` | ✅ | Translated (numeric core; plot dropped) — Gaussian kernel density on two correlated series |
| `stats/` | `lasso1.m` | ✅ | Translated (numeric core; 501-point plot sweep dropped) — extends lasso2.py's identical 5-lambda/dataset run with R²/df/complexity reporting |
| `stats/` | `lasso2.m` | ✅ | Translated (penalized-form lasso path via `lasso_ccd`); the tau-constrained comparison and 501-point plot sweep in the original are not re-translated separately |
| `stats/` | `lasso3.m` | ⬜ | Uses `selectLasso` (lasso-path variable-selection ordering) — not currently ported as a standalone function |
| `stats/` | `ml_ols.m` | ⬜ | Helper function (per-observation log-density) for `cov1.m`, not a standalone example — inlined directly into `cov1.py` |
| `stats/` | `pca1.m` | ✅ | Translated |
| `stats/` | `qreg1.m` | ✅ | Translated — linear quantile regression at 9 tau levels |
| `stats/` | `qreg2.m` | ✅ | Translated (numeric core; plot dropped) — local-linear/quadratic kernel mean & quantile regression vs. population curves |
| `stats/` | `quantile1.m` | ⬜ | Exercises `quantile_classification`/`unique2` — not ported, no quanttoolbox equivalent (superseded by `pd.qcut`/`np.unique`) |
| `stats/` | `quantile2.m` | ⬜ | Same reason as `quantile1.m` |
| `stats/` | `ridge1.m` | ✅ | See `examples/regression.md` (numeric core; plot dropped) |
| `stats/` | `ridge2.m` | ✅ | Translated — `ridge_tau_targeted` cross-checked against fixed-lambda `ridge`, absolute and relative tau, two lambda-search grids |

### svm/ (12 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `svm/` | `svm1.m` | ✅ | See `examples/svm.md` |
| `svm/` | `svm2.m` | ✅ | Translated |
| `svm/` | `svm3.m` | ✅ | Translated — matches svm4.py exactly (primal/dual duality) |
| `svm/` | `svm4.m` | ✅ | Translated — matches svm3.py exactly (primal/dual duality) |
| `svm/` | `svm5.m` | ✅ | Translated (numeric core; original's C-sweep plot not re-translated separately, covered by svm3/svm4 at 4 representative C values) |
| `svm/` | `svm6.m` | ✅ | Translated — OLS/LAD/quantile/SVM comparison; SVM-LS matches OLS at high C as expected |
| `svm/` | `svm7.m` | ✅ | Translated (dual formulation) — matches svm6.py's primal results exactly |
| `svm/` | `svm8.m` | ✅ | Translated — fixed-seed substitution (`default_rng(0)`) for the n=1000 synthetic dataset; OLS/SVM-LS and Quantile/SVM-eps pairs agree at large C as expected |
| `svm/` | `svm_regression_dual_theo.m` | ✅ | Function source file (not an example script) — the `theo/` duplicate already merged into `svm.py`'s `svm_regression_dual` |
| `svm/` | `svm_regression_primal_theo.m` | ✅ | Function source file (not an example script) — already merged into `svm.py`'s `svm_regression_primal` |
| `svm/` | `theo6.m` | ✅ | Duplicate of svm6.m's exact scenario — covered by svm6.py |
| `svm/` | `theo7.m` | ✅ | Duplicate of svm6.m's exact scenario — covered by svm6.py |

### tools/ (6 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `tools/` | `hline1.m` | ⬜ | display/formatting helper, not a ported function |
| `tools/` | `indnv1.m` | ⬜ | `indnv` not ported (superseded by numpy indexing — see migration map) |
| `tools/` | `latex1.m` | ⬜ | LaTeX table export, not ported (use `pandas.to_latex`) |
| `tools/` | `recode1.m` | ⬜ | `recode` not ported (superseded by `numpy.where`/`pandas` idioms) |
| `tools/` | `retcode1.m` | ⬜ | return-code display helper, not a standalone function |

---

### Summary

| Folder | Total | ✅ Translated | ⬜ Remaining |
|---|---|---|---|
| `backtest/` | 9 | 7 | 2 |
| `dates/` | 5 | 3 | 2 |
| `ects/` | 39 | 37 | 2 |
| `maths/` | 9 | 9 | 0 |
| `matrix/` | 9 | 5 | 4 |
| `optim/` | 11 | 10 | 1 |
| `rpb/` | 21 | 21 | 0 |
| `stats/` | 18 | 13 | 5 |
| `svm/` | 12 | 12 | 0 |
| `tools/` | 5 | 0 | 5 |
| **Total** | **138** | **117** | **21** |

*(`stats/ridge.inc` excluded as a data snippet, not a standalone example
— 138 actual `.m` example scripts plus that one data-only file. The
`tutorial/` folder is tracked separately, see above.)*

## Notes for translators

General guidance for anyone extending or re-verifying this port:

- **Gap closed:** `portfolio/mean_variance.py` and
  `portfolio/tracking_error.py` were marked 🟨 for a while because they
  only implemented the "gamma-problem" frontier mode (evaluate at a
  given risk-aversion value). Several original examples (`test_bl3.m`
  through `test_bl6.m`, `test_mvo3.m`, part of `test_mvo2.m`) needed a
  "sigma-problem"/TE-target mode instead — bisecting an internal
  risk-aversion parameter to hit a target portfolio volatility or
  tracking error directly, the same pattern
  `portfolio.risk_budgeting.risk_budgeting_target` already implements
  for risk budgeting. Both modules now have this via
  `mvo_target_portfolio`/`te_target_portfolio`, closing every ⬜ in
  `rpb/` above. `stats/regression/lasso.py` is still marked 🟨, for a
  smaller, unrelated reason: `selectLasso.m` (lasso-path
  variable-*selection-ordering*, not the coefficient path itself) isn't
  ported as a standalone function.

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
