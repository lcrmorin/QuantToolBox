# Migration Map: MATLAB → Python

`quanttoolbox` is a Python port of two original MATLAB code bases:
Roncalli's general **QuantToolBox**, and the **Handbook of Sustainable
Finance** toolbox from [`hfs-archive`](https://github.com/lcrmorin/hfs-archive).
Both are tracked together below as one module map — from the Python side
they're just `quanttoolbox`, and which original repo a given file came
from is a historical detail, not something a user of the package needs
to think about.

See also [Library alternatives](library_alternatives.md) for which
ported modules should be replaced with mature Python libraries versus
which genuinely justify staying custom, and [MATLAB bugs
found](matlab_bugs_found.md) for defects discovered in the original
source during porting.

## Module map

Status: ✅ ported & tested · 🟨 partially ported (see Notes). Every
MATLAB source file is accounted for below, either with a Python module
or in the [Not ported](#not-ported) list — so the only outstanding work
in the whole port is whatever isn't ✅ here.

| Python module | Original MATLAB source | Status | Notes |
|---|---|---|---|
| `backtest/reporting.py` | `generate_backtest.m`, `generate_backtest2.m`, `backtest_reporting.m` | ✅ | |
| `backtest/returns.py` | `price2return.m`, `price2return2.m`, `return2price.m`, `price2unfunded.m`, `unfunded2price.m`, `capitalized_libor*.m` | ✅ | |
| `backtest/stats.py` | `maximum_drawdown.m`, `static_turnover.m`, `annualized_turnover.m`, `average_return.m`, `monthly_statistics.m`, `yearly_statistics.m`, `index_repeated_data.m` | ✅ | |
| `bond/pricing.py` | `bond/compute_bond_price.m`, `compute_bond_ytm.m`, `compute_coupon_yield.m`, `quadratic_form_bond_portfolio1.m`, `quadratic_form_bond_portfolio2.m` | ✅ | `quadratic_form_bond_portfolio*.m` also exists under `hsf/` — ported once |
| `copula/dependence.py` | `copula/KendallCopula*.m`, `SpearmanCopula*.m`, `dependogram.m`, `DebyeFunction.m`, `diLogFunction.m` | ✅ | |
| `copula/families.py` | `copula/cdfCopula*.m`, `pdfCopula*.m`, `cdfConditionalCopula*.m`, `cdfSloaneCopula.m`, `contourCopula*.m`, `singularCopula*.m` | ✅ | Gumbel3 PDF not ported — original bug, see MATLAB bugs found #7 |
| `copula/simulate.py` | `copula/rndCopula*.m`, `rndnCopula.m` | ✅ | |
| `credit/reduced_form.py` | `credit/Density_Markov_Generator.m`, `Hazard_Markov_Generator.m`, `Survival_Markov_Generator.m`, `cdfExponential.m`, `pdfExponential.m`, `invExponential.m`, `rndExponential.m`, `survivalExponential.m` | ✅ | |
| `credit/structural.py` | `credit/Black_Scholes_Model.m`, `PD_Merton_Model.m`, `B0_Extended_Merton_Model.m`, `E0_Extended_Merton_Model.m`, `PD_Extended_Merton_Model.m`, `PD_Black_Cox_Model.m`, `Merton_Jump_Model.m`, `Merton_Jump_Climate_Model.m`, `Reinders_Credit_Model.m` | ✅ | |
| `dates/convert.py` | `dates/Excel2Matlab_Dates.m`, `Matlab2Excel_Dates.m`, `is_yyyymmdd.m`, `numdate.m`, `datenum2.m`, `excel_column.m` | ✅ | |
| `dates/rebalancing.py` | `dates/generic_rebalancing.m`, `annual_rebalancing.m`, `monthly_rebalancing.m`, `quarterly_rebalancing.m`, `semi_annual_rebalancing.m`, `weekly_rebalancing.m`, `generate_trading_dates.m` | ✅ | |
| `econometrics/estimation.py` | `ects/ols_estimation.m`, `ols_constrained_estimation.m`, `gmm_estimation.m`, `gmm_constrained_estimation.m`, `ml_estimation.m`, `ml_constrained_estimation.m`, `wald_test.m` | ✅ | |
| `econometrics/kalman.py` | `ects/state_space_model.m`, `ssm_set.m`, `ssm_steady_state.m`, `Kalman_filtering.m` | ✅ | |
| `econometrics/tests.py` | `ects/adf_test.m` | ✅ | Uses `statsmodels.tsa.stattools.adfuller` |
| `econometrics/var.py` | `ects/varx_cls.m`, `varx_cml.m`, `varx_ls.m`, `varx_ml.m`, `varx_order.m`, `varx_constrained_estimation_onestep.m`, `var_constrained_estimation_onestep.m` | ✅ | |
| `econometrics/whittle.py` | `ects/whittle_estimation.m`, `whittle_constrained_estimation.m`, `whittle_local_level.m`, `whittle_local_linear_trend.m`, `maths/pdgm.m`, `maths/periodogram.m` | ✅ | Jacobian bug fixed, see MATLAB bugs found #1 |
| `linalg/special_matrices.py` | `matrix/vec.m`, `vech.m`, `vecr.m`, `xpnd.m`, `commutation_matrix.m`, `duplication_matrix.m`, `elimination_matrix.m`, `reshapec.m`, `reshaper.m`, `diagrv.m`, `lowmat.m`, `upmat.m`, `design.m` | ✅ | |
| `maths/numerical_diff.py` | `maths/numerical_gradient.m`, `numerical_hessian.m`, `numerical_jacobian.m`, `sign_operator.m` | ✅ | |
| `maths/simulation.py` | `maths/simulate_gbm.m`, `simulate_gbm2.m`, `simulate_multi_gbm.m`, `compute_ewma.m`, `momentum_ewma.m`, `volatility_target.m`, `algebraic_riccati_equation.m`, `lyapunov_equation.m` | ✅ | `simulate_multi_gbm` reimplemented correctly, see MATLAB bugs found #2 |
| `mixtures/gaussian_mixture.py` | `mixture/mixture_*.m`, `estimate_em_mixture.m`, `logl_em_mixture.m` | ✅ | |
| `mixtures/jump_diffusion.py` | `mixture/jump_*.m`, `bivariate_lognormal_skewness.m`, `lognormal_moments.m`, `lognormal_skewness.m` | ✅ | |
| `optim/bisection.py` | `optim/bisection.m`, `bisection2.m`, `explicit2implicit.m`, `implicit2explicit.m` | ✅ | |
| `optim/projection.py` | `optim/projection_L1.m`, `projection_L2.m`, `projection_Linfinity.m`, `projection_box_L2.m` | ✅ | |
| `optim/proximal.py` | `optim/proximal_L1.m`, `proximal_L2.m`, `proximal_Linfinity.m`, `proximal_bounds.m`, `proximal_equality.m`, `proximal_inequality.m`, `proximal_linear_constraints.m`, `proximal_max.m`, `proximal_turnover.m`, `soft_thresholding.m` | ✅ | Known convergence-check limitation kept as-is, see MATLAB bugs found #5 |
| `optim/quadprog.py` | `optim/quadprog_bc_ccd.m`, `quadprog_lasso.m`, `quadprog_mixed_norm.m`, `quadprog_mixed2_norm.m`, `quadprog_ridge.m`, `quadprog_turnover.m`, `qp_hyperplane.m` | ✅ | |
| `portfolio/black_litterman.py` | `rpb/compute_Black_Litterman_moments.m`, `implied_risk_premia.m` | ✅ | |
| `portfolio/erc_mdp.py` | `rpb/compute_erc_portfolio.m`, `mloapa/compute_ERC_*.m`, `compute_MDP_ADMM.m` | ✅ | |
| `portfolio/mean_variance.py` | `rpb/compute_mvo_portfolio*.m`, `compute_minvar_portfolio.m`, `mloapa/compute_MinVar_ADMM*.m`, `compute_MDP_ADMM.m`, `compute_mdp_*.m` | ✅ | |
| `portfolio/risk_budgeting.py` | `rpb/compute_rb_*.m`, `crb/compute_rb_sd_*.m`, `mloapa/compute_ERC_*.m`, `lagrange_rb_sd.m` | ✅ | Newton positivity floor added, see MATLAB bugs found #3 |
| `portfolio/tracking_error.py` | `rpb/compute_te_portfolio*.m`, `compute_minimum_te_portfolio.m`, `compute_te_portfolio_mixed_norm.m` | ✅ | |
| `spline/spline.py` | `spline/csspline.m`, `dspline.m`, `fspline.m`, `intspline.m`, `invspline.m`, `band.m`, `bandrv.m`, `bandsolpd.m`, `rotater.m` | ✅ | |
| `stats/distributions.py` | `stats/cdfn.m`, `cdfni.m`, `cdft.m`, `cdfti.m`, `cdftc.m`, `cdfchi2.m`, `cdfchi2c.m`, `cdff.m`, `cdffc.m`, `cdfmvn.m`, `pdfmvn.m`, `pdfn.m`, `rndmvn.m`, `gqf1_*.m`, `gqf2_*.m`, `cdfSN*.m`/`pdfSN.m`/`momSN.m`/`rndSN.m`, `cdfST*.m`/`pdfST.m`/`momST.m`/`rndST.m`, `cdfBates.m`/`pdfBates.m`, `cdfbeta.m`/`pdfbeta.m`, `cdfig.m`/`pdfig.m`, `cdfln.m`/`pdfln.m`, `cdfNormalRatio.m`/`pdfNormalRatio.m`, `pdfPoissonBinomial.m`, `cdfchi2i.m`, `pdft.m`, `compute_cdf_order_statistics.m`, `compute_inv_cdf_order_statistics.m`, `constant_correlation_matrix.m` | ✅ | |
| `stats/dose_response.py` | `stats/drcHormetic1.m`, `drcHormetic2.m`, `drcLogLogistic.m`, `drcLogNormal.m`, `drcWeibull1.m`, `drcWeibull2.m` | ✅ | |
| `stats/moments.py` | `stats/skewness_coefficient.m`, `kurtosis_coefficient.m`, `herfindahl_index.m`, `mean_absolute_difference.m`, `cov2cor.m`, `cor2cov.m`, `corrx.m`, `pearson_correlation.m`, `active_share*.m`, `asynchronous_cov.m`, `weekly_cov.m`, `rolling_correlation.m`, `rolling_volatility.m` | ✅ | |
| `stats/multivariate.py` | `stats/cdfbvn.m`, `pdfbvn.m`, `cdfbvt.m` | ✅ | `genz/*.m` superseded by `scipy`, see Library alternatives |
| `stats/regression/kernel.py` | `stats/regKernelDensity.m`, `regKernelMean.m`, `regKernelQuantile.m`, `regKernelPayoff.m` | ✅ | |
| `stats/regression/lasso.py` | `stats/regLassoCCD.m`, `regLassoADMM.m`, `regLasso.m`, `selectLasso.m`, `regElasticNet.m` | 🟨 | `selectLasso.m` (lasso-path variable-selection ordering) not yet a standalone function — the one open item in the whole port |
| `stats/regression/ols.py` | `stats/regOLS.m`, `regCenter.m`, `regStandardize.m`, `regCND.m`, `regPCA.m` | ✅ | |
| `stats/regression/quantile.py` | `stats/quantile_regression.m`, `qrCopulaNormal.m`, `qrCopulaStudent.m` | ✅ | |
| `stats/regression/ridge.py` | `stats/regRidge.m`, `regRidge2.m` | ✅ | |
| `stats/regression/robust.py` | `ects/robust_regression.m`, `robust_huber_regression.m`, `robust_lad_regression.m`, `robust_quantile_regression.m`, `robust_inverse_quantile_regression.m` | ✅ | |
| `sustainable_finance/carbon.py` | `hsf/carbon_budget_linear.m`, `carbon_budget_linear_reduction.m`, `carbon_budget_linear_trend.m`, `carbon_budget_piecewise.m`, `carbon_budget_compound_reduction.m`, `carbon_budget_Reduction.m` | ✅ | |
| `sustainable_finance/climate.py` | `hsf/dice_temperature_matrix.m`, `dice_temperature_simulation.m` | ✅ | |
| `sustainable_finance/ecology.py` | `hsf/species_area_relationship.m`, `endemics_area_relationship.m`, `species_abundance_distribution.m`, `hurlbert.m` | ✅ | |
| `sustainable_finance/entropy.py` | `hsf/shannon_entropy.m`, `shannon_entropy_markov_chain.m`, `estimate_markov_generator.m` | ✅ | |
| `sustainable_finance/esg.py` | `hsf/compute_esg_beta_star.m`, `compute_esg_minimum_variance.m`, `compute_pedersen_portfolio.m` | ✅ | `cdp_filter.m` not ported — tied to an unshipped dataset |
| `sustainable_finance/risk.py` | `hsf/quadratic_form.m`, `quadratic_form_risk.m`, `bond_portfolio_metrics.m` | ✅ | |
| `svm/svm.py` | `svm/svm_classification_dual.m`, `svm_classification_primal.m`, `svm_regression_dual.m`, `svm_regression_primal.m` | ✅ | |
| `viz/export.py` | `tools/save_graphic.m`, `save_graphic2.m` | ✅ | |

Two loose scripts sit outside any module and aren't tracked above:
`copula1.m`-`copula4.m` (worked NORTA examples, in French, at the
`hfs-archive` toolbox root) are future material for an HSF-Notebooks
notebook rather than library code, and `maths/arccosh.m`/`arcsinh.m`
appear in both source trees but are covered by the [Not
ported](#not-ported) list below (superseded by NumPy either way).

## Not ported

MATLAB source with no Python equivalent — each one because a standard
library already covers it, not because of missing effort.

| Original MATLAB source | Superseded by |
|---|---|
| `matrix/rows.m`, `cols.m`, `sumc.m`, `meanc.m`, `stdc.m`, `maxc.m`, `minc.m`, `prodc.m` | Native NumPy (`.shape`, `.sum()`, `.mean()`, `.std()`, ...) |
| `tools/packr.m`, `selif.m`, `seqa.m`, `lag1.m`, `lagn.m`, `missrv.m`, `miss.m`, `rev.m`, `trimr.m` | pandas/NumPy idioms (`.dropna()`, boolean indexing, `np.arange`, `.shift()`, `.fillna()`, slicing) |
| `export/*.m` (16 files) | `matplotlib.pyplot.savefig(dpi=..., bbox_inches="tight")` |
| `latex/*.m`, `tools/latex_sa.m`, `latex_tabular.m` | `pandas.DataFrame.to_latex()` / Jinja2 templates |
| `init_color.m` | matplotlib style sheets |
| `init_global.m` | `quanttoolbox.config` dataclasses |
| `maths/arccosh.m`, `arcsinh.m` | `numpy.arccosh`/`numpy.arcsinh` |

## Example translation tracker

Tracks every script in the original MATLAB `Examples/` folder (138
files across 10 subfolders) against its Python translation. The
`tutorial/` folder (11 generic MATLAB-101 lessons, unrelated to any
`quanttoolbox` function) is out of scope — see
[HSF-Notebooks](https://github.com/lcrmorin/HSF-Notebooks)'s
`notebooks/hsf/00_tutorial.ipynb` for its Python translation.

Legend: **Port?** Yes = a Python translation belongs here; No =
intentionally skipped (reason in Notes) and excluded from remaining-work
counts. **Status** ✅ done · ⬜ not done. The only cell that's `Port:
Yes` and `Status: ⬜` anywhere in this tracker is `stats/lasso3.m` — the
same open item as `stats/regression/lasso.py` above.

The **Notes** column also flags examples that called `plot`/`figure` or
`xlsread`/`xlswrite`/`readtable`/`writetable` in the original: these are
translated for their *numeric* logic only, with plotting/Excel I/O
dropped.

---

### backtest/ (9 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `backtest1.m` | No | — | Exercises `fillmiss`/`findnomiss`, now private helpers, not public functions |
| `backtest2.m` | Yes | ✅ | 3 rebalancing schedules (single, 4 fixed positions, date-subset) |
| `backtest3.m` | Yes | ✅ | `generate_backtest` at 4 rebalancing frequencies |
| `backtest4.m` | Yes | ✅ | Per-asset bid/ask transaction costs; turnover cross-checked against `static_turnover` |
| `backtest5.m` | Yes | ✅ | Flat transaction cost under the actually-exercised rebalancing schedule |
| `fillmiss1.m` | No | — | Same reason as `backtest1.m` |
| `mdd1.m` | Yes | ✅ | `maximum_drawdown`, relative mode |
| `unfunded1.m` | Yes | ✅ | Funded vs. two unfunded formulations, cross-checked against each other |
| `unfunded2.m` | Yes | ✅ | Small hand-traceable n=8 version of `unfunded1.py`'s round-trip |

### dates/ (5 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `date1.m` | No | — | Excel I/O only, no numeric core to keep |
| `date2.m` | No | — | Excel I/O only |
| `date3.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `date4.m` | Yes | ✅ | Identical `generate_trading_dates` call already covered by `date3.m` |
| `rebalancing_dates.m` | Yes | ✅ | Reads a MATLAB `.mat`/`datetime` file; underlying functions covered by the test suite with synthetic data instead |

### ects/ (39 `.m` files; 9 `.asc` data files are not examples)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `gmm1.m` | Yes | ✅ | OLS/MLE/GMM, unconstrained and restricted, one missing y |
| `kalman1a.m` | Yes | ✅ | Byte-identical to `panel1.m` — already translated as `panel1.py` |
| `kalman1b.m` | Yes | ✅ | Time-domain ML under two Kalman a0 choices, compared post-hoc |
| `kalman1c.m` | Yes | ✅ | Time-domain vs. frequency-domain (Whittle) ML, cross-checked against `whittle1.py` |
| `kalman2a.m` | Yes | ✅ | Local Linear Trend Kalman filter on Gnp.asc, fixed variance parameters |
| `kalman2b.m` | Yes | ✅ | LLT model, Whittle (frequency-domain) ML on Gnp.asc |
| `kalman2c.m` | Yes | ✅ | LLT model, time-domain ML; cross-checked against `kalman2b.py` |
| `kalman3a.m` | Yes | ✅ | 2-state/2-obs SSM filter on Kalman3.asc, from steady-state initial mean/covariance |
| `kalman3b.m` | Yes | ✅ | Same model/data as `kalman3a.py` via the time-varying code path; reproduces it exactly |
| `kalman3c.m` | Yes | ✅ | Filtered state + 95% confidence band |
| `kalman3d.m` | Yes | ✅ | ML recovery of all 11 free SSM parameters |
| `kalman4a.m` | Yes | ✅ | Simulated random-walk-coefficient regression; time-varying-Z Kalman filter recovers the path |
| `kalman4b.m` | Yes | ✅ | Same simulation as `kalman4a.py`, with variances estimated by ML |
| `ml1.m` | No | — | Near-duplicate of `ml2.m`; its distinctive feature (string-name function dispatch) has no Python equivalent |
| `ml1_fn.m` | No | — | Helper for `ml1.m`, not a standalone example |
| `ml2.m` | Yes | ✅ | OLS vs. Gaussian-MLE covariance, Hessian/OPG/HC estimators |
| `ml3.m` | Yes | ✅ | Beta-distribution MLE (LGD modeling), Hessian/OPG/HC covariance |
| `ml4.m` | Yes | ✅ | Numerical vs. analytical Jacobian/Hessian, all 3 covariance estimators |
| `ols1.m` | Yes | ✅ | See `examples/regression.md` |
| `panel1.m` | Yes | ✅ | Local-level Kalman filter example (Harvey 1990), not panel data despite the filename |
| `quantile1.m` | Yes | ✅ | Pinball-loss minimization vs. sorted-sample quantile vs. true Normal quantile |
| `quantile2.m` | Yes | ✅ | Monte Carlo OLS vs. LAD under heteroskedastic noise |
| `robust1.m` | Yes | ✅ | See `examples/regression.md` |
| `robust2.m` | Yes | ✅ | OLS/median/LAD/Huber regression |
| `robust3.m` | Yes | ✅ | OLS/median/LAD, plus quantile regression via IRLS and the exact LP formulation |
| `varx1a.m` | Yes | ✅ | VAR(2) on Lutkepohl data via `varx_estimate(..., method="ls")` |
| `varx1b.m` | Yes | ✅ | Wald test for no Granger-causality via `wald_test` |
| `varx1c.m` | Yes | ✅ | Wald test for no instantaneous causality |
| `varx1d.m` | Yes | ✅ | Restricted VAR(2) estimated by both LS and concentrated ML |
| `varx1e.m` | Yes | ✅ | Lag-order selection via `varx_order` |
| `varx2a.m` | Yes | ✅ | Reduced form of a dynamic simultaneous-equations system |
| `varx2b.m` | Yes | ✅ | `varx2a.py`'s model with 4 coefficients restricted |
| `varx2c.m` | Yes | ✅ | Same restricted model via concentrated ML; cross-checked against `varx2b.py` |
| `varx3.m` | Yes | ✅ | Trivial 5-obs OLS vs. `varx_estimate(..., p=0)` equivalence check |
| `varx4a.m` | Yes | ✅ | Restricted SUR (2-eq), two-step GLS |
| `varx4b.m` | Yes | ✅ | Restricted SUR (3-eq), two Sigma variants compared |
| `varx5a.m` | Yes | ✅ | Simultaneous-equations system, OLS/GLS-2S/GLS-3S/LIML compared |
| `whittle1.m` | Yes | ✅ | Whittle local-level MLE on the same data as `panel1.m` |
| `whittle2.m` | Yes | ✅ | Bloomfield exponential spectral density, custom `sdf_fn` |

### maths/ (9 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `grad1.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `grad2.m` | Yes | ✅ | Scalar function of 2 variables (poly + log + exp) |
| `grad3.m` | Yes | ✅ | Elementwise function summed first, same trick as `grad1.m` |
| `grad4.m` | Yes | ✅ | Same elementwise-via-sum trick as `grad3.m` |
| `grad5.m` | Yes | ✅ | Both scalar and explicit-sum formulations |
| `hess1.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `hess2.m` | Yes | ✅ | Numerical gradient + Hessian vs. analytical |
| `pdgm1.m` | Yes | ✅ | Raw periodogram of a 4-obs series, verified against the source's own comment |
| `pdgm2.m` | Yes | ✅ | Same as `pdgm1.py`, 8-obs series |

### matrix/ (9 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `design1.m` | No | — | `design()`'s rounding behavior is exercised directly by the test suite instead |
| `matrix1.m` | Yes | ✅ | Elimination/duplication/commutation matrix shapes and sums |
| `matrix2.m` | Yes | ✅ | 7 matrix identities checked for M=1..10 |
| `reshape1.m` | Yes | ✅ | vech/xpnd (both orderings) and reshaper/reshapec |
| `shiftc1.m` | No | — | Exercises `shiftc`, superseded by `numpy.roll`/slicing |
| `shiftr1.m` | No | — | Exercises `shiftr`, same reason |
| `submat1.m` | No | — | Exercises `submat`, superseded by NumPy fancy indexing |
| `vec1.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `vech1.m` | Yes | ✅ | See `examples/building_blocks.md` |

### optim/ (11 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `bisection1.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `explicit1.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `explicit2.m` | Yes | ✅ | Explicit/implicit round-trip for one constraint |
| `explicit3.m` | Yes | ✅ | Explicit-to-implicit for 3 simultaneous zero-restrictions |
| `prox_L1.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `prox_L2.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `prox_Linfinity.m` | Yes | ✅ | See `examples/building_blocks.md` |
| `prox_turnover1.m` | Yes | ✅ | Proximal-L1 vs. both projection-L1 algorithms |
| `prox_turnover2.m` | No | — | Exercises `proximal_turnover` via MATLAB-only `fminunc`/`optimoptions`; function already covered by the test suite |
| `proximal1.m` | Yes | ✅ | Bounds, inequalities, equalities, combined linear constraints |
| `turnover1.m` | Yes | ✅ | See `examples/building_blocks.md` |

### rpb/ (21 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `test_bl1.m` | Yes | ✅ | See `examples/black_litterman.md` |
| `test_bl2.m` | Yes | ✅ | Sensitivity analysis across 5 view scenarios |
| `test_bl3.m` | Yes | ✅ | Sigma-target MVO mode |
| `test_bl4.m` | Yes | ✅ | TE-target-matching mode |
| `test_bl5.m` | Yes | ✅ | TE-target-matching plus a full-view Black-Litterman example |
| `test_bl6.m` | Yes | ✅ | Raw `quadprog` call plus TE-target-matching and a gamma-recovery check |
| `test_box1.m` | Yes | ✅ | ERC + box-constrained RB at progressively wider bounds |
| `test_erc1.m` | Yes | ✅ | See `examples/risk_budgeting.md` |
| `test_erc2.m` | Yes | ✅ | |
| `test_erc3.m` | Yes | ✅ | |
| `test_lasso1.m` | Yes | ✅ | Unconstrained/budget-constrained MVO plus ridge/lasso |
| `test_lasso2.m` | Yes | ✅ | Representative points from the ridge/lasso static/dynamic sweep |
| `test_lasso3.m` | Yes | ✅ | Ridge/lasso penalties via `solve_qp` |
| `test_lasso4.m` | Yes | ✅ | Byte-identical to `test_lasso3.m` |
| `test_lasso5.m` | Yes | ✅ | Mixed ridge+lasso penalties toward two target vectors |
| `test_minvar1.m` | Yes | ✅ | See `examples/mean_variance.md` |
| `test_minvar2.m` | Yes | ✅ | MinVar with general linear constraints |
| `test_mvo1.m` | Yes | ✅ | See `examples/mean_variance.md` |
| `test_mvo2.m` | Yes | ✅ | Gamma-, mu-, and sigma-problem modes |
| `test_mvo3.m` | Yes | ✅ | Sigma-problem under 3 weight-bound configurations |
| `test_te1.m` | Yes | ✅ | See `examples/mean_variance.md` |

### stats/ (19 files; `ridge.inc` is a shared data snippet, not an example)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `count1.m` | No | — | Exercises `counts`/`countmiss`, superseded by `np.histogram`/`pd.value_counts` |
| `cov1.m` | Yes | ✅ | Closed-form OLS vs. Gaussian MLE, Hessian/OPG/HC standard errors |
| `elasticnet1.m` | Yes | ✅ | Elastic net path (alpha=0.5) |
| `elasticnet2.m` | Yes | ✅ | Same translation as `elasticnet1.m` (alpha=0.25) |
| `kalman1.m` | Yes | ✅ | Time-varying-beta model, box-constrained MLE |
| `kalman2.m` | Yes | ✅ | Same model as `kalman1.py`, deliberately bad start |
| `kernel1.m` | Yes | ✅ | Gaussian kernel density on two correlated series |
| `lasso1.m` | Yes | ✅ | Lambda-sweep with R²/df/complexity reporting |
| `lasso2.m` | Yes | ✅ | Penalized-form lasso path |
| `lasso3.m` | Yes | ⬜ | Uses `selectLasso` (lasso-path variable-selection ordering) — the one open item, see `stats/regression/lasso.py` above |
| `ml_ols.m` | No | — | Helper for `cov1.m`, inlined directly into `cov1.py` |
| `pca1.m` | Yes | ✅ | |
| `qreg1.m` | Yes | ✅ | Linear quantile regression at 9 tau levels |
| `qreg2.m` | Yes | ✅ | Local-linear/quadratic kernel mean & quantile regression |
| `quantile1.m` | No | — | Exercises `quantile_classification`/`unique2`, superseded by `pd.qcut`/`np.unique` |
| `quantile2.m` | No | — | Same reason as `quantile1.m` |
| `ridge1.m` | Yes | ✅ | See `examples/regression.md` |
| `ridge2.m` | Yes | ✅ | `ridge_tau_targeted` cross-checked against fixed-lambda `ridge` |

### svm/ (12 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `svm1.m` | Yes | ✅ | See `examples/svm.md` |
| `svm2.m` | Yes | ✅ | |
| `svm3.m` | Yes | ✅ | Matches `svm4.py` exactly (primal/dual duality) |
| `svm4.m` | Yes | ✅ | Matches `svm3.py` exactly |
| `svm5.m` | Yes | ✅ | Covered by `svm3`/`svm4` at 4 representative C values |
| `svm6.m` | Yes | ✅ | OLS/LAD/quantile/SVM comparison |
| `svm7.m` | Yes | ✅ | Dual formulation, matches `svm6.py`'s primal results |
| `svm8.m` | Yes | ✅ | Fixed-seed n=1000 synthetic dataset |
| `svm_regression_dual_theo.m` | Yes | ✅ | Function source, not an example — merged into `svm.py` |
| `svm_regression_primal_theo.m` | Yes | ✅ | Function source, not an example — merged into `svm.py` |
| `theo6.m` | Yes | ✅ | Duplicate of `svm6.m`'s scenario |
| `theo7.m` | Yes | ✅ | Duplicate of `svm6.m`'s scenario |

### tools/ (5 files)

| MATLAB file | Port? | Status | Notes |
|---|---|---|---|
| `hline1.m` | No | — | Display/formatting helper, not a ported function |
| `indnv1.m` | No | — | `indnv` superseded by NumPy indexing |
| `latex1.m` | No | — | LaTeX table export, use `pandas.to_latex` |
| `recode1.m` | No | — | `recode` superseded by `numpy.where`/pandas idioms |
| `retcode1.m` | No | — | Return-code display helper, not a standalone function |

---

### Summary

| Folder | Total | Not ported | To port | Done | Remaining |
|---|---|---|---|---|---|
| `backtest/` | 9 | 2 | 7 | 7 | 0 |
| `dates/` | 5 | 2 | 3 | 3 | 0 |
| `ects/` | 39 | 2 | 37 | 37 | 0 |
| `maths/` | 9 | 0 | 9 | 9 | 0 |
| `matrix/` | 9 | 4 | 5 | 5 | 0 |
| `optim/` | 11 | 1 | 10 | 10 | 0 |
| `rpb/` | 21 | 0 | 21 | 21 | 0 |
| `stats/` | 18 | 4 | 14 | 13 | 1 |
| `svm/` | 12 | 0 | 12 | 12 | 0 |
| `tools/` | 5 | 5 | 0 | 0 | 0 |
| **Total** | **138** | **20** | **118** | **117** | **1** |

The single remaining item is `stats/lasso3.m` / `selectLasso.m` — the
same gap as `stats/regression/lasso.py`'s 🟨 status above.

## Notes for translators

- Every module still using MATLAB `global` state (ADMM/CCD tolerances,
  MVO problem context, GMM/ML/Whittle settings) should take the
  corresponding dataclass from `quanttoolbox.config` as an explicit
  argument with a default instance.
- `crb/` and `rpb/` together account for ~95 MATLAB files, many of them
  near-identical solver variants (ADMM/CCD/Newton/fmincon x
  unconstrained/box/linear constraints). These collapse into
  `portfolio/risk_budgeting.py` as a single class — check the original
  file names only to confirm behavioral parity during testing, not to
  preserve a 1:1 file structure.
- MATLAB is 1-indexed and column-major; watch for off-by-one loop
  bounds and `reshape`/`vec` ordering differences when porting
  arithmetic directly.
- `quadprog(...)` calls (MATLAB Optimization Toolbox) map to
  `qpsolvers.solve_qp` or `cvxpy` — solver choice affects numerical
  tolerance, so regression-test against the MATLAB output where
  precision matters (e.g. SVM support vectors).
- When translating a formula that depends on a specific
  scaling/normalization convention (e.g. a `2π` factor, a `1/n`
  normalization), verify it against a known closed-form result or a
  numerical derivative rather than trusting the transcription —
  several bugs found during this port were exactly this kind of silent
  scaling-factor mismatch.
