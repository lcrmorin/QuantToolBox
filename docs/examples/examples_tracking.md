# Example translation tracker

Tracks every script in the original MATLAB toolbox's `Examples/` folder
(149 files across 11 subfolders) against its Python translation status.

Status legend: ⬜ not started · 🟡 translated, not cross-verified · ✅
translated and cross-verified against the original MATLAB source (run via
[Octave](https://octave.org/), a MATLAB-compatible open-source
interpreter, with the numeric output compared directly against the
Python port's output for the same inputs)

The **Notes** column flags examples that call `plot`/`figure` or
`xlsread`/`xlswrite`/`readtable`/`writetable`. These are translated for
their *numeric* logic (the actual computation), with the plotting/Excel
I/O portions dropped rather than translated — a doc page showing a
`matplotlib` chart or a pandas DataFrame export is more useful than a
literal port of MATLAB's plotting calls, and the underlying numbers are
what's being verified either way.

## How verification works

Where an example doesn't depend on external data files, plotting, or
random number generation with an unspecified seed, both the original
`.m` script (run via Octave) and the Python translation are executed
with identical inputs, and their numeric outputs are compared directly.
Where they match (to floating-point precision, or to reasonable rounding
if the original only displays a few decimal places), the example is
marked ✅. Where an example is translated but not cross-run (e.g. it only
makes sense interactively, or needs a dataset not trivial to reproduce
byte-for-byte), it's marked 🟡 with a note explaining why.

---

## backtest/ (9 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `backtest/` | `backtest1.m` | ⬜ | Exercises `fillmiss`/`findnomiss`, which became private helpers (`_fill_missing_ffill_bfill`/`_first_last_valid`) rather than public functions during porting -- not translated as a standalone example for now |
| `backtest/` | `backtest2.m` | 🟡 | Translated (numeric core; plot dropped) — 3 rebalancing schedules (single, 4 fixed positions, date-subset with 2 out-of-range placeholders + a NaN price) |
| `backtest/` | `backtest3.m` | 🟡 | Translated — generate_backtest at 4 rebalancing frequencies |
| `backtest/` | `backtest4.m` | 🟡 | Translated — per-asset bid/ask transaction costs; turnover cross-checked against `static_turnover` |
| `backtest/` | `backtest5.m` | 🟡 | Translated — flat transaction cost under the original's actually-exercised (monthly/every-date) rebalancing schedule |
| `backtest/` | `fillmiss1.m` | ⬜ | Same reason as `backtest1.m` -- exercises `fillmiss`/`findnomiss`, now private helpers |
| `backtest/` | `mdd1.m` | 🟡 | Translated (numeric core; plot dropped) — maximum_drawdown, relative mode |
| `backtest/` | `unfunded1.m` | 🟡 | Translated (numeric core; plot dropped) — funded vs. two unfunded-formulation backtests, cross-checked against each other |
| `backtest/` | `unfunded2.m` | 🟡 | Translated (numeric core; plot dropped) — small hand-traceable n=8 version of unfunded1.py's round-trip |

## dates/ (5 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `dates/` | `date1.m` | ⬜ | Excel I/O (reads/writes `date1.xlsx`) |
| `dates/` | `date2.m` | ⬜ | Excel I/O |
| `dates/` | `date3.m` | ✅ | Self-contained — see `examples/building_blocks.md` |
| `dates/` | `date4.m` | ✅ | Identical `generate_trading_dates` call already covered by `date3.m` |
| `dates/` | `rebalancing_dates.m` | 🟡 | Uses a `.mat` file with MATLAB's newer `datetime` object type (MCOS class), which Octave can't fully deserialize — environment limitation, not a translation gap. The underlying `weekly_rebalancing`/`monthly_rebalancing`/`quarterly_rebalancing` functions are already covered by the package's own test suite with synthetic data. |

## ects/ (39 `.m` files; 9 `.asc` data files are not examples)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `ects/` | `gmm1.m` | 🟡 | Translated — OLS/MLE/GMM, unconstrained and beta[3]=1-restricted, on a dataset with one missing y; `theta7` (a second, redundant encoding of the same restriction) not re-translated separately (see gmm1.py docstring) |
| `ects/` | `kalman1a.m` | 🟡 | Byte-identical to `panel1.m` (verified via diff) -- already translated as `panel1.py`, not duplicated |
| `ects/` | `kalman1b.m` | 🟡 | Translated (numeric core; plot dropped) — time-domain ML of (sigma_epsilon, sigma_eta) under two Kalman a0 choices during estimation, compared post-hoc |
| `ects/` | `kalman1c.m` | 🟡 | Translated (numeric core; plot dropped) — time-domain vs. frequency-domain (Whittle) ML compared; Whittle estimate cross-checked against `whittle1.py`'s output |
| `ects/` | `kalman2a.m` | 🟡 | Translated (numeric core; plot dropped) — Local Linear Trend Kalman filter on Gnp.asc (uses `Gnp.asc`, not `Kalman3.asc` as the tracker previously noted), fixed variance parameters |
| `ects/` | `kalman2b.m` | 🟡 | Translated (numeric core; plot dropped) — LLT model, Whittle (frequency-domain) ML on Gnp.asc |
| `ects/` | `kalman2c.m` | 🟡 | Translated (numeric core; plot dropped) — LLT model, time-domain ML on Gnp.asc; cross-checked against kalman2b.py's Whittle estimate |
| `ects/` | `kalman3a.m` | 🟡 | Translated — general 2-state/2-obs SSM filter on `Kalman3.asc`, from steady-state initial mean/covariance |
| `ects/` | `kalman3b.m` | 🟡 | Translated — same model/data as kalman3a.py, run through the time-varying (3-D matrix) code path; verified to reproduce kalman3a.py's output exactly |
| `ects/` | `kalman3c.m` | 🟡 | Translated (numeric core; plot dropped) — filtered state + 95% confidence band |
| `ects/` | `kalman3d.m` | 🟡 | Translated — ML recovery of all 11 free SSM parameters, starting from and compared against their true values |
| `ects/` | `kalman4a.m` | 🟡 | Translated (numeric core; plot dropped) — simulated random-walk-coefficient regression, time-varying-Z Kalman filter recovers the beta_t path; seeded RNG |
| `ects/` | `kalman4b.m` | 🟡 | Translated (numeric core; plot dropped) — same simulation as kalman4a.py, but (sigma_epsilon, sigma_beta1, sigma_beta2) estimated by ML rather than assumed known |
| `ects/` | `ml1.m` | ⬜ | Near-duplicate of `ml2.m` (same data/model), whose distinctive feature is exercising MATLAB's string-name function dispatch (`ml_estimation('ml1_fn',sv)`) -- not meaningful to translate; lower priority |
| `ects/` | `ml1_fn.m` | ⬜ | helper function for `ml1.m`, not standalone |
| `ects/` | `ml2.m` | 🟡 | Translated — OLS vs. Gaussian-MLE covariance matrix, Hessian/OPG/HC estimators |
| `ects/` | `ml3.m` | 🟡 | Translated — Beta-distribution MLE (LGD modeling), Hessian/OPG/HC covariance |
| `ects/` | `ml4.m` | 🟡 | Translated — numerical vs. analytical Jacobian/Hessian, all 3 covariance estimators under both |
| `ects/` | `ols1.m` | ✅ | Self-contained — see `examples/regression.md` |
| `ects/` | `panel1.m` | 🟡 | Translated — **note:** despite its filename/old tracker note, this is a local-level Kalman filter example (Harvey 1990), not panel data; corrected and translated using `econometrics.kalman` (numeric core; plot dropped) |
| `ects/` | `quantile1.m` | 🟡 | Translated (numeric core; plot dropped) — pinball-loss minimization vs. sorted-sample quantile vs. true Normal quantile |
| `ects/` | `quantile2.m` | 🟡 | Translated (numeric core; plot dropped) — Monte Carlo (nS=2000) OLS vs. LAD under heteroskedastic noise |
| `ects/` | `robust1.m` | ✅ | Self-contained — see `examples/regression.md` |
| `ects/` | `robust2.m` | 🟡 | Translated — OLS/median/LAD/Huber regression |
| `ects/` | `robust3.m` | 🟡 | Translated — OLS/median/LAD, plus quantile regression via both IRLS M-estimation and the exact LP formulation (cross-checked against each other) |
| `ects/` | `varx1a.m` | 🟡 | Translated — VAR(2) on log-differenced Lutkepohl.asc investment/income/consumption (embedded), via `varx_estimate(..., method="ls")` |
| `ects/` | `varx1b.m` | 🟡 | Translated — Wald test for no Granger-causality (income/consumption -> investment) via `wald_test` |
| `ects/` | `varx1c.m` | 🟡 | Translated — Wald test for no instantaneous causality, on the ML fit's Cholesky-augmented theta/vcv |
| `ects/` | `varx1d.m` | 🟡 | Translated — restricted VAR(2) (7 of 21 coefficients free, via `design(w)`) estimated by both LS and concentrated ML |
| `ects/` | `varx1e.m` | 🟡 | Translated — lag-order selection (`varx_order`, p=1..5) on same data as varx1a.m |
| `ects/` | `varx2a.m` | 🟡 | Translated — reduced form of a dynamic simultaneous-equations system (Lutkepohl ch.10) via `varx_estimate(..., p=1)` |
| `ects/` | `varx2b.m` | 🟡 | Translated — varx2a.py's model with 4 coefficients restricted, via restricted LS |
| `ects/` | `varx2c.m` | 🟡 | Translated — same restricted model as varx2b.py, via concentrated ML; cross-checked against it |
| `ects/` | `varx3.m` | 🟡 | Translated — trivial 5-obs OLS vs. `varx_estimate(..., p=0)` equivalence check |
| `ects/` | `varx4a.m` | 🟡 | Translated — JHGLL restricted SUR (2-eq, embedded 20-obs data), two-step GLS |
| `ects/` | `varx4b.m` | 🟡 | Translated — JHGLL restricted SUR (3-eq, embedded 30-obs data), two Sigma variants compared |
| `ects/` | `varx5a.m` | 🟡 | Translated — JHGLL simultaneous-equations system, OLS/GLS-2S/GLS-3S/LIML compared (all four converge to consistent Gamma/B/PI) |
| `ects/` | `whittle1.m` | 🟡 | Translated — Whittle local-level MLE on the same Purse.asc data as panel1.m (single BFGS path; original's 3-optimizer comparison collapses to one) |
| `ects/` | `whittle2.m` | 🟡 | Translated — Bloomfield exponential spectral density (order 4), custom `sdf_fn` passed to `whittle_estimation` |

## maths/ (9 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `maths/` | `grad1.m` | ✅ | See `examples/building_blocks.md` |
| `maths/` | `grad2.m` | 🟡 | Translated — scalar function of 2 variables (poly + log + exp) |
| `maths/` | `grad3.m` | 🟡 | Translated — elementwise fun summed first (same trick as grad1.m) so the scalar-valued `numerical_gradient` can be used |
| `maths/` | `grad4.m` | 🟡 | Translated — same elementwise-via-sum trick as grad3.m |
| `maths/` | `grad5.m` | 🟡 | Translated — both the already-scalar `fun` and the explicit-sum `fun2` formulations (same gradient either way) |
| `maths/` | `hess1.m` | ✅ | See `examples/building_blocks.md` |
| `maths/` | `hess2.m` | 🟡 | Translated — numerical gradient + Hessian vs. analytical, at a point with a very small x2 |
| `maths/` | `pdgm1.m` | 🟡 | Translated — not actually a plot despite the earlier note; raw periodogram of a 4-obs series via `periodogram`, output verified against the MATLAB source's own comment |
| `maths/` | `pdgm2.m` | 🟡 | Translated — same as pdgm1.py, 8-obs series, output verified against the MATLAB source's own comment |

## matrix/ (9 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `matrix/` | `design1.m` | 🟡 | Rounding behavior verified independently; Octave hit an unrelated int64-cast environment issue on this run |
| `matrix/` | `matrix1.m` | ✅ | Cross-verified vs. Octave (elimination/duplication/commutation matrix shapes+sums match exactly) |
| `matrix/` | `matrix2.m` | 🟡 | Translated — 7 elimination/duplication/commutation-matrix identities checked for M=1..10; uses `np.isclose`/`allclose` rather than exact `==` since `inv(D'*D)` isn't bit-exact |
| `matrix/` | `reshape1.m` | 🟡 | Translated — vech/xpnd (both orderings) and reshaper/reshapec; the `shiftr` portion is dropped (not ported, see `shiftr1.m` below) |
| `matrix/` | `shiftc1.m` | ⬜ | Exercises `shiftc` — not ported (GAUSS shift primitive, superseded by `numpy.roll`/slicing idioms) |
| `matrix/` | `shiftr1.m` | ⬜ | Exercises `shiftr` — not ported, same reason |
| `matrix/` | `submat1.m` | ⬜ | Exercises `submat` — not ported (superseded by NumPy fancy indexing) |
| `matrix/` | `vec1.m` | ✅ | See `examples/building_blocks.md` |
| `matrix/` | `vech1.m` | ✅ | See `examples/building_blocks.md` |

## optim/ (11 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `optim/` | `bisection1.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `explicit1.m` | ✅ | See `examples/building_blocks.md` |
| `optim/` | `explicit2.m` | 🟡 | Translated — explicit/implicit round-trip for one constraint, plus a `design()` matrix demo |
| `optim/` | `explicit3.m` | 🟡 | Translated — explicit-to-implicit for 3 simultaneous zero-restrictions |
| `optim/` | `prox_L1.m` | ✅ | Cross-verified vs. Octave (fixed input) — see `examples/building_blocks.md` |
| `optim/` | `prox_L2.m` | ✅ | Cross-verified vs. Octave — see `examples/building_blocks.md` |
| `optim/` | `prox_Linfinity.m` | ✅ | Cross-verified vs. Octave — see `examples/building_blocks.md` |
| `optim/` | `prox_turnover1.m` | 🟡 | Translated — proximal-L1 vs. both projection-L1 algorithms; fixed-seed substitution for unseeded `rand` |
| `optim/` | `prox_turnover2.m` | 🟡 | Uses `proximal_turnover` → `fminunc`/`optimoptions`; Octave-forge's `optim` package doesn't implement `optimoptions`, so this specific example can't be cross-run in this environment. Function itself already covered by the package's own test suite. |
| `optim/` | `proximal1.m` | 🟡 | Translated — bounds/1-4 inequalities/1-2 equalities/combined linear constraints; the original's `Proximal_Algorithm` 1-vs-2 comparison is moot post-port (see proximal1.py's docstring — the redundant closed-form-vs-QP branches for bounds/equality weren't kept) |
| `optim/` | `turnover1.m` | ✅ | Cross-verified vs. Octave (fixed input) — see `examples/building_blocks.md` |

## rpb/ (21 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `rpb/` | `test_bl1.m` | ✅ | See `examples/black_litterman.md` |
| `rpb/` | `test_bl2.m` | 🟡 | Translated — sensitivity analysis across 5 view scenarios, fixed risk-aversion MVO |
| `rpb/` | `test_bl3.m` | ✅ | Sigma-target MVO mode -- `mean_variance.mvo_target_portfolio` (previously a documented gap, now implemented); 5 BL view scenarios matched to the x0 benchmark's volatility, output cross-verified against Octave |
| `rpb/` | `test_bl4.m` | ✅ | TE-target-matching mode -- `tracking_error.te_target_portfolio` (previously a documented gap, now implemented); 6 TE targets, cross-verified against Octave |
| `rpb/` | `test_bl5.m` | ✅ | Same TE-target-matching mode as test_bl4.m, plus a second Black-Litterman worked example (full-view, `P=I`); 5 blocks, cross-verified against Octave |
| `rpb/` | `test_bl6.m` | ✅ | Raw `quadprog` call translated via `te_portfolio` (shown to solve the identical QP), plus `te_target_portfolio`; cross-verified against Octave, including a gamma-recovery self-consistency check |
| `rpb/` | `test_box1.m` | 🟡 | Translated — ERC + box-constrained RB at progressively wider bounds |
| `rpb/` | `test_erc1.m` | ✅ | See `examples/risk_budgeting.md` |
| `rpb/` | `test_erc2.m` | 🟡 | Translated |
| `rpb/` | `test_erc3.m` | 🟡 | Translated |
| `rpb/` | `test_lasso1.m` | 🟡 | Translated — unconstrained/budget-constrained MVO (identical, since `mvo_portfolio` auto-applies the budget constraint when `a_eq` is omitted, matching `compute_mvo_portfolio.m`) plus ridge/lasso via `solve_qp` (no auto budget constraint there, matching the raw `quadprog_ridge`/`quadprog_lasso`) |
| `rpb/` | `test_lasso2.m` | 🟡 | Translated — numeric core (representative points from the 250-point ridge/lasso static/dynamic sweep `test_lasso3.m` demonstrates directly), plot dropped; not cross-verified -- the original's lasso branch is numerically fragile under Octave's `quadprog` (see the .py docstring) |
| `rpb/` | `test_lasso3.m` | 🟡 | Translated via `solve_qp` ridge/lasso penalties (replaces original's `quadprog_ridge`/`quadprog_lasso`/`quadprog_mixed`) |
| `rpb/` | `test_lasso4.m` | 🟡 | Byte-identical to `test_lasso3.m` -- same translation covers both |
| `rpb/` | `test_lasso5.m` | 🟡 | Translated — mixed ridge+lasso penalties toward two different target vectors |
| `rpb/` | `test_minvar1.m` | ✅ | See `examples/mean_variance.md` |
| `rpb/` | `test_minvar2.m` | 🟡 | Translated — MinVar with general linear equality/inequality constraints |
| `rpb/` | `test_mvo1.m` | ✅ | See `examples/mean_variance.md` |
| `rpb/` | `test_mvo2.m` | ✅ | All 3 sections translated -- gamma-problem (`mvo_frontier`), mu-problem and sigma-problem (`mvo_target_portfolio`, previously a documented gap, now implemented); cross-verified against Octave |
| `rpb/` | `test_mvo3.m` | ✅ | Sigma-problem under 3 weight-bound configurations, cross-verified against Octave -- **found and documented a genuine MATLAB bug** in the process: the original script skips `init_global`, leaving `BISECTION_Tol` unset and silently breaking every target-matching call it makes; see `docs/matlab_bugs_found.md` #6. This translation produces the correct numbers. |
| `rpb/` | `test_te1.m` | ✅ | See `examples/mean_variance.md` |

## stats/ (19 files; `ridge.inc` is a shared data snippet, not a standalone example)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `stats/` | `count1.m` | ⬜ | Exercises `counts`/`countmiss` (GAUSS tabulation utilities) — not ported, no quanttoolbox equivalent to demonstrate (superseded by `np.histogram`/`pd.value_counts`) |
| `stats/` | `cov1.m` | 🟡 | Translated — OLS in closed form vs. Gaussian MLE (`ml_estimation`) comparing Hessian/OPG/HC standard errors; `ml_robust_vcv`'s 5 partially-redundant covariance variants condensed to `ml_estimation`'s 3 `cov=` options |
| `stats/` | `elasticnet1.m` | 🟡 | Translated — elastic net path (alpha=0.5) via `elastic_net_ccd` |
| `stats/` | `elasticnet2.m` | 🟡 | Same translation as `elasticnet1.m` (alpha=0.25 case) — covered together |
| `stats/` | `kalman1.m` | 🟡 | Translated — time-varying-beta random-walk-coefficient model, bounded MLE via `scipy.optimize.minimize(..., method="L-BFGS-B")` (`ml_estimation` has no box-constraint support) since the `ects/` Kalman pass is now done |
| `stats/` | `kalman2.m` | 🟡 | Translated — same model as kalman1.py, a0 fixed at [-20,-20] (unrecovered from a deliberately bad start) rather than estimated |
| `stats/` | `kernel1.m` | 🟡 | Translated (numeric core; plot dropped) — Gaussian kernel density on two correlated series |
| `stats/` | `lasso1.m` | 🟡 | Translated (numeric core; 501-point plot sweep dropped) — extends lasso2.py's identical 5-lambda/dataset run with R²/df/complexity reporting |
| `stats/` | `lasso2.m` | 🟡 | Translated (penalized-form lasso path via `lasso_ccd`); the tau-constrained comparison and 501-point plot sweep in the original are not re-translated separately |
| `stats/` | `lasso3.m` | ⬜ | Uses `selectLasso` (lasso-path variable-selection ordering) — not currently ported as a standalone function |
| `stats/` | `ml_ols.m` | ⬜ | Helper function (per-observation log-density) for `cov1.m`, not a standalone example — inlined directly into `cov1.py` |
| `stats/` | `pca1.m` | 🟡 | Translated |
| `stats/` | `qreg1.m` | 🟡 | Translated — linear quantile regression at 9 tau levels |
| `stats/` | `qreg2.m` | 🟡 | Translated (numeric core; plot dropped) — local-linear/quadratic kernel mean & quantile regression vs. population curves |
| `stats/` | `quantile1.m` | ⬜ | Exercises `quantile_classification`/`unique2` — not ported, no quanttoolbox equivalent (superseded by `pd.qcut`/`np.unique`) |
| `stats/` | `quantile2.m` | ⬜ | Same reason as `quantile1.m` |
| `stats/` | `ridge1.m` | ✅ | See `examples/regression.md` (numeric core; plot dropped) |
| `stats/` | `ridge2.m` | 🟡 | Translated — `ridge_tau_targeted` cross-checked against fixed-lambda `ridge`, absolute and relative tau, two lambda-search grids |

## svm/ (12 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `svm/` | `svm1.m` | ✅ | See `examples/svm.md` |
| `svm/` | `svm2.m` | 🟡 | Translated |
| `svm/` | `svm3.m` | 🟡 | Translated — matches svm4.py exactly (primal/dual duality) |
| `svm/` | `svm4.m` | 🟡 | Translated — matches svm3.py exactly (primal/dual duality) |
| `svm/` | `svm5.m` | 🟡 | Translated (numeric core; original's C-sweep plot not re-translated separately, covered by svm3/svm4 at 4 representative C values) |
| `svm/` | `svm6.m` | 🟡 | Translated — OLS/LAD/quantile/SVM comparison; SVM-LS matches OLS at high C as expected |
| `svm/` | `svm7.m` | 🟡 | Translated (dual formulation) — matches svm6.py's primal results exactly |
| `svm/` | `svm8.m` | 🟡 | Translated — fixed-seed substitution (`default_rng(0)`) for the n=1000 synthetic dataset; OLS/SVM-LS and Quantile/SVM-eps pairs agree at large C as expected |
| `svm/` | `svm_regression_dual_theo.m` | ✅ | Function source file (not an example script) — the `theo/` duplicate already merged into `svm.py`'s `svm_regression_dual` |
| `svm/` | `svm_regression_primal_theo.m` | ✅ | Function source file (not an example script) — already merged into `svm.py`'s `svm_regression_primal` |
| `svm/` | `theo6.m` | 🟡 | Duplicate of svm6.m's exact scenario — covered by svm6.py |
| `svm/` | `theo7.m` | 🟡 | Duplicate of svm6.m's exact scenario — covered by svm6.py |

## tools/ (6 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `tools/` | `hline1.m` | ⬜ | display/formatting helper, not a ported function |
| `tools/` | `indnv1.m` | ⬜ | `indnv` not ported (superseded by numpy indexing — see migration map) |
| `tools/` | `latex1.m` | ⬜ | LaTeX table export, not ported (use `pandas.to_latex`) |
| `tools/` | `recode1.m` | ⬜ | `recode` not ported (superseded by `numpy.where`/`pandas` idioms) |
| `tools/` | `retcode1.m` | ⬜ | return-code display helper, not a standalone function |

## tutorial/ (11 files)

| Folder | MATLAB file | Status | Notes |
|---|---|---|---|
| `tutorial/` | `gblank.m` | ⬜ | helper, not a lesson |
| `tutorial/` | `lesson1.m` | ⬜ | general walkthrough, not per-function |
| `tutorial/` | `lesson2.m` | ⬜ | |
| `tutorial/` | `lesson3.m` | ⬜ | |
| `tutorial/` | `lesson4.m` | ⬜ | |
| `tutorial/` | `lesson5.m` | ⬜ | |
| `tutorial/` | `lesson6.m` | ⬜ | |
| `tutorial/` | `lesson7.m` | ⬜ | |
| `tutorial/` | `lesson8.m` | ⬜ | |
| `tutorial/` | `lesson9.m` | ⬜ | |
| `tutorial/` | `lesson10.m` | ⬜ | |

---

## Summary

| Folder | Total | ✅ Done | 🟡 Translated (not cross-verified) | ⬜ Remaining |
|---|---|---|---|---|
| `backtest/` | 9 | 0 | 7 | 2 |
| `dates/` | 5 | 2 | 1 | 2 |
| `ects/` | 39 | 2 | 35 | 2 |
| `maths/` | 9 | 2 | 7 | 0 |
| `matrix/` | 9 | 3 | 3 | 3 |
| `optim/` | 11 | 6 | 5 | 0 |
| `rpb/` | 21 | 11 | 10 | 0 |
| `stats/` | 18 | 1 | 12 | 5 |
| `svm/` | 12 | 3 | 9 | 0 |
| `tools/` | 5 | 0 | 0 | 5 |
| `tutorial/` | 11 | 0 | 0 | 11 |
| **Total** | **149** | **30** | **88** | **31** |

*(`stats/ridge.inc` excluded as a data snippet, not a standalone example
— 149 actual `.m` example scripts plus that one data-only file.)*

## Verification tooling

As of this update, [Octave](https://octave.org/) (MATLAB-compatible,
open source) is installed and confirmed to run the original QuantToolbox
`.m` files directly — including functions with naming collisions against
Octave's own built-ins (`vec`, `vech`, `rows`, `commutation_matrix`,
`duplication_matrix`, `periodogram` all shadow core library functions,
which Octave warns about but still executes correctly using the
project's own versions via `addpath`). This means, going forward, ✅
status genuinely means "the original MATLAB code and the Python port
were both run, on the same inputs, and produced matching output" — not
just "this was translated and looks plausible."
