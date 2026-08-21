# Python library alternatives: what to keep, what to switch

For every ported module, this document names the closest existing Python
library alternative (if one exists) and makes an honest call on whether
to keep the ported implementation, switch to the library, or use both
for different purposes. The guiding question throughout is: **does the
ported code do something the alternative genuinely can't, or is it mostly
duplicating well-trodden ground?**

Three verdicts are used:

- **KEEP** — the ported code does something the alternative doesn't
  (a specific parameterization, a quant-specific measure, a missing
  feature), so there's a real reason to maintain it.
- **SWITCH** — a mature, well-tested library already does this better;
  the ported code is mostly worth keeping as a reference/fallback, not
  as the primary path.
- **HYBRID** — use the library for the common case, keep the ported code
  for a specific capability it has that the library lacks.

---

## Dates (`dates/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `convert.py` | `pandas` (already the base) | **KEEP** | Pandas has no built-in Excel-serial-date conversion; this is a thin, genuinely useful wrapper. |
| `rebalancing.py` | [`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) | **HYBRID** | Our rebalancing dates are weekday-only (no exchange holiday calendar) — that's a real gap versus the original MATLAB behavior too. `pandas_market_calendars` gives real NYSE/LSE/etc. holiday calendars; worth wiring in as an optional calendar source for production backtests, while keeping our nearest-available-date snapping logic (which the library doesn't replicate exactly). |

---

## Stats (`stats/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `distributions.py` — simple wrappers (normal/t/chi2/F/MVN) | `scipy.stats` (already the backend) | **KEEP** | These are one-line wrappers around scipy; no separate module needed downstream, but no reason to remove them either — call-site compatibility with the original toolbox. |
| `distributions.py` — GQF1/GQF2 | *(none)* | **KEEP** | Genuinely niche (generalized quadratic-form distributions for Delta-Gamma VaR). Nothing in scipy, statsmodels, or elsewhere in the ecosystem does this. |
| `moments.py` — `rolling_correlation`/`rolling_volatility` | `pandas.DataFrame.rolling().corr()`/`.std()` | **SWITCH** | Pandas' rolling ops are implemented in Cython and are meaningfully faster than our Python-loop version for anything beyond toy sizes. The only reason to keep ours is the specific `method=2` "compute returns within each window" variant for illiquid series, which pandas doesn't offer directly. |
| `moments.py` — `active_share`, `herfindahl_index`, `asynchronous_cov`, `weekly_cov` | *(none)* | **KEEP** | Portfolio-construction-specific measures with no general-purpose library equivalent. |
| `regression/ols.py` | `statsmodels.OLS`/`WLS` | **HYBRID** | For plain (unrestricted) OLS, statsmodels is more complete (more diagnostics, better-tested edge cases). Ours is worth keeping specifically for the `restriction=(RR, r)` linear-restriction parameterization, which statsmodels doesn't support as directly. |
| `regression/ridge.py` | `sklearn.linear_model.Ridge`/`RidgeCV` | **HYBRID** | sklearn's solver is more optimized for large/sparse problems. Keep ours for the `ridge_tau_targeted` (L2-norm-budget, not penalty) parameterization — sklearn has no equivalent for that. |
| `regression/lasso.py` | `sklearn.linear_model.Lasso`/`ElasticNet`/`LassoCV` | **SWITCH** (for the penalized-form solvers) | sklearn's coordinate descent is Cython-compiled and extensively battle-tested; there's no good reason to keep hand-rolled `lasso_ccd`/`lasso_admm` for the standard penalized case — recommend routing those through sklearn directly. **KEEP** `lasso_tau_constrained` specifically, since sklearn has no L1-*budget* (as opposed to L1-*penalty*) interface. |
| `regression/kernel.py` | `statsmodels.nonparametric.KernelReg` | **HYBRID** | statsmodels' version supports automatic bandwidth selection via cross-validation and both local-constant/local-linear estimators — more complete than our fixed-bandwidth port. Worth switching to for general use; keep ours only where the exact original bandwidth formula needs to be reproduced for parity with existing MATLAB-based results. |
| `regression/quantile.py` | `statsmodels.regression.quantile_regression.QuantReg`, or `sklearn.linear_model.QuantileRegressor` (newer sklearn) | **SWITCH** | Both are more battle-tested than our `scipy.optimize.linprog`-based implementation, handle edge cases (rank-deficient design matrices, ties) more robustly, and QuantReg in particular has been used in production statsmodels code for years. Keep ours only if the exact slack-variable (`u`, `v`) outputs are needed downstream. |
| `regression/robust.py` | `statsmodels.robust.robust_linear_model.RLM` | **HYBRID** | RLM covers Huber, Tukey biweight, Andrew's wave, Hampel, and trimmed-mean M-estimators via IRLS — a superset of our Huber implementation, and more robustly tested. It does *not* cover LAD or quantile M-estimation directly (though `QuantReg(q=0.5)` is exactly LAD). Recommend: `RLM` for Huber/general M-estimation, keep our `lad_regression`/`quantile_m_regression`/`inverse_quantile_m_regression` for those specific losses. |

---

## Econometrics (`econometrics/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `estimation.py` (GMM/ML) | `statsmodels.sandbox.regression.gmm.GMM`, `statsmodels.base.model.GenericLikelihoodModel`, or the [`linearmodels`](https://github.com/bashtage/linearmodels) package (more modern GMM/IV support) | **HYBRID** | These libraries are more mature for standard GMM/MLE use cases. Keep ours specifically for the explicit `theta = RR @ gamma + r` linear-restriction interface, which none of the alternatives expose as directly — that parameterization is the main reason this module exists as custom code at all. |
| `var.py` | `statsmodels.tsa.api.VAR` | **SWITCH** for the unrestricted case | statsmodels' VAR is comprehensive: automatic lag-order selection, impulse response functions, forecast error variance decomposition, forecasting — well beyond what we ported, and it's the standard tool the econometrics community actually uses. **KEEP** `varx_estimate`'s linear-restriction support (`a_eq`/`b_eq` on the stacked coefficient vector) — statsmodels' VAR doesn't support arbitrary parameter restrictions. |
| `kalman.py` | `statsmodels.tsa.statespace.MLEModel`/`kalman_filter` | **SWITCH** for anything beyond simple filtering | statsmodels' state-space framework is dramatically more capable: smoothing (not just filtering), MLE parameter fitting built in, diffuse initialization, a compiled Cython backend for real performance on long series. Our `kalman_filter` is fine for simple, transparent filtering tasks or minimal-dependency use, but statsmodels is the better choice for anything production-scale. |
| `whittle.py` | *(none found)* | **KEEP** | Whittle (frequency-domain) estimation isn't implemented in statsmodels, `arch`, or other common packages. Genuinely fills a gap. |
| `tests.py` (ADF) | `statsmodels.tsa.stattools.adfuller` (already used) | **SWITCH** (already done) | Also worth knowing: the [`arch`](https://github.com/bashtage/arch) package's `arch.unitroot` module has a wider family of unit-root tests (ADF, Phillips-Perron, DFGLS, KPSS, Zivot-Andrews) if more than ADF is ever needed — not currently ported, but worth knowing about. |

---

## Optimization (`optim/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `proximal.py`, `projection.py` | [`pyproximal`](https://github.com/PyLops/pyproximal) covers some overlapping norm/constraint proximal operators | **KEEP** | Low-level building blocks with no strong off-the-shelf equivalent covering our exact operator set (turnover constraints, combined Dykstra-projected linear+box systems for portfolio construction specifically). `pyproximal` is oriented toward signal-processing/inverse-problems use cases, not portfolio constraints. |
| `quadprog.py` (`solve_qp`) | `qpsolvers.solve_qp` directly (already a declared dependency, currently used *underneath* cvxpy) | **HYBRID** | For hot loops that call `solve_qp` many times (e.g. risk-budgeting's inner ADMM loop), calling `qpsolvers.solve_qp` directly — bypassing cvxpy's DSL-parsing overhead — would likely be measurably faster. Worth profiling if performance in tight loops becomes a concern; keep the cvxpy-based version for its more expressive constraint-building (ridge/lasso penalty terms, arbitrary constraint composition), which raw `qpsolvers` doesn't offer. |
| `bisection.py` | `scipy.optimize.brentq`/`bisect` | **HYBRID** | scipy's `brentq` converges faster (superlinear, not just bisection) for scalar root-finding, and is compiled. **KEEP** our vectorized (array-broadcast, many-roots-at-once) version — scipy's root finders are scalar-only, and vectorizing over many independent brackets is exactly what our version adds. |

---

## Portfolio (`portfolio/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `mean_variance.py`, `black_litterman.py`, `tracking_error.py` | [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt) | **HYBRID** | PyPortfolioOpt is a mature, actively maintained library covering mean-variance optimization, Black-Litterman, CVaR, discrete allocation, and more — genuinely worth using directly for standard portfolio construction tasks. Keep ours for tighter integration with the rest of this codebase (shared `solve_qp`, direct ridge/lasso penalty composability) and where the exact original parameterization needs to match existing analysis. |
| `risk_budgeting.py` | [`riskparityportfolio`](https://github.com/mirca/riskparityportfolio) (PyPI, narrower scope) | **KEEP** | `riskparityportfolio` covers basic risk parity but not the box-constrained/general-linear-constrained/VaR-ES/target-matching breadth this module has. This is the single largest and most-tested piece of custom work in the whole port (75+ original files consolidated) — genuinely the strongest "keep" case in this document. |

---

## Mixtures (`mixtures/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `gaussian_mixture.py` — `estimate_em_mixture` | `sklearn.mixture.GaussianMixture` | **HYBRID** | sklearn's EM implementation is more numerically robust (covariance regularization, multiple initializations, convergence diagnostics) and well-tested for the *general* n-component case. Worth switching to for the pure model-fitting step. **KEEP** everything downstream of fitting — VaR/ES, risk contribution, risk budgeting, PDF/skewness under the mixture — since sklearn's `GaussianMixture` has none of that; it only fits parameters. |
| `jump_diffusion.py` | *(none)* | **KEEP** | Jump-diffusion-specific risk measures; no general-purpose equivalent exists. |

---

## SVM (`svm/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `svm.py` | `sklearn.svm.SVC`/`SVR` | **SWITCH** for standalone use | sklearn's SVM is backed by `libsvm`/`liblinear` (compiled C, extremely well optimized and battle-tested) — already verified to match our implementation to 3+ decimal places. For any standalone classification/regression task, use sklearn directly; it will be faster and more robust to edge cases (kernel tricks, class imbalance handling, probability calibration — none of which we ported). **KEEP** ours only if the SVM needs to be composed with other constraints inside the same `solve_qp`/cvxpy optimization (e.g. embedding an SVM-like margin constraint inside a larger portfolio problem) — that composability is the one thing sklearn's opaque solver can't offer. |

---

## Spline (`spline/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `spline.py` | `scipy.interpolate.CubicSpline` (pure interpolation), `scipy.interpolate.UnivariateSpline`/`make_smoothing_spline` (smoothing) | **SWITCH** for general use | Already verified our `p=1` case matches `CubicSpline` to machine precision — scipy's tools are more mature, better-tested, and actively maintained. The one real friction point: our `p` parameter (a `[0,1]` interpolation/smoothing blend) and scipy's `s` parameter (a target sum-of-squared-residuals) are *different parameterizations* of smoothness, not directly interchangeable. **KEEP** ours only where exact `p`-parameterized behavior needs to match existing MATLAB-based configs or analysis. |

---

## Maths (`maths/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `numerical_diff.py` | [`numdifftools`](https://github.com/pbrod/numdifftools) | **HYBRID** | `numdifftools` uses adaptive step sizing and Richardson extrapolation — meaningfully more accurate than our fixed-step-size approach, especially for ill-conditioned functions. Worth using where precision matters (e.g. as an alternative Hessian source for MLE standard errors). Keep ours for the specific magnitude-scaled step convention already wired into `econometrics.estimation`/`whittle`, to avoid adding a dependency for something already working correctly. |
| `simulation.py` — `compute_ewma` | `pandas.DataFrame.ewm()` | **HYBRID** | Pandas' `.ewm()` is a highly optimized, feature-complete exponentially-weighted accessor. The friction: pandas parameterizes by `alpha`/`span`/`halflife` (smoothing factor), while our `compute_ewma` uses `lambda_` as a mean-reversion *rate* with explicit `dt` scaling — a different but related convention requiring a small translation layer (`alpha = lambda_ * dt`, roughly) to switch cleanly. Worth doing if performance on very long series matters; keep ours as-is for now given the interface is already wired into `volatility_target`/`momentum_ewma`. |
| `simulation.py` — GBM simulators, Riccati/Lyapunov | `scipy.linalg` (already the backend for Riccati/Lyapunov) | **KEEP** | Riccati/Lyapunov already route through `scipy.linalg.solve_continuous_are`/`solve_lyapunov` — this *is* the "switch" already done. GBM simulation has no strong standard-library equivalent at this level of simplicity (QuantLib-Python exists but is a much heavier dependency for far more sophisticated PDE/Monte Carlo needs than this module targets). |

---

## Backtest (`backtest/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `returns.py`, `stats.py`, `reporting.py` | [`vectorbt`](https://github.com/polakowo/vectorbt), `bt`, `backtrader`, `zipline` | **KEEP** | These are full backtesting *frameworks* — heavier dependencies, different paradigms (event-driven for backtrader/zipline vs. vectorized), and a much bigger surface area than this module's lightweight, transparent, vectorized style suits. `vectorbt` specifically is worth knowing about if backtest performance on very large universes/long histories becomes a bottleneck (it's Numba-accelerated), but it's a genuinely different tool, not a drop-in replacement for this module's scope. |

---

## Linear algebra (`linalg/`)

| Module | Alternative | Verdict | Why |
|---|---|---|---|
| `special_matrices.py` | *(none found as a standalone public utility)* | **KEEP** | `vec`/`vech`/`xpnd`/commutation/duplication/elimination matrices show up as private internals scattered inside packages like `linearmodels`, but there's no clean, importable public utility module for these. Genuinely fills a gap other libraries solve ad hoc internally rather than exposing. |

---

## Summary: where the strongest "switch" cases are

If prioritizing effort, these five would give the most practical benefit
for the least risk, roughly in order of impact:

1. **`stats/regression/lasso.py`** → route the penalized-form solvers
   through `sklearn.linear_model.Lasso`/`ElasticNet` (keep
   `lasso_tau_constrained` as-is).
2. **`econometrics/var.py`** → use `statsmodels.tsa.api.VAR` for anything
   unrestricted (keep `varx_estimate` for the restricted case).
3. **`econometrics/kalman.py`** → use
   `statsmodels.tsa.statespace.MLEModel` for anything beyond simple
   filtering.
4. **`svm/svm.py`** → recommend `sklearn.svm.SVC`/`SVR` directly to users
   for standalone SVM tasks; keep the module for QP-composability use
   cases only.
5. **`spline/spline.py`** → recommend `scipy.interpolate.CubicSpline` for
   pure interpolation; keep the smoothing-spline path only where the
   `p`-parameterization specifically matters.

The strongest "keep, no serious alternative" cases are `stats/
distributions.py`'s GQF family, `portfolio/risk_budgeting.py`,
`linalg/special_matrices.py`, and `econometrics/whittle.py` — these are
the places where the port is providing something genuinely absent
elsewhere in the Python ecosystem, not just re-deriving what a
well-known library already does.
