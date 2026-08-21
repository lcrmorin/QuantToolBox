# Python library alternatives: what to keep, what to switch

For every ported module, this document names the closest existing Python
library alternative (if one exists), makes an honest call on whether to
keep the ported code, switch to the library, or use both for different
purposes, and — for anything worth keeping — asks a further question:
could the gap it fills be contributed back upstream, so the alternative
library covers it too, instead of it staying locked inside this port?
The guiding question throughout the first three columns is: **does the
ported code do something the alternative genuinely can't, or is it
mostly duplicating well-trodden ground?** The **Upstream potential**
column then asks, for whatever survives that question: is this a clean
enough, generalizable enough gap that a maintainer would plausibly
accept it, or is it too tied to this port's specific conventions to be
worth proposing?

Three verdicts are used in the **Verdict** column:

- **KEEP** — the ported code does something the alternative doesn't
  (a specific parameterization, a quant-specific measure, a missing
  feature), so there's a real reason to maintain it.
- **SWITCH** — a mature, well-tested library already does this better;
  the ported code is mostly worth keeping as a reference/fallback, not
  as the primary path.
- **HYBRID** — use the library for the common case, keep the ported code
  for a specific capability it has that the library lacks.

A **SWITCH** verdict almost always means **N/A** in the Upstream
potential column: if the alternative library already does the job
better, there's nothing this port has to offer it back. Upstream
potential mostly applies to **KEEP** and **HYBRID** cases, where this
port has a capability the alternative lacks.

---

## Dates (`dates/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `convert.py` | `pandas` (already the base) | **KEEP** | Pandas has no built-in Excel-serial-date conversion; this is a thin, genuinely useful wrapper. | Thin enough (one conversion function) to read as a personal utility rather than a generalizable pandas capability — not a strong candidate on its own. |
| `rebalancing.py` | [`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) | **HYBRID** | Our rebalancing dates are weekday-only (no exchange holiday calendar) — that's a real gap versus the original MATLAB behavior too. `pandas_market_calendars` gives real NYSE/LSE/etc. holiday calendars; worth wiring in as an optional calendar source for production backtests, while keeping our nearest-available-date snapping logic (which the library doesn't replicate exactly). | The nearest-available-date snapping logic has no direct equivalent in `pandas_market_calendars` either — worth raising as a feature request there once the HYBRID switch is done, rather than upstreaming preemptively. |

---

## Stats (`stats/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `distributions.py` — simple wrappers (normal/t/chi2/F/MVN) | `scipy.stats` (already the backend) | **KEEP** | These are one-line wrappers around scipy; no separate module needed downstream, but no reason to remove them either — call-site compatibility with the original toolbox. | Not worth upstreaming — kept only for call-site compatibility, not a capability gap in scipy. |
| `distributions.py` — GQF1/GQF2 | *(none)* | **KEEP** | Genuinely niche (generalized quadratic-form distributions for Delta-Gamma VaR). Nothing in scipy, statsmodels, or elsewhere in the ecosystem does this. | No clean existing home found. `scipy.stats` is conservative about adding new distribution families, and no general econometrics/risk package is a natural fit either — more realistic as its own small standalone release than a PR into an existing library. |
| `moments.py` — `rolling_correlation`/`rolling_volatility` | `pandas.DataFrame.rolling().corr()`/`.std()` | **SWITCH** | Pandas' rolling ops are implemented in Cython and are meaningfully faster than our Python-loop version for anything beyond toy sizes. The only reason to keep ours is the specific `method=2` "compute returns within each window" variant for illiquid series, which pandas doesn't offer directly. | N/A for the switched core. The `method=2` variant is real but narrow enough (illiquid-series-specific) that it's not obviously worth a pandas PR — a documented recipe is more realistic than a new `.rolling()` option. |
| `moments.py` — `active_share`, `herfindahl_index`, `asynchronous_cov`, `weekly_cov` | *(none)* | **KEEP** | Portfolio-construction-specific measures with no general-purpose library equivalent. | Narrow enough that no single library is a natural home; if pursued, `active_share` specifically fits performance-analytics packages like `empyrical`/`pyfolio-reloaded` better than a general stats library. |
| `regression/ols.py` | `statsmodels.OLS`/`WLS` | **HYBRID** | For plain (unrestricted) OLS, statsmodels is more complete (more diagnostics, better-tested edge cases). Ours is worth keeping specifically for the `restriction=(RR, r)` linear-restriction parameterization, which statsmodels doesn't support as directly. | statsmodels' API design opinions run strong; worth raising as an issue before assuming a PR would land. |
| `regression/ridge.py` | `sklearn.linear_model.Ridge`/`RidgeCV` | **HYBRID** | sklearn's solver is more optimized for large/sparse problems. Keep ours for the `ridge_tau_targeted` (L2-norm-budget, not penalty) parameterization — sklearn has no equivalent for that. | `ridge_tau_targeted` is an L2-*budget* parameterization; sklearn core is conservative about scope creep, so a scikit-learn-contrib package is the more realistic landing spot than sklearn itself. |
| `regression/lasso.py` | `sklearn.linear_model.Lasso`/`ElasticNet`/`LassoCV` | **SWITCH** (for the penalized-form solvers) | sklearn's coordinate descent is Cython-compiled and extensively battle-tested; there's no good reason to keep hand-rolled `lasso_ccd`/`lasso_admm` for the standard penalized case — recommend routing those through sklearn directly. **KEEP** `lasso_tau_constrained` specifically, since sklearn has no L1-*budget* (as opposed to L1-*penalty*) interface. | N/A for the switched solvers. `lasso_tau_constrained` (L1-budget) is the one piece worth offering — again more realistic via scikit-learn-contrib than sklearn core. |
| `regression/kernel.py` | `statsmodels.nonparametric.KernelReg` | **HYBRID** | statsmodels' version supports automatic bandwidth selection via cross-validation and both local-constant/local-linear estimators — more complete than our fixed-bandwidth port. Worth switching to for general use; keep ours only where the exact original bandwidth formula needs to be reproduced for parity with existing MATLAB-based results. | Our fixed-bandwidth formula is a strict subset of what `KernelReg` already does — nothing to contribute back. |
| `regression/quantile.py` | `statsmodels.regression.quantile_regression.QuantReg`, or `sklearn.linear_model.QuantileRegressor` (newer sklearn) | **SWITCH** | Both are more battle-tested than our `scipy.optimize.linprog`-based implementation, handle edge cases (rank-deficient design matrices, ties) more robustly, and QuantReg in particular has been used in production statsmodels code for years. Keep ours only if the exact slack-variable (`u`, `v`) outputs are needed downstream. | N/A — both alternatives already cover this more robustly; the slack-variable output convention is too implementation-specific to propose upstream. |
| `regression/robust.py` | `statsmodels.robust.robust_linear_model.RLM` | **HYBRID** | RLM covers Huber, Tukey biweight, Andrew's wave, Hampel, and trimmed-mean M-estimators via IRLS — a superset of our Huber implementation, and more robustly tested. It does *not* cover LAD or quantile M-estimation directly (though `QuantReg(q=0.5)` is exactly LAD). Recommend: `RLM` for Huber/general M-estimation, keep our `lad_regression`/`quantile_m_regression`/`inverse_quantile_m_regression` for those specific losses. | LAD/quantile M-estimation aren't in `RLM` today — worth raising as a statsmodels issue, though its estimator set is deliberately curated around IRLS M-estimators, so acceptance isn't a given. |

---

## Econometrics (`econometrics/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `estimation.py` (GMM/ML) | `statsmodels.sandbox.regression.gmm.GMM`, `statsmodels.base.model.GenericLikelihoodModel`, or the [`linearmodels`](https://github.com/bashtage/linearmodels) package (more modern GMM/IV support) | **HYBRID** | These libraries are more mature for standard GMM/MLE use cases. Keep ours specifically for the explicit `theta = RR @ gamma + r` linear-restriction interface, which none of the alternatives expose as directly — that parameterization is the main reason this module exists as custom code at all. | Same restriction interface as `regression/ols.py` above; worth a combined issue/PR rather than two separate proposals, but statsmodels' opinionated API design means this is a discussion, not a quick patch. |
| `var.py` | `statsmodels.tsa.api.VAR` | **SWITCH** for the unrestricted case | statsmodels' VAR is comprehensive: automatic lag-order selection, impulse response functions, forecast error variance decomposition, forecasting — well beyond what we ported, and it's the standard tool the econometrics community actually uses. **KEEP** `varx_estimate`'s linear-restriction support (`a_eq`/`b_eq` on the stacked coefficient vector) — statsmodels' VAR doesn't support arbitrary parameter restrictions. | A genuinely useful gap for applied macro/finance users — `VAR` has no arbitrary-restriction support today — but `VAR` is mature and stable, so this is a bigger API-design conversation than a quick patch, best opened as an issue first. |
| `kalman.py` | `statsmodels.tsa.statespace.MLEModel`/`kalman_filter` | **SWITCH** for anything beyond simple filtering | statsmodels' state-space framework is dramatically more capable: smoothing (not just filtering), MLE parameter fitting built in, diffuse initialization, a compiled Cython backend for real performance on long series. Our `kalman_filter` is fine for simple, transparent filtering tasks or minimal-dependency use, but statsmodels is the better choice for anything production-scale. | N/A — statsmodels' framework is already strictly more capable; there's nothing here to contribute back. |
| `whittle.py` | *(none found)* | **KEEP** | Whittle (frequency-domain) estimation isn't implemented in statsmodels, `arch`, or other common packages. Genuinely fills a gap. | Strongest candidate in this file. Either a new estimator class following statsmodels' `Model`/`Results` convention, or a standalone function in `arch`. Would need generalizing from this port's specific local-level/local-linear-trend/custom-`sdf_fn` interface to whatever calling convention the host library expects — substantial enough to warrant a scoping conversation with a maintainer before an unsolicited PR. |
| `tests.py` (ADF) | `statsmodels.tsa.stattools.adfuller` (already used) | **SWITCH** (already done) | Also worth knowing: the [`arch`](https://github.com/bashtage/arch) package's `arch.unitroot` module has a wider family of unit-root tests (ADF, Phillips-Perron, DFGLS, KPSS, Zivot-Andrews) if more than ADF is ever needed — not currently ported, but worth knowing about. | N/A — already routes through statsmodels; nothing custom left to offer back. |

---

## Optimization (`optim/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `proximal.py`, `projection.py` | [`pyproximal`](https://github.com/PyLops/pyproximal) covers some overlapping norm/constraint proximal operators | **KEEP** | Low-level building blocks with no strong off-the-shelf equivalent covering our exact operator set (turnover constraints, combined Dykstra-projected linear+box systems for portfolio construction specifically). `pyproximal` is oriented toward signal-processing/inverse-problems use cases, not portfolio constraints. | `pyproximal`'s API is built around signal-processing/inverse-problems operators, so our portfolio-specific set (turnover, Dykstra-projected linear+box systems) doesn't map cleanly onto it — a targeted PR isn't obviously easy. More realistic as a standalone niche package if this is ever generalized beyond this port. |
| `quadprog.py` (`solve_qp`) | `qpsolvers.solve_qp` directly (already a declared dependency, currently used *underneath* cvxpy) | **HYBRID** | For hot loops that call `solve_qp` many times (e.g. risk-budgeting's inner ADMM loop), calling `qpsolvers.solve_qp` directly — bypassing cvxpy's DSL-parsing overhead — would likely be measurably faster. Worth profiling if performance in tight loops becomes a concern; keep the cvxpy-based version for its more expressive constraint-building (ridge/lasso penalty terms, arbitrary constraint composition), which raw `qpsolvers` doesn't offer. | N/A — this module is built on top of `qpsolvers`/`cvxpy` rather than replacing anything either one does; there's no independent capability to contribute back. |
| `bisection.py` | `scipy.optimize.brentq`/`bisect` | **HYBRID** | scipy's `brentq` converges faster (superlinear, not just bisection) for scalar root-finding, and is compiled. **KEEP** our vectorized (array-broadcast, many-roots-at-once) version — scipy's root finders are scalar-only, and vectorizing over many independent brackets is exactly what our version adds. | The easiest "quick win" of this whole document: small, self-contained, dependency-free. Array-broadcast bisection (many independent brackets solved at once) has come up as a recurring, low-priority SciPy feature request, and this module already does it cleanly. |

---

## Portfolio (`portfolio/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `mean_variance.py`, `black_litterman.py`, `tracking_error.py` | [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt) | **HYBRID** | PyPortfolioOpt is a mature, actively maintained library covering mean-variance optimization, Black-Litterman, CVaR, discrete allocation, and more — genuinely worth using directly for standard portfolio construction. Keep ours for tighter integration with the rest of this codebase (shared `solve_qp`, direct ridge/lasso penalty composability) and where the exact original parameterization needs to match existing analysis. | `mvo_target_portfolio`/`te_target_portfolio` (bisection-based mu-problem/sigma-problem target-matching, added this cycle) are the candidate. PyPortfolioOpt already has `efficient_risk`/`efficient_return` covering similar ground for plain MVO — compare directly before assuming this is additive. The tracking-error-*relative* version (`te_target_portfolio`) is the more likely genuine gap, since PyPortfolioOpt's target-matching isn't benchmark-relative in most places. |
| `risk_budgeting.py` | [`riskparityportfolio`](https://github.com/mirca/riskparityportfolio) (PyPI, narrower scope) | **KEEP** | `riskparityportfolio` covers basic risk parity but not the box-constrained/general-linear-constrained/VaR-ES/target-matching breadth this module has. This is the single largest and most-tested piece of custom work in the whole port (75+ original files consolidated) — genuinely the strongest "keep" case in this document. | Propose the constrained solvers as additional solver options on `riskparityportfolio` rather than pitching a from-scratch new library — same problem domain, narrower existing scope. This is the single largest, most-tested "keep" case in the whole port, substantial enough that it deserves a scoping conversation with the maintainer before any PR, not a quick drive-by. |

---

## Mixtures (`mixtures/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `gaussian_mixture.py` — `estimate_em_mixture` | `sklearn.mixture.GaussianMixture` | **HYBRID** | sklearn's EM implementation is more numerically robust (covariance regularization, multiple initializations, convergence diagnostics) and well-tested for the *general* n-component case. Worth switching to for the pure model-fitting step. **KEEP** everything downstream of fitting — VaR/ES, risk contribution, risk budgeting, PDF/skewness under the mixture — since sklearn's `GaussianMixture` has none of that; it only fits parameters. | N/A for the fitting step (switching to sklearn). The downstream risk layer (VaR/ES, risk contribution, PDF/skewness under a mixture) is mixture-specific enough that no active general-purpose project is an obvious target — see `jump_diffusion.py` below for the same conclusion in a related module. |
| `jump_diffusion.py` | *(none)* | **KEEP** | Jump-diffusion-specific risk measures; no general-purpose equivalent exists. | Niche enough that no actively-maintained project in this exact space is an obvious target — probably not worth pursuing as a standalone contribution right now. |

---

## SVM (`svm/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `svm.py` | `sklearn.svm.SVC`/`SVR` | **SWITCH** for standalone use | sklearn's SVM is backed by `libsvm`/`liblinear` (compiled C, extremely well optimized and battle-tested) — already verified to match our implementation to 3+ decimal places. For any standalone classification/regression task, use sklearn directly; it will be faster and more robust to edge cases (kernel tricks, class imbalance handling, probability calibration — none of which we ported). **KEEP** ours only if the SVM needs to be composed with other constraints inside the same `solve_qp`/cvxpy optimization (e.g. embedding an SVM-like margin constraint inside a larger portfolio problem) — that composability is the one thing sklearn's opaque solver can't offer. | N/A — sklearn's SVM is already the better standalone tool. The one thing ours offers (composing an SVM-like margin constraint inside a larger cvxpy/`solve_qp` problem) doesn't fit sklearn's opaque-solver API at all, so there's no narrow feature to propose there. |

---

## Spline (`spline/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `spline.py` | `scipy.interpolate.CubicSpline` (pure interpolation), `scipy.interpolate.UnivariateSpline`/`make_smoothing_spline` (smoothing) | **SWITCH** for general use | Already verified our `p=1` case matches `CubicSpline` to machine precision — scipy's tools are more mature, better-tested, and actively maintained. The one real friction point: our `p` parameter (a `[0,1]` interpolation/smoothing blend) and scipy's `s` parameter (a target sum-of-squared-residuals) are *different parameterizations* of smoothness, not directly interchangeable. **KEEP** ours only where exact `p`-parameterized behavior needs to match existing MATLAB-based configs or analysis. | N/A — scipy already covers general use well; the `p`-parameterization is a different (not clearly superior) smoothness convention, not an obvious missing option in scipy's API. |

---

## Maths (`maths/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `numerical_diff.py` | [`numdifftools`](https://github.com/pbrod/numdifftools) | **HYBRID** | `numdifftools` uses adaptive step sizing and Richardson extrapolation — meaningfully more accurate than our fixed-step-size approach, especially for ill-conditioned functions. Worth using where precision matters (e.g. as an alternative Hessian source for MLE standard errors). Keep ours for the specific magnitude-scaled step convention already wired into `econometrics.estimation`/`whittle`, to avoid adding a dependency for something already working correctly. | N/A — `numdifftools`' adaptive approach is already the more general, more accurate one; our fixed convention is specific to this port's estimation code, not a missing capability worth proposing there. |
| `simulation.py` — `compute_ewma` | `pandas.DataFrame.ewm()` | **HYBRID** | Pandas' `.ewm()` is a highly optimized, feature-complete exponentially-weighted accessor. The friction: pandas parameterizes by `alpha`/`span`/`halflife` (smoothing factor), while our `compute_ewma` uses `lambda_` as a mean-reversion *rate* with explicit `dt` scaling — a different but related convention requiring a small translation layer (`alpha = lambda_ * dt`, roughly) to switch cleanly. Worth doing if performance on very long series matters; keep ours as-is for now given the interface is already wired into `volatility_target`/`momentum_ewma`. | N/A — `.ewm()` is already more complete; the `lambda_`/`dt` convention is just a different parameterization, not a missing pandas capability. |
| `simulation.py` — GBM simulators, Riccati/Lyapunov | `scipy.linalg` (already the backend for Riccati/Lyapunov) | **KEEP** | Riccati/Lyapunov already route through `scipy.linalg.solve_continuous_are`/`solve_lyapunov` — this *is* the "switch" already done. GBM simulation has no strong standard-library equivalent at this level of simplicity (QuantLib-Python exists but is a much heavier dependency for far more sophisticated PDE/Monte Carlo needs than this module targets). | Too simple (a few lines of Cholesky decomposition + `cumsum`) to need a dependency; most users would rather inline it than take one. `QuantLib-Python` already covers this need at a much higher sophistication level for anyone who actually needs that — probably not worth pursuing. |

---

## Backtest (`backtest/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `returns.py`, `stats.py`, `reporting.py` | [`vectorbt`](https://github.com/polakowo/vectorbt), `bt`, `backtrader`, `zipline` | **KEEP** | These are full backtesting *frameworks* — heavier dependencies, different paradigms (event-driven for backtrader/zipline vs. vectorized), and a much bigger surface area than this module's lightweight, transparent, vectorized style suits. `vectorbt` specifically is worth knowing about if backtest performance on very large universes/long histories becomes a bottleneck (it's Numba-accelerated), but it's a genuinely different tool, not a drop-in replacement for this module's scope. | `vectorbt`/`backtrader`/`zipline` are deliberately different tools (event-driven or heavier-dependency, vs. this module's lightweight vectorized style); contributing into one of them would fight its design philosophy rather than fill a gap in it — probably not worth pursuing. |

---

## Linear algebra (`linalg/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `special_matrices.py` | *(none found as a standalone public utility)* | **KEEP** | `vec`/`vech`/`xpnd`/commutation/duplication/elimination matrices show up as private internals scattered inside packages like `linearmodels`, but there's no clean, importable public utility module for these. Genuinely fills a gap other libraries solve ad hoc internally rather than exposing. | The cleanest PR candidate in this whole document: pure, dependency-free NumPy functions, no state. [`linearmodels`](https://github.com/bashtage/linearmodels) is the more directly relevant target (already has near-equivalents to consolidate against); `scipy.linalg` is the higher-visibility one. Small enough for a maintainer to review in one sitting. |

---

## Bond & sustainable finance (`bond/`, `sustainable_finance/`)

| Module | Alternative | Verdict | Why | Upstream potential |
|---|---|---|---|---|
| `bond/pricing.py` — `bond_price`, `bond_ytm`, `coupon_yield` | [`QuantLib-Python`](https://github.com/lballabio/QuantLib-SWIG) | **HYBRID** | QuantLib has far more complete bond conventions (day-count fractions, calendars, callable/amortizing structures) than this flat-rate, cash-flow-list version. Its bond machinery is also a much heavier dependency for the simple flat-discounting case this module targets. Keep this module for quick, dependency-free pricing/YTM against an explicit cash-flow schedule; reach for QuantLib once real-world conventions (day counts, holiday calendars, embedded options) matter. | N/A — QuantLib already covers the general case more completely; this module's simplicity is the point, not a gap to contribute. |
| `bond/pricing.py` — `bond_portfolio_quadratic_form(_vs_benchmark)`, `sustainable_finance/risk.py` | *(none found)* | **KEEP** | Sector-level modified-duration/DTS-targeting quadratic risk forms (with an optional active-share term against a benchmark) are specific to fixed-income portfolio construction and don't appear as a public utility in any general optimization or fixed-income library surveyed. | Narrow and tied to this port's (Q, R, c) quadratic-form convention (shared with `portfolio/risk_budgeting.py`) — more realistic as a documented pattern within this port than a standalone proposal. |

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

## Summary: where the strongest upstreaming candidates are

These are the same "keep, no serious alternative" cases, now looked at
from the other direction: which of them are worth actually proposing to
an upstream project, rather than staying locked inside this port?

1. **`linalg/special_matrices.py`** → [`linearmodels`](https://github.com/bashtage/linearmodels)
   or `scipy.linalg`. Pure, dependency-free NumPy functions with no
   state — the cleanest PR of this whole document.
2. **`optim/bisection.py`** (the vectorized version) → `scipy.optimize`.
   Small, self-contained, and array-broadcast bisection has come up as a
   recurring (if low-priority) SciPy feature request already.
3. **`econometrics/whittle.py`** → `statsmodels.tsa` or [`arch`](https://github.com/bashtage/arch).
   Frequency-domain estimation isn't implemented in either — genuinely
   fills a gap, but needs generalizing beyond this port's specific
   interface, so this is a scoping conversation, not a drive-by PR.
4. **`portfolio/risk_budgeting.py`** → [`riskparityportfolio`](https://github.com/mirca/riskparityportfolio).
   The single largest, most-tested piece of custom work in the whole
   port (75+ original MATLAB files consolidated) — substantial enough
   that it deserves its own scoping conversation with the maintainer.
5. **`stats/distributions.py`**'s GQF1/GQF2 family — no clean existing
   home found anywhere in the ecosystem; more realistic as a small
   standalone release than a PR into an existing library.

A few narrower gaps are real but smaller, and land better as an
scikit-learn-contrib package or a statsmodels issue than a from-scratch
library: the `restriction=(RR, r)` linear-restriction interface shared
by `stats/regression/ols.py` and `econometrics/estimation.py`; the
`a_eq`/`b_eq` restriction support in `econometrics/var.py`; the
tau-*budget* parameterizations in `stats/regression/ridge.py`/`lasso.py`;
and `portfolio/mean_variance.py`/`tracking_error.py`'s new
`mvo_target_portfolio`/`te_target_portfolio` functions, worth comparing
directly against PyPortfolioOpt's `efficient_risk`/`efficient_return`
before assuming they're additive (the tracking-error-relative version is
the more likely genuine gap). See each module's row above for the
specific target and caveat.

**Probably not worth pursuing right now:** `backtest/` (fights the
design philosophy of the event-driven/heavier frameworks it would land
in), `maths/simulation.py`'s GBM simulators (too simple to need a
dependency), and `mixtures/jump_diffusion.py` (niche enough that no
actively-maintained project in this exact space is an obvious target).

None of this is a promise any of these would be accepted — each library
has its own contribution norms (issue-first discussion, its own
test/type conventions, maintainer bandwidth), and "no existing
alternative" doesn't automatically mean "wanted upstream." Start with
the clean, dependency-free entries (`special_matrices.py`,
`bisection.py`) — small enough for a maintainer to review in one
sitting — before attempting the larger modules (`risk_budgeting.py`,
`whittle.py`), which are substantial enough to warrant a scoping
conversation with maintainers first, rather than an unsolicited PR.
