# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- `portfolio.tracking_error.minimum_te_portfolio`: new `on_infeasible`
  option (`"raise"`, the default, unchanged; `"nearest"` catches
  `solve_qp`'s `RuntimeError` on an infeasible/non-converged input and
  instead solves the same tracking-error QP via `scipy.optimize.minimize`
  (SLSQP) starting from the benchmark weights, replicating the
  constraint-handling of HSF-Notebooks chapter 11h's local
  `min_te_portfolio` (including pruning rank-deficient equality rows) to
  return the "nearest" near-feasible point instead of raising). Not
  ported from the MATLAB HSF toolbox; promoted from HSF-Notebooks
  chapter 11h.
- `portfolio.tracking_error.solve_mixed_norm_te_portfolio`: L1 mixed
  active-share/MD/DTS bond tracking-error objective under a budget and
  carbon-intensity-reduction-target constraint, solved both by direct
  nonsmooth SLSQP minimization and by an epigraph LP reformulation
  (`scipy.optimize.linprog`) -- kept as two separate solve paths since
  the MD/DTS terms are L1 norms of non-diagonal sector-aggregation
  matrices, not expressible via `solve_qp`'s elementwise `lasso_penalty`.
  Not ported from the MATLAB HSF toolbox; promoted from HSF-Notebooks
  chapter 11b/16f.
- `sustainable_finance.impact.blended_finance_simulation`: two-tranche
  (junior/senior) cash-flow waterfall for a blended-finance portfolio,
  allocating cumulative losses and coupon income between tranches given
  a realized cumulative-default path. Ported from `HSF/5. Impact
  Investing/blended_finance_simulation.m`; promoted from HSF-Notebooks
  chapter 05a.
- `sustainable_finance.policy.weitzman_example`: Weitzman (1974)
  "prices vs. quantities" comparison of a price instrument (tax) against
  a quantity instrument (cap) for correcting a linear externality under
  marginal-abatement-cost uncertainty, including the closed-form
  expected-welfare-loss identity. Not ported from a single MATLAB HSF
  toolbox function; promoted from HSF-Notebooks chapter 10e.
- `sustainable_finance.welfare.cobb_douglas_welfare`: Marshallian
  demand, indirect utility, and closed-form Compensating/Equivalent
  Variation welfare measures for a 2-good Cobb-Douglas consumer facing a
  relative-price shock. Not ported from a single MATLAB HSF toolbox
  function; promoted from HSF-Notebooks chapter 16e.
- `copula.families.gaussian_copula_pdf_grid`: evaluates the bivariate
  Gaussian copula density over the full outer-product grid of two 1-D
  `u1`/`u2` arrays (a `numpy.meshgrid` wrapper delegating to the
  existing `gaussian_copula_pdf`, since that function takes paired
  `(n_obs, 2)` columns rather than a broadcasting row/column pair). Not
  ported from the MATLAB HSF toolbox as a standalone function; promoted
  from HSF-Notebooks chapter 15a.
- `copula.families.frank_level_curve`: closed-form inversion of the
  Frank copula CDF tracing the level curve where the copula equals a
  target value `t`, vectorized over a grid of `u1` against a vector of
  `theta` scenarios. Shares its formula with the already-ported
  `frank_contour` (from MATLAB's `copula/contourCopulaFrank.m`) but
  without that function's extra out-of-``[0,1]`` clamp, matching the
  source notebook exactly; promoted from HSF-Notebooks chapter 15a.
- `optim.l1_minimize.minimize_l1_objective`: direct nonsmooth SLSQP
  minimization of a sum of weighted L1-norm penalty terms (each an
  optionally linearly-transformed deviation from a reference point) under
  linear equality/inequality constraints and box bounds. Ported from
  `HSF/11. Portfolio Optimization/chap11_net_zero5.m`'s `fmincon(R_Mix,
  ...)` branch (not discoverable by filename search, and not in
  `docs/migration_map.md`'s `hsf/` toolbox-library mapping, since it's a
  chapter run-script rather than a named library function -- found by
  content search instead); promoted from HSF-Notebooks chapter 11h's
  bond-specific `solve_bond_l1_duration_neutral_smooth`, generalized to
  plain array arguments; kept as an independent SLSQP-based algorithm
  rather than routed through `optim.quadprog.solve_qp`, since the
  notebook uses it as an algorithmically-distinct cross-check against its
  LP-based counterpart (`chap11_net_zero5.m`'s own `linprog` branch).
- `stats.regression.lasso.lasso_regularization_path`: loops the toolbox's
  own `lasso_ccd` coordinate-descent solver over a lambda grid to build
  the full regularization path (coefficient matrix, L1 norm, degrees of
  freedom, and complexity per lambda). Ported from the MATLAB HSF
  toolbox's `regLasso2.m` (translated using `lasso_ccd` rather than the
  original's `quadprog`-based QP, the same substitution
  `docs/examples/stats/lasso2.py` already makes); promoted from
  HSF-Notebooks chapter 15b's `reg_lasso2`.
- `credit.reduced_form.build_climate_stressed_generator`: scales the
  default-column entries of a continuous-time Markov generator matrix by
  a stress multiplier `beta` and re-balances the diagonal so every row
  still sums to zero. Not ported from the MATLAB HSF toolbox -- promoted
  from HSF-Notebooks chapter 13f.
- `stats.scoring`: rank- (`qscore`), standardization- (`zscore`), and
  Beta-shaped (`bz_score`) transforms of a raw criterion onto a common
  scale, plus their normal-CDF/quantile compositions (`qz_score`,
  `zq_score`) and weighted aggregation of already-computed scores
  (`aggregate`). Ported from the MATLAB HSF toolbox's nested functions in
  `chap2_scoring9.m`/`chap2_scoring10.m` ("2. ESG Scoring"); promoted from
  HSF-Notebooks chapter 02b's Python translation of those functions (also
  reused, as `standardize`, in chapter 15b).
- `credit.vasicek`: single-factor Gaussian-copula credit-portfolio model
  -- `thresholds_from_matrix` (discretizes a rating-transition matrix
  into conditioning thresholds), `invcdf_default_rate` (the Vasicek
  large-homogeneous-portfolio default-rate quantile, i.e. the Basel IRB
  worst-case default rate), and `vasicek_density` (the matching limiting
  density). Not ported from the MATLAB toolbox -- promoted from
  HSF-Notebooks chapters 13g/13i/13j, where three notebooks had
  independently hand-rolled the same closed forms.
- `credit.markov_chain.expected_hitting_time`: expected number of steps
  to reach a target rating/state set from each state of a discrete-time
  Markov chain, via first-step analysis. Promoted from HSF-Notebooks
  chapter 2a.
- `portfolio.attribution.beta_pi_alpha`: CAPM beta/priced-risk/alpha
  decomposition of every asset against an arbitrary reference portfolio
  (Pastor & Pedersen (2022)-style). Promoted from HSF-Notebooks chapter
  3b.
- `linalg.rank_reduction.reduce_to_full_row_rank`: greedily drops rows
  from an equality-constraint system `a @ x == b` that don't increase
  `matrix_rank(a)`, returning an equivalent full-row-rank system. Guards
  `scipy.optimize.minimize(..., method="SLSQP")` against rank-deficient
  equality constraints, which otherwise leave it silently returning the
  initial guess unchanged instead of raising. Not ported from the MATLAB
  toolbox -- promoted from HSF-Notebooks chapter 11h.
- `sustainable_finance.physical_risk`: physical/geophysical-science
  formulas, as distinct from `sustainable_finance.esg`/`climate` --
  `ice_albedo`/`ice_albedo_tau` (bistable ice-albedo energy-balance model:
  equilibrium condition and local relaxation timescale),
  `compute_horizontal_resolution` (IPCC-style grid-cell horizontal
  resolution at the equator/45 degrees latitude), `resolution_to_angular`
  (rhomboidal/triangular-spectral/plain-angular grid-resolution label to a
  degree pair), and `wiechel_forward` (closed-form polar-aspect Wiechel
  map projection). Promoted from HSF-Notebooks chapters 08c/12b/12c/12d.
  `ice_albedo`/`ice_albedo_tau` and `compute_horizontal_resolution` turn
  out to have MATLAB antecedents in `hfs-archive` after all (an inline
  local function duplicated across `HSF/8. Economic Modeling/
  chap8_physic4{d,e,h,i,j}.m`, and a standalone `HSF/12. Physical Risk/
  compute_horizontal_resolution.m`, respectively) -- neither lives in the
  `hsf/` toolbox library folder `docs/migration_map.md` tracks, so both
  are new to that map; `resolution_to_angular` and `wiechel_forward` are
  genuinely original to the notebooks.
- `stats.distributions.poisson_binomial_pmf_dp`,
  `poisson_binomial_pmf_dft`, and `poisson_binomial_pmf_brute_force`:
  three independent exact-PMF algorithms for the Poisson-binomial
  distribution (direct O(n^2) DP/convolution recursion, DFT of the
  characteristic function, and brute-force enumeration), kept separate
  from -- and cross-checked against -- the existing scipy-backed
  `poisson_binomial_pmf`. Not ported from the MATLAB toolbox -- promoted
  from HSF-Notebooks chapter 16d as reference implementations for a
  documentation example showing the three derivations agree with the
  packaged function.
- `sustainable_finance.esg.esg_transition_matrix`: ESG score-bucket
  transition-probability matrix and per-bucket/aggregate turnover under a
  bivariate-normal "true score"/noisy-remeasurement model. Not ported
  from a standalone MATLAB HSF toolbox library function -- the identical
  algorithm appears as a local function inside `HSF/16. Exercise
  Solution/chap16_chap2_exercise4.m`, of which HSF-Notebooks chapter 16a
  is itself a direct Python port; promoted here from that notebook.
- `sustainable_finance.esg.mc_weighted_score`: Monte Carlo simulation of
  a power-law-weighted portfolio's score, vectorized over the draw axis,
  with an optional correlated-factor mode. Not ported from a standalone
  MATLAB HSF toolbox library function -- `HSF/16. Exercise
  Solution/chap16_chap2_exercise4.m` performs the same computation
  inline; promoted from HSF-Notebooks chapter 16a, which factored it into
  a reusable function.
- `sustainable_finance.ecology.fn_delta_t`: net species-richness growth
  rate `delta(S) = lambda(S) - mu(S)` of a colonization-extinction
  equilibrium model (theory of island biogeography), used for
  equilibrium root-finding, ODE trajectory integration, and
  comparative-statics sweeps. Not ported from a standalone MATLAB HSF
  toolbox library function -- the identical model appears as local
  functions inside `HSF/16. Exercise Solution/chap16_chap5_exercise2.m`,
  of which HSF-Notebooks chapter 16c is itself a direct Python port;
  promoted here from that notebook.

### Fixed

- `sustainable_finance.ecology.species_abundance_distribution`: the
  `"octave"` and custom-breakpoint (`breaks=<array>`) modes' bucket-
  assignment rule was corrected to match numpy's own `np.histogram`
  half-open-bin convention exactly (previously used a different,
  non-numpy-native inclusive convention). HSF-Notebooks chapter 05c's
  independent reimplementation of this same function was cross-checked
  against the old toolbox convention on real survey data (Whittaker's
  Siskiyou Mountains tree survey and Condit's Barro Colorado Island
  forest census) and found to classify some individual counts into
  different abundance buckets; the two now agree exactly.

## [0.3.0] - Copula module, Python 3.13 support, documentation cleanup

### Added

- `copula` module: Clayton, Frank, Gumbel, Gaussian, Student-t, and 13
  further named bivariate copula families (AMH, Gumbel-Barnett,
  Galambos, Husler-Reiss, Plackett, FGM, Cubic, logistic-Gumbel,
  Marshall-Olkin, Sloane, nested Gumbel, plus the Fréchet-Hoeffding
  bounds and independence copula), Kendall's tau / Spearman's rho
  dependence measures, and generic conditional-CDF-inversion
  simulation — 73 tests, cross-checked against `statsmodels` where an
  equivalent exists.
- Python 3.13 support — full test suite, mypy, and ruff verified clean
  under a fresh 3.13 install; added to `classifiers` in
  `pyproject.toml`.

### Fixed

- `bond.pricing`, `stats.distributions.lognormal_cdf`, and
  `stats.dose_response.drc_log_normal`: 3 mypy type errors.
- `stats.distributions.skew_t_ppf`: the Newton loop's default tolerance
  (`1e-8`) was tighter than `bvt_cdf`'s ~1e-4 QMC noise floor could ever
  satisfy, so every call silently burned its full iteration budget for
  no accuracy gain (~4x slower than necessary). Loosened the default to
  `tol=1e-4`, matching the CDF's actual achievable precision.
- `copula.families.pdfCopulaGumbel3`-equivalent: found (via finite
  difference and `sympy`) that the original MATLAB PDF formula doesn't
  match the derivative of its own CDF — not ported; see
  `docs/matlab_bugs_found.md` #7.
- Closed the remaining test-coverage gaps in `bond`, `copula`, `credit`,
  and `sustainable_finance` — those four modules are now at 100%.

### Docs

- Consistent "QuantToolBox" (3 capitals) capitalization across all docs
  and source docstrings; replaced the placeholder README.
- Merged the migration map's two source-repo tables into one and added
  a `Port?` column, so remaining work is a precise, checkable count
  (currently 1 item: `stats/lasso3.m`) instead of an ambiguous mix of
  "not done" and "never going to be done."
- Reordered the featured examples into a simplest-first narrative:
  price a bond → build a portfolio → budget its risk → blend in market
  views → building blocks → regressions → Whittle estimation → SVM.

## [0.2.0] - Target-matching portfolios, wider example coverage

### Added

- `portfolio.mean_variance.mvo_target_portfolio` and
  `portfolio.tracking_error.te_target_portfolio`: mu-problem/sigma-problem
  target-matching (bisect on risk-aversion gamma to hit a target expected
  return, volatility, or tracking error) — closes a gap that was
  previously documented as not ported.

### Fixed

- mypy `Cannot infer type of lambda` at the two new bisection call sites.

### Docs

- Every translated example script now shows up live-synced, directly on
  its matching API reference page — the full ~150-script original
  example set, not just the curated subset. Added a curated deep dive on
  Whittle (frequency-domain) estimation, and a new
  `docs/upstreaming_candidates.md` identifying which modules do something
  no existing Python library does.
- 13 more example scripts translated and cross-verified against the
  original MATLAB source via Octave, including the `ects/` VARX/Kalman/
  Whittle cluster and the `rpb/` Black-Litterman/MVO target-matching
  set — tracker now at 30/149 cross-verified, 0 remaining untranslated
  in `rpb/`. Found and documented a genuine bug in the original
  `Examples/rpb/test_mvo3.m` along the way (missing `init_global` call
  silently breaks target-matching — see `docs/matlab_bugs_found.md`).
- Fixed a broken `mkdocs build --strict`, corrected the site title
  casing, and reorganized the nav (translation tracker moved into
  "Notes for translators").

## [0.1.0] - Initial release

Complete Python port of the MATLAB QuantToolBox library. See
`docs/migration_map.md` for the full file-by-file mapping from the
original MATLAB source.

### Included

- `dates` — Excel/date conversion, rebalancing calendars
- `stats` — distributions (incl. the GQF distribution family), moments,
  regression (OLS/ridge/lasso/robust/kernel/quantile)
- `backtest` — return/price conversion, drawdown, turnover, backtest
  simulation and reporting
- `optim` — proximal operators, projections, a consolidated QP solver
  (`solve_qp`), bisection
- `portfolio` — risk budgeting (unconstrained/box/general-constrained,
  VaR/ES, target-matching), mean-variance, tracking error,
  Black-Litterman
- `econometrics` — OLS/GMM/ML estimation, VAR/VARX, Kalman filter,
  Whittle estimation, ADF test
- `svm` — SVM classification/regression, primal and dual
- `spline` — cubic smoothing splines
- `maths` — numerical differentiation, GBM simulation, EWMA/momentum,
  Riccati/Lyapunov equation solvers
- `mixtures` — Gaussian mixture and jump-diffusion risk measures
- `linalg` — vec/vech/xpnd and special matrices
- `viz` — figure export

### Notes

- Consolidates ~900 original MATLAB files, with many near-duplicate
  solver variants merged into single, parameterized implementations
  (see `docs/migration_map.md` for specifics on `optim.quadprog.solve_qp`
  and `portfolio.risk_budgeting`, the two largest consolidations).
- A handful of genuine bugs in the original MATLAB source were found and
  fixed during porting (documented in `docs/migration_map.md`'s
  "Notes for translators" section), verified via Monte Carlo
  cross-checks, known analytical solutions, and agreement with
  scikit-learn/scipy reference implementations where applicable.
