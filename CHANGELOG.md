# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- Python 3.13 support — full test suite, mypy, and ruff verified clean;
  added to `classifiers` in `pyproject.toml`.

### Docs

- Consistent "QuantToolBox" (3 capitals) capitalization across all docs
  and source docstrings; replaced the placeholder README; merged the
  migration map's two source-repo tables into one and added a `Port?`
  column so remaining work is a precise, checkable count; reordered the
  featured examples into a simplest-first narrative (bond pricing →
  portfolio construction → risk budgeting → ...).

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
