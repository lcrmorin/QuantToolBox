# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

## [0.1.0] - Initial release

Complete Python port of the MATLAB QuantToolbox library. See
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
