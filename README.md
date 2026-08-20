# QuantToolbox (Python)

Python port of the MATLAB `QuantToolbox` library: econometrics, portfolio
optimization / risk budgeting, distribution utilities, and backtesting tools.

This is a from-scratch redesign, not a line-by-line transliteration. The
original MATLAB code relied on:

- a hand-built "GAUSS-style" primitive layer (`rows`, `cols`, `sumc`,
  `packr`, `selif`, `seqa`, `lag1`, ...) — replaced here with native
  NumPy / pandas operations.
- `global` variables for solver configuration (ADMM/CCD tolerances,
  iteration limits, MVO problem state) — replaced here with explicit
  config dataclasses passed into functions/classes.
- many near-duplicate solver variants (e.g. ~40 files implementing risk
  budgeting under different constraint/algorithm combinations) —
  consolidated here into single classes parameterized by `method=` /
  `constraints=`.

See `docs/migration_map.md` for the file-by-file mapping from the original
MATLAB source to this package, including specific translation notes.

## Install (editable, dev)

```bash
pip install -e ".[dev,viz]"
```

## Project layout

```
src/quanttoolbox/
├── config.py            # dataclasses replacing MATLAB `global` blocks
├── dates/                Excel<->date conversion, rebalancing calendars
├── backtest/              return/price series, drawdown, turnover, reporting
├── stats/                 distributions, moments, regression (OLS/ridge/lasso/robust/kernel)
├── econometrics/          OLS/GMM/ML/Whittle estimation, VAR/VARX, Kalman filter, ADF/Wald tests
├── optim/                 proximal operators, projections, QP wrapper, bisection
├── portfolio/             risk budgeting, mean-variance, tracking error, Black-Litterman
├── mixtures/               Gaussian-mixture & jump-diffusion risk measures
├── svm/                    SVM primal/dual (classification & regression)
├── spline/                cubic smoothing splines, banded solver
├── maths/                  numerical differentiation, GBM simulation
├── linalg/                 vec/vech/xpnd, commutation/duplication matrices
└── viz/                    figure export helpers
```

## Status

Scaffold stage — module structure and public APIs are being ported
incrementally from the MATLAB source. See `docs/migration_map.md` for
per-module progress.

## Testing

```bash
pytest
```
