# Examples

The original MATLAB toolbox ships an `Examples/` folder with ~150
scripts, one or a few per function, many referencing published textbook
results (mostly Roncalli, T. (2013), *Introduction to Risk Parity and
Budgeting*). This section translates a curated set of them into Python,
each reproducing the same published/computed numbers as a built-in
sanity check.

Available so far:

- [Risk budgeting (ERC / VaR / ES)](risk_budgeting.md) — reproduces
  Roncalli (2013), Tables 2.2–2.4.
- [Black-Litterman](black_litterman.md) — reproduces Roncalli (2013),
  page 24.
- [Mean-variance, minimum-variance, tracking error](mean_variance.md)
- [Ridge, OLS, and robust regression](regression.md)
- [SVM classification](svm.md)
- [Building blocks: bisection, linear algebra, numerical differentiation](building_blocks.md)

**Not yet translated** (real-data examples from the `ects/` folder —
VARX, Kalman filtering, and Whittle estimation on classic time-series
datasets like the Lynx/Sunspots/GNP series — plus the `tutorial/` lesson
series, which is more of a general walkthrough than per-function
examples): see `Examples/ects/*.asc` and `Examples/tutorial/` in the
original [`hfs-archive`](https://github.com/lcrmorin/hfs-archive) repo
for the source material if you'd like to contribute a translation.

The full set of MATLAB scripts these are translated from is organized by
module: `rpb/`, `optim/`, `stats/`, `svm/`, `ects/`, `matrix/`, `dates/`,
`tools/`, `backtest/`, `maths/`.
