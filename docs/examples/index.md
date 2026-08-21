# Examples

The original MATLAB toolbox ships an `Examples/` folder with ~150
scripts, one or a few per function, many referencing published textbook
results (mostly Roncalli, T. (2013), *Introduction to Risk Parity and
Budgeting*). This section translates a curated set of them into Python,
each reproducing the same published numbers as a built-in sanity check.

Available so far:

- [Risk budgeting (ERC / VaR / ES)](risk_budgeting.md) — reproduces
  Roncalli (2013), Tables 2.2–2.4.
- [Black-Litterman](black_litterman.md) — reproduces Roncalli (2013),
  page 24.

More are being added over time — see `Examples/` in the original
[`hfs-archive`](https://github.com/lcrmorin/hfs-archive) repository for
the full set of MATLAB scripts these are translated from (organized by
module: `rpb/`, `optim/`, `stats/`, `svm/`, `ects/`, `matrix/`, `dates/`,
`tools/`, `backtest/`, `maths/`).
