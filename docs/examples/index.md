# Examples

Two things live under "Examples" here:

- **This section** — a handful of curated deep dives, each walking
  through one published result (often a textbook table) with real output
  and explanation. Start here to understand *why* a module exists,
  especially the ones flagged in [Library
  alternatives](../library_alternatives.md) as doing something no other
  Python library does.
- **The API Reference** — every function with a translated example shows
  it directly on its reference page, live-synced from the same `.py`
  files so it can't drift. That's the exhaustive set: essentially all
  ~150 scripts from the original MATLAB `Examples/` folder (see the
  [example translation tracker](../migration_map.md#example-translation-tracker)
  for file-by-file status), filed under whichever module they exercise.

In short: come here for the guided tour, go to the API Reference for the
exhaustive catalog.

## Available deep dives

- [Risk budgeting (ERC / VaR / ES)](risk_budgeting.md) — reproduces
  Roncalli (2013), Tables 2.2–2.4. `portfolio.risk_budgeting` is the
  single largest "no real alternative" case in the whole port.
- [Black-Litterman](black_litterman.md) — reproduces Roncalli (2013),
  page 24.
- [Mean-variance, minimum-variance, tracking error](mean_variance.md)
- [Whittle (frequency-domain) estimation](whittle.md) —
  `econometrics.whittle` is the other clean "nothing else does this"
  case: no `statsmodels`/`arch` equivalent exists.
- [Ridge, OLS, and robust regression](regression.md)
- [SVM classification](svm.md)
- [Building blocks: bisection, linear algebra, numerical
  differentiation](building_blocks.md) — `linalg.special_matrices` is
  a third "fills a real gap" case (vec/vech/duplication/elimination
  matrices have no clean public equivalent elsewhere).
- [Bond pricing and sector-level risk](bond.md) — the first example
  from the [HSF toolbox port](../migration_map.md), not part of the
  `Examples/` translation tracker below.

The full set of original MATLAB scripts these examples translate from
is organized by module: `rpb/`, `optim/`, `stats/`, `svm/`, `ects/`,
`matrix/`, `dates/`, `tools/`, `backtest/`, `maths/` — see the
[example translation tracker](../migration_map.md#example-translation-tracker)
for exactly which files are done, and the
[`hfs-archive`](https://github.com/lcrmorin/hfs-archive) repo for the
original source material.
