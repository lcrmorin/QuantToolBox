# Examples

Two different things live under "Examples" in this documentation, and
it's worth being clear about which is which:

- **This section** — a small set of hand-curated, narrated deep dives.
  Each one picks a specific result (often a published textbook table),
  walks through the code, shows the actual output, and explains what it
  means. They're a good starting point for understanding *why* a module
  exists, especially the ones flagged in [Library
  alternatives](../library_alternatives.md) as doing something no other
  Python library does.
- **The API Reference** — every quanttoolbox function that has a
  translated example gets it shown directly on its own reference page,
  under a collapsible "Examples" heading. That covers the full breadth:
  essentially every translated script from the original MATLAB
  `Examples/` folder (~150 scripts total; see the [translation
  tracker](examples_tracking.md) for the file-by-file status), filed
  under whichever module it actually exercises. Source is pulled in
  live from the same `.py` files in the repo, so it can't drift out of
  sync — go there for the complete, exhaustive set.

In short: come here for a guided tour of the interesting parts, go to
the API Reference for the exhaustive, per-function catalog.

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

The full set of original MATLAB scripts these examples translate from
is organized by module: `rpb/`, `optim/`, `stats/`, `svm/`, `ects/`,
`matrix/`, `dates/`, `tools/`, `backtest/`, `maths/` — see the
[translation tracker](examples_tracking.md) for exactly which files are
done, and the [`hfs-archive`](https://github.com/lcrmorin/hfs-archive)
repo for the original source material.
