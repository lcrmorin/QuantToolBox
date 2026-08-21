# hsf_toolbox_matlab (reference only)

MATLAB source for the function library behind Thierry Roncalli's
*Handbook of Sustainable Finance*, kept here as reference material
while it's ported into `quanttoolbox`.

This folder is **not part of the installable package**. It's plain
MATLAB source, not importable Python, and it's excluded from the wheel
build via `pyproject.toml`'s existing package scoping
(`packages = ["src/quanttoolbox"]` — nothing outside `src/` is ever
packaged, so no extra exclusion rule was needed).

See [`docs/migration_map.md`](../../docs/migration_map.md)'s "HSF
toolbox port (planned)" section for the source folder layout and the
planned Python module mapping.

## A few things to check before porting

- `quadratic_form_bond_portfolio1.m`/`2.m` are duplicated verbatim in
  both `bond/` and `hsf/` — port once, under `bond/pricing.py`.
- `genz/qsimvn*.m`, `copula/cdfmvn.m`, and `stats/cdfmvn.m`/`pdfmvn.m`
  all look like variants of multivariate-normal CDF machinery that's
  already ported once, as `quanttoolbox.stats.distributions.cdfmvn`/
  `pdfmvn` (from the *original* `QuantToolbox/stats/cdfmvn.m`). Diff
  these against each other and against the already-ported version
  before assuming each is a distinct algorithm worth its own port.
