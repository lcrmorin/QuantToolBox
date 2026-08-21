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
toolbox port (planned)" section for the planned Python module mapping,
and [HSF-Notebooks](https://github.com/lcrmorin/HSF-Notebooks)'s
`CHAPTERS.md` (chapter 0) for the matching notebook-side entry.

## Contents

| Folder | Files | Covers |
|---|---|---|
| `bond/` | 5 | Bond pricing, YTM, coupon yield, quadratic-form bond-portfolio risk |
| `copula/` | 67 | CDFs/PDFs for ~23 copula families, Kendall's tau / Spearman's rho closed forms, empirical dependograms, copula simulation |
| `credit/` | 17 | Structural credit models (Merton, Black-Cox, jump-diffusion) and reduced-form/Markov-generator hazard-rate models |
| `genz/` | 10 | Genz-Bretz quadrature for multivariate normal/Student CDFs |
| `hsf/` | 24 | Carbon budgets, ESG beta/minimum-variance/Pedersen portfolios, DICE temperature model, species-area/abundance ecology measures, Shannon entropy |
| `maths/` | 2 | `arccosh`/`arcsinh` — not ported, see `docs/migration_map.md`'s "Not ported" table |
| `stats/` | 38 | Skew-normal/skew-t, Bates, bivariate normal/t, order statistics, dose-response curves, and other distributions not in the original `QuantToolbox` |
| `copula1.m`-`copula4.m` (loose, at this folder's root) | 4 | Worked copula-simulation examples (French), not a module — see the note in `docs/migration_map.md` |

163 module files + these 4 loose examples = 167 total.

## A few things to check before porting

- `quadratic_form_bond_portfolio1.m`/`2.m` are duplicated verbatim in
  both `bond/` and `hsf/` — port once, under `bond/pricing.py`.
- `genz/qsimvn*.m`, `copula/cdfmvn.m`, and `stats/cdfmvn.m`/`pdfmvn.m`
  all look like variants of multivariate-normal CDF machinery that's
  already ported once, as `quanttoolbox.stats.distributions.cdfmvn`/
  `pdfmvn` (from the *original* `QuantToolbox/stats/cdfmvn.m`). Diff
  these against each other and against the already-ported version
  before assuming each is a distinct algorithm worth its own port.
