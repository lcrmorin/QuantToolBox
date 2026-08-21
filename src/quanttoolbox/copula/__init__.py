"""Bivariate copula families ported from the HSF toolbox (`hfs-archive`'s
`HSF/0. Toolbox/copula/`): CDFs/PDFs for ~20 named families
(`families.py`), Kendall's tau / Spearman's rho and the empirical
dependogram (`dependence.py`), and copula simulation (`simulate.py`).

Architecture note (see each module's docstring for detail): rather than
23 independent, boilerplate-heavy per-family modules, the families that
already have a tested, `numpy`/`scipy`-backed implementation elsewhere in
this package are built as thin wrappers around it --
`quanttoolbox.stats.multivariate.bvn_cdf`/`bvt_cdf` for the Gaussian/
Student copulas, `quanttoolbox.stats.distributions.mvn_cdf` for the
n-dimensional Gaussian case -- instead of re-deriving bivariate-normal/
Student CDF machinery a second time. `quanttoolbox.optim.bisection.
bisection` (already vectorized) backs the one generic conditional-
inversion simulator (`simulate.simulate_from_conditional_cdf`) shared by
every family lacking a closed-form quantile inversion, rather than a
family-by-family hand-rolled root-finding loop. See docs/migration_map.md,
"HSF toolbox port", for the full source breakdown.
"""
