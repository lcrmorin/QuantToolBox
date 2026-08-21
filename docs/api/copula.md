# `quanttoolbox.copula`

## `copula.families`

!!! info "Python alternatives"
    **Hybrid**: `statsmodels.distributions.copula.api` (already a dependency of this package) implements Gaussian, Student-t, Clayton, Frank, and Gumbel copulas and was used to numerically verify `clayton_cdf`/`clayton_pdf`, `frank_cdf`/`frank_pdf`, `gumbel_cdf`/`gumbel_pdf`, and `gaussian_copula_cdf`/`gaussian_copula_pdf` to full float64 precision. It is not wrapped directly: `statsmodels`' classes fix one `theta`/`corr` per instance, while this port (like the original MATLAB) vectorizes `theta` per observation; and `StudentTCopula.cdf()` raises `NotImplementedError` in `statsmodels` (no closed form is implemented there), so `student_copula_cdf` had to be self-implemented via `scipy.stats.multivariate_t`/a bivariate wrapper regardless. `gaussian_copula_cdf`/`_pdf` and `student_copula_cdf`/`_pdf` reuse `quanttoolbox.stats.multivariate.bvn_cdf`/`bvn_pdf`/`bvt_cdf` for the bivariate case and `quanttoolbox.stats.distributions.mvn_cdf` for the general Gaussian case, instead of a third from-scratch implementation of bivariate/multivariate-normal-CDF machinery already present twice elsewhere in this package. The remaining 13 families (AMH, Gumbel-Barnett, Galambos, Husler-Reiss, Plackett, FGM, Cubic, logistic-Gumbel, Marshall-Olkin, Sloane, nested Gumbel, and the Fréchet-Hoeffding/independence bounds) have no equivalent in any Python copula library surveyed (`statsmodels`, `copulas`, `pyvinecopulib`) -- **keep**.

    The original MATLAB source's own `cdfCopulaGumbel3.m`/`pdfCopulaGumbel3.m` pair (a 3-variable nested/hierarchical Gumbel copula) has a genuine bug: the shipped PDF formula does not match `d³C/du1du2du3` of its own CDF, verified independently via both a 3-D central finite difference and exact `sympy` symbolic differentiation (both agree with each other at ≈1.2096 for `theta1=1.5, theta2=3.0, u=(0.3,0.5,0.7)`; the original formula gives ≈1.0064). `nested_gumbel_cdf` is ported; no PDF is shipped for it, rather than risk a second, differently-wrong hand derivation. Separately, several families (AMH, Husler-Reiss, Marshall-Olkin, FGM, Sloane) never had a PDF in the original at all -- no PDF is fabricated for those either.

::: quanttoolbox.copula.families

## `copula.dependence`

!!! info "Python alternatives"
    **Keep** -- Kendall's tau and Spearman's rho for the families above, plus the Debye/dilogarithm special functions their closed forms need and the empirical dependogram (pseudo-observations via marginal ranks). `clayton_tau`/`frank_tau`/`gumbel_tau`/`gaussian_tau` were verified against `statsmodels`' `.tau()` methods. The original `SpearmanCopula.m` is itself a *generic* double-integral estimator (`rho = 12 * integral2(C,0,1,0,1) - 3`, for any copula CDF `C`) that the original only ever plugged Clayton and Gumbel into (both lack closed-form Spearman's rho); `spearman_rho_numeric` here (via `scipy.integrate.dblquad`) keeps that genericity explicit and usable for any family in `families.py`, not just the two the original wired up -- `clayton_rho`/`gumbel_rho` are one-line instantiations of it.

::: quanttoolbox.copula.dependence

## `copula.simulate`

!!! info "Python alternatives"
    **Keep** -- no general-purpose Python library exposes generic conditional-CDF-inversion copula simulation the way this module and its MATLAB source do. The original `rndCopula2.m` is itself a *generic* bivariate engine (draw `u1, v2 ~ U(0,1)`, solve `conditional_cdf(u1, u2) = v2` for `u2` by bisection) that the original only ever plugged Gumbel into; `simulate_from_conditional_cdf` here (built on this package's already-vectorized `quanttoolbox.optim.bisection.bisection`) keeps that genericity explicit -- it works for any family exposing a conditional CDF (e.g. `families.amh_conditional_cdf`), not just Gumbel. AMH and Frank have closed-form quantile inversions in the original and use those directly instead (`simulate_amh`, `simulate_frank`) -- faster, and exact rather than iterative. `simulate_gaussian_copula`/`simulate_student_copula` merge the n-dimensional and bivariate-special-case MATLAB originals into one Cholesky-based function each.

::: quanttoolbox.copula.simulate
