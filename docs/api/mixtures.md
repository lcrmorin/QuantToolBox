# `quanttoolbox.mixtures`

## `mixtures.gaussian_mixture`

!!! info "Python alternatives"
    **Hybrid**: `sklearn.mixture.GaussianMixture` is more numerically robust (covariance regularization, multiple initializations, convergence diagnostics) for the *general* n-component EM-fitting step (`estimate_em_mixture`). **Keep** everything downstream of fitting — VaR/ES, risk contribution, risk budgeting, PDF/skewness under the mixture — since sklearn's `GaussianMixture` only fits parameters, nothing else.

::: quanttoolbox.mixtures.gaussian_mixture

## `mixtures.jump_diffusion`

!!! info "Python alternatives"
    **Keep** — jump-diffusion-specific risk measures with no general-purpose equivalent.

::: quanttoolbox.mixtures.jump_diffusion
