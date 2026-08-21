# `quanttoolbox.optim`

## `optim.proximal` / `optim.projection`

!!! info "Python alternatives"
    **Keep** — [`pyproximal`](https://github.com/PyLops/pyproximal) covers some overlapping norm/constraint proximal operators, but is oriented toward signal-processing/inverse-problems use cases, not portfolio constraints (turnover, combined Dykstra-projected linear+box systems). No strong off-the-shelf equivalent for this exact operator set.

::: quanttoolbox.optim.proximal

::: quanttoolbox.optim.projection

## `optim.quadprog` (`solve_qp`)

!!! info "Python alternatives"
    **Hybrid**: for hot loops calling `solve_qp` many times (e.g. risk-budgeting's inner ADMM loop), calling `qpsolvers.solve_qp` directly — bypassing cvxpy's DSL-parsing overhead — would likely be faster. Worth profiling if performance matters; keep the cvxpy-based version for its more expressive constraint composition (ridge/lasso penalties, arbitrary constraints).

::: quanttoolbox.optim.quadprog

## `optim.bisection`

!!! info "Python alternatives"
    **Hybrid**: `scipy.optimize.brentq`/`bisect` converge faster (superlinear) for scalar root-finding and are compiled. **Keep** our vectorized (array-broadcast, many-roots-at-once) version — scipy's root finders are scalar-only.

::: quanttoolbox.optim.bisection
