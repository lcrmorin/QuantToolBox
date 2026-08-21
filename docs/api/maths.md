# `quanttoolbox.maths`

## `maths.numerical_diff`

!!! info "Python alternatives"
    **Hybrid**: [`numdifftools`](https://github.com/pbrod/numdifftools) uses adaptive step sizing and Richardson extrapolation — meaningfully more accurate than this module's fixed-step approach. Worth using where precision matters; keep this module for the magnitude-scaled step convention already wired into `econometrics.estimation`/`whittle`.

::: quanttoolbox.maths.numerical_diff

## `maths.simulation`

!!! info "Python alternatives"
    `compute_ewma`: **hybrid** — `pandas.DataFrame.ewm()` is highly optimized but uses a different parameterization (`alpha`/`span`/`halflife` vs. this module's mean-reversion-rate `lambda_` + explicit `dt`); a small translation layer would be needed to switch. GBM simulation and the Riccati/Lyapunov solvers: **keep** — the latter already route through `scipy.linalg.solve_continuous_are`/`solve_lyapunov` (that *is* the switch), and GBM simulation has no equally-simple standard-library equivalent (QuantLib-Python is a much heavier dependency).

::: quanttoolbox.maths.simulation
