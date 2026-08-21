# `quanttoolbox.spline`

!!! info "Python alternatives"
    **Switch** to `scipy.interpolate.CubicSpline` for pure interpolation (already verified to match to machine precision) or `scipy.interpolate.UnivariateSpline`/`make_smoothing_spline` for smoothing — more mature and actively maintained. The friction: our `p` parameter (a `[0,1]` blend) and scipy's `s` parameter (a target residual sum-of-squares) are different smoothness parameterizations, not directly interchangeable. **Keep** this module only where exact `p`-parameterized behavior needs to match existing MATLAB-based analysis.

## `spline.spline`

::: quanttoolbox.spline.spline
