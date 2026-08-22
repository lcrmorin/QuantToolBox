"""Physical/geophysical-science formulas used in this book's physical-risk
chapters -- as distinct from `sustainable_finance.esg` (ESG portfolio
scoring) and `sustainable_finance.climate` (the DICE economic-climate
model). Covers a bistable ice-albedo energy-balance model
(`ice_albedo`/`ice_albedo_tau`), IPCC-style climate-model grid horizontal
resolution (`compute_horizontal_resolution`, `resolution_to_angular`), and
the closed-form polar-aspect Wiechel map projection (`wiechel_forward`).

Port-status note (checked directly against `hfs-archive`, since none of
these four functions live in the `hsf/` toolbox library folder tracked by
`docs/migration_map.md`):

- `ice_albedo`/`ice_albedo_tau` *do* have a MATLAB antecedent -- an inline
  local function (byte-identical across all five copies), not a
  standalone toolbox file: `HSF/8. Economic Modeling/chap8_physic4{d,e,h,
  i,j}.m`. Promoted here from HSF-Notebooks chapter 08c's own NumPy
  translation of that inline formula.
- `compute_horizontal_resolution` *does* have a MATLAB antecedent: `HSF/12.
  Physical Risk/compute_horizontal_resolution.m`, called from
  `chap12_grid2e.m`/`chap12_grid3{a,b,c,d}.m`. Promoted here from
  HSF-Notebooks chapter 12c's own translation, which reproduces the same
  formula (verified against the `.m` source).
- `resolution_to_angular` and `wiechel_forward` have no MATLAB antecedent
  at all: `resolution_to_angular`'s three-notation lookup is original to
  the notebook, and the one MATLAB reference to Wiechel
  (`HSF/12. Physical Risk/chap12_coordinate6.m`) only calls MATLAB Mapping
  Toolbox's built-in `axesm('wiechel', ...)` -- no formula is implemented
  in the archive. Not ported from the MATLAB HSF toolbox -- no `.m` file
  in `hfs-archive` implements this. Promoted from HSF-Notebooks chapter
  12d/12b respectively.

Translation notes:

- `ice_albedo`/`ice_albedo_tau` keep the notebook's own vectorization
  choices verbatim: `ice_albedo` uses a two-level `np.where` (rather than
  MATLAB's three-mask sum) and clips the `(T_warm - T_s)` base to
  non-negative before raising to the (possibly non-integer) power `eta`,
  avoiding `nan`s from a negative base outside the mixed-phase band.
  `ice_albedo_tau` reshapes any scalar/1-D parameter to a column vector
  (`_as_column`) when `T_s` is 2-D, exactly matching the notebook's own
  broadcasting helper -- needed because the calling notebook evaluates the
  relaxation time at all three temperature branches (columns) across a
  swept parameter (rows) at once.
- `compute_horizontal_resolution` hardcodes `length_phi = 111.2` km/deg
  (i.e. Earth radius `R = 6371` km), matching both the MATLAB `.m` file
  and the notebook's own derivation to 1 decimal place.
- `resolution_to_angular` returns `(nan, nan)` for an unrecognized
  resolution string, matching the notebook's own fallback (used so a
  `pandas` column of mixed/unknown resolution labels degrades to `NaN`
  rather than raising).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Rhomboidal spectral-truncation resolutions -> (delta_phi, delta_lambda) in degrees.
RHOMBOIDAL_RESOLUTIONS: dict[str, tuple[float, float]] = {
    "R15": (4.5, 7.5),
    "R21": (3.2, 5.6),
    "R30": (2.25, 3.75),
}

#: Triangular spectral-truncation resolutions -> delta_phi == delta_lambda, in degrees.
TRIANGULAR_RESOLUTIONS: dict[str, float] = {
    "T21": 5.6,
    "T30": 4.0,
    "T31": 3.9,
    "T32": 3.8,
    "T42": 2.8,
    "T47": 2.5,
    "T63": 1.9,
    "T85": 1.4,
    "T106": 1.1,
    "T126": 0.9375,
    "T159": 0.75,
    "T213": 0.5625,
    "T319": 0.375,
}


def _as_column(v: np.ndarray | float, n: int) -> np.ndarray:
    """Reshape a scalar/1-D array of length `n` to an `(n, 1)` column so it
    broadcasts against an `(n, k)` array; anything else passes through
    unchanged."""
    v = np.asarray(v, dtype=float)
    return v.reshape(n, 1) if v.ndim == 1 and v.shape[0] == n else v


def ice_albedo(
    t_s: np.ndarray | float,
    alpha_cold: np.ndarray | float,
    t_cold: np.ndarray | float,
    alpha_warm: np.ndarray | float,
    t_warm: np.ndarray | float,
    eta: np.ndarray | float,
    s0: np.ndarray | float,
    epsilon: np.ndarray | float,
    sigma: np.ndarray | float,
    c: np.ndarray | float,
    d_e: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """Net top-of-atmosphere energy imbalance `dT/dt` of the bistable
    ice-albedo energy-balance model at surface temperature `t_s`: incoming
    solar flux `0.25 * s0 * (1 - alpha(t_s))` minus outgoing longwave flux
    `(2 - epsilon) / 2 * sigma * t_s**4`, plus an optional exogenous
    forcing perturbation `d_e`, divided by heat capacity `c`.

    The planetary albedo `alpha(t_s)` is piecewise: `alpha_cold` below
    `t_cold`, `alpha_warm` above `t_warm`, and a smooth power-law blend of
    the two (exponent `eta`) in between. A root of this function (found
    e.g. via `quanttoolbox.optim.bisection.bisection`) is an equilibrium
    surface temperature; the model is bistable for `eta` past a critical
    threshold (three roots -- two stable, one unstable -- rather than one).

    Not ported from a standalone MATLAB toolbox file -- promoted from
    HSF-Notebooks chapter 08c. A byte-identical inline local function
    (MATLAB argument order `T_s, alpha_cold, T_cold, alpha_warm, T_warm,
    eta, S0, epsilon, sigma, c, dE`) appears in `HSF/8. Economic Modeling/
    chap8_physic4{d,e,h,i,j}.m` in `hfs-archive`, but not as a standalone
    `hsf/` toolbox file.
    """
    t_s = np.asarray(t_s, dtype=float)
    alpha = np.where(
        t_s < t_cold,
        alpha_cold,
        np.where(
            t_s > t_warm,
            alpha_warm,
            alpha_warm
            + (alpha_cold - alpha_warm)
            * (np.clip(t_warm - t_s, 0, None) ** eta)
            / ((t_warm - t_cold) ** eta),
        ),
    )
    f_solar = 0.25 * s0 * (1 - alpha)
    f_lw = (2 - epsilon) / 2 * sigma * t_s**4
    return (f_solar - f_lw + d_e) / c


@dataclass
class IceAlbedoRelaxationResult:
    """Local relaxation timescale `tau` and linearized feedback
    `lambda_` of the ice-albedo model at a given surface temperature (or
    array of temperatures) -- see `ice_albedo_tau`."""

    tau: np.ndarray | float
    lambda_: np.ndarray | float


def ice_albedo_tau(
    t_s: np.ndarray | float,
    alpha_cold: np.ndarray | float,
    t_cold: np.ndarray | float,
    alpha_warm: np.ndarray | float,
    t_warm: np.ndarray | float,
    eta: np.ndarray | float,
    s0: np.ndarray | float,
    epsilon: np.ndarray | float,
    sigma: np.ndarray | float,
    c: np.ndarray | float,
) -> IceAlbedoRelaxationResult:
    """Local relaxation timescale `tau = |c / lambda_|` and linearized
    feedback parameter `lambda_ = d(dT/dt)/d(t_s)` of the `ice_albedo`
    model at surface temperature `t_s` -- i.e. the first-order stability
    of an equilibrium found from `ice_albedo`: `lambda_ < 0` is a stable
    equilibrium (perturbations decay with timescale `tau`), `lambda_ > 0`
    is unstable.

    When `t_s` is a 2-D array (e.g. one row per swept-parameter value,
    one column per equilibrium branch), the other parameters are
    broadcast as column vectors if given as a 1-D array of the same
    length as `t_s`'s first axis -- matching the notebook's own
    `_col`/broadcasting convention.

    Not ported from a standalone MATLAB toolbox file -- promoted from
    HSF-Notebooks chapter 08c. A matching inline local function (MATLAB
    argument order `T_s, alpha_cold, T_cold, alpha_warm, T_warm, eta, S0,
    epsilon, sigma, c, dE`, returning `[tau, lambda]`) appears in `HSF/8.
    Economic Modeling/chap8_physic4j.m` in `hfs-archive`.
    """
    t_s = np.asarray(t_s, dtype=float)
    n = t_s.shape[0]
    alpha_cold, t_cold, alpha_warm, t_warm, eta, s0, epsilon = (
        _as_column(v, n) for v in (alpha_cold, t_cold, alpha_warm, t_warm, eta, s0, epsilon)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        d_alpha = np.where(
            (t_s >= t_cold) & (t_s <= t_warm),
            -eta
            * (alpha_cold - alpha_warm)
            * (np.clip(t_warm - t_s, 0, None) ** (eta - 1))
            / ((t_warm - t_cold) ** eta),
            0.0,
        )
    lambda_ = -0.25 * s0 * d_alpha - 4 * ((2 - epsilon) / 2) * sigma * t_s**3
    tau = np.abs(c / lambda_)
    return IceAlbedoRelaxationResult(tau=tau, lambda_=lambda_)


@dataclass
class HorizontalResolutionResult:
    """IPCC-style grid-cell horizontal resolution (km) at the equator
    (`r_0`), at 45 degrees latitude (`r_45`), and the latitude-averaged
    shortcut `r_avg` -- see `compute_horizontal_resolution`."""

    r_0: np.ndarray | float
    r_45: np.ndarray | float
    r_avg: np.ndarray | float


def compute_horizontal_resolution(
    delta_phi: np.ndarray | float, delta_lambda: np.ndarray | float
) -> HorizontalResolutionResult:
    """IPCC-style grid-cell horizontal resolution (km) for a
    `delta_phi` x `delta_lambda` degree grid cell: `R(phi) = length_phi *
    sqrt(delta_phi * delta_lambda * cos(phi))`, with `length_phi = 111.2`
    km/deg (`pi * R_earth / 180` for `R_earth = 6371` km), evaluated at
    the equator (`r_0`, `phi = 0`) and at 45 degrees latitude (`r_45`,
    `phi = pi / 4`), plus the latitude-averaged shortcut `r_avg = 0.7628 *
    r_0`.

    Original: `HSF/12. Physical Risk/compute_horizontal_resolution.m` in
    `hfs-archive` (not part of the `hsf/` toolbox library folder tracked
    by `docs/migration_map.md`, but a real MATLAB source file
    nonetheless). Promoted here from HSF-Notebooks chapter 12c, which
    reproduces the same formula.
    """
    length_phi = 111.2
    r_0 = length_phi * np.sqrt(delta_phi * delta_lambda * np.cos(0.0))
    r_45 = length_phi * np.sqrt(delta_phi * delta_lambda * np.cos(np.pi / 4))
    r_avg = 0.7628 * r_0
    return HorizontalResolutionResult(r_0=r_0, r_45=r_45, r_avg=r_avg)


@dataclass
class AngularResolution:
    """A grid-resolution label converted to a `(delta_phi, delta_lambda)`
    degree pair -- see `resolution_to_angular`."""

    delta_phi: float
    delta_lambda: float


def resolution_to_angular(res: str) -> AngularResolution:
    """Convert a climate-model grid-resolution label to a `(delta_phi,
    delta_lambda)` degree pair, in any of three notations: rhomboidal
    spectral truncation (`"R15"`, `"R21"`, `"R30"`), triangular spectral
    truncation (`"T21"` ... `"T319"`), or a plain angular-degree string
    (`"4°x5°"`, `"2.5°x3.75°"`, with `delta_phi` first). Returns
    `(nan, nan)` for an unrecognized label.

    Not ported from the MATLAB HSF toolbox -- no `.m` file in
    `hfs-archive` implements this. Promoted from HSF-Notebooks chapter
    12d.
    """
    res = str(res)
    if res in RHOMBOIDAL_RESOLUTIONS:
        delta_phi, delta_lambda = RHOMBOIDAL_RESOLUTIONS[res]
        return AngularResolution(delta_phi=delta_phi, delta_lambda=delta_lambda)
    if res in TRIANGULAR_RESOLUTIONS:
        d = TRIANGULAR_RESOLUTIONS[res]
        return AngularResolution(delta_phi=d, delta_lambda=d)
    if "x" in res:
        phi_str, lambda_str = res.split("x")
        return AngularResolution(
            delta_phi=float(phi_str.replace("°", "")),
            delta_lambda=float(lambda_str.replace("°", "")),
        )
    return AngularResolution(delta_phi=np.nan, delta_lambda=np.nan)


@dataclass
class WiechelProjection:
    """Projected planar coordinates `(x, y)` from the polar-aspect
    Wiechel map projection -- see `wiechel_forward`."""

    x: np.ndarray | float
    y: np.ndarray | float


def wiechel_forward(
    lon_deg: np.ndarray | float, lat_deg: np.ndarray | float, R: float = 1.0
) -> WiechelProjection:
    """Forward polar-aspect Wiechel map projection (H. Wiechel, 1879): a
    pseudoazimuthal, equal-area projection with semicircular meridians
    arranged in a pinwheel around the pole, centered on the North Pole.

    `x = R * (sin(lambda) * cos(phi) - (1 - sin(phi)) * cos(lambda))`,
    `y = -R * (cos(lambda) * cos(phi) + (1 - sin(phi)) * sin(lambda))`,
    with `phi`/`lambda` the latitude/longitude (converted here from
    `lat_deg`/`lon_deg` in degrees) and `R` the sphere radius. Satisfies
    `x**2 + y**2 == 2 * R**2 * (1 - sin(phi))`, the same radial equal-area
    function as the polar Lambert azimuthal projection.

    Not ported from the MATLAB HSF toolbox -- no `.m` file in
    `hfs-archive` implements this (the one MATLAB reference,
    `HSF/12. Physical Risk/chap12_coordinate6.m`, only calls MATLAB
    Mapping Toolbox's built-in `axesm('wiechel', ...)`; no formula is
    given). Promoted from HSF-Notebooks chapter 12b.
    """
    lam = np.radians(lon_deg)
    phi = np.radians(lat_deg)
    x = R * (np.sin(lam) * np.cos(phi) - (1 - np.sin(phi)) * np.cos(lam))
    y = -R * (np.cos(lam) * np.cos(phi) + (1 - np.sin(phi)) * np.sin(lam))
    return WiechelProjection(x=x, y=y)
