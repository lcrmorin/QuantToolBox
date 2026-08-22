"""Species-area/endemics-area relationships, species-abundance
distributions, Hurlbert's rarefaction estimator, and a colonization-
extinction equilibrium model's net species-richness growth rate --
classic biodiversity measures from ecology, used in this toolbox's
biodiversity-risk chapters.

Ported from HSF toolbox `hsf/{species_area_relationship,
endemics_area_relationship,species_abundance_distribution,hurlbert}.m`.
`fn_delta_t` has no standalone `.m` source in `hfs-archive` -- see its own
docstring -- and is instead promoted from HSF-Notebooks chapter 16c.

Translation notes:

- `species_area_relationship`/`endemics_area_relationship` accept `a` as a
  scalar or array of target sub-areas, and can work either directly from
  per-species individual counts `n_i`, or from a pre-binned
  species-abundance histogram (`s_j` = number of species with exactly `j`
  individuals) -- typically the output of `species_abundance_distribution`.
  Pass `s_j=None` for the first mode.
- `species_abundance_distribution.m`'s three-way branch on its second
  argument (`brk`) is preserved as three explicit modes rather than
  MATLAB's type-based dispatch: `breaks=None` (unique-count histogram),
  `breaks="octave"` (Preston's log2 "octave" abundance classes), or
  `breaks=<array>` (custom breakpoints). The `"octave"` and `<array>`
  modes' bucket-assignment rule was corrected to match numpy's own
  `np.histogram` half-open-bin convention exactly (`floor(log2(n))` for
  octaves; `np.histogram(n_i, bins=[0, *breaks])` for custom breakpoints)
  -- the original bucket-assignment rule here used a different,
  non-numpy-native inclusive convention, which HSF-Notebooks chapter 05c's
  independent reimplementation of this same function was found to
  disagree with on real (Whittaker/BCI) survey data before this fix; see
  `species_abundance_distribution`'s own docstring.
- `hurlbert.m`'s `method=2` (log-gamma) branch is numerically more stable
  than `method=1`'s direct `scipy.special.comb` ratio for large sample
  sizes (avoids overflow in the individual binomial coefficients); both
  compute the same quantity and are verified to agree in this module's
  tests.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb, gammaln


def species_area_relationship(
    n_i: np.ndarray, s_j: np.ndarray | None, area_total: float, area: np.ndarray | float
) -> np.ndarray:
    """Expected number of species found in a sub-area `area` out of a total
    surveyed area `area_total`, given either per-species individual counts
    `n_i` (`s_j=None`), or a species-abundance histogram `s_j` (number of
    species with exactly `j` = 1, 2, ... individuals).

    Original: hsf/species_area_relationship.m
    """
    area = np.asarray(area, dtype=float)
    ratio = 1.0 - area / area_total

    if s_j is None:
        n_i = np.asarray(n_i, dtype=float)
        total_species = n_i.shape[0]
        term = np.sum(ratio[..., None] ** n_i, axis=-1)
        return total_species - term

    s_j = np.asarray(s_j, dtype=float)
    total_species = np.sum(s_j)
    j = np.arange(1, s_j.shape[0] + 1)
    term = np.sum(s_j * ratio[..., None] ** j, axis=-1)
    return total_species - term


def endemics_area_relationship(
    n_i: np.ndarray, s_j: np.ndarray | None, area_total: float, area: np.ndarray | float
) -> np.ndarray:
    """Expected number of species confined entirely within a sub-area
    `area` out of a total surveyed area `area_total` ("endemics"), given
    either per-species individual counts `n_i` (`s_j=None`), or a
    species-abundance histogram `s_j`.

    Original: hsf/endemics_area_relationship.m
    """
    area = np.asarray(area, dtype=float)
    ratio = area / area_total

    if s_j is None:
        n_i = np.asarray(n_i, dtype=float)
        return np.sum(ratio[..., None] ** n_i, axis=-1)

    s_j = np.asarray(s_j, dtype=float)
    j = np.arange(1, s_j.shape[0] + 1)
    return np.sum(s_j * ratio[..., None] ** j, axis=-1)


def species_abundance_distribution(
    n_i: np.ndarray, breaks: np.ndarray | str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bin per-species individual counts `n_i` into a species-abundance
    histogram: `s[k]` species fall into abundance class `j[k]`.

    - `breaks=None` (default): one class per distinct count value in `n_i`.
    - `breaks="octave"`: Preston's log2 "octave" classes, class `k`
      (`k=0,1,2,...`) covering `2**k <= n < 2**(k+1)` (so `k=0` is `n=1`;
      `k=1` is `n` in `{2,3}`; `k=2` is `n` in `{4,...,7}`; etc.), assigned
      via `k = floor(log2(n))`.
    - `breaks=<array>`: custom upper breakpoints; classes are the
      `numpy.histogram` bins of `n_i` over `[0, *breaks]` (all classes
      half-open `[lower, upper)` except the last, which is closed at both
      ends); `j[k]` is the midpoint of class `k`'s range.

    Returns `(j, s, breaks)` -- `breaks` is `None` for the default mode.

    Original: hsf/species_abundance_distribution.m

    Note: the `"octave"` and `<array>` modes' bucket-assignment rule below
    was corrected to this numpy-native `np.histogram`/`floor(log2(n))`
    convention from an earlier, non-numpy-native inclusive rule (each
    class taken as `(prev_break, this_break]`, closed at the top and open
    at the bottom, with the first class starting from `n=1`/`n=0` rather
    than a numpy-style half-open `[0, this_break)`). On real survey data
    with non-integer abundance counts (Whittaker's Siskiyou Mountains tree
    survey), the two conventions disagree: HSF-Notebooks chapter 05c's
    independent reimplementation of this function -- built and used
    without reference to this toolbox version -- was cross-checked against
    it and found to classify some individual counts into different
    buckets. This toolbox's binning was corrected to match the notebook's
    (and numpy's own) convention exactly; see
    `tests/sustainable_finance/test_ecology.py` for the real-data
    cross-check.
    """
    n_i = np.asarray(n_i, dtype=float)

    if breaks is None:
        j = np.unique(n_i)
        s = np.array([np.sum(n_i == jj) for jj in j], dtype=float)
        return j, s, None

    if isinstance(breaks, str):
        if breaks.lower() != "octave":
            raise ValueError("breaks must be None, 'octave', or an array of breakpoints")
        k = np.floor(np.log2(n_i)).astype(int)
        k_max = int(k.max())
        octaves = np.arange(0, k_max + 1)
        s = np.array([np.sum(k == kk) for kk in octaves], dtype=float)
        octave_breaks = 2.0 ** (octaves + 1)
        return octaves.astype(float), s, octave_breaks

    breaks = np.asarray(breaks, dtype=float)
    edges = np.concatenate([[0.0], breaks])
    s, _ = np.histogram(n_i, bins=edges)
    j = 0.5 * (edges[:-1] + edges[1:])
    return j, s.astype(float), breaks


def hurlbert(n_i: np.ndarray, m: int, method: int = 1) -> float:
    """Hurlbert's rarefaction estimator: expected number of species present
    in a random sample of `m` individuals drawn (without replacement) from
    a community with per-species counts `n_i`.

    `method=2` uses a log-gamma formulation (via `scipy.special.gammaln`)
    for numerical stability at large sample sizes; `method=1` (default)
    computes the binomial-coefficient ratio directly.

    Original: hsf/hurlbert.m
    """
    n_i = np.asarray(n_i, dtype=float)
    n = np.sum(n_i)

    sac = 0.0
    for n_s in n_i:
        if (n - n_s) >= m:
            if method == 2:
                log_q = (
                    gammaln(n - n_s + 1)
                    + gammaln(n - m + 1)
                    - (gammaln(n + 1) + gammaln(n - n_s - m + 1))
                )
                q = np.exp(log_q)
            else:
                q = comb(n - n_s, m) / comb(n, m)
            sac += 1.0 - q
        else:
            sac += 1.0

    return float(sac)


def _fn_lambda_t(S_t: np.ndarray | float, lambda_0: float, beta1: float, S_s: float) -> np.ndarray:
    """Colonization rate `lambda(S)`: `=lambda_0` at `S=0`, decaying to `0`
    at `S=S_s` (a saturating island); `=0` for `S>S_s`. Internal helper for
    `fn_delta_t`."""
    S_t = np.asarray(S_t, dtype=float)
    lambda_t = (
        lambda_0
        * (np.exp(-((S_t / S_s) ** (beta1 * lambda_0))) - np.exp(-1.0))
        / (1.0 - np.exp(-1.0))
    )
    lambda_s = 0.0
    e = S_t <= S_s
    return np.where(e, lambda_t, lambda_s)


def _fn_mu_long_t(S_t: np.ndarray | float, mu_s: float, beta2: float, S_s: float) -> np.ndarray:
    """Long-term extinction rate `mu^long(S)`: `=0` at `S=0`, rising to
    `mu_s` at `S=S_s`; `=mu_s` for `S>S_s`. Internal helper for
    `fn_delta_t`."""
    S_t = np.asarray(S_t, dtype=float)
    mu_long_t = mu_s * (np.exp((S_t / S_s) ** (beta2 * mu_s)) - 1.0) / (np.exp(1.0) - 1.0)
    e = S_t <= S_s
    return np.where(e, mu_long_t, mu_s)


def _fn_mu_short_t(
    S_t: np.ndarray | float,
    lambda_t: np.ndarray | float,
    beta3: float,
    beta4: float,
    S_s: float,
) -> np.ndarray:
    """Short-term extinction rate `mu^short(S) = beta3*lambda(S)*exp(-beta4*S)`
    -- proportional to the colonization rate (recently-arrived species are
    more extinction-prone); `=0` for `S>S_s`. Internal helper for
    `fn_delta_t`."""
    S_t = np.asarray(S_t, dtype=float)
    mu_short_t = beta3 * lambda_t * np.exp(-beta4 * S_t)
    eta_s = 0.0
    e = S_t <= S_s
    return np.where(e, mu_short_t, eta_s)


def fn_delta_t(
    S_t: np.ndarray | float,
    lambda_0: float,
    beta1: float,
    mu_s: float,
    beta2: float,
    beta3: float,
    beta4: float,
    S_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Net species-richness growth rate `delta(S) = lambda(S) - mu(S)` of a
    colonization-extinction equilibrium model (the theory of island
    biogeography, TIB), where `mu(S) = mu^long(S) + mu^short(S)`:

    - `lambda(S)`: colonization rate, `=lambda_0` at `S=0`, decaying to
      `0` at the saturation richness `S=S_s`.
    - `mu^long(S)`: long-term extinction rate, `=0` at `S=0`, rising to
      `mu_s` at `S=S_s`.
    - `mu^short(S) = beta3*lambda(S)*exp(-beta4*S)`: a short-term
      extinction component proportional to the colonization rate
      (recently-arrived species are more extinction-prone).

    `S_t` is clipped to `S_t = max(0, S_t)` before evaluation, so the
    ODE/root-finder driving `dS/dt = delta(S)` never evaluates the model
    at a negative richness -- this clip protects only this one function
    evaluation, not the ODE trajectory as a whole (an unclipped `S(t)` can
    still run negative between evaluations if `delta(S)` stays negative
    all the way down).

    Returns `(delta_t, lambda_t, mu_long_t, mu_short_t, mu_t)`. Used for
    equilibrium root-finding (`delta_t == 0`), ODE trajectory integration
    (`dS/dt = delta_t`), and comparative-statics sweeps over the model
    parameters.

    Not ported from a standalone MATLAB HSF toolbox library function -- no
    `hsf/fn_delta_t.m` file exists in `hfs-archive`. The identical model
    (`fn_lambda_t`, `fn_mu_long_t`, `fn_mu_short_t`, `fn_delta_t`) does
    appear as local (embedded) functions inside `HSF/16. Exercise
    Solution/chap16_chap5_exercise2.m`, of which HSF-Notebooks chapter 16c
    is itself a direct Python port; promoted here from that notebook.
    """
    S_t = np.maximum(0.0, np.asarray(S_t, dtype=float))
    lambda_t = _fn_lambda_t(S_t, lambda_0, beta1, S_s)
    mu_long_t = _fn_mu_long_t(S_t, mu_s, beta2, S_s)
    mu_short_t = _fn_mu_short_t(S_t, lambda_t, beta3, beta4, S_s)
    mu_t = mu_long_t + mu_short_t
    delta_t = lambda_t - mu_t
    return delta_t, lambda_t, mu_long_t, mu_short_t, mu_t
